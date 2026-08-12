"""Gene (feature) presence matrix from per-study measured-gene lists.

For multi-study atlases each study measures a different subset of genes; genes
a study did not measure are *structural* zeros (not biological zeros) and must
be masked out of a reconstruction loss (BAScVI ``feature_presence_mask``:
``reconst_loss * feature_presence_mask``).

Presence is derived from each study's **var.csv** (the genes that study actually
measured) -- NOT from expression non-zeros, because a measured gene that simply
isn't expressed in any cell would otherwise be wrongly marked absent.

    present[row, gene] = 1  iff  gene is in the var.csv of that row's study

Row granularity
---------------
Rows are indexed by the integer obs column named in ``row_col``:

  * ``"sample_id"`` (default) -> ``[n_samples, n_genes]``
  * ``"study_idx"``           -> ``[n_studies, n_genes]``

Because presence is derived from a **per-study** var.csv, every sample within a
study gets an identical row. Sample granularity therefore stores the same
information redundantly -- on a typical integrated atlas with many samples per
study the redundancy factor is (n_samples / n_studies), often 10-50x -- and is
only worth paying for if per-sample gene panels are introduced later. Study
granularity is the compact equivalent otherwise.

Output: SciPy sparse ``.npz`` (CSR, uint8), gene columns in atlas
``var.parquet`` order. Load with ``scipy.sparse.load_npz(path).toarray()`` and
index ``[row_col_value, :]``.

Note on BAScVI
--------------
This is **not** interchangeable with the matrix BAScVI's ``ZdataDataModule``
loads. That datamodule reads a dense ``feature_presence_matrix.npy`` of shape
``[n_studies, n_genes]`` (float32) written by its own
``scripts/build_zdata_feature_presence.py``, and indexes it by ``study_idx``.
To feed it from here, build with ``row_col="study_idx"`` and convert:
``np.save(path, load_npz(out).toarray().astype(np.float32))``.
"""
from __future__ import annotations

import csv
import gzip
import os

import numpy as np
import polars as pl
import scipy.sparse as sp


def _read_var_genes(var_path: str) -> list[str]:
    """Read the 'gene' column from a var.csv[.gz] (header with a 'gene' field)."""
    op = gzip.open if var_path.endswith(".gz") else open
    with op(var_path, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            gi = header.index("gene")
        except ValueError:
            gi = len(header) - 1  # fall back to last column
        return [row[gi] for row in reader if row]


def _index_var_folders(var_source_dirs) -> dict[str, str]:
    """Map folder basename -> path to its var.csv[.gz], scanning the given dirs."""
    folders: dict[str, str] = {}
    for d in var_source_dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            sub = os.path.join(d, name)
            if not os.path.isdir(sub):
                continue
            for fn in ("var.csv.gz", "var.csv"):
                p = os.path.join(sub, fn)
                if os.path.exists(p):
                    folders[name] = p
                    break
    return folders


def _resolve_folder(study: str, folders: dict[str, str], overrides: dict | None):
    if overrides and study in overrides and overrides[study] in folders:
        return folders[overrides[study]]
    for cand in (study,
                 study.removeprefix("external_") if study.startswith("external_") else None,
                 study.removeprefix("internal_") if study.startswith("internal_") else None):
        if cand and cand in folders:
            return folders[cand]
    return None


def build_feature_presence_matrix(
    zdata_path: str,
    var_source_dirs,
    sample_col: str = "sample_id",
    row_col: str | None = None,
    study_col: str = "study_name",
    output_filename: str = "feature_presence_matrix.npz",
    folder_overrides: dict | None = None,
    obs_filename: str = "obs.parquet",
    var_filename: str = "var.parquet",
    verbose: bool = True,
):
    """Build a [n_samples x n_genes] presence matrix from per-study var.csv lists.

    Parameters
    ----------
    zdata_path
        Built zdata directory (obs.parquet + var.parquet).
    var_source_dirs
        One or more directories, each containing ``<study_folder>/var.csv[.gz]``
        (e.g. the per-study MTX export dirs). A study folder is matched to a
        ``study_col`` value by basename, stripping an ``external_``/``internal_``
        prefix; use ``folder_overrides={study_name: folder_name}`` for the rest.
    row_col
        Integer obs column whose values index the rows of the output. Use
        ``"sample_id"`` for a per-sample matrix or ``"study_idx"`` for the
        compact per-study equivalent (see the module docstring). Defaults to
        ``sample_col`` for backward compatibility.
    sample_col
        Deprecated alias for ``row_col``; retained so existing callers keep
        working. ``row_col`` wins when both are given.
    study_col
        obs column naming each cell's study (used to look up its var.csv).
    """
    if isinstance(var_source_dirs, (str, os.PathLike)):
        var_source_dirs = [var_source_dirs]
    row_col = row_col or sample_col

    ref_genes = pl.read_parquet(os.path.join(zdata_path, var_filename))["gene"].to_list()
    gidx = {g: i for i, g in enumerate(ref_genes)}
    n_genes = len(ref_genes)

    # dict.fromkeys dedupes while preserving order: row_col may name the same
    # column as study_col, which polars rejects as a duplicate selection.
    obs = pl.read_parquet(os.path.join(zdata_path, obs_filename),
                          columns=list(dict.fromkeys([study_col, row_col])))
    samp = obs[row_col].to_numpy()
    if not np.issubdtype(samp.dtype, np.integer):
        raise TypeError(f"row_col '{row_col}' must be integer; got {samp.dtype}")
    n_samples = int(samp.max()) + 1
    study = obs[study_col].to_numpy()

    folders = _index_var_folders(var_source_dirs)
    if verbose:
        print(f"feature_presence(var.csv): {n_samples} rows ({row_col}), {n_genes} ref genes, "
              f"{len(folders)} study folders found", flush=True)

    present = np.zeros((n_samples, n_genes), dtype=bool)
    # study -> the sample_ids belonging to it
    sdf = pl.DataFrame({"_sid": samp, "_st": study}).unique()
    study_to_sids: dict[str, list[int]] = {}
    for sid, st in sdf.iter_rows():
        study_to_sids.setdefault(st, []).append(int(sid))

    unresolved, fmt_mismatch = [], []
    for st, sids in study_to_sids.items():
        vp = _resolve_folder(st, folders, folder_overrides)
        if vp is None:
            unresolved.append(st)
            present[sids, :] = True   # conservative: don't mask if unknown
            continue
        genes = _read_var_genes(vp)
        cols = [gidx[g] for g in genes if g in gidx]
        if not cols and genes:
            # genes present but none map to the reference symbols -> gene-ID
            # format mismatch (e.g. Ensembl IDs vs symbol reference). Cannot map
            # reliably here; treat as fully measured (don't mask) + flag.
            fmt_mismatch.append(st)
            present[sids, :] = True
            continue
        present[np.ix_(sids, cols)] = True
        if verbose:
            print(f"  {st}: {len(cols)}/{len(genes)} measured genes in ref "
                  f"-> {len(sids)} samples", flush=True)

    if unresolved or fmt_mismatch:
        import warnings
        if unresolved:
            warnings.warn(f"{len(unresolved)} studies had no matching var.csv folder "
                          f"(marked fully-present): {unresolved}")
        if fmt_mismatch:
            warnings.warn(f"{len(fmt_mismatch)} studies' var.csv used non-symbol gene "
                          f"IDs (e.g. Ensembl) -> couldn't map, marked fully-present: "
                          f"{fmt_mismatch}")

    out = sp.csr_matrix(present.astype(np.uint8))
    out_path = os.path.join(zdata_path, output_filename)
    sp.save_npz(out_path, out)
    if verbose:
        print(f"feature_presence: wrote {out_path}  shape={present.shape}  "
              f"density={present.mean():.3f}  unresolved_studies={len(unresolved)}",
              flush=True)
    return present, out_path
