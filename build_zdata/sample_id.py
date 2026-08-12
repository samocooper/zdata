"""Globally-unique, monotonic sample identifier for multi-study atlases.

A per-study ``sample_name`` (or a within-study ``sample_idx``) is NOT unique
across an integrated atlas: the same sample id recurs in different studies, and
some studies leave ``sample_name`` null. ML batch correction (and the
per-sample feature-presence matrix) needs one globally-unique sample key.

This adds two columns to obs.parquet:

  * ``sample_id``  : monotonically increasing integer, unique per
    (study, effective-sample), ordered by study then sample (so a study's
    samples occupy a contiguous id range).
  * ``sample_uid`` : human-readable unique string; the effective sample name,
    prefixed with a short study token only where the bare name collides across
    studies (e.g. ``sc-PSC014`` / ``sn-PSC014``).

Effective sample per cell, in order of preference:
    sample_name  ->  "donor:" + donor_id  ->  "sidx:" + sample_idx
(so null/blank sample_name is recovered from donor, else the within-study idx).

Implementation note: the derivation is fully vectorised in polars. An earlier
row-at-a-time version allocated three Python object arrays of length n_cells,
which at atlas scale (2e7 cells) cost minutes of wall time and gigabytes of
pointers.
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Values that count as "no sample name recorded".
_BLANK_TOKENS = ("", "nan", "none", "null", "na")

# Separator between study and sample in the internal grouping key. Unit
# Separator (0x1F) cannot occur in a study or sample name.
_KEY_SEP = "\x1f"


def _study_token(study: str, external_prefix: str = "external_") -> str:
    """Short human-readable token for a study, used to break uid collisions."""
    if study.endswith("_scRNA"):
        return "sc"
    if study.endswith("_snRNA"):
        return "sn"
    s = study.removeprefix(external_prefix)
    return s.split("_")[0]


def _blank_mask(col: str) -> pl.Expr:
    """True where the column holds no usable value."""
    s = pl.col(col).cast(pl.String, strict=False).str.strip_chars()
    return s.is_null() | s.str.to_lowercase().is_in(_BLANK_TOKENS)


def _effective_sample_expr(
    sample_col: str, donor_col: str | None, fallback_idx_col: str | None,
    available: set[str],
) -> pl.Expr:
    """sample_name -> 'donor:'+donor_id -> 'sidx:'+sample_idx, first non-blank."""
    expr = (
        pl.when(_blank_mask(sample_col))
        .then(None)
        .otherwise(pl.col(sample_col).cast(pl.String, strict=False).str.strip_chars())
    )
    if donor_col and donor_col in available:
        expr = expr.fill_null(
            pl.when(_blank_mask(donor_col))
            .then(None)
            .otherwise(
                pl.concat_str([
                    pl.lit("donor:"),
                    pl.col(donor_col).cast(pl.String, strict=False).str.strip_chars(),
                ])
            )
        )
    if fallback_idx_col and fallback_idx_col in available:
        expr = expr.fill_null(
            pl.concat_str([
                pl.lit("sidx:"),
                pl.col(fallback_idx_col).cast(pl.String, strict=False),
            ])
        )
    return expr.fill_null(pl.lit("sidx:unknown")).alias("_eff")


def assign_global_sample_id(
    zdata_path: str,
    study_col: str = "study_name",
    sample_col: str = "sample_name",
    donor_col: str | None = "donor_id",
    fallback_idx_col: str | None = "sample_idx",
    id_col: str = "sample_id",
    uid_col: str = "sample_uid",
    external_prefix: str = "external_",
    obs_filename: str = "obs.parquet",
    overwrite: bool = False,
    verbose: bool = True,
) -> int:
    """Add ``sample_id`` / ``sample_uid`` columns to a built zdata obs.parquet.

    Parameters
    ----------
    overwrite
        Recompute and replace the columns if they already exist. Default False
        raises instead, so a re-run cannot silently leave stale ids in place.

    Returns
    -------
    int
        Number of unique samples assigned.
    """
    obs_path = os.path.join(zdata_path, obs_filename)

    schema = pl.read_parquet_schema(obs_path)
    available = set(schema)
    if study_col not in available:
        raise ValueError(f"study_col '{study_col}' not found in {obs_filename}")
    if sample_col not in available:
        raise ValueError(f"sample_col '{sample_col}' not found in {obs_filename}")

    existing = [c for c in (id_col, uid_col) if c in available]
    if existing and not overwrite:
        raise ValueError(
            f"column(s) {existing} already present in {obs_filename}; "
            f"pass overwrite=True to recompute"
        )

    read_cols = [c for c in (study_col, sample_col, donor_col, fallback_idx_col)
                 if c and c in available]
    o = pl.read_parquet(obs_path, columns=read_cols)
    n = o.height

    # --- effective sample + grouping key (vectorised) ------------------------
    o = o.with_columns(
        _effective_sample_expr(sample_col, donor_col, fallback_idx_col, available)
    ).with_columns(
        pl.concat_str([pl.col(study_col).cast(pl.String), pl.col("_eff")],
                      separator=_KEY_SEP).alias("_key")
    )

    # --- monotonic id from sorted (study, effective-sample) ------------------
    uniq = (o.select("_key").unique().sort("_key")
              .with_row_index(name=id_col))
    o = o.join(uniq, on="_key", how="left")
    n_samples = uniq.height

    # --- readable uid, disambiguating cross-study collisions -----------------
    # A bare sample name that occurs in more than one study is ambiguous; for
    # external studies we prefix a short study token. Only the colliding names
    # need per-study treatment, so this stays a small join.
    collisions = (
        o.select(["_eff", study_col]).unique()
         .group_by("_eff").agg(pl.col(study_col).n_unique().alias("_nstudies"))
         .filter(pl.col("_nstudies") > 1)
         .select("_eff")
    )
    n_collisions = collisions.height

    if n_collisions:
        studies = o.select(study_col).unique().to_series().to_list()
        token_map = pl.DataFrame({
            study_col: studies,
            "_token": [_study_token(str(s), external_prefix) for s in studies],
        })
        o = (o.join(collisions.with_columns(pl.lit(True).alias("_collides")),
                    on="_eff", how="left")
               .join(token_map, on=study_col, how="left")
               .with_columns(
                   pl.when(
                       pl.col("_collides").fill_null(False)
                       & pl.col(study_col).cast(pl.String).str.starts_with(external_prefix)
                   )
                   .then(pl.concat_str([pl.col("_token"), pl.col("_eff")], separator="-"))
                   .otherwise(pl.col("_eff"))
                   .alias(uid_col)
               ))
    else:
        o = o.with_columns(pl.col("_eff").alias(uid_col))

    sample_id = o[id_col].to_numpy()
    uid = o[uid_col].to_list()

    # ids run 0..n_samples-1, so the count bounds the largest value.
    if n_samples > np.iinfo(np.int32).max:
        raise ValueError(
            f"{n_samples} samples exceeds int32 range for column '{id_col}'"
        )

    if verbose:
        print(f"assign_global_sample_id: {n_samples:,} unique samples "
              f"(disambiguated {n_collisions} colliding names)", flush=True)

    _append_columns(obs_path, id_col, uid_col, sample_id, uid, n,
                    drop_existing=existing)
    return n_samples


def _append_columns(obs_path, id_col, uid_col, sample_id, uid, n, drop_existing=()):
    """Stream-append the two columns, preserving dtypes/encoding of the rest."""
    pf = pq.ParquetFile(obs_path)
    names = [c for c in pf.schema_arrow.names if c not in drop_existing]
    tmp = obs_path + ".sid.tmp"
    w = None
    off = 0
    try:
        for b in pf.iter_batches(batch_size=500_000):
            t = pa.Table.from_batches([b])
            if drop_existing:
                t = t.select(names)
            nb = t.num_rows
            t = t.append_column(id_col, pa.array(sample_id[off:off + nb], type=pa.int32()))
            t = t.append_column(uid_col, pa.array(uid[off:off + nb], type=pa.large_string()))
            if w is None:
                # Dictionary-encode everything except a delta-friendly row index.
                dict_cols = [c for c in names if c != "_row_index"] + [uid_col]
                kwargs = {"compression": "zstd", "use_dictionary": dict_cols}
                if "_row_index" in names:
                    kwargs["column_encoding"] = {"_row_index": "DELTA_BINARY_PACKED"}
                w = pq.ParquetWriter(tmp, t.schema, **kwargs)
            w.write_table(t)
            off += nb
        if off != n:
            raise RuntimeError(
                f"row count mismatch while rewriting {obs_path}: wrote {off}, expected {n}"
            )
    except BaseException:
        if w is not None:
            w.close()
            w = None
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    finally:
        if w is not None:
            w.close()
    os.replace(tmp, obs_path)
