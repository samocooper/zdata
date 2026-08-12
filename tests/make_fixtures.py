#!/usr/bin/env python3
"""Generate the synthetic test fixtures used by the zdata test-suite.

The fixtures are **entirely synthetic**: random counts with generated barcodes,
gene symbols and study names. Nothing here derives from a real dataset, so the
repository carries no third-party expression data, no donor identifiers, and no
curation flags -- and a clone stays small.

Regenerate with::

    python tests/make_fixtures.py

Output (all under ``tests/``)::

    h5ad_test_dir/<name>.h5          AnnData HDF5
    mtx_test_dir/<name>/             matrix.mtx + obs.csv + var.csv
    zarr_test_dir/<name>.zarr/       AnnData Zarr

Determinism matters: a fixed seed keeps fixtures byte-stable across
regenerations, so they do not churn in git.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite

TESTS_DIR = Path(__file__).parent

# Deliberately small. The suite exercises block/chunk boundaries, not scale.
# 5 x 256 = 1280 cells matches the dimensions the tests assert on; density is
# set just high enough that every gene slab of the column-major build is
# non-empty, and no higher (it dominates fixture size).
N_DATASETS = 5
N_CELLS = 256
# Genes span the FULL reference list rather than a prefix: the column-major
# build processes fixed-width gene slabs and errors on an empty one, so the
# data must be spread across the whole aligned gene space.
N_GENES = None          # None -> use the entire reference gene list
DENSITY = 0.002
SEED = 20240101

# Synthetic study names. The mixed prefixes exercise the external_/internal_
# stripping in feature_presence without naming any real study.
STUDY_NAMES = [
    "external_alpha_2020_00000001",
    "external_beta_2021_00000002",
    "internal_gamma_2022",
    "external_delta_2023_00000004",
]

CELL_TYPES = ["T cell", "B cell", "Macrophage", "Fibroblast", "Endothelial"]
TISSUES = ["liver", "lung", "colon", "kidney"]
PROTOCOLS = ["10x_3prime_v3", "10x_5prime_v2"]


def _gene_names(n: int) -> list[str]:
    """First ``n`` symbols from the shipped reference gene list.

    These must be *real* symbols, not invented ones: the build pipeline aligns
    each dataset to ``files/2ks10c_genes.txt``, and genes absent from that list
    are dropped -- invented symbols would align to an all-zero matrix. Gene
    symbols are public reference data, not dataset content.
    """
    gene_list = TESTS_DIR.parent / "files" / "2ks10c_genes.txt"
    with open(gene_list) as f:
        symbols = [line.strip() for line in f if line.strip()]
    if n is None:
        return symbols
    if len(symbols) < n:
        raise RuntimeError(f"reference gene list has only {len(symbols)} symbols")
    return symbols[:n]


def _barcodes(n: int, suffix: int) -> list[str]:
    """Deterministic pseudo-barcodes in the usual 16bp-plus-lane shape."""
    rng = np.random.default_rng(SEED + suffix)
    letters = np.array(list("ACGT"))
    return [
        "".join(rng.choice(letters, 16)) + f"-1_{suffix}"
        for _ in range(n)
    ]


def _make_adata(idx: int) -> ad.AnnData:
    rng = np.random.default_rng(SEED + idx)
    genes = _gene_names(N_GENES)
    n_genes = len(genes)
    X = sp.random(N_CELLS, n_genes, density=DENSITY, format="csr",
                  random_state=SEED + idx)
    X.data = rng.integers(1, 500, size=X.data.shape[0]).astype(np.float32)

    n_samples = 2 + (idx % 2)
    sample_idx = rng.integers(0, n_samples, N_CELLS)

    obs = pd.DataFrame({
        "barcode": _barcodes(N_CELLS, idx),
        "study_name": STUDY_NAMES[idx % len(STUDY_NAMES)],
        "sample_name": [f"sample_{s}" for s in sample_idx],
        "sample_idx": sample_idx.astype(np.int64),
        "donor_id": [f"donor_{s}" for s in sample_idx],
        "standard_true_celltype": rng.choice(CELL_TYPES, N_CELLS),
        "authors_celltype": rng.choice(CELL_TYPES, N_CELLS),
        "cells_or_nuclei": rng.choice(["cells", "nuclei"], N_CELLS),
        "tissue_collected": rng.choice(TISSUES, N_CELLS),
        "tissue_site": rng.choice(TISSUES, N_CELLS),
        "scrnaseq_protocol": rng.choice(PROTOCOLS, N_CELLS),
        "nnz": np.asarray((X > 0).sum(axis=1)).ravel().astype(np.int64),
        "soma_joinid": np.arange(N_CELLS, dtype=np.int64),
        "batch_name": f"batch_{idx}",
    })
    obs.index = obs["barcode"].to_numpy()

    var = pd.DataFrame({"gene": genes})
    var.index = var["gene"].to_numpy()

    return ad.AnnData(X=X, obs=obs, var=var)


def main() -> None:
    h5ad_dir = TESTS_DIR / "h5ad_test_dir"
    mtx_dir = TESTS_DIR / "mtx_test_dir"
    zarr_dir = TESTS_DIR / "zarr_test_dir"
    for d in (h5ad_dir, mtx_dir, zarr_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for i in range(N_DATASETS):
        name = f"synthetic_{i:02d}_test"
        adata = _make_adata(i)

        adata.write_h5ad(h5ad_dir / f"{name}.h5", compression="gzip")
        adata.write_zarr(zarr_dir / f"{name}.zarr")

        sub = mtx_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        mmwrite(str(sub / "matrix.mtx"), adata.X.astype(np.int32))
        adata.obs.to_csv(sub / "obs.csv", index=False)
        adata.var.to_csv(sub / "var.csv", index=False)

        print(f"  {name}: {adata.n_obs} cells x {adata.n_vars} genes, {adata.X.nnz} nnz")

    total = sum(f.stat().st_size for d in (h5ad_dir, mtx_dir, zarr_dir)
                for f in d.rglob("*") if f.is_file())
    n_files = sum(1 for d in (h5ad_dir, mtx_dir, zarr_dir)
                  for f in d.rglob("*") if f.is_file())
    print(f"\nwrote {n_files} files, {total/1e6:.2f} MB total")


if __name__ == "__main__":
    main()
