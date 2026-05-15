#!/usr/bin/env python3
"""
Build a complete zdata object from a directory of MTX+CSV subdirectories.

Each subdirectory must contain:
  - matrix.mtx (or matrix.mtx.gz): sparse expression matrix, MatrixMarket format
  - obs.csv (or obs.csv.gz): observation (cell) metadata
  - var.csv (or var.csv.gz): variable (gene) metadata

The build is fully streaming -- aligned matrix data is piped straight into the
``mtx_to_zdata`` compressor over a binary COO pipe, so no intermediate ``.mtx``
text files are ever written to disk:

1. Stream X_RM: each dataset is read, its genes aligned to the standard gene
   list, and the aligned rows are accumulated into fixed-size super-chunks that
   are streamed directly into the compressor.
2. Stream X_CM: the just-built (compressed, seekable) X_RM is read back in
   gene-column slabs, transposed, and streamed into the compressor.
3. obs.parquet is concatenated from the obs.csv files (with per-row nnz /
   total-count stats computed during step 1).
4. var.parquet is written from the standard gene list + per-gene nnz.

Output: a complete .zdata directory with X_RM, X_CM, metadata.json,
obs.parquet, var.parquet.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.io import mmread
from scipy.sparse import csr_matrix, vstack

from zdata.build_zdata.align_mtx import (
    _reorder_matrix_columns,
    get_default_gene_list_path,
)
from zdata.build_zdata.build_x import _stream_csr_to_zdata


# ---------------------------------------------------------------------------
# Input discovery / gzipped-CSV resolution
# ---------------------------------------------------------------------------

def _resolve_csv(subdir: Path, base: str) -> Path:
    """Return the path to ``{base}`` or ``{base}.gz`` inside ``subdir``.

    pandas.read_csv transparently decompresses based on the ``.gz`` suffix, so
    callers just need the resolved path.
    """
    plain = subdir / base
    if plain.exists():
        return plain
    gz = subdir / (base + ".gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"Neither {base} nor {base}.gz found in {subdir}")


def discover_mtx_csv_directories(input_dir: str) -> list[Path]:
    """
    Discover subdirectories containing a matrix, obs and var file.

    Each of obs/var may be plain (``obs.csv``) or gzipped (``obs.csv.gz``);
    the matrix may be ``matrix.mtx`` or ``matrix.mtx.gz``.

    Args:
        input_dir: Path to directory containing subdirectories

    Returns:
        Sorted list of subdirectory paths

    Raises:
        FileNotFoundError: If input_dir does not exist
        ValueError: If no valid subdirectories are found
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_path.is_dir():
        raise ValueError(f"Path is not a directory: {input_dir}")

    valid_dirs = []
    for subdir in sorted(input_path.iterdir()):
        if not subdir.is_dir():
            continue
        has_mtx = bool(list(subdir.glob("matrix.mtx*")))
        has_obs = (subdir / "obs.csv").exists() or (subdir / "obs.csv.gz").exists()
        has_var = (subdir / "var.csv").exists() or (subdir / "var.csv.gz").exists()
        if has_mtx and has_obs and has_var:
            valid_dirs.append(subdir)

    if not valid_dirs:
        raise ValueError(
            f"No valid MTX+CSV subdirectories found in {input_dir}. Each "
            f"subdirectory must contain matrix.mtx, obs.csv and var.csv "
            f"(optionally gzipped)."
        )

    return valid_dirs


def read_gene_list_from_var_csv(var_csv_path: Path) -> list[str]:
    """
    Read gene names from a var.csv (or var.csv.gz) file.

    Expects a CSV with a 'gene' column, or falls back to the index.

    Args:
        var_csv_path: Path to var.csv / var.csv.gz file

    Returns:
        List of gene name strings
    """
    df = pd.read_csv(var_csv_path, index_col=0)
    if "gene" in df.columns:
        return df["gene"].astype(str).tolist()
    return df.index.astype(str).tolist()


# ---------------------------------------------------------------------------
# Alignment: yield aligned CSR row-chunks for one dataset
# ---------------------------------------------------------------------------

def _iter_dataset_aligned_chunks(subdir: Path, gene_list: list[str], n_genes: int,
                                 mtx_chunk_size: int):
    """
    Read one MTX+CSV subdirectory, align its genes to ``gene_list`` and yield
    the aligned matrix in CSR row-chunks of at most ``mtx_chunk_size`` rows.

    Yields:
        csr_matrix chunks of shape (<=mtx_chunk_size, n_genes)
    """
    file_genes = read_gene_list_from_var_csv(_resolve_csv(subdir, "var.csv"))
    gene_to_old = {gene: idx for idx, gene in enumerate(file_genes)}
    old_to_new_idx = {
        gene_to_old[gene]: new_idx
        for new_idx, gene in enumerate(gene_list)
        if gene in gene_to_old
    }
    matched = len(old_to_new_idx)
    print(f"    {len(file_genes)} genes in var.csv; "
          f"{matched}/{n_genes} standard genes matched "
          f"({matched / n_genes * 100:.1f}%)")

    mtx_files = sorted(subdir.glob("matrix.mtx*"))
    X = mmread(str(mtx_files[0]))
    if not isinstance(X, csr_matrix):
        X = X.tocsr()
    # int64 count data -> int32 halves memory; final clamp happens at stream time
    if np.issubdtype(X.data.dtype, np.integer) and X.data.dtype.itemsize > 4:
        X.data = X.data.astype(np.int32)
    n_rows = X.shape[0]
    print(f"    matrix: {n_rows} rows x {X.shape[1]} cols, {X.nnz} nnz")

    for start in range(0, n_rows, mtx_chunk_size):
        end = min(start + mtx_chunk_size, n_rows)
        X_csc = X[start:end].tocsc()
        X_aligned = _reorder_matrix_columns(X_csc, old_to_new_idx, n_genes)
        del X_csc
        gc.collect()
        yield X_aligned
        del X_aligned
        gc.collect()

    del X
    gc.collect()


def _take_rows(buffer: list, n: int):
    """
    Take exactly ``n`` rows off the front of ``buffer`` (a list of CSR
    matrices), splitting a matrix if necessary.

    Returns:
        (super_chunk_csr, remaining_buffer_list)
    """
    taken = []
    taken_rows = 0
    idx = 0
    while idx < len(buffer) and taken_rows < n:
        mat = buffer[idx]
        need = n - taken_rows
        if mat.shape[0] <= need:
            taken.append(mat)
            taken_rows += mat.shape[0]
            idx += 1
        else:
            taken.append(mat[:need])
            buffer[idx] = mat[need:]
            taken_rows += need
            break
    remaining = buffer[idx:]
    super_chunk = vstack(taken, format="csr") if len(taken) > 1 else taken[0].tocsr()
    return super_chunk, remaining


# ---------------------------------------------------------------------------
# Streaming + metadata helpers
# ---------------------------------------------------------------------------

def _stream_and_place(csr, zdata_dir: Path, subdir: str, axis_offset: int,
                      block: int, max_per_chunk: int, dtype: str,
                      meta_list: list, zstd_level: int = 1) -> None:
    """
    Stream ``csr`` into ``mtx_to_zdata`` and move the produced chunk files into
    ``{zdata_dir}/{subdir}/`` with globally-correct chunk numbers.

    ``axis_offset`` is the global index of ``csr``'s first row (cells for X_RM,
    genes for X_CM) and must be a multiple of ``max_per_chunk``.
    """
    if axis_offset % max_per_chunk != 0:
        raise ValueError(
            f"axis_offset ({axis_offset}) must be a multiple of "
            f"max_per_chunk ({max_per_chunk})"
        )
    global_chunk_start = axis_offset // max_per_chunk
    total = csr.shape[0]

    tmp_dir = zdata_dir / ".tmp_stream"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    try:
        local_bins = _stream_csr_to_zdata(
            csr, tmp_dir, subdir, block, max_per_chunk, axis_offset, dtype,
            zstd_level=zstd_level,
        )
        dest_dir = zdata_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        for local_bin in local_bins:
            local = int(local_bin.stem)
            gnum = global_chunk_start + local
            os.replace(str(local_bin), str(dest_dir / f"{gnum}.bin"))
            start = axis_offset + local * max_per_chunk
            end = min(start + max_per_chunk, axis_offset + total)
            n_in_chunk = end - start
            meta_list.append({
                "chunk_num": gnum,
                "file": f"{gnum}.bin",
                "blocks": (n_in_chunk + block - 1) // block,
                "start_row": start,
                "end_row": end,
            })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_metadata(zdata_dir: Path, dtype: str, total_rows: int, n_genes: int,
                    total_nnz: int, rm_chunks: list, cm_chunks: list | None,
                    block_rows: int, block_columns: int,
                    max_rows: int, max_columns: int,
                    source_names: list[str]) -> None:
    """Write metadata.json. If ``cm_chunks`` is None a provisional RM-only file
    is written (enough for ZData to open and read rows during the X_CM pass)."""
    rm_chunks = sorted(rm_chunks, key=lambda c: c["chunk_num"])
    metadata: dict = {
        "version": 1,
        "format": "zdata",
        "dtype": dtype,
        "shape": [total_rows, n_genes],
        "nnz_total": total_nnz,
        "num_chunks_rm": len(rm_chunks),
        "total_blocks_rm": sum(c["blocks"] for c in rm_chunks),
        "blocks_per_chunk": max_rows // block_rows,
        "block_rows": block_rows,
        "block_columns": block_columns,
        "max_rows_per_chunk": max_rows,
        "max_columns_per_chunk": max_columns,
        "chunks_rm": rm_chunks,
        "source_files_rm": list(source_names),
    }
    if cm_chunks is not None:
        cm_chunks = sorted(cm_chunks, key=lambda c: c["chunk_num"])
        metadata["num_chunks_cm"] = len(cm_chunks)
        metadata["total_blocks_cm"] = sum(c["blocks"] for c in cm_chunks)
        metadata["chunks_cm"] = cm_chunks
        metadata["cm_chunk_ranges"] = sorted(
            [[c["start_row"], c["end_row"], c["chunk_num"]] for c in cm_chunks]
        )
        metadata["source_files_cm"] = list(source_names)

    with open(zdata_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Phase 1: stream X_RM
# ---------------------------------------------------------------------------

def _build_x_rm(subdirs: list[Path], gene_list: list[str], zdata_dir: Path,
                block_rows: int, max_rows: int, mtx_chunk_size: int,
                dtype: str, zstd_level: int):
    """
    Stream the row-major store. Returns
    ``(rm_chunks, total_rows, total_nnz, row_nnz, row_counts, column_nnz)``.
    """
    n_genes = len(gene_list)
    rm_chunks: list = []
    buffer: list = []
    buffer_rows = 0
    cursor = 0  # global row index of the next super-chunk (multiple of mtx_chunk_size)

    row_nnz_parts: list = []
    row_counts_parts: list = []
    column_nnz = np.zeros(n_genes, dtype=np.int64)

    rm_dir = zdata_dir / "X_RM"
    if rm_dir.exists():
        shutil.rmtree(rm_dir)
    rm_dir.mkdir(parents=True, exist_ok=True)

    for ds_idx, subdir in enumerate(subdirs):
        print(f"\n  [{ds_idx + 1}/{len(subdirs)}] {subdir.name}")
        for X_aligned in _iter_dataset_aligned_chunks(
            subdir, gene_list, n_genes, mtx_chunk_size
        ):
            row_nnz_parts.append(np.diff(X_aligned.indptr).astype(np.uint32))
            row_counts_parts.append(
                np.asarray(X_aligned.sum(axis=1)).ravel().astype(np.float32)
            )
            column_nnz += X_aligned.getnnz(axis=0).astype(np.int64)

            buffer.append(X_aligned)
            buffer_rows += X_aligned.shape[0]

            while buffer_rows >= mtx_chunk_size:
                super_chunk, buffer = _take_rows(buffer, mtx_chunk_size)
                buffer_rows -= mtx_chunk_size
                _stream_and_place(super_chunk, zdata_dir, "X_RM", cursor,
                                  block_rows, max_rows, dtype, rm_chunks,
                                  zstd_level=zstd_level)
                cursor += mtx_chunk_size
                print(f"    streamed X_RM rows {cursor - mtx_chunk_size}-{cursor - 1}")
                del super_chunk
                gc.collect()

    # flush the final (partial) super-chunk
    if buffer_rows > 0:
        super_chunk = (
            vstack(buffer, format="csr") if len(buffer) > 1 else buffer[0].tocsr()
        )
        _stream_and_place(super_chunk, zdata_dir, "X_RM", cursor,
                          block_rows, max_rows, dtype, rm_chunks,
                          zstd_level=zstd_level)
        print(f"    streamed X_RM rows {cursor}-{cursor + buffer_rows - 1} (final)")
        cursor += buffer_rows
        del super_chunk
        gc.collect()

    total_rows = cursor
    row_nnz = (
        np.concatenate(row_nnz_parts) if row_nnz_parts
        else np.zeros(0, dtype=np.uint32)
    )
    row_counts = (
        np.concatenate(row_counts_parts) if row_counts_parts
        else np.zeros(0, dtype=np.float32)
    )
    total_nnz = int(row_nnz.sum())
    return rm_chunks, total_rows, total_nnz, row_nnz, row_counts, column_nnz


# ---------------------------------------------------------------------------
# Phase 2: stream X_CM (read X_RM back, transpose in gene slabs)
# ---------------------------------------------------------------------------

def _build_x_cm(zdata_dir: Path, total_rows: int, n_genes: int, total_nnz: int,
                block_columns: int, max_columns: int, dtype: str,
                slab_genes: int | None, zstd_level: int):
    """
    Build the column-major store by reading the compressed X_RM back in
    gene-column slabs, transposing each slab and streaming it into the
    compressor. Returns the list of X_CM chunk metadata entries.
    """
    from zdata.core import ZData

    if slab_genes is None:
        # aim for ~3B nnz per slab (~18 GB as CSR); never below one CM chunk
        avg_per_gene = max(total_nnz / max(n_genes, 1), 1.0)
        slab_genes = max(max_columns, int(3_000_000_000 / avg_per_gene))
    # round up to a whole number of CM chunks so chunk numbering stays clean
    slab_genes = ((slab_genes + max_columns - 1) // max_columns) * max_columns
    slab_genes = min(slab_genes, ((n_genes + max_columns - 1) // max_columns) * max_columns)

    cm_dir = zdata_dir / "X_CM"
    if cm_dir.exists():
        shutil.rmtree(cm_dir)
    cm_dir.mkdir(parents=True, exist_ok=True)

    zd = ZData(str(zdata_dir))
    read_block = zd.max_rows_per_chunk * 8  # rows per read-back batch

    cm_chunks: list = []
    n_slabs = (n_genes + slab_genes - 1) // slab_genes
    print(f"  slab width: {slab_genes} genes ({n_slabs} slab(s) over X_RM)")

    for slab_idx, c0 in enumerate(range(0, n_genes, slab_genes)):
        c1 = min(c0 + slab_genes, n_genes)
        parts = []
        for rb in range(0, total_rows, read_block):
            rb_end = min(rb + read_block, total_rows)
            sub = zd.read_rows_csr(slice(rb, rb_end))
            parts.append(sub[:, c0:c1])
            del sub
        slab = vstack(parts, format="csr") if len(parts) > 1 else parts[0].tocsr()
        del parts
        gc.collect()

        slab_t = slab.T.tocsr()  # genes-as-rows
        del slab
        gc.collect()

        _stream_and_place(slab_t, zdata_dir, "X_CM", c0,
                          block_columns, max_columns, dtype, cm_chunks,
                          zstd_level=zstd_level)
        print(f"    streamed X_CM genes {c0}-{c1 - 1} "
              f"(slab {slab_idx + 1}/{n_slabs})")
        del slab_t
        gc.collect()

    return cm_chunks


# ---------------------------------------------------------------------------
# obs.parquet / var.parquet
# ---------------------------------------------------------------------------

def _write_obs_parquet(subdirs: list[Path], zdata_dir: Path, join_strategy: str,
                       output_filename: str, row_nnz: np.ndarray,
                       row_counts: np.ndarray, min_nnz: int | None) -> str:
    """Concatenate obs.csv files, attach per-row stats, optionally filter, and
    write obs.parquet. Row order matches the streamed X_RM row order."""
    print(f"\n  Concatenating obs metadata from {len(subdirs)} subdirectories "
          f"(strategy: {join_strategy})...")

    frames: list[pl.DataFrame] = []
    for subdir in subdirs:
        obs_df = pd.read_csv(_resolve_csv(subdir, "obs.csv"), index_col=0)
        # Pandas 'object' columns frequently mix int/str values; polars
        # rejects those when converting to UTF-8. Coerce to str up-front so
        # the per-column unification pass below has clean dtypes to work with.
        for col in obs_df.columns:
            if obs_df[col].dtype == object:
                obs_df[col] = obs_df[col].astype(str)
        df = pl.from_pandas(obs_df)
        df = df.with_columns(pl.lit(subdir.name).alias("_source_dir"))
        frames.append(df)

    # Per-column union-dtype pass: across the 62 datasets the same column name
    # can appear with different types (e.g. patient_id is int in some files,
    # str in others). polars' diagonal_relaxed only upcasts within numeric
    # types; for mixed numeric/string (or bool/string) we have to cast to a
    # common supertype ourselves before concat or the concat will raise.
    _INT = {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
    _FLOAT = {pl.Float32, pl.Float64}
    _NUMERIC = _INT | _FLOAT

    col_dtypes: dict = {}
    for df in frames:
        for c, dt in df.schema.items():
            col_dtypes.setdefault(c, set()).add(dt)

    target_dtype: dict = {}
    for c, dts in col_dtypes.items():
        if len(dts) == 1:
            target_dtype[c] = next(iter(dts))
        elif all(d in _INT for d in dts):
            target_dtype[c] = pl.Int64
        elif all(d in _NUMERIC for d in dts):
            target_dtype[c] = pl.Float64
        else:
            target_dtype[c] = pl.String

    for i, df in enumerate(frames):
        cast_exprs = []
        for c, dt in df.schema.items():
            tgt = target_dtype[c]
            if dt != tgt:
                cast_exprs.append(pl.col(c).cast(tgt, strict=False))
        if cast_exprs:
            frames[i] = df.with_columns(cast_exprs)

    if join_strategy == "inner":
        common = set(frames[0].columns)
        for df in frames[1:]:
            common &= set(df.columns)
        common.discard("_source_dir")
        ordered = sorted(common)
        selected = [
            df.select(ordered + (["_source_dir"] if "_source_dir" in df.columns else []))
            for df in frames
        ]
        combined = pl.concat(selected, how="diagonal_relaxed")
    else:
        combined = pl.concat(frames, how="diagonal_relaxed")

    if combined.height != len(row_nnz):
        raise ValueError(
            f"obs row count ({combined.height}) does not match streamed row "
            f"count ({len(row_nnz)})"
        )

    combined = combined.with_columns([
        pl.Series("nnz", row_nnz, dtype=pl.UInt32),
        pl.Series("total_counts", row_counts, dtype=pl.Float32),
    ])
    combined = combined.with_columns(
        pl.when(pl.col("total_counts") > 0)
        .then(10000.0 / pl.col("total_counts"))
        .otherwise(None)
        .alias("scaling_factor")
    )
    combined = combined.with_row_index("_row_index")

    if min_nnz is not None and min_nnz > 0:
        before = combined.height
        combined = combined.filter(pl.col("nnz") >= min_nnz)
        print(f"    filtered cells with nnz < {min_nnz}: "
              f"removed {before - combined.height}, kept {combined.height}")

    output_path = zdata_dir / output_filename
    combined.write_parquet(str(output_path), compression="zstd")
    print(f"    wrote {combined.height} rows to {output_path}")
    return str(output_path)


def _write_var_parquet(zdata_dir: Path, gene_list: list[str],
                       column_nnz: np.ndarray) -> str:
    """Write var.parquet from the standard gene list + per-gene nnz counts."""
    if len(column_nnz) != len(gene_list):
        raise ValueError(
            f"column nnz count ({len(column_nnz)}) != gene count "
            f"({len(gene_list)})"
        )
    var_df = pl.DataFrame({
        "gene": gene_list,
        "index": list(range(len(gene_list))),
        "nnz": column_nnz.astype(np.uint32),
    })
    var_path = zdata_dir / "var.parquet"
    var_df.write_parquet(str(var_path), compression="zstd")
    print(f"  var.parquet saved ({len(gene_list)} genes)")
    return str(var_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_zdata_from_mtx_csv(
    input_dir: str,
    output_name: str,
    gene_list_path: str | None = None,
    block_rows: int = 16,
    block_columns: int | None = None,
    max_rows: int = 8192,
    max_columns: int = 256,
    obs_join_strategy: str = "outer",
    obs_output_filename: str = "obs.parquet",
    cleanup_temp: bool = True,
    mtx_chunk_size: int = 131072,
    mtx_temp_dir: str | None = None,
    min_nnz: int | None = 300,
    cm_slab_genes: int | None = None,
    zstd_level: int = 3,
) -> Path:
    """
    Build a complete zdata object from a directory of MTX+CSV subdirectories.

    The build is fully streaming: aligned matrix data is piped straight into the
    compressor, so no intermediate ``.mtx`` text files touch disk regardless of
    dataset size.

    Args:
        input_dir: Directory containing subdirectories with MTX+CSV files.
        output_name: Output directory name for the zdata object.
        gene_list_path: Path to standard gene list (default: package default).
        block_rows: Rows per block for row-major (X_RM) files.
        block_columns: Rows per block for column-major (X_CM) files
            (default: same as block_rows).
        max_rows: Max rows per X_RM chunk file.
        max_columns: Max genes per X_CM chunk file.
        obs_join_strategy: "inner" or "outer" join for obs columns.
        obs_output_filename: Name of the obs parquet file.
        cleanup_temp: Accepted for backwards compatibility (the streaming build
            writes no intermediate files, so there is nothing to clean up).
        mtx_chunk_size: Rows per streamed X_RM super-chunk. Rounded up to a
            multiple of ``max_rows``.
        mtx_temp_dir: Accepted for backwards compatibility; unused.
        min_nnz: Minimum nnz for cell filtering in obs.parquet (None to disable).
        cm_slab_genes: Genes per X_CM build slab (default: auto-sized from the
            matrix density). Rounded up to a multiple of ``max_columns``.
        zstd_level: zstd compression level for the chunk archives (1-22).
            Higher = smaller output, more CPU. The archives stay seekable at
            any level.

    Returns:
        Path to the created zdata directory.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_path.is_dir():
        raise ValueError(f"Path is not a directory: {input_dir}")

    if block_columns is None:
        block_columns = block_rows
    # super-chunks must be a whole number of X_RM chunks for clean numbering
    if mtx_chunk_size % max_rows != 0:
        mtx_chunk_size = ((mtx_chunk_size + max_rows - 1) // max_rows) * max_rows

    subdirs = discover_mtx_csv_directories(input_dir)

    if gene_list_path is None:
        gene_list_path = str(get_default_gene_list_path())
    if not os.path.exists(gene_list_path):
        raise FileNotFoundError(f"Gene list file not found: {gene_list_path}")
    with open(gene_list_path, "r") as f:
        gene_list = [line.strip() for line in f if line.strip()]
    if not gene_list:
        raise ValueError(f"No genes found in {gene_list_path}")
    n_genes = len(gene_list)

    dtype = "uint16"  # MTX count data; matches the zarr/h5ad pipelines

    print("=" * 70)
    print("Building zdata from MTX+CSV directories (streaming)")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_name}")
    print(f"Subdirectories:   {len(subdirs)}")
    print(f"Standard genes:   {n_genes}")

    zdata_dir = Path(output_name)
    zdata_dir.mkdir(parents=True, exist_ok=True)
    # clear any stale metadata/parquet from a previous build
    for stale in zdata_dir.glob("*.json"):
        stale.unlink()
    for stale in zdata_dir.glob("*.parquet"):
        stale.unlink()

    # --- Phase 1: stream X_RM ------------------------------------------------
    print(f"\n{'=' * 70}\nPhase 1: streaming X_RM\n{'=' * 70}")
    (rm_chunks, total_rows, total_nnz,
     row_nnz, row_counts, column_nnz) = _build_x_rm(
        subdirs, gene_list, zdata_dir, block_rows, max_rows, mtx_chunk_size,
        dtype, zstd_level
    )
    print(f"\n  X_RM complete: {total_rows} rows, {n_genes} genes, "
          f"{total_nnz} nnz, {len(rm_chunks)} chunk files")

    # provisional metadata + obs/var so ZData can open the half-built atlas
    source_names = [d.name for d in subdirs]
    _write_metadata(zdata_dir, dtype, total_rows, n_genes, total_nnz,
                    rm_chunks, None, block_rows, block_columns,
                    max_rows, max_columns, source_names)
    pl.DataFrame({"_row_index": np.arange(total_rows, dtype=np.uint32)}).write_parquet(
        str(zdata_dir / "obs.parquet"), compression="zstd"
    )
    pl.DataFrame({
        "gene": gene_list,
        "index": list(range(n_genes)),
    }).write_parquet(str(zdata_dir / "var.parquet"), compression="zstd")

    # --- Phase 2: stream X_CM ------------------------------------------------
    print(f"\n{'=' * 70}\nPhase 2: streaming X_CM (transpose via X_RM read-back)\n{'=' * 70}")
    cm_chunks = _build_x_cm(zdata_dir, total_rows, n_genes, total_nnz,
                            block_columns, max_columns, dtype, cm_slab_genes,
                            zstd_level)
    print(f"\n  X_CM complete: {len(cm_chunks)} chunk files")

    # --- final metadata ------------------------------------------------------
    _write_metadata(zdata_dir, dtype, total_rows, n_genes, total_nnz,
                    rm_chunks, cm_chunks, block_rows, block_columns,
                    max_rows, max_columns, source_names)

    # --- obs.parquet / var.parquet ------------------------------------------
    print(f"\n{'=' * 70}\nPhase 3: obs.parquet / var.parquet\n{'=' * 70}")
    _write_obs_parquet(subdirs, zdata_dir, obs_join_strategy, obs_output_filename,
                       row_nnz, row_counts, min_nnz)
    _write_var_parquet(zdata_dir, gene_list, column_nnz)

    print(f"\n{'=' * 70}")
    print("zdata object built successfully from MTX+CSV")
    print("=" * 70)
    print(f"Output directory: {zdata_dir}")
    for label in ("metadata.json", "obs.parquet", "var.parquet"):
        if (zdata_dir / label).exists():
            print(f"  ok {label}")
    for sub in ("X_RM", "X_CM"):
        d = zdata_dir / sub
        if d.exists():
            print(f"  ok {sub}/ ({len(list(d.glob('*.bin')))} chunk files)")

    return zdata_dir


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build a zdata object from a directory of MTX+CSV "
        "subdirectories. Each subdirectory must contain matrix.mtx, obs.csv "
        "and var.csv (optionally gzipped). The build is fully streaming -- no "
        "intermediate .mtx files are written to disk."
    )
    parser.add_argument("input_dir", type=str,
                        help="Directory of subdirectories with matrix.mtx, "
                             "obs.csv, var.csv")
    parser.add_argument("output_name", type=str,
                        help='Output directory name (e.g. "atlas.zdata")')
    parser.add_argument("--gene-list", type=str, default=None,
                        help="Path to standard gene list (default: package default)")
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument("--block-columns", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=8192)
    parser.add_argument("--max-columns", type=int, default=256)
    parser.add_argument("--obs-join-strategy", choices=["inner", "outer"],
                        default="outer")
    parser.add_argument("--min-nnz", type=int, default=300,
                        help="Minimum nnz for cell filtering (0 to disable)")
    parser.add_argument("--mtx-chunk-size", type=int, default=131072,
                        help="Rows per streamed X_RM super-chunk")
    parser.add_argument("--cm-slab-genes", type=int, default=None,
                        help="Genes per X_CM build slab (default: auto)")
    parser.add_argument("--zstd-level", type=int, default=3,
                        help="zstd compression level 1-22 (default: 3)")

    args = parser.parse_args()

    try:
        zdata_dir = build_zdata_from_mtx_csv(
            args.input_dir,
            args.output_name,
            gene_list_path=args.gene_list,
            block_rows=args.block_rows,
            block_columns=args.block_columns,
            max_rows=args.max_rows,
            max_columns=args.max_columns,
            obs_join_strategy=args.obs_join_strategy,
            mtx_chunk_size=args.mtx_chunk_size,
            min_nnz=args.min_nnz if args.min_nnz > 0 else None,
            cm_slab_genes=args.cm_slab_genes,
            zstd_level=args.zstd_level,
        )
        print(f"\nBuild complete! Output: {zdata_dir}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
