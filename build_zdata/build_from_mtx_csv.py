#!/usr/bin/env python3
"""
Build complete zdata object from a directory of MTX+CSV subdirectories.

Each subdirectory should contain:
  - matrix.mtx: sparse expression matrix in MatrixMarket format
  - obs.csv: observation (cell) metadata
  - var.csv: variable (gene) metadata

This module orchestrates the full pipeline:
1. Reads each subdirectory's matrix.mtx and var.csv to determine gene lists
2. Aligns genes to a standard gene list and writes aligned MTX files
3. Compresses aligned MTX files into zdata format (via build_x)
4. Concatenates obs.csv files into obs.parquet
5. Writes var.parquet from the standard gene list

Input: Directory of subdirectories, each containing matrix.mtx, obs.csv, var.csv
Output: Complete .zdata directory with X_RM, X_CM, metadata.json, obs.parquet, var.parquet
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, csr_matrix, hstack, vstack

from zdata.build_zdata.align_mtx import (
    _reorder_matrix_columns,
    create_column_major_fragments,
    get_default_gene_list_path,
)
from zdata.build_zdata.build_x import build_zdata
import pandas as pd


def discover_mtx_csv_directories(input_dir: str) -> list[Path]:
    """
    Discover subdirectories containing matrix.mtx, obs.csv, and var.csv.

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
        # Check for required files (matrix.mtx may have been gzipped by mmwrite)
        mtx_files = list(subdir.glob("matrix.mtx*"))
        has_obs = (subdir / "obs.csv").exists()
        has_var = (subdir / "var.csv").exists()
        if mtx_files and has_obs and has_var:
            valid_dirs.append(subdir)

    if not valid_dirs:
        raise ValueError(
            f"No valid MTX+CSV subdirectories found in {input_dir}. "
            f"Each subdirectory must contain matrix.mtx, obs.csv, and var.csv."
        )

    return valid_dirs


def read_gene_list_from_var_csv(var_csv_path: Path) -> list[str]:
    """
    Read gene names from a var.csv file.

    Expects a CSV with a 'gene' column or uses the index if no 'gene' column exists.

    Args:
        var_csv_path: Path to var.csv file

    Returns:
        List of gene name strings
    """
    df = pd.read_csv(var_csv_path, index_col=0)
    if "gene" in df.columns:
        return df["gene"].tolist()
    # Fall back to the index (which is often the gene name)
    return df.index.tolist()


def align_mtx_csv_directory_to_mtx(
    input_dir: str,
    gene_list_path: str,
    output_dir: str,
    chunk_size: int = 131072,
) -> str:
    """
    Read MTX+CSV subdirectories, align genes to a standard gene list, and write
    aligned MTX files in the same format produced by align_zarr_directory_to_mtx.

    Args:
        input_dir: Directory containing subdirectories with matrix.mtx, obs.csv, var.csv
        gene_list_path: Path to standard gene list (one gene per line)
        output_dir: Directory where aligned MTX files will be written
        chunk_size: Maximum rows per aligned MTX file (default: 131072)

    Returns:
        Path to manifest.json
    """
    # Load standard gene list
    with open(gene_list_path, "r") as f:
        gene_list = [line.strip() for line in f if line.strip()]
    if not gene_list:
        raise ValueError(f"No genes found in {gene_list_path}")

    n_new_cols = len(gene_list)
    print(f"Standard gene list contains {n_new_cols} genes")

    # Discover input directories
    subdirs = discover_mtx_csv_directories(input_dir)
    print(f"Found {len(subdirs)} MTX+CSV subdirectories to process")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    mtx_output_dir = os.path.join(output_dir, "rm_mtx_files")
    os.makedirs(mtx_output_dir, exist_ok=True)

    print(f"\nProcessing MTX+CSV files and creating aligned MTX files (max {chunk_size} rows per file)")

    output_files = []
    manifest_data = []
    column_nnz_accumulator = np.zeros(n_new_cols, dtype=np.uint32)

    # Accumulator for chunking across multiple input files
    current_chunk_rows: list[csr_matrix] = []
    current_chunk_sources: list[dict] = []
    current_row_start = 0

    def _write_mtx_chunk(chunk_rows, chunk_sources, row_start):
        """Write accumulated rows as a single aligned MTX file with stats."""
        if not chunk_rows:
            return row_start, None, np.zeros(n_new_cols, dtype=np.uint32)

        combined = vstack(chunk_rows, format="csr")
        del chunk_rows
        gc.collect()

        row_end = row_start + combined.shape[0] - 1
        chunk_path = os.path.join(mtx_output_dir, f"rows_{row_start}_{row_end}.mtx")

        row_nnz = np.diff(combined.indptr).astype(np.uint32)
        col_nnz = combined.getnnz(axis=0).astype(np.uint32)
        row_total_counts = np.array(combined.sum(axis=1)).flatten().astype(np.float32)

        print(f"  Writing MTX file: {os.path.basename(chunk_path)}")
        mmwrite(chunk_path, combined)
        print(f"  ✓ {combined.shape[0]} rows × {n_new_cols} cols, {combined.nnz} non-zeros")

        # Row stats file (nnz + total_counts)
        stats_path = os.path.join(mtx_output_dir, f"rows_{row_start}_{row_end}_stats.txt")
        stats_data = np.column_stack([row_nnz, row_total_counts])
        np.savetxt(stats_path, stats_data, fmt="%u %.6f", delimiter="\t",
                   header="nnz\ttotal_counts", comments="")

        entry = {
            "mtx_file": os.path.basename(chunk_path),
            "mtx_path": chunk_path,
            "row_start": row_start,
            "row_end": row_end,
            "n_rows": combined.shape[0],
            "row_stats_file": os.path.basename(stats_path),
            "source_files": list(chunk_sources),
        }

        del combined
        gc.collect()

        return row_end + 1, entry, col_nnz

    for dir_idx, subdir in enumerate(subdirs):
        print(f"\n[{dir_idx + 1}/{len(subdirs)}] Processing MTX+CSV: {subdir.name}")

        # Read gene list from var.csv
        file_genes = read_gene_list_from_var_csv(subdir / "var.csv")
        print(f"  {len(file_genes)} genes in var.csv")

        # Build column-reorder mapping: old column index → new column index
        gene_to_old_idx = {gene: idx for idx, gene in enumerate(file_genes)}
        old_to_new_idx = {}
        for new_idx, gene in enumerate(gene_list):
            if gene in gene_to_old_idx:
                old_to_new_idx[gene_to_old_idx[gene]] = new_idx

        matched = len(old_to_new_idx)
        print(f"  {matched}/{n_new_cols} standard genes matched ({matched / n_new_cols * 100:.1f}%)")

        # Read sparse matrix
        mtx_files = sorted(subdir.glob("matrix.mtx*"))
        X = mmread(str(mtx_files[0]))
        if not isinstance(X, csr_matrix):
            X = X.tocsr()
        n_rows = X.shape[0]
        print(f"  Matrix: {n_rows} rows × {X.shape[1]} cols, {X.nnz} nnz")

        # Process in row chunks
        for chunk_start in range(0, n_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_rows)
            X_chunk = X[chunk_start:chunk_end]

            # Align columns
            X_csc = X_chunk.tocsc()
            del X_chunk
            X_aligned = _reorder_matrix_columns(X_csc, old_to_new_idx, n_new_cols)
            del X_csc
            gc.collect()

            current_chunk_rows.append(X_aligned)
            current_chunk_sources.append({
                "file": subdir.name,
                "file_type": "mtx_csv",
                "rows_in_chunk": X_aligned.shape[0],
            })

            total_accumulated = sum(m.shape[0] for m in current_chunk_rows)
            if total_accumulated >= chunk_size:
                current_row_start, entry, col_nnz = _write_mtx_chunk(
                    current_chunk_rows, current_chunk_sources, current_row_start
                )
                output_files.append(entry["mtx_path"])
                column_nnz_accumulator += col_nnz
                manifest_data.append(entry)
                current_chunk_rows = []
                current_chunk_sources = []

        del X
        gc.collect()

    # Write remaining rows
    if current_chunk_rows:
        remaining = sum(m.shape[0] for m in current_chunk_rows)
        print(f"\nWriting final chunk with {remaining} rows")
        current_row_start, entry, col_nnz = _write_mtx_chunk(
            current_chunk_rows, current_chunk_sources, current_row_start
        )
        output_files.append(entry["mtx_path"])
        column_nnz_accumulator += col_nnz
        manifest_data.append(entry)

    # Write column nnz
    col_nnz_path = os.path.join(output_dir, "column_nnz.txt")
    np.savetxt(col_nnz_path, column_nnz_accumulator, fmt="%u", delimiter="\n")
    print(f"  ✓ Column nnz saved to {os.path.basename(col_nnz_path)}")

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {
        "gene_list_file": str(gene_list_path),
        "n_genes": n_new_cols,
        "source_directories": [str(d) for d in subdirs],
        "source_directory_names": [d.name for d in subdirs],
        "input_type": "mtx_csv",
        "mtx_files": manifest_data,
        "total_mtx_files": len(output_files),
        "chunk_size": chunk_size,
        "column_nnz_file": "column_nnz.txt",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_rows = sum(e["n_rows"] for e in manifest_data)
    print(f"\n✓ Successfully wrote {len(output_files)} aligned MTX file(s)")
    print(f"  Total rows: {total_rows}, Total columns: {n_new_cols}")

    # Create column-major fragments
    print(f"\n{'=' * 70}")
    print("Creating column-major fragments for efficient column access...")
    print(f"{'=' * 70}")
    create_column_major_fragments(output_dir, mtx_output_dir, output_files, n_new_cols)

    return manifest_path


def concat_obs_from_mtx_csv_directory(
    input_dir: str,
    output_dir: str,
    join_strategy: str = "outer",
    output_filename: str = "obs.parquet",
    row_nnz_files: list[str] | None = None,
    min_nnz: int | None = 300,
    directories_filter: list[str] | None = None,
) -> str:
    """
    Read obs.csv files from MTX+CSV subdirectories, concatenate, and save as parquet.

    Args:
        input_dir: Directory containing subdirectories with obs.csv files
        output_dir: Directory where parquet file will be saved
        join_strategy: How to join columns: "inner" or "outer"
        output_filename: Name of output parquet file
        row_nnz_files: List of row stats files to merge (nnz + total_counts)
        min_nnz: Minimum nnz threshold for cell filtering (None to disable)
        directories_filter: Optional list of subdirectory names to process

    Returns:
        Path to created parquet file
    """
    subdirs = discover_mtx_csv_directories(input_dir)

    if directories_filter is not None:
        filter_set = set(directories_filter)
        subdirs = [d for d in subdirs if d.name in filter_set]

    if not subdirs:
        raise ValueError(f"No matching MTX+CSV subdirectories found in {input_dir}")

    print(f"Reading obs data from {len(subdirs)} subdirectories...")

    dataframes: list[pl.DataFrame] = []
    for idx, subdir in enumerate(subdirs):
        print(f"  [{idx + 1}/{len(subdirs)}] Reading obs.csv from {subdir.name}")
        obs_df = pd.read_csv(subdir / "obs.csv", index_col=0)
        df = pl.from_pandas(obs_df)

        # Normalize integer types
        for col in df.columns:
            dtype = df[col].dtype
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))

        # Add source tracking column
        df = df.with_columns(pl.lit(subdir.name).alias("_source_dir"))
        print(f"    ✓ {df.height} rows, {len(df.columns)} columns")
        dataframes.append(df)

    # Concatenate
    print(f"\nConcatenating obs data (strategy: {join_strategy})...")
    if join_strategy == "inner":
        common_cols = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_cols &= set(df.columns)
        common_cols.discard("_source_dir")
        common_cols = sorted(common_cols)
        selected = []
        for df in dataframes:
            cols = list(common_cols)
            if "_source_dir" in df.columns:
                cols.append("_source_dir")
            selected.append(df.select(cols))
        combined_df = pl.concat(selected, how="diagonal")
    else:
        combined_df = pl.concat(dataframes, how="diagonal")

    print(f"  ✓ Combined: {combined_df.height} rows × {len(combined_df.columns)} columns")

    # Add row stats (nnz + total_counts)
    if row_nnz_files:
        print(f"\nLoading row stats from {len(row_nnz_files)} file(s)...")
        all_nnz = []
        all_total_counts = []
        for stats_file in sorted(row_nnz_files):
            if not os.path.exists(stats_file):
                raise FileNotFoundError(f"Row stats file not found: {stats_file}")
            stats = np.loadtxt(stats_file, skiprows=1)
            all_nnz.extend(stats[:, 0].astype(np.uint32).tolist())
            all_total_counts.extend(stats[:, 1].astype(np.float32).tolist())

        if len(all_nnz) != combined_df.height:
            raise ValueError(
                f"Row stats count ({len(all_nnz)}) != obs rows ({combined_df.height})"
            )

        combined_df = combined_df.with_columns([
            pl.Series("nnz", all_nnz, dtype=pl.UInt32),
            pl.Series("total_counts", all_total_counts, dtype=pl.Float32),
        ])
        combined_df = combined_df.with_columns(
            pl.when(pl.col("total_counts") > 0)
            .then(10000.0 / pl.col("total_counts"))
            .otherwise(None)
            .alias("scaling_factor")
        )
        print(f"  ✓ Added nnz, total_counts, scaling_factor columns")

    # Add row index before filtering
    combined_df = combined_df.with_row_index("_row_index")

    # Filter low-quality cells
    if min_nnz is not None and min_nnz > 0 and "nnz" in combined_df.columns:
        before = combined_df.height
        combined_df = combined_df.filter(pl.col("nnz") >= min_nnz)
        print(f"  ✓ Filtered cells with nnz < {min_nnz}: removed {before - combined_df.height}, kept {combined_df.height}")

    # Write parquet
    output_path = Path(output_dir) / output_filename
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    combined_df.write_parquet(str(output_path), compression="zstd")
    print(f"  ✓ Wrote {combined_df.height} rows to {output_path}")

    return str(output_path)


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
) -> Path:
    """
    Build a complete zdata object from a directory of MTX+CSV subdirectories.

    Each subdirectory must contain matrix.mtx, obs.csv, and var.csv.

    Args:
        input_dir: Directory containing subdirectories with MTX+CSV files
        output_name: Output directory name for the zdata object
        gene_list_path: Path to standard gene list (default: package default)
        block_rows: Rows per block for row-major files (default: 16)
        block_columns: Rows per block for column-major files (default: same as block_rows)
        max_rows: Max rows per chunk for row-major files (default: 8192)
        max_columns: Max rows per chunk for column-major files (default: 256)
        obs_join_strategy: Strategy for joining obs columns: "inner" or "outer"
        obs_output_filename: Name of obs parquet file (default: "obs.parquet")
        cleanup_temp: Whether to clean up temporary MTX files (default: True)
        mtx_chunk_size: Max rows per intermediate MTX file (default: 131072)
        mtx_temp_dir: Optional persistent directory for intermediate MTX files
        min_nnz: Minimum nnz for cell filtering (None to disable, default: 300)

    Returns:
        Path to created zdata directory
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_path.is_dir():
        raise ValueError(f"Path is not a directory: {input_dir}")

    # Validate we have valid subdirectories
    subdirs = discover_mtx_csv_directories(input_dir)

    print("=" * 70)
    print("Building zdata from MTX+CSV directories")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output zdata directory: {output_name}")
    print(f"Found {len(subdirs)} MTX+CSV subdirectories")

    # Use default gene list if not provided
    if gene_list_path is None:
        gene_list_path = str(get_default_gene_list_path())
    if not os.path.exists(gene_list_path):
        raise FileNotFoundError(f"Gene list file not found: {gene_list_path}")

    # Set up temp directory for aligned MTX files
    use_custom_mtx_dir = mtx_temp_dir is not None
    temp_dir_context = None
    temp_dir_context_entered = False

    if use_custom_mtx_dir:
        temp_mtx_dir = Path(mtx_temp_dir)
        temp_mtx_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir_context = tempfile.TemporaryDirectory(prefix="zdata_build_mtxcsv_")
        temp_mtx_dir = Path(temp_dir_context.__enter__())
        temp_dir_context_entered = True

    try:
        # Step 1: Align MTX files to standard gene list
        print(f"\n{'=' * 70}")
        print("Step 1: Aligning genes to standard gene list")
        print("=" * 70)

        manifest_path = align_mtx_csv_directory_to_mtx(
            input_dir, gene_list_path, str(temp_mtx_dir), chunk_size=mtx_chunk_size
        )
        print(f"\n✓ Alignment complete! Manifest: {manifest_path}")

        # Step 2: Build zdata from aligned MTX files
        print(f"\n{'=' * 70}")
        print("Step 2: Building zdata from aligned MTX files")
        print("=" * 70)

        zdata_dir = build_zdata(
            str(temp_mtx_dir),
            output_name,
            block_rows=block_rows,
            block_columns=block_columns,
            max_rows=max_rows,
            max_columns=max_columns,
        )
        zdata_dir = Path(zdata_dir)
        print(f"\n✓ Zdata build complete! Output: {zdata_dir}")

        # Step 3: Concatenate obs from CSV files
        print(f"\n{'=' * 70}")
        print("Step 3: Concatenating obs metadata from CSV files")
        print("=" * 70)

        # Find row stats files from manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        row_nnz_files = []
        mtx_dir = os.path.join(str(temp_mtx_dir), "rm_mtx_files")
        for mtx_entry in manifest.get("mtx_files", []):
            stats_file = mtx_entry.get("row_stats_file")
            if stats_file:
                stats_path = os.path.join(mtx_dir, stats_file)
                if os.path.exists(stats_path):
                    row_nnz_files.append(stats_path)

        # Get list of successfully processed directories from manifest
        processed_dirs = manifest.get("source_directory_names", [])

        obs_output_path = concat_obs_from_mtx_csv_directory(
            input_dir,
            str(zdata_dir),
            join_strategy=obs_join_strategy,
            output_filename=obs_output_filename,
            row_nnz_files=row_nnz_files,
            min_nnz=min_nnz,
            directories_filter=processed_dirs if processed_dirs else None,
        )
        print(f"\n✓ Obs concatenation complete! Output: {obs_output_path}")

        # Step 4: Save gene list as var.parquet
        print(f"\n{'=' * 70}")
        print("Step 4: Saving gene list as var.parquet")
        print("=" * 70)

        with open(gene_list_path, "r") as f:
            genes = [line.strip() for line in f if line.strip()]

        col_nnz_path = os.path.join(str(temp_mtx_dir), "column_nnz.txt")
        if not os.path.exists(col_nnz_path):
            raise FileNotFoundError(f"Column nnz file not found: {col_nnz_path}")

        column_nnz = np.loadtxt(col_nnz_path, dtype=np.uint32)
        if len(column_nnz) != len(genes):
            raise ValueError(
                f"Column nnz count ({len(column_nnz)}) != gene count ({len(genes)})"
            )

        var_df = pl.DataFrame({
            "gene": genes,
            "index": range(len(genes)),
            "nnz": column_nnz.tolist(),
        })
        var_path = zdata_dir / "var.parquet"
        var_df.write_parquet(str(var_path), compression="zstd")
        print(f"✓ var.parquet saved ({len(genes)} genes)")

    finally:
        if temp_dir_context_entered and cleanup_temp:
            try:
                temp_dir_context.__exit__(None, None, None)
            except Exception:
                pass

    print(f"\n{'=' * 70}")
    print("✓ Complete zdata object built successfully from MTX+CSV!")
    print("=" * 70)
    print(f"Output directory: {zdata_dir}")

    # Verify output
    for label, path in [
        ("metadata.json", zdata_dir / "metadata.json"),
        (obs_output_filename, zdata_dir / obs_output_filename),
        ("var.parquet", zdata_dir / "var.parquet"),
    ]:
        if path.exists():
            print(f"  ✓ {label}")
    xrm = zdata_dir / "X_RM"
    if xrm.exists():
        bins = list(xrm.glob("*.bin"))
        print(f"  ✓ X_RM/ ({len(bins)} chunk files)")
    xcm = zdata_dir / "X_CM"
    if xcm.exists():
        bins = list(xcm.glob("*.bin"))
        print(f"  ✓ X_CM/ ({len(bins)} chunk files)")

    return zdata_dir


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build a zdata object from a directory of MTX+CSV subdirectories. "
        "Each subdirectory must contain matrix.mtx, obs.csv, and var.csv."
    )
    parser.add_argument(
        "input_dir", type=str,
        help="Directory containing subdirectories with matrix.mtx, obs.csv, var.csv",
    )
    parser.add_argument(
        "output_name", type=str,
        help='Output directory name (e.g., "atlas.zdata" or "atlas")',
    )
    parser.add_argument("--gene-list", type=str, default=None,
                        help="Path to standard gene list file (default: package default)")
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument("--block-columns", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=8192)
    parser.add_argument("--max-columns", type=int, default=256)
    parser.add_argument("--obs-join-strategy", choices=["inner", "outer"], default="outer")
    parser.add_argument("--min-nnz", type=int, default=300,
                        help="Minimum nnz for cell filtering (0 to disable)")
    parser.add_argument("--mtx-chunk-size", type=int, default=131072)
    parser.add_argument("--mtx-temp-dir", type=str, default=None)
    parser.add_argument("--no-cleanup-temp", action="store_true")

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
            cleanup_temp=not args.no_cleanup_temp,
            mtx_chunk_size=args.mtx_chunk_size,
            mtx_temp_dir=args.mtx_temp_dir,
            min_nnz=args.min_nnz if args.min_nnz > 0 else None,
        )
        print(f"\n✓ Build complete! Output: {zdata_dir}")
        return 0
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
