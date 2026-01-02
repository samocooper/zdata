#!/usr/bin/env python3
"""
Concatenate obs/metadata from multiple zarr files into a single polars DataFrame.

Reads obs data from all zarr files in a directory, converts to polars format,
joins them according to specified strategy (inner, outer, or by specified columns),
and saves the result as a parquet file in the zdata directory.
"""

import zarr
import anndata as ad
import polars as pl
import numpy as np
import sys
import os
import argparse
from pathlib import Path
from typing import List, Optional


def read_obs_from_h5ad(h5ad_path: str, h5ad_name: str) -> pl.DataFrame:
    """
    Read obs/metadata from a single h5ad file and convert to polars DataFrame.
    
    Args:
        h5ad_path: Path to the h5ad file (.h5 or .hdf5)
        h5ad_name: Name of the h5ad file (for source tracking)
    
    Returns:
        polars DataFrame with obs data
    
    Raises:
        RuntimeError: If reading fails
    """
    try:
        # Use full loading instead of backed mode for better type consistency
        adata = ad.read_h5ad(h5ad_path)
        
        # Get obs DataFrame (pandas)
        obs_df = adata.obs
        
        # Convert to polars
        df = pl.from_pandas(obs_df)
        
        # Normalize integer types to avoid schema conflicts when concatenating
        # Cast all integer types to Int64 for consistency
        for col in df.columns:
            dtype = df[col].dtype
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
        
        # Add source column
        df = df.with_columns([
            pl.lit(h5ad_name).alias("_source_file")
        ])
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Failed to read obs data from {h5ad_name}: {e}") from e

def read_obs_from_zarr(zarr_path: str, zarr_name: str) -> pl.DataFrame:
    """
    Read obs/metadata from a single zarr file and convert to polars DataFrame.
    
    Args:
        zarr_path: Path to the zarr file
        zarr_name: Name of the zarr file (for source tracking)
    
    Returns:
        polars DataFrame with obs data
    
    Raises:
        RuntimeError: If reading fails
    """
    zarr_group = zarr.open(zarr_path, mode='r')
    
    try:
        if 'obs' not in zarr_group:
            raise ValueError(f"'obs' group not found in {zarr_name}. All zarr files must have obs data.")
        
        obs_group = zarr_group['obs']
        obs_keys = list(obs_group.keys())
        
        # Filter out _index as it's metadata, not a data column
        obs_columns = [key for key in obs_keys if key != '_index']
        
        if not obs_columns:
            raise ValueError(f"No obs columns found in {zarr_name}. Zarr file must have at least one obs column.")
        
        # Build dictionary of column data
        column_data = {}
        skipped_columns = []
        
        for col_name in obs_columns:
            col_obj = obs_group[col_name]
            
            # Check if it's a categorical (Group) or simple array
            if isinstance(col_obj, zarr.Group):
                # Categorical data: has 'categories' and 'codes' arrays
                if 'categories' in col_obj and 'codes' in col_obj:
                    try:
                        categories = col_obj['categories'][:]
                        codes = col_obj['codes'][:]
                        # Handle empty categories array (can happen with some zarr files)
                        if len(categories) == 0:
                            # If no categories, all codes are invalid - use None for all
                            values = [None] * len(codes)
                        else:
                            # Map codes to category strings - handle invalid codes
                            values = []
                            for code in codes:
                                if code >= 0 and code < len(categories):
                                    values.append(categories[code])
                                else:
                                    # Invalid code (negative or out of range) - use None
                                    # This can happen with missing data encoded as -1
                                    values.append(None)
                        column_data[col_name] = values
                    except Exception as e:
                        raise RuntimeError(f"Failed to read categorical column '{col_name}' from {zarr_name}: {e}") from e
                else:
                    # Group but not standard categorical - skip with warning but don't fail
                    skipped_columns.append(col_name)
            else:
                # Simple array - read directly
                try:
                    values = col_obj[:]
                    # Convert numpy array to list for polars
                    if isinstance(values, np.ndarray):
                        # Handle object arrays (strings) specially
                        if values.dtype == object:
                            values = values.tolist()
                        else:
                            values = values.tolist()
                    column_data[col_name] = values
                except Exception as e:
                    raise RuntimeError(f"Failed to read column '{col_name}' from {zarr_name}: {e}") from e
        
        if not column_data:
            raise ValueError(f"No readable columns found in {zarr_name} (skipped {len(skipped_columns)} non-standard columns)")
        
        if skipped_columns:
            print(f"  WARNING: Skipped {len(skipped_columns)} non-standard columns: {', '.join(skipped_columns[:5])}")
        
        # Determine number of rows from first column
        n_rows = len(list(column_data.values())[0])
        
        # Verify all columns have same length
        for col_name, values in column_data.items():
            if len(values) != n_rows:
                raise ValueError(f"Column '{col_name}' has inconsistent length ({len(values)}) in {zarr_name} (expected {n_rows})")
        
        # Create polars DataFrame
        df = pl.DataFrame(column_data)
        
        # Add source column to track which zarr file this came from
        df = df.with_columns([
            pl.lit(zarr_name).alias("_source_zarr")
        ])
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Failed to read obs data from {zarr_name}: {e}") from e


def concat_obs_dataframes(
    dataframes: List[pl.DataFrame],
    join_strategy: str = "outer",
    join_on: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Concatenate multiple polars DataFrames using specified join strategy.
    
    For obs/metadata, this concatenates rows (stacks vertically) rather than
    merging horizontally. The join strategy determines which columns to keep.
    
    Args:
        dataframes: List of polars DataFrames to join
        join_strategy: One of "inner", "outer", or "columns"
        join_on: If join_strategy is "columns", list of column names that must be present
    
    Returns:
        Combined polars DataFrame with concatenated rows
    """
    if not dataframes:
        raise ValueError("No dataframes provided")
    
    if len(dataframes) == 1:
        return dataframes[0]
    
    if join_strategy == "outer":
        # Outer: keep all columns from all dataframes, fill missing with null
        # Normalize integer types across all dataframes before concatenation
        # to avoid schema conflicts (Int8 vs Int64, etc.)
        normalized_dfs = []
        for df in dataframes:
            # Cast all integer types to Int64 for consistency
            cast_exprs = []
            for col in df.columns:
                dtype = df[col].dtype
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32]:
                    cast_exprs.append(pl.col(col).cast(pl.Int64))
                else:
                    cast_exprs.append(pl.col(col))
            if cast_exprs:
                df = df.with_columns(cast_exprs)
            normalized_dfs.append(df)
        
        # Use concat with how='diagonal' to align columns and fill missing values
        result = pl.concat(normalized_dfs, how='diagonal')
        return result
    
    elif join_strategy == "inner":
        # Inner: only keep columns present in all dataframes
        # Find common columns (excluding _source_zarr which we always keep)
        common_cols = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_cols = common_cols.intersection(set(df.columns))
        
        # Always include _source_zarr if it exists
        if "_source_zarr" not in common_cols:
            # Check if any dataframe has it
            if any("_source_zarr" in df.columns for df in dataframes):
                # Add it to common columns if at least one has it
                # But we'll handle it separately
                pass
        
        # Remove _source_zarr from common columns for now
        has_source = "_source_zarr" in common_cols
        common_cols.discard("_source_zarr")
        common_cols = sorted(list(common_cols))
        
        if not common_cols and not has_source:
            raise ValueError("No common columns found for inner join")
        
        # Select only common columns from each dataframe
        # Add _source_zarr if it exists in that dataframe
        selected_dfs = []
        for df in dataframes:
            cols_to_select = common_cols.copy()
            if "_source_zarr" in df.columns:
                cols_to_select.append("_source_zarr")
            if "_source_file" in df.columns:
                cols_to_select.append("_source_file")
            df_selected = df.select(cols_to_select)
            
            # Normalize integer types
            cast_exprs = []
            for col in df_selected.columns:
                dtype = df_selected[col].dtype
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32]:
                    cast_exprs.append(pl.col(col).cast(pl.Int64))
                else:
                    cast_exprs.append(pl.col(col))
            if cast_exprs:
                df_selected = df_selected.with_columns(cast_exprs)
            selected_dfs.append(df_selected)
        
        # Concatenate rows
        result = pl.concat(selected_dfs, how='diagonal')
        return result
    
    elif join_strategy == "columns":
        if not join_on:
            raise ValueError("join_on must be specified when join_strategy is 'columns'")
        
        # Columns: only keep specified columns (must be present in all dataframes)
        # Verify all dataframes have the required columns
        for i, df in enumerate(dataframes):
            missing_cols = [col for col in join_on if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"DataFrame {i} missing required columns: {missing_cols}"
                )
        
        # Select only the specified columns (plus _source_zarr if available)
        selected_dfs = []
        for df in dataframes:
            cols_to_select = list(join_on)
            if "_source_zarr" in df.columns:
                cols_to_select.append("_source_zarr")
            if "_source_file" in df.columns:
                cols_to_select.append("_source_file")
            df_selected = df.select(cols_to_select)
            
            # Normalize integer types
            cast_exprs = []
            for col in df_selected.columns:
                dtype = df_selected[col].dtype
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32]:
                    cast_exprs.append(pl.col(col).cast(pl.Int64))
                else:
                    cast_exprs.append(pl.col(col))
            if cast_exprs:
                df_selected = df_selected.with_columns(cast_exprs)
            selected_dfs.append(df_selected)
        
        # Concatenate rows
        result = pl.concat(selected_dfs, how='diagonal')
        return result
    
    else:
        raise ValueError(f"Unknown join_strategy: {join_strategy}. Must be 'inner', 'outer', or 'columns'")


def concat_obs_from_zarr_directory(
    zarr_dir: str,
    output_dir: str,
    join_strategy: str = "outer",
    join_on: Optional[List[str]] = None,
    output_filename: str = "obs.parquet",
    zarr_files_filter: Optional[set] = None,
    row_nnz_files: Optional[List[str]] = None
):
    """
    Read obs data from all zarr or h5ad files in a directory, join them, and save to parquet.
    Auto-detects file type based on extensions: .zarr (directories) or .h5/.hdf5 (h5ad files).
    
    Args:
        zarr_dir: Directory containing .zarr files (directories) or .h5/.hdf5 files (h5ad)
        output_dir: Directory where parquet file will be saved (typically zdata directory)
        join_strategy: One of "inner", "outer", or "columns"
        join_on: If join_strategy is "columns", list of column names to join on
        output_filename: Name of output parquet file
        zarr_files_filter: Optional set of file names to process (filters available files)
        row_nnz_files: Optional list of text files containing row nnz values (one per line)
                      These will be merged into the obs DataFrame as an 'nnz' column
    
    Returns:
        Path to created parquet file
    """
    from zdata.build_zdata.align_mtx import _detect_file_type
    
    zarr_dir_path = Path(zarr_dir)
    
    # Find all files (zarr directories and h5ad files)
    all_zarr_files = sorted([f for f in zarr_dir_path.glob("*.zarr") if f.is_dir()])
    all_h5ad_files = sorted([f for f in zarr_dir_path.iterdir() 
                              if f.is_file() and (f.suffix in ['.h5', '.hdf5'] or f.name.endswith('.h5ad'))])
    
    all_files = []
    file_types = {}
    
    for f in all_zarr_files:
        all_files.append(f)
        file_types[f] = 'zarr'
    
    for f in all_h5ad_files:
        all_files.append(f)
        file_types[f] = 'h5ad'
    
    # Filter to only process files that were successfully processed in alignment step
    if zarr_files_filter is not None:
        files = [f for f in all_files if f.name in zarr_files_filter]
        if len(files) < len(all_files):
            skipped = len(all_files) - len(files)
            print(f"Filtering files: {len(files)} to process, {skipped} skipped (not in alignment manifest)")
    else:
        files = all_files
    
    if not files:
        if zarr_files_filter:
            raise ValueError(f"No files found matching filter. Available: {[f.name for f in all_files]}")
        else:
            raise ValueError(f"No .zarr files (directories) or .h5/.hdf5 files (h5ad) found in {zarr_dir}")
    
    print(f"Found {len([f for f in files if file_types[f] == 'zarr'])} zarr file(s) and {len([f for f in files if file_types[f] == 'h5ad'])} h5ad file(s) to process")
    print(f"Join strategy: {join_strategy}")
    if join_on:
        print(f"Join columns: {', '.join(join_on)}")
    
    # Read obs data from each file
    print(f"\nReading obs data from files...")
    dataframes = []
    
    for file_idx, file_path in enumerate(files):
        file_name = file_path.name
        file_type = file_types[file_path]
        print(f"  [{file_idx + 1}/{len(files)}] Processing {file_type.upper()}: {file_name}")
        
        if file_type == 'zarr':
            df = read_obs_from_zarr(str(file_path), file_name)
        elif file_type == 'h5ad':
            df = read_obs_from_h5ad(str(file_path), file_name)
        else:
            raise ValueError(f"Unknown file type for {file_name}")
        
        if df is None:
            raise RuntimeError(f"Failed to read obs data from {file_name}. This file must be processed successfully - skipping is not allowed.")
        
        print(f"    ✓ Read {df.height} rows, {len(df.columns)} columns")
        dataframes.append(df)
    
    if not dataframes:
        raise ValueError("No obs data successfully read from any files")
    
    print(f"\nSuccessfully read obs data from {len(dataframes)} file(s)")
    
    # Join dataframes
    print(f"\nJoining dataframes...")
    combined_df = concat_obs_dataframes(dataframes, join_strategy, join_on)
    
    print(f"  ✓ Combined DataFrame: {combined_df.height} rows × {len(combined_df.columns)} columns")
    
    # Add row nnz (number of non-zeros per row/cell) - required
    if not row_nnz_files:
        raise ValueError(
            "row_nnz_files must be provided. Row nnz values are required for obs DataFrame."
        )
    
    print(f"\nLoading row nnz values from {len(row_nnz_files)} file(s)...")
    all_row_nnz = []
    for nnz_file in sorted(row_nnz_files):
        if not os.path.exists(nnz_file):
            raise FileNotFoundError(f"Row nnz file not found: {nnz_file}")
        
        nnz_values = np.loadtxt(nnz_file, dtype=np.uint32)
        all_row_nnz.extend(nnz_values.tolist())
        print(f"  ✓ Loaded {len(nnz_values)} nnz values from {os.path.basename(nnz_file)}")
    
    if len(all_row_nnz) != combined_df.height:
        raise ValueError(
            f"Row nnz count ({len(all_row_nnz)}) doesn't match obs rows ({combined_df.height}). "
            f"This indicates a mismatch in the alignment process."
        )
    
    combined_df = combined_df.with_columns([
        pl.Series("nnz", all_row_nnz, dtype=pl.UInt32)
    ])
    print(f"  ✓ Added nnz column to obs DataFrame ({len(all_row_nnz)} values)")
    
    # Add explicit integer index column to prevent pandas from inferring string index
    # This ensures clean conversion to pandas without ImplicitModificationWarning
    combined_df = combined_df.with_row_index("_row_index")
    
    # Show column summary
    print(f"  Columns: {', '.join(combined_df.columns[:10])}" + 
          (f" ... ({len(combined_df.columns)} total)" if len(combined_df.columns) > 10 else ""))
    
    # Ensure output directory exists
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Write to parquet
    output_path = output_dir_path / output_filename
    print(f"\nWriting to parquet: {output_path}")
    
    combined_df.write_parquet(str(output_path), compression="zstd")
    
    print(f"  ✓ Successfully wrote {combined_df.height} rows to {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    return str(output_path)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Concatenate obs/metadata from multiple zarr files into a single polars DataFrame and save as parquet.'
    )
    parser.add_argument(
        'zarr_dir',
        type=str,
        help='Directory containing .zarr files to process'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory where parquet file will be saved (typically zdata directory)'
    )
    parser.add_argument(
        '--join-strategy',
        type=str,
        choices=['inner', 'outer', 'columns'],
        default='outer',
        help='Join strategy: "inner" (only columns present in all files), "outer" (all columns, fill missing with null), or "columns" (only specified columns). Rows are concatenated (stacked). Default: outer'
    )
    parser.add_argument(
        '--join-on',
        type=str,
        nargs='+',
        default=None,
        help='Column names that must be present when --join-strategy is "columns". Only these columns (plus _source_zarr) will be kept. Example: --join-on barcode sample_id'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default='obs.parquet',
        help='Name of output parquet file. Default: obs.parquet'
    )
    parser.add_argument(
        '--row-nnz-files',
        type=str,
        nargs='+',
        default=None,
        help='Optional list of row nnz text files to merge into obs DataFrame'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    zarr_dir_path = Path(args.zarr_dir)
    if not zarr_dir_path.exists():
        print(f"ERROR: Zarr directory not found: {args.zarr_dir}")
        sys.exit(1)
    
    if not zarr_dir_path.is_dir():
        print(f"ERROR: Path is not a directory: {args.zarr_dir}")
        sys.exit(1)
    
    # Validate join strategy
    if args.join_strategy == 'columns' and not args.join_on:
        print("ERROR: --join-on must be specified when --join-strategy is 'columns'")
        sys.exit(1)
    
    try:
        output_path = concat_obs_from_zarr_directory(
            str(zarr_dir_path),
            args.output_dir,
            args.join_strategy,
            args.join_on,
            args.output_filename,
            row_nnz_files=args.row_nnz_files
        )
        print(f"\n✓ Concatenation complete!")
        print(f"  Output file: {output_path}")
        return 0
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

