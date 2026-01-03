#!/usr/bin/env python3
"""
Build complete zdata object from zarr files.

This script orchestrates the full pipeline:
1. Aligns zarr files to standard gene list and converts to MTX format
2. Builds zdata directory from MTX files
3. Concatenates obs/metadata from zarr files and saves to parquet

Input: Directory of zarr files
Output: Complete .zdata directory with X_RM, X_CM, metadata.json, and obs.parquet
"""

import sys
import os
import argparse
import tempfile
from pathlib import Path

# Import functions from other build scripts
from zdata.build_zdata.align_mtx import align_zarr_directory_to_mtx
from zdata.build_zdata.build_x import build_zdata
from zdata.build_zdata.concat_obs import concat_obs_from_zarr_directory

import polars as pl


def build_zdata_from_zarr(
    zarr_dir: str,
    output_name: str,
    gene_list_path: str = None,
    block_rows: int = 16,
    block_columns: int = None,
    max_rows: int = 8192,
    max_columns: int = 256,
    obs_join_strategy: str = "outer",
    obs_join_on: list = None,
    obs_output_filename: str = "obs.parquet",
    cleanup_temp: bool = True,
    mtx_chunk_size: int = 131072,
    mtx_temp_dir: str = None
):
    """
    Build complete zdata object from directory of zarr or h5ad files.
    Auto-detects file type based on extensions: .zarr (directories) or .h5/.hdf5 (h5ad files).
    
    Args:
        zarr_dir: Directory containing .zarr files (directories) or .h5/.hdf5 files (h5ad format)
        output_name: Output directory name (can include .zdata suffix, e.g., "atlas.zdata")
        gene_list_path: Path to standard gene list file (default: uses package default)
        block_rows: Number of rows per block for row-major (X_RM) files (default: 16)
        block_columns: Number of rows per block for column-major (X_CM) files (default: None, uses block_rows)
        max_rows: Maximum rows per chunk for row-major files (default: 8192)
        max_columns: Maximum rows per chunk for column-major files (default: 256)
        obs_join_strategy: Strategy for joining obs data ("inner", "outer", or "columns")
        obs_join_on: If obs_join_strategy is "columns", list of column names to join on
        obs_output_filename: Name of obs parquet file (default: "obs.parquet")
        cleanup_temp: Whether to clean up temporary MTX files (default: True). 
                      Ignored if mtx_temp_dir is specified (files are always preserved).
        mtx_chunk_size: Maximum rows per MTX file during alignment (default: 131072)
        mtx_temp_dir: Optional path to directory for MTX files. If specified:
                      - MTX files will be preserved (not cleaned up) even on failure
                      - Existing MTX files in this directory will be reused if manifest exists
                      - Allows rerunning pipeline from aligned MTX files
    
    Returns:
        Path to created zdata directory (as Path object)
    """
    zarr_dir_path = Path(zarr_dir)
    if not zarr_dir_path.exists():
        raise FileNotFoundError(f"Zarr directory not found: {zarr_dir}")
    
    if not zarr_dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {zarr_dir}")
    
    # Auto-detect and check for files (zarr directories and h5ad files)
    zarr_files = sorted([f for f in zarr_dir_path.glob("*.zarr") if f.is_dir()])
    h5ad_files = sorted([f for f in zarr_dir_path.iterdir() 
                         if f.is_file() and (f.suffix in ['.h5', '.hdf5'] or f.name.endswith('.h5ad'))])
    
    if not zarr_files and not h5ad_files:
        raise ValueError(f"No .zarr files (directories) or .h5/.hdf5 files (h5ad) found in {zarr_dir}")
    
    print("=" * 70)
    print("Building zdata from files")
    print("=" * 70)
    print(f"Input directory: {zarr_dir}")
    print(f"Output zdata directory: {output_name}")
    print(f"Found {len(zarr_files)} zarr file(s) and {len(h5ad_files)} h5ad file(s)")
    
    # Step 1: Align zarr files to standard gene list and convert to MTX
    print("\n" + "=" * 70)
    print("Step 1: Aligning zarr files to standard gene list and converting to MTX")
    print("=" * 70)
    
    # Use default gene list if not provided
    if gene_list_path is None:
        from zdata.build_zdata.align_mtx import get_default_gene_list_path
        gene_list_path = str(get_default_gene_list_path())
    
    if not os.path.exists(gene_list_path):
        raise FileNotFoundError(f"Gene list file not found: {gene_list_path}")
    
    # Determine MTX directory: use custom if provided, otherwise create temp
    use_custom_mtx_dir = mtx_temp_dir is not None
    if use_custom_mtx_dir:
        temp_mtx_dir = Path(mtx_temp_dir)
        temp_mtx_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using custom MTX directory: {temp_mtx_dir}")
        print(f"  MTX files will be preserved (not cleaned up)")
        
        # Check if MTX files already exist (for rerun capability)
        manifest_path = temp_mtx_dir / "manifest.json"
        if manifest_path.exists():
            print(f"  Found existing manifest: {manifest_path}")
            print(f"  Checking if MTX files can be reused...")
            mtx_output_dir = temp_mtx_dir / "rm_mtx_files"
            if mtx_output_dir.exists() and list(mtx_output_dir.glob("*.mtx")):
                mtx_count = len(list(mtx_output_dir.glob("*.mtx")))
                print(f"  Found {mtx_count} existing MTX file(s) - will reuse them")
                reuse_existing = True
            else:
                print(f"  No existing MTX files found - will regenerate")
                reuse_existing = False
        else:
            reuse_existing = False
        temp_dir_context = None
        temp_dir_context_entered = False
    else:
        # Use temporary directory that will be cleaned up
        temp_dir_context = tempfile.TemporaryDirectory(prefix="zdata_build_")
        temp_mtx_dir = Path(temp_dir_context.__enter__())
        print(f"Using temporary directory for MTX files: {temp_mtx_dir}")
        reuse_existing = False
        temp_dir_context_entered = True
    
    try:
        # Align zarr files to MTX format (skip if reusing existing)
        if not reuse_existing:
            try:
                manifest_path = align_zarr_directory_to_mtx(
                    str(zarr_dir_path),
                    gene_list_path,
                    str(temp_mtx_dir),
                    tmp_dir=None,
                    chunk_size=mtx_chunk_size
                )
                print(f"\n✓ Alignment complete! Manifest: {manifest_path}")
            except Exception as e:
                print(f"\n✗ ERROR in alignment step: {e}")
                import traceback
                traceback.print_exc()
                if use_custom_mtx_dir:
                    print(f"\n⚠ MTX files preserved in: {temp_mtx_dir}")
                    print(f"  You can rerun the pipeline with --mtx-temp-dir {temp_mtx_dir} to continue from MTX files")
                raise
        else:
            # Reusing existing MTX files
            manifest_path = str(temp_mtx_dir / "manifest.json")
            print(f"\n✓ Reusing existing MTX files from: {temp_mtx_dir}")
            print(f"  Manifest: {manifest_path}")
        
        # Step 2: Build zdata from MTX files
        print("\n" + "=" * 70)
        print("Step 2: Building zdata from MTX files")
        print("=" * 70)
        
        try:
            zdata_dir = build_zdata(
                str(temp_mtx_dir),
                output_name,
                block_rows=block_rows,
                block_columns=block_columns,
                max_rows=max_rows,
                max_columns=max_columns
            )
            # Convert to Path if not already
            zdata_dir = Path(zdata_dir)
            print(f"\n✓ Zdata build complete! Output: {zdata_dir}")
        except Exception as e:
            print(f"\n✗ ERROR in zdata build step: {e}")
            import traceback
            traceback.print_exc()
            if use_custom_mtx_dir:
                print(f"\n⚠ MTX files preserved in: {temp_mtx_dir}")
                print(f"  You can rerun the pipeline with --mtx-temp-dir {temp_mtx_dir} to continue from MTX files")
            raise
        
        # Step 3: Concatenate obs/metadata from zarr files
        # IMPORTANT: Only process zarr files that were successfully processed in alignment step
        # This ensures obs data and expression data are in sync
        print("\n" + "=" * 70)
        print("Step 3: Concatenating obs/metadata from zarr files")
        print("=" * 70)
        
        try:
            # Read manifest to get list of successfully processed zarr files
            # Since we now fail on errors, all files in manifest were successfully processed
            import json
            successfully_processed_zarrs = set()
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    # Collect all source files that contributed to MTX files
                    # Support both old format ('zarr_files') and new format ('source_files')
                    for mtx_entry in manifest.get('mtx_files', []):
                        source_files = mtx_entry.get('source_files', mtx_entry.get('zarr_files', []))
                        for file_info in source_files:
                            # Support both old format ('zarr_file') and new format ('file')
                            file_name = file_info.get('file', file_info.get('zarr_file', ''))
                            if file_name:
                                successfully_processed_zarrs.add(file_name)
            
            if successfully_processed_zarrs:
                print(f"Processing obs from {len(successfully_processed_zarrs)} file(s) that were successfully aligned")
                
                # Find row nnz files from MTX processing
                row_nnz_files = []
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        mtx_output_dir = os.path.join(temp_mtx_dir, "rm_mtx_files")
                        for mtx_entry in manifest.get('mtx_files', []):
                            mtx_file = mtx_entry.get('mtx_file', '')
                            if mtx_file:
                                # Extract row range from filename (e.g., rows_0_131071.mtx -> rows_0_131071_nnz.txt)
                                base_name = os.path.splitext(mtx_file)[0]
                                nnz_file = os.path.join(mtx_output_dir, f"{base_name}_nnz.txt")
                                if os.path.exists(nnz_file):
                                    row_nnz_files.append(nnz_file)
                
                obs_output_path = concat_obs_from_zarr_directory(
                    str(zarr_dir_path),
                    str(zdata_dir),
                    join_strategy=obs_join_strategy,
                    join_on=obs_join_on,
                    output_filename=obs_output_filename,
                    zarr_files_filter=successfully_processed_zarrs,  # Only process these files
                    row_nnz_files=row_nnz_files  # Add row nnz values
                )
                print(f"\n✓ Obs concatenation complete! Output: {obs_output_path}")
            else:
                raise RuntimeError("No source files found in manifest. This indicates a problem with the alignment step.")
        except Exception as e:
            print(f"\n✗ ERROR: Obs concatenation failed: {e}")
            import traceback
            traceback.print_exc()
            raise  # Fail the build if obs concatenation fails
        
        # Step 4: Load column nnz and save gene list as var.parquet
        print("\n" + "=" * 70)
        print("Step 4: Loading column nnz and saving gene list as var.parquet")
        print("=" * 70)
        
        try:
            with open(gene_list_path, 'r') as f:
                genes = [line.strip() for line in f if line.strip()]
            
            if not genes:
                raise ValueError(f"No genes found in gene list file: {gene_list_path}")
            
            # Load column nnz from file (calculated during MTX processing)
            import numpy as np
            column_nnz_path = os.path.join(str(temp_mtx_dir), "column_nnz.txt")
            
            if not os.path.exists(column_nnz_path):
                raise FileNotFoundError(
                    f"Column nnz file not found: {column_nnz_path}. "
                    f"This file should have been created during MTX alignment."
                )
            
            print(f"\nLoading column nnz from {os.path.basename(column_nnz_path)}...")
            column_nnz = np.loadtxt(column_nnz_path, dtype=np.uint32)
            
            if len(column_nnz) != len(genes):
                raise ValueError(
                    f"Column nnz count ({len(column_nnz)}) doesn't match gene count ({len(genes)}). "
                    f"This indicates a mismatch in the alignment process."
                )
            
            print(f"  ✓ Loaded nnz for {len(column_nnz)} columns")
            
            # Create var DataFrame with gene list and nnz column
            var_df = pl.DataFrame({
                'gene': genes,
                'index': range(len(genes)),
                'nnz': column_nnz.tolist()
            })
            
            var_output_path = zdata_dir / "var.parquet"
            var_df.write_parquet(str(var_output_path), compression="zstd")
            print(f"\n✓ Gene list saved to var.parquet")
            print(f"  Output: {var_output_path}")
            print(f"  Genes: {len(genes)}")
            print(f"  Column nnz: included")
        except Exception as e:
            print(f"\n✗ ERROR: Failed to save var.parquet: {e}")
            import traceback
            traceback.print_exc()
            if use_custom_mtx_dir:
                print(f"\n⚠ MTX files preserved in: {temp_mtx_dir}")
                print(f"  You can rerun the pipeline with --mtx-temp-dir {temp_mtx_dir} to continue from MTX files")
            raise
    
    finally:
        # Clean up temporary directory only if:
        # 1. temp_dir_context was actually created (not using custom dir)
        # 2. cleanup_temp is True
        if temp_dir_context_entered and cleanup_temp:
            try:
                temp_dir_context.__exit__(None, None, None)
            except:
                pass
    
    print("\n" + "=" * 70)
    print("✓ Complete zdata object built successfully!")
    print("=" * 70)
    # Ensure zdata_dir is a Path object
    zdata_dir = Path(zdata_dir)
    print(f"Output directory: {zdata_dir}")
    
    # Verify output structure
    zdata_path = zdata_dir
    xrm_dir = zdata_path / "X_RM"
    xcm_dir = zdata_path / "X_CM"
    metadata_file = zdata_path / "metadata.json"
    obs_file = zdata_path / obs_output_filename
    var_file = zdata_path / "var.parquet"
    
    print(f"\nOutput structure:")
    if metadata_file.exists():
        print(f"  ✓ metadata.json")
    if xrm_dir.exists() and list(xrm_dir.glob("*.bin")):
        print(f"  ✓ X_RM/ ({len(list(xrm_dir.glob('*.bin')))} chunk files)")
    if xcm_dir.exists() and list(xcm_dir.glob("*.bin")):
        print(f"  ✓ X_CM/ ({len(list(xcm_dir.glob('*.bin')))} chunk files)")
    if obs_file.exists():
        print(f"  ✓ {obs_output_filename}")
    if var_file.exists():
        print(f"  ✓ var.parquet")
    
    return zdata_dir


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Build complete zdata object from directory of zarr or h5ad files. '
                    'Auto-detects file type based on extensions: .zarr (directories) or .h5/.hdf5 (h5ad files). '
                    'This orchestrates the full pipeline: alignment, zdata build, and obs concatenation.'
    )
    parser.add_argument(
        'zarr_dir',
        type=str,
        help='Directory containing .zarr files (directories) or .h5/.hdf5 files (h5ad format) to process'
    )
    parser.add_argument(
        'output_name',
        type=str,
        help='Output directory name (can include .zdata suffix, e.g., "atlas.zdata" or "atlas")'
    )
    parser.add_argument(
        '--gene-list',
        type=str,
        default=None,
        help='Path to file containing standard gene list (one gene per line). Default: uses package default'
    )
    parser.add_argument(
        '--block-rows',
        type=int,
        default=16,
        help='Number of rows per block for row-major (X_RM) files (default: 16)'
    )
    parser.add_argument(
        '--block-columns',
        type=int,
        default=None,
        help='Number of rows per block for column-major (X_CM) files (default: same as --block-rows)'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=8192,
        help='Maximum rows per chunk for row-major files (default: 8192)'
    )
    parser.add_argument(
        '--max-columns',
        type=int,
        default=256,
        help='Maximum rows per chunk for column-major files (default: 256)'
    )
    parser.add_argument(
        '--obs-join-strategy',
        type=str,
        choices=['inner', 'outer', 'columns'],
        default='outer',
        help='Strategy for joining obs data: "inner" (only common columns), "outer" (all columns), or "columns" (only specified columns). Default: outer'
    )
    parser.add_argument(
        '--obs-join-on',
        type=str,
        nargs='+',
        default=None,
        help='Column names to join on when --obs-join-strategy is "columns". Example: --obs-join-on barcode sample_id'
    )
    parser.add_argument(
        '--obs-output-filename',
        type=str,
        default='obs.parquet',
        help='Name of obs parquet file (default: obs.parquet)'
    )
    parser.add_argument(
        '--mtx-chunk-size',
        type=int,
        default=131072,
        help='Maximum rows per MTX file during alignment (default: 131072)'
    )
    parser.add_argument(
        '--mtx-temp-dir',
        type=str,
        default=None,
        help='Directory to store MTX files. If specified, MTX files will be preserved (not cleaned up) '
             'even on failure, allowing the pipeline to be rerun from aligned MTX files. '
             'If existing MTX files are found, they will be reused.'
    )
    parser.add_argument(
        '--no-cleanup-temp',
        action='store_true',
        help='Do not clean up temporary MTX files (only applies when --mtx-temp-dir is not specified)'
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
    
    # Validate obs join strategy
    if args.obs_join_strategy == 'columns' and not args.obs_join_on:
        print("ERROR: --obs-join-on must be specified when --obs-join-strategy is 'columns'")
        sys.exit(1)
    
    # Validate parameters
    if args.block_rows < 1 or args.block_rows > 256:
        print(f"ERROR: block_rows must be between 1 and 256, got {args.block_rows}")
        sys.exit(1)
    
    block_columns = args.block_columns if args.block_columns is not None else args.block_rows
    if block_columns < 1 or block_columns > 256:
        print(f"ERROR: block_columns must be between 1 and 256, got {block_columns}")
        sys.exit(1)
    
    if args.max_rows < 1 or args.max_rows > 1000000:
        print(f"ERROR: max_rows must be between 1 and 1000000, got {args.max_rows}")
        sys.exit(1)
    
    if args.max_columns < 1 or args.max_columns > 1000000:
        print(f"ERROR: max_columns must be between 1 and 1000000, got {args.max_columns}")
        sys.exit(1)
    
    try:
        zdata_dir = build_zdata_from_zarr(
            str(zarr_dir_path),
            args.output_name,
            gene_list_path=args.gene_list,
            block_rows=args.block_rows,
            block_columns=block_columns,
            max_rows=args.max_rows,
            max_columns=args.max_columns,
            obs_join_strategy=args.obs_join_strategy,
            obs_join_on=args.obs_join_on,
            obs_output_filename=args.obs_output_filename,
            cleanup_temp=not args.no_cleanup_temp,
            mtx_chunk_size=args.mtx_chunk_size,
            mtx_temp_dir=args.mtx_temp_dir
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

