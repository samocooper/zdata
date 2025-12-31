#!/usr/bin/env python3
"""
Align columns (genes) in zarr files to a standard gene list and output as MTX format.
Processes a directory of zarr files, concatenating them into MTX files (max 131072 rows each).
Outputs multiple MTX files named by row range (e.g., rows_0_131071.mtx) and a manifest file.
These files can be converted to zdata format using mtx_to_zdata.
"""

import zarr
from scipy.sparse import csr_matrix, csc_matrix, vstack
from scipy.io import mmwrite
import numpy as np
import sys
import os
import json
import argparse
from pathlib import Path

# Default gene list path (relative to zdata package)
_DEFAULT_GENE_LIST = Path(__file__).parent.parent / "files" / "2ks10c_genes.txt"

def process_zarr_file(zarr_path, gene_list_path, old_to_new_idx, n_new_cols):
    """
    Process a single zarr file and return aligned CSR matrix.
    Only keeps CSR matrix in memory, frees zarr resources immediately.
    
    Returns:
        csr_matrix: Aligned CSR matrix, or None if error
        n_rows: Number of rows processed
    """
    try:
        zarr_group = zarr.open(zarr_path, mode='r')
        
        # Get dimensions
        n_cols = zarr_group["var"]["gene"].shape[0]
        n_rows = zarr_group["obs"]["barcode"].shape[0]
        
        X = zarr_group['X']
        
        # Load entire zarr file as CSR (memory efficient for sparse data)
        indptr = X["indptr"][:]
        data = X["data"][:]
        indices = X["indices"][:]
        
        # Rebase indptr to start at zero
        if len(indptr) > 0:
            start = indptr[0]
            indptr = indptr - start
        
        X_csr = csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))
        
        # Convert to CSC for column reordering
        X_csc = X_csr.tocsc()
        
        # Get CSC arrays
        old_data = X_csc.data
        old_indices = X_csc.indices
        old_indptr = X_csc.indptr
        n_old_cols = X_csc.shape[1]
        
        # Initialize column storage
        new_col_data = [[] for _ in range(n_new_cols)]
        new_col_indices = [[] for _ in range(n_new_cols)]
        
        # Iterate through old columns and map to new positions
        for old_col in range(n_old_cols):
            if old_col in old_to_new_idx:
                new_col = old_to_new_idx[old_col]
                col_start = old_indptr[old_col]
                col_end = old_indptr[old_col + 1]
                new_col_data[new_col].extend(old_data[col_start:col_end])
                new_col_indices[new_col].extend(old_indices[col_start:col_end])
        
        # Build new CSC arrays
        new_data = []
        new_indices = []
        new_indptr = [0]
        
        for new_col in range(n_new_cols):
            new_data.extend(new_col_data[new_col])
            new_indices.extend(new_col_indices[new_col])
            new_indptr.append(len(new_data))
        
        # Create new CSC matrix and convert to CSR
        X_chunk_reordered = csc_matrix((new_data, new_indices, new_indptr), 
                                       shape=(n_rows, n_new_cols)).tocsr()
        
        # Close zarr file to free memory
        del zarr_group
        del X_csr
        del X_csc
        
        return X_chunk_reordered, n_rows
        
    except Exception as e:
        print(f"  ERROR processing {zarr_path}: {e}")
        return None, 0

def align_zarr_directory_to_mtx(zarr_dir, gene_list_path, output_dir, tmp_dir=None):
    """
    Process a directory of zarr files, align columns to standard gene list, and write as MTX files.
    Concatenates multiple zarr files into single MTX files (max 131072 rows each).
    
    Args:
        zarr_dir: Directory containing .zarr files to process
        gene_list_path: Path to file containing standard gene list (one per line)
        output_dir: Directory where output MTX files will be written
        tmp_dir: Optional temporary directory for intermediate files
    
    Returns:
        Path to manifest file
    """
    # Read standard gene list
    with open(gene_list_path, 'r') as f:
        gene_list = [line.strip() for line in f if line.strip()]
    
    if not gene_list:
        raise ValueError(f"No genes found in {gene_list_path}")
    
    print(f"Standard gene list contains {len(gene_list)} genes")
    n_new_cols = len(gene_list)
    
    # Find all zarr files in directory (alphabetical order)
    zarr_dir_path = Path(zarr_dir)
    zarr_files = sorted([f for f in zarr_dir_path.glob("*.zarr") if f.is_dir()])
    
    if not zarr_files:
        raise ValueError(f"No .zarr files found in {zarr_dir}")
    
    print(f"Found {len(zarr_files)} zarr file(s) to process")
    
    # Build gene mapping from first zarr file (all should have same genes)
    print(f"\nReading gene mapping from first zarr file: {zarr_files[0].name}")
    first_zarr = zarr.open(str(zarr_files[0]), mode='r')
    gene_array = first_zarr['var']['gene']
    gene_values = gene_array[:].tolist()
    del first_zarr
    
    # Create mapping: old_col_idx -> new_col_idx
    gene_to_old_idx = {gene: idx for idx, gene in enumerate(gene_values)}
    old_to_new_idx = {}
    matched_genes = 0
    for new_idx, gene in enumerate(gene_list):
        if gene in gene_to_old_idx:
            old_to_new_idx[gene_to_old_idx[gene]] = new_idx
            matched_genes += 1
    
    print(f"Matched {matched_genes} genes from standard list to zarr files")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Process zarr files and accumulate rows
    chunk_size = 131072
    print(f"\nProcessing zarr files and creating MTX files (max {chunk_size} rows per file)")
    
    output_files = []
    manifest_data = []
    
    current_chunk_rows = []
    current_chunk_zarrs = []  # Track which zarr files contributed to current chunk
    current_row_start = 0
    mtx_file_idx = 0
    
    for zarr_idx, zarr_path in enumerate(zarr_files):
        print(f"\n[{zarr_idx + 1}/{len(zarr_files)}] Processing: {zarr_path.name}")
        
        # Process this zarr file
        X_aligned, n_rows = process_zarr_file(str(zarr_path), gene_list_path, old_to_new_idx, n_new_cols)
        
        if X_aligned is None:
            print(f"  Skipping {zarr_path.name} due to error")
            continue
        
        print(f"  Processed {n_rows} rows from {zarr_path.name}")
        
        # Add to current chunk
        current_chunk_rows.append(X_aligned)
        current_chunk_zarrs.append({
            'zarr_file': zarr_path.name,
            'zarr_path': str(zarr_path),
            'rows_in_chunk': n_rows,
            'row_start_in_chunk': sum(m.shape[0] for m in current_chunk_rows[:-1])
        })
        
        total_rows_in_chunk = sum(m.shape[0] for m in current_chunk_rows)
        
        # If we've reached or exceeded chunk size, write MTX file
        if total_rows_in_chunk >= chunk_size:
            # Concatenate matrices
            print(f"  Concatenating {len(current_chunk_rows)} matrices ({total_rows_in_chunk} total rows)")
            combined_matrix = vstack(current_chunk_rows, format='csr')
            
            # Determine which zarr files contributed to this chunk (before potential split)
            zarr_files_for_manifest = current_chunk_zarrs.copy()
            
            # Determine which zarr files contributed to the written portion
            rows_written = 0
            zarr_files_written = []
            remainder_zarr_info = None
            
            # Trim to exactly chunk_size if needed
            if combined_matrix.shape[0] > chunk_size:
                # Split: write full chunk, keep remainder
                matrix_to_write = combined_matrix[:chunk_size]
                remainder = combined_matrix[chunk_size:]
                
                # Calculate which zarr files contributed to written portion and remainder
                for zarr_info in current_chunk_zarrs:
                    zarr_rows = zarr_info['rows_in_chunk']
                    if rows_written + zarr_rows <= chunk_size:
                        # Entire zarr file is in written portion
                        zarr_files_written.append(zarr_info.copy())
                        rows_written += zarr_rows
                    else:
                        # This zarr file is split
                        rows_from_this_zarr_in_written = chunk_size - rows_written
                        remainder_rows = zarr_rows - rows_from_this_zarr_in_written
                        
                        # Add partial zarr info to written portion
                        partial_info = zarr_info.copy()
                        partial_info['rows_in_chunk'] = rows_from_this_zarr_in_written
                        zarr_files_written.append(partial_info)
                        
                        # Create remainder info
                        remainder_zarr_info = {
                            'zarr_file': zarr_info['zarr_file'],
                            'zarr_path': zarr_info['zarr_path'],
                            'rows_in_chunk': remainder_rows,
                            'row_start_in_chunk': 0
                        }
                        break
                
                # Update tracking for remainder
                current_chunk_rows = [remainder]
                current_chunk_zarrs = [remainder_zarr_info] if remainder_zarr_info else []
            else:
                matrix_to_write = combined_matrix
                zarr_files_written = current_chunk_zarrs.copy()
                current_chunk_rows = []
                current_chunk_zarrs = []
            
            # Write MTX file
            row_end = current_row_start + matrix_to_write.shape[0] - 1
            chunk_output_path = os.path.join(output_dir, f"rows_{current_row_start}_{row_end}.mtx")
            output_files.append(chunk_output_path)
            
            print(f"  Writing MTX file: {os.path.basename(chunk_output_path)}")
            mmwrite(chunk_output_path, matrix_to_write)
            print(f"  ✓ {matrix_to_write.shape[0]} rows × {n_new_cols} cols, {matrix_to_write.nnz} non-zeros")
            
            # Record in manifest
            manifest_data.append({
                'mtx_file': os.path.basename(chunk_output_path),
                'mtx_path': chunk_output_path,
                'row_start': current_row_start,
                'row_end': row_end,
                'n_rows': matrix_to_write.shape[0],
                'zarr_files': zarr_files_written
            })
            
            current_row_start = row_end + 1
            mtx_file_idx += 1
            
            # Free memory
            del combined_matrix
            del matrix_to_write
    
    # Write final chunk if there are remaining rows
    if current_chunk_rows:
        print(f"\nWriting final chunk with {sum(m.shape[0] for m in current_chunk_rows)} rows")
        combined_matrix = vstack(current_chunk_rows, format='csr')
        
        row_end = current_row_start + combined_matrix.shape[0] - 1
        chunk_output_path = os.path.join(output_dir, f"rows_{current_row_start}_{row_end}.mtx")
        output_files.append(chunk_output_path)
        
        print(f"  Writing MTX file: {os.path.basename(chunk_output_path)}")
        mmwrite(chunk_output_path, combined_matrix)
        print(f"  ✓ {combined_matrix.shape[0]} rows × {n_new_cols} cols, {combined_matrix.nnz} non-zeros")
        
        # Record in manifest
        manifest_data.append({
            'mtx_file': os.path.basename(chunk_output_path),
            'mtx_path': chunk_output_path,
            'row_start': current_row_start,
            'row_end': row_end,
            'n_rows': combined_matrix.shape[0],
            'zarr_files': current_chunk_zarrs
        })
        
        del combined_matrix
    
    # Write manifest file
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {
        'gene_list_file': str(gene_list_path),
        'n_genes': n_new_cols,
        'matched_genes': matched_genes,
        'zarr_files_processed': [str(f) for f in zarr_files],
        'zarr_files_order': [f.name for f in zarr_files],
        'mtx_files': manifest_data,
        'total_mtx_files': len(output_files),
        'chunk_size': chunk_size
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Calculate total rows processed
    total_rows = sum(entry['n_rows'] for entry in manifest_data)
    
    print(f"\n✓ Successfully wrote {len(output_files)} aligned MTX file(s)")
    print(f"  Total rows processed: {total_rows}")
    print(f"  Total columns: {n_new_cols}")
    print(f"  Output directory: {output_dir}")
    print(f"  Manifest file: {manifest_path}")
    
    return manifest_path

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Align zarr file(s) columns to standard gene list and output as MTX format. '
                    'Processes a directory of zarr files, concatenating them into MTX files (max 131072 rows each).'
    )
    parser.add_argument(
        'zarr_input',
        type=str,
        help='Path to input .zarr file or directory containing .zarr files'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory where MTX files will be written (files named by row range, e.g., rows_0_131071.mtx)'
    )
    parser.add_argument(
        '--gene-list',
        type=str,
        default=str(_DEFAULT_GENE_LIST),
        help=f'Path to file containing standard gene list (one gene per line). Default: {_DEFAULT_GENE_LIST}'
    )
    parser.add_argument(
        '--tmp-dir',
        type=str,
        default=None,
        help='Optional temporary directory for intermediate files'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    zarr_input_path = Path(args.zarr_input)
    if not zarr_input_path.exists():
        print(f"ERROR: Zarr file/directory not found: {args.zarr_input}")
        sys.exit(1)
    
    gene_list_path = args.gene_list
    if not os.path.exists(gene_list_path):
        print(f"ERROR: Gene list file not found: {gene_list_path}")
        print(f"  Expected default location: {_DEFAULT_GENE_LIST}")
        print(f"  You can specify a different file with --gene-list")
        sys.exit(1)
    
    try:
        # Determine if input is a directory or single zarr file
        if zarr_input_path.is_dir():
            # Directory of zarr files
            align_zarr_directory_to_mtx(
                str(zarr_input_path),
                gene_list_path,
                args.output_dir,
                args.tmp_dir
            )
        elif str(zarr_input_path).endswith('.zarr'):
            # Single zarr file - treat parent directory as input (will only find this one file)
            zarr_dir = zarr_input_path.parent
            align_zarr_directory_to_mtx(
                str(zarr_dir),
                gene_list_path,
                args.output_dir,
                args.tmp_dir
            )
        else:
            print(f"ERROR: Input must be a directory containing .zarr files or a .zarr file/directory")
            sys.exit(1)
        print("\n✓ Alignment complete!")
        return 0
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
