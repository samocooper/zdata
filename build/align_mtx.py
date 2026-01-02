#!/usr/bin/env python3
"""
Align columns (genes) in zarr files to a standard gene list and output as MTX format.
Processes a directory of zarr files, concatenating them into MTX files (max 131072 rows each by default).
Outputs multiple MTX files named by row range (e.g., rows_0_131071.mtx) and a manifest file.
These files can be converted to zdata format using mtx_to_zdata.
"""

import zarr
from scipy.sparse import csr_matrix, csc_matrix, vstack, hstack
from scipy.io import mmwrite, mmread
import numpy as np
import sys
import os
import json
import argparse
from pathlib import Path
import shutil

# Default gene list path (relative to zdata package)
# This file is required and must be included in the package distribution
_DEFAULT_GENE_LIST = Path(__file__).parent.parent / "files" / "2ks10c_genes.txt"

def get_default_gene_list_path() -> Path:
    """Get the path to the default gene list file.
    
    This function handles both development and installed package scenarios.
    The file must be included in the package distribution.
    
    Returns:
        Path to the default gene list file
        
    Raises:
        FileNotFoundError: If the default gene list file is not found
    """
    gene_list_path = _DEFAULT_GENE_LIST
    
    # If file doesn't exist at expected location, try to find it
    if not gene_list_path.exists():
        # Try alternative locations (for installed packages)
        import zdata
        package_dir = Path(zdata.__file__).parent
        alternative_path = package_dir / "files" / "2ks10c_genes.txt"
        if alternative_path.exists():
            return alternative_path
    
    if not gene_list_path.exists():
        raise FileNotFoundError(
            f"Default gene list file not found at {gene_list_path}. "
            f"This file (files/2ks10c_genes.txt) is required and must be included in the package distribution."
        )
    
    return gene_list_path

def process_zarr_file(zarr_path, gene_list_path, old_to_new_idx, n_new_cols):
    """
    Process a single zarr file and return aligned CSR matrix.
    Only keeps CSR matrix in memory, frees zarr resources immediately.
    
    Returns:
        csr_matrix: Aligned CSR matrix
        n_rows: Number of rows processed
    
    Raises:
        RuntimeError: If processing fails
    """
    zarr_group = zarr.open(zarr_path, mode='r')
    
    try:
        n_cols = zarr_group["var"]["gene"].shape[0]
        n_rows = zarr_group["obs"]["barcode"].shape[0]
        
        X = zarr_group['X']
        indptr = X["indptr"][:]
        data = X["data"][:]
        indices = X["indices"][:]
        
        if len(indptr) > 0:
            start = indptr[0]
            indptr = indptr - start
        
        X_csr = csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))
        X_csc = X_csr.tocsc()
        
        old_data = X_csc.data
        old_indices = X_csc.indices
        old_indptr = X_csc.indptr
        n_old_cols = X_csc.shape[1]
        
        new_col_data = [[] for _ in range(n_new_cols)]
        new_col_indices = [[] for _ in range(n_new_cols)]
        
        for old_col in range(n_old_cols):
            if old_col in old_to_new_idx:
                new_col = old_to_new_idx[old_col]
                col_start = old_indptr[old_col]
                col_end = old_indptr[old_col + 1]
                new_col_data[new_col].extend(old_data[col_start:col_end])
                new_col_indices[new_col].extend(old_indices[col_start:col_end])
        
        new_data = []
        new_indices = []
        new_indptr = [0]
        
        for new_col in range(n_new_cols):
            new_data.extend(new_col_data[new_col])
            new_indices.extend(new_col_indices[new_col])
            new_indptr.append(len(new_data))
        
        X_chunk_reordered = csc_matrix((new_data, new_indices, new_indptr), 
                                       shape=(n_rows, n_new_cols)).tocsr()
        
        del X_csr, X_csc
        
        return X_chunk_reordered, n_rows
        
    except Exception as e:
        raise RuntimeError(f"Failed to process zarr file {zarr_path}: {e}") from e

def align_zarr_directory_to_mtx(zarr_dir, gene_list_path, output_dir, tmp_dir=None, chunk_size=131072):
    """
    Process a directory of zarr files, align columns to standard gene list, and write as MTX files.
    Concatenates multiple zarr files into single MTX files (max chunk_size rows each).
    
    Args:
        zarr_dir: Directory containing .zarr files to process
        gene_list_path: Path to file containing standard gene list (one per line)
        output_dir: Directory where output MTX files will be written
        tmp_dir: Optional temporary directory for intermediate files
        chunk_size: Maximum number of rows per MTX file (default: 131072)
    
    Returns:
        Path to manifest file
    """
    with open(gene_list_path, 'r') as f:
        gene_list = [line.strip() for line in f if line.strip()]
    
    if not gene_list:
        raise ValueError(f"No genes found in {gene_list_path}")
    
    print(f"Standard gene list contains {len(gene_list)} genes")
    n_new_cols = len(gene_list)
    
    zarr_dir_path = Path(zarr_dir)
    zarr_files = sorted([f for f in zarr_dir_path.glob("*.zarr") if f.is_dir()])
    
    if not zarr_files:
        raise ValueError(f"No .zarr files found in {zarr_dir}")
    
    print(f"Found {len(zarr_files)} zarr file(s) to process")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    mtx_output_dir = os.path.join(output_dir, "rm_mtx_files")
    if not os.path.exists(mtx_output_dir):
        os.makedirs(mtx_output_dir, exist_ok=True)
        print(f"Created MTX output directory: {mtx_output_dir}")
    
    print(f"\nProcessing zarr files and creating MTX files (max {chunk_size} rows per file)")
    
    output_files = []
    manifest_data = []
    column_nnz_accumulator = np.zeros(n_new_cols, dtype=np.uint32)
    
    current_chunk_rows = []
    current_chunk_zarrs = []  # Track which zarr files contributed to current chunk
    current_row_start = 0
    mtx_file_idx = 0
    
    for zarr_idx, zarr_path in enumerate(zarr_files):
        print(f"\n[{zarr_idx + 1}/{len(zarr_files)}] Processing: {zarr_path.name}")
        
        zarr_group = zarr.open(str(zarr_path), mode='r')
        if 'var' not in zarr_group or 'gene' not in zarr_group['var']:
            raise ValueError(f"Zarr file {zarr_path.name} is missing required 'var/gene' array. All zarr files must have this structure.")
        
        zarr_genes = zarr_group['var']['gene'][:].tolist()
        
        gene_to_old_idx = {gene: idx for idx, gene in enumerate(zarr_genes)}
        old_to_new_idx = {}
        for new_idx, gene in enumerate(gene_list):
            if gene in gene_to_old_idx:
                old_col_idx = gene_to_old_idx[gene]
                old_to_new_idx[old_col_idx] = new_idx
        
        X_aligned, n_rows = process_zarr_file(str(zarr_path), gene_list_path, old_to_new_idx, n_new_cols)
        
        if X_aligned is None:
            raise RuntimeError(f"Failed to process zarr file {zarr_path.name}. This file must be processed successfully - skipping is not allowed.")
        
        print(f"  Processed {n_rows} rows from {zarr_path.name}")
        
        current_chunk_rows.append(X_aligned)
        current_chunk_zarrs.append({
            'zarr_file': zarr_path.name,
            'zarr_path': str(zarr_path),
            'rows_in_chunk': n_rows,
            'row_start_in_chunk': sum(m.shape[0] for m in current_chunk_rows[:-1])
        })
        
        total_rows_in_chunk = sum(m.shape[0] for m in current_chunk_rows)
        
        if total_rows_in_chunk >= chunk_size:
            print(f"  Concatenating {len(current_chunk_rows)} matrices ({total_rows_in_chunk} total rows)")
            combined_matrix = vstack(current_chunk_rows, format='csr')
            
            rows_written = 0
            zarr_files_written = []
            remainder_zarr_info = None
            
            if combined_matrix.shape[0] > chunk_size:
                matrix_to_write = combined_matrix[:chunk_size]
                remainder = combined_matrix[chunk_size:]
                
                for zarr_info in current_chunk_zarrs:
                    zarr_rows = zarr_info['rows_in_chunk']
                    if rows_written + zarr_rows <= chunk_size:
                        zarr_files_written.append(zarr_info.copy())
                        rows_written += zarr_rows
                    else:
                        rows_from_this_zarr_in_written = chunk_size - rows_written
                        remainder_rows = zarr_rows - rows_from_this_zarr_in_written
                        
                        partial_info = zarr_info.copy()
                        partial_info['rows_in_chunk'] = rows_from_this_zarr_in_written
                        zarr_files_written.append(partial_info)
                        
                        remainder_zarr_info = {
                            'zarr_file': zarr_info['zarr_file'],
                            'zarr_path': zarr_info['zarr_path'],
                            'rows_in_chunk': remainder_rows,
                            'row_start_in_chunk': 0
                        }
                        break
                
                current_chunk_rows = [remainder]
                current_chunk_zarrs = [remainder_zarr_info] if remainder_zarr_info else []
            else:
                matrix_to_write = combined_matrix
                zarr_files_written = current_chunk_zarrs.copy()
                current_chunk_rows = []
                current_chunk_zarrs = []
            
            row_nnz = np.diff(matrix_to_write.indptr).astype(np.uint32)
            column_nnz_chunk = matrix_to_write.getnnz(axis=0).astype(np.uint32)
            column_nnz_accumulator += column_nnz_chunk
            
            row_end = current_row_start + matrix_to_write.shape[0] - 1
            chunk_output_path = os.path.join(mtx_output_dir, f"rows_{current_row_start}_{row_end}.mtx")
            output_files.append(chunk_output_path)
            
            print(f"  Writing MTX file: {os.path.basename(chunk_output_path)}")
            mmwrite(chunk_output_path, matrix_to_write)
            print(f"  ✓ {matrix_to_write.shape[0]} rows × {n_new_cols} cols, {matrix_to_write.nnz} non-zeros")
            
            row_nnz_path = os.path.join(mtx_output_dir, f"rows_{current_row_start}_{row_end}_nnz.txt")
            np.savetxt(row_nnz_path, row_nnz, fmt='%u', delimiter='\n')
            
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
            
            del combined_matrix, matrix_to_write
    
    # Write final chunk if there are remaining rows
    if current_chunk_rows:
        print(f"\nWriting final chunk with {sum(m.shape[0] for m in current_chunk_rows)} rows")
        combined_matrix = vstack(current_chunk_rows, format='csr')
        
        row_end = current_row_start + combined_matrix.shape[0] - 1
        chunk_output_path = os.path.join(mtx_output_dir, f"rows_{current_row_start}_{row_end}.mtx")
        output_files.append(chunk_output_path)
        
        # Calculate row nnz (number of non-zeros per row) from CSR indptr
        row_nnz = np.diff(combined_matrix.indptr).astype(np.uint32)
        
        # Calculate column nnz directly from CSR matrix (no transposition needed)
        column_nnz_chunk = combined_matrix.getnnz(axis=0).astype(np.uint32)
        column_nnz_accumulator += column_nnz_chunk
        
        print(f"  Writing MTX file: {os.path.basename(chunk_output_path)}")
        mmwrite(chunk_output_path, combined_matrix)
        print(f"  ✓ {combined_matrix.shape[0]} rows × {n_new_cols} cols, {combined_matrix.nnz} non-zeros")
        
        # Write row nnz to temporary file (will be merged into obs.parquet later)
        row_nnz_path = os.path.join(mtx_output_dir, f"rows_{current_row_start}_{row_end}_nnz.txt")
        np.savetxt(row_nnz_path, row_nnz, fmt='%u', delimiter='\n')
        
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
    
    # Write column nnz to file (accumulated across all MTX files)
    column_nnz_path = os.path.join(output_dir, "column_nnz.txt")
    np.savetxt(column_nnz_path, column_nnz_accumulator, fmt='%u', delimiter='\n')
    print(f"  ✓ Column nnz saved to {os.path.basename(column_nnz_path)}")
    
    # Write manifest file
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {
        'gene_list_file': str(gene_list_path),
        'n_genes': n_new_cols,
        'zarr_files_processed': [str(f) for f in zarr_files],
        'zarr_files_order': [f.name for f in zarr_files],
        'mtx_files': manifest_data,
        'total_mtx_files': len(output_files),
        'chunk_size': chunk_size,
        'column_nnz_file': 'column_nnz.txt'
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Calculate total rows processed
    total_rows = sum(entry['n_rows'] for entry in manifest_data)
    
    print(f"\n✓ Successfully wrote {len(output_files)} aligned MTX file(s)")
    print(f"  Total rows processed: {total_rows}")
    print(f"  Total columns: {n_new_cols}")
    print(f"  Output directory: {output_dir}")
    print(f"  MTX files directory: {mtx_output_dir}")
    print(f"  Manifest file: {manifest_path}")
    
    # Second pass: Create column-major fragments for efficient column access
    print(f"\n{'='*70}")
    print("Creating column-major fragments for efficient column access...")
    print(f"{'='*70}")
    create_column_major_fragments(output_dir, mtx_output_dir, output_files, n_new_cols)
    
    return manifest_path

def create_column_major_fragments(output_dir, mtx_output_dir, mtx_files, n_cols):
    """
    Create column-major fragments by:
    1. Fragmenting each MTX file into 256-column chunks
    2. Transposing each fragment (CSR -> CSC -> CSR for transposed view)
    3. Saving transposed fragments to tmp directory
    4. Combining fragments covering the same column range
    5. Saving combined fragments to cm_mtx_files directory
    
    Args:
        output_dir: Base output directory
        mtx_output_dir: Directory containing row-major MTX files
        mtx_files: List of MTX file paths
        n_cols: Total number of columns
    """
    # Create tmp and cm_mtx_files directories
    tmp_dir = os.path.join(output_dir, "tmp")
    cm_mtx_dir = os.path.join(output_dir, "cm_mtx_files")
    
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(cm_mtx_dir, exist_ok=True)
    
    print(f"  Created tmp directory: {tmp_dir}")
    print(f"  Created cm_mtx_files directory: {cm_mtx_dir}")
    
    # Fragment size (columns per fragment)
    fragment_cols = 256
    num_fragments = (n_cols + fragment_cols - 1) // fragment_cols
    
    print(f"\n  Fragmenting {len(mtx_files)} MTX files into {num_fragments} column fragments ({fragment_cols} cols each)...")
    
    # Step 1: Fragment and transpose each MTX file, save to tmp
    # Structure: tmp/{mtx_file_basename}_frag_{frag_idx}.mtx
    fragment_map = {}  # Maps (frag_idx) -> list of (mtx_file, frag_file_path)
    
    for mtx_idx, mtx_file in enumerate(mtx_files):
        mtx_basename = os.path.basename(mtx_file)
        print(f"    [{mtx_idx + 1}/{len(mtx_files)}] Processing {mtx_basename}...")
        
        # Load MTX file and convert to CSC for efficient column indexing
        matrix = mmread(mtx_file)
        # Convert to CSC format for efficient column indexing (mmread may return COO)
        if not isinstance(matrix, csc_matrix):
            matrix = matrix.tocsc()
        n_rows, n_cols_file = matrix.shape
        
        # Fragment by columns
        for frag_idx in range(num_fragments):
            col_start = frag_idx * fragment_cols
            col_end = min(col_start + fragment_cols, n_cols_file)
            
            if col_start >= n_cols_file:
                break
            
            # Extract column fragment using CSC indexing (efficient column access)
            col_fragment = matrix[:, col_start:col_end]
            
            # Transpose: rows become columns (for column-major access)
            # Transpose returns CSR format
            transposed_fragment = col_fragment.T.tocsr()
            
            # Save transposed fragment to tmp
            frag_filename = f"{os.path.splitext(mtx_basename)[0]}_frag_{frag_idx}.mtx"
            frag_path = os.path.join(tmp_dir, frag_filename)
            mmwrite(frag_path, transposed_fragment)
            
            # Track this fragment
            if frag_idx not in fragment_map:
                fragment_map[frag_idx] = []
            fragment_map[frag_idx].append((mtx_file, frag_path))
            
            # Free memory
            del col_fragment
            del transposed_fragment
        
        # Free original matrix
        del matrix
    
    print(f"  ✓ Created {sum(len(frags) for frags in fragment_map.values())} transposed fragments in tmp directory")
    
    # Step 2: Combine fragments covering the same column range
    print(f"\n  Combining fragments by column range...")
    
    for frag_idx in range(num_fragments):
        if frag_idx not in fragment_map:
            continue
        
        frag_list = fragment_map[frag_idx]
        col_start = frag_idx * fragment_cols
        col_end = min(col_start + fragment_cols, n_cols)
        
        print(f"    Fragment {frag_idx}: columns {col_start}-{col_end-1} ({len(frag_list)} fragments to combine)...")
        
        # Load and combine all fragments for this column range
        combined_fragments = []
        for mtx_file, frag_path in frag_list:
            frag_matrix = mmread(frag_path)
            # Convert to CSR format if needed
            if not isinstance(frag_matrix, csr_matrix):
                frag_matrix = frag_matrix.tocsr()
            combined_fragments.append(frag_matrix)
            # Free memory immediately
            del frag_matrix
        
        # Stack fragments horizontally (hstack): each fragment adds more columns (cells)
        # After transposition: rows = genes (256), columns = cells
        # Combining fragments from different MTX files adds more cells (columns)
        if combined_fragments:
            combined_matrix = hstack(combined_fragments, format='csr')
            
            # Save combined fragment
            cm_filename = f"cols_{col_start}_{col_end-1}.mtx"
            cm_path = os.path.join(cm_mtx_dir, cm_filename)
            mmwrite(cm_path, combined_matrix)
            
            print(f"      ✓ Combined {len(combined_fragments)} fragments -> {combined_matrix.shape[0]} rows × {combined_matrix.shape[1]} cols")
            
            # Free memory
            del combined_matrix
            del combined_fragments
    
    print(f"\n  ✓ Created {num_fragments} column-major MTX files in {cm_mtx_dir}")
    print(f"  ✓ Column-major fragments allow efficient access to gene columns")
    
    # Clean up temporary fragments
    print(f"\n  Cleaning up temporary fragments...")
    try:
        shutil.rmtree(tmp_dir)
        print(f"  ✓ Removed temporary directory: {tmp_dir}")
    except Exception as e:
        print(f"  WARNING: Could not remove temporary directory {tmp_dir}: {e}")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Align zarr file(s) columns to standard gene list and output as MTX format. '
                    'Processes a directory of zarr files, concatenating them into MTX files (max 131072 rows each by default).'
    )
    parser.add_argument(
        'zarr_input',
        type=str,
        help='Path to input .zarr file or directory containing .zarr files'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory where MTX files will be written in rm_mtx_files subdirectory (files named by row range, e.g., rows_0_131071.mtx)'
    )
    parser.add_argument(
        '--gene-list',
        type=str,
        default=None,
        help='Path to file containing standard gene list (one gene per line). Default: uses package default (files/2ks10c_genes.txt)'
    )
    parser.add_argument(
        '--tmp-dir',
        type=str,
        default=None,
        help='Optional temporary directory for intermediate files'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=131072,
        help='Maximum number of rows per MTX file (default: 131072)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    zarr_input_path = Path(args.zarr_input)
    if not zarr_input_path.exists():
        print(f"ERROR: Zarr file/directory not found: {args.zarr_input}")
        sys.exit(1)
    
    # Use default gene list if not provided
    if args.gene_list is None:
        try:
            gene_list_path = str(get_default_gene_list_path())
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        gene_list_path = args.gene_list
    
    if not os.path.exists(gene_list_path):
        print(f"ERROR: Gene list file not found: {gene_list_path}")
        if args.gene_list is None:
            print(f"  This is the default gene list file that should be included in the package.")
        else:
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
                args.tmp_dir,
                args.chunk_size
            )
        elif str(zarr_input_path).endswith('.zarr'):
            # Single zarr file - treat parent directory as input (will only find this one file)
            zarr_dir = zarr_input_path.parent
            align_zarr_directory_to_mtx(
                str(zarr_dir),
                gene_list_path,
                args.output_dir,
                args.tmp_dir,
                args.chunk_size
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
