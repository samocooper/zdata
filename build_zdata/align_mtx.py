#!/usr/bin/env python3
"""
Align columns (genes) in zarr or h5ad files to a standard gene list and output as MTX format.
Processes a directory of .zarr files (directories) or .h5/.hdf5 files (h5ad format),
concatenating them into MTX files (max 131072 rows each by default).
Auto-detects file type based on extensions: .zarr (directories) or .h5/.hdf5 (h5ad files).
Outputs multiple MTX files named by row range (e.g., rows_0_131071.mtx) and a manifest file.
These files can be converted to zdata format using mtx_to_zdata.
"""

import zarr
import anndata as ad
from scipy.sparse import csr_matrix, csc_matrix, vstack, hstack
from scipy.io import mmwrite, mmread
import numpy as np
import sys
import os
import json
import argparse
from pathlib import Path
import shutil
import gc

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

def _detect_file_type(file_path):
    """
    Detect file type based on extension and structure.
    
    Args:
        file_path: Path to file or directory
        
    Returns:
        str: 'zarr', 'h5ad', or None if unknown
    """
    path = Path(file_path)
    
    # Check if it's a directory (zarr files are directories)
    if path.is_dir() and path.name.endswith('.zarr'):
        return 'zarr'
    
    # Check file extensions for h5ad
    if path.is_file():
        if path.suffix in ['.h5', '.hdf5'] or path.name.endswith('.h5ad'):
            return 'h5ad'
        elif path.name.endswith('.zarr'):
            return 'zarr'
    
    return None

def _reorder_matrix_columns(X_csc, old_to_new_idx, n_new_cols):
    """
    Reorder columns of a CSC matrix according to old_to_new_idx mapping.
    Shared helper function for both zarr and h5ad processing.
    Uses efficient numpy arrays instead of Python lists to avoid memory bloat.
    
    Args:
        X_csc: CSC sparse matrix to reorder
        old_to_new_idx: Dictionary mapping old column indices to new column indices
        n_new_cols: Number of columns in the output matrix
    
    Returns:
        csr_matrix: Reordered matrix in CSR format
    """
    n_rows, n_old_cols = X_csc.shape
    
    # Use views to access data efficiently (no copying overhead)
    # Extract the data we need and release all references before returning
    nnz = X_csc.nnz
    old_data = X_csc.data
    old_indices = X_csc.indices
    old_indptr = X_csc.indptr
    
    # Step 1: Count non-zeros per new column from old_indptr (no data copying yet)
    col_counts = np.zeros(n_new_cols, dtype=np.int64)
    for old_col in range(n_old_cols):
        if old_col in old_to_new_idx:
            new_col = old_to_new_idx[old_col]
            col_start = old_indptr[old_col]
            col_end = old_indptr[old_col + 1]
            col_counts[new_col] += (col_end - col_start)
    
    # Step 2: Build new_indptr from counts (vectorized)
    new_indptr = np.zeros(n_new_cols + 1, dtype=np.int64)
    new_indptr[1:] = np.cumsum(col_counts)
    total_nnz = new_indptr[-1]
    
    # Step 3: Pre-allocate new arrays
    new_data = np.empty(total_nnz, dtype=old_data.dtype)
    new_indices = np.empty(total_nnz, dtype=old_indices.dtype)
    
    # Step 4: Track current position in each new column
    col_positions = np.zeros(n_new_cols, dtype=np.int64)
    
    # Step 5: Directly copy data from old positions to new positions using indptr mapping
    for old_col in range(n_old_cols):
        if old_col in old_to_new_idx:
            new_col = old_to_new_idx[old_col]
            old_start = old_indptr[old_col]
            old_end = old_indptr[old_col + 1]
            col_len = old_end - old_start
            
            if col_len > 0:
                # Calculate destination position in new arrays
                new_start = new_indptr[new_col] + col_positions[new_col]
                new_end = new_start + col_len
                
                # Direct copy from old position to new position (no intermediate lists!)
                new_data[new_start:new_end] = old_data[old_start:old_end]
                new_indices[new_start:new_end] = old_indices[old_start:old_end]
                
                col_positions[new_col] += col_len
    
    # Free old array views now that we've copied the data we need
    # This allows X_csc to be garbage collected
    del old_data, old_indices, old_indptr
    
    # Trim arrays if needed (in case some columns weren't mapped)
    actual_nnz = new_indptr[-1]
    if actual_nnz < nnz:
        new_data = new_data[:actual_nnz]
        new_indices = new_indices[:actual_nnz]

    # Build CSC matrix from reordered data
    X_chunk_reordered_csc = csc_matrix((new_data, new_indices, new_indptr), 
                                       shape=(n_rows, n_new_cols))
    
    # Convert to CSR
    X_chunk_reordered = X_chunk_reordered_csc.tocsr()
    del X_chunk_reordered_csc
    gc.collect()
    
    return X_chunk_reordered

def process_h5ad_file(h5ad_path, gene_list_path, old_to_new_idx, n_new_cols):
    """
    Process a single h5ad file and return aligned CSR matrix.
    
    Args:
        h5ad_path: Path to .h5 or .hdf5 file (h5ad format)
        gene_list_path: Path to gene list file (for consistency with zarr function)
        old_to_new_idx: Dictionary mapping old column indices to new column indices
        n_new_cols: Number of columns in the aligned matrix
    
    Returns:
        csr_matrix: Aligned CSR matrix
        n_rows: Number of rows processed
    
    Raises:
        RuntimeError: If processing fails
    """
    try:
        # Read h5ad file (h5ad files are loaded into memory by anndata)
        adata = ad.read_h5ad(h5ad_path)
        
        # Get matrix and close file
        X = adata.X
        adata.file.close()
        
        # Convert to CSR if needed
        if not isinstance(X, csr_matrix):
            X_csr = X.tocsr()
            if hasattr(X, 'shape') and X is not X_csr:
                del X
        else:
            X_csr = X
        
        n_rows, n_old_cols = X_csr.shape
        del adata
        
        # Convert to CSC and reorder columns
        X_csc = X_csr.tocsc()
        del X_csr
        gc.collect()
        
        X_chunk_reordered = _reorder_matrix_columns(X_csc, old_to_new_idx, n_new_cols)
        del X_csc
        gc.collect()
        
        return X_chunk_reordered, n_rows
        
    except Exception as e:
        raise RuntimeError(f"Failed to process h5ad file {h5ad_path}: {e}") from e

def process_zarr_file_chunks(zarr_path, gene_list_path, old_to_new_idx, n_new_cols, mtx_chunk_size=131072):
    """
    Process a single zarr file in chunks matching MTX file size.
    Yields aligned CSR matrices in chunks to avoid loading entire compressed zarr into memory.
    
    Args:
        zarr_path: Path to zarr file
        gene_list_path: Path to gene list (unused, kept for compatibility)
        old_to_new_idx: Dictionary mapping old column indices to new column indices
        n_new_cols: Number of columns in the aligned matrix
        mtx_chunk_size: Number of rows per chunk (default: 131072, matching MTX file size)
    
    Yields:
        tuple: (csr_matrix, n_rows_in_chunk, total_rows_in_file)
    
    Raises:
        RuntimeError: If processing fails
    """
    zarr_group = zarr.open(zarr_path, mode='r')
    
    try:
        n_cols = zarr_group["var"]["gene"].shape[0]
        n_rows = zarr_group["obs"]["barcode"].shape[0]
        
        X = zarr_group['X']
        # Get indptr to determine row boundaries (need full indptr for indexing)
        indptr = X["indptr"][:]
        
        if len(indptr) > 0:
            start = indptr[0]
            indptr = indptr - start
        
        # Process in row chunks matching MTX file size to avoid loading entire compressed file
        for chunk_start in range(0, n_rows, mtx_chunk_size):
            chunk_end = min(chunk_start + mtx_chunk_size, n_rows)
            chunk_n_rows = chunk_end - chunk_start
            
            # Extract row chunk boundaries from indptr
            chunk_indptr_start = indptr[chunk_start]
            chunk_indptr_end = indptr[chunk_end]
            
            # Load only the data/indices needed for this chunk (zarr decompresses on access)
            chunk_data = X["data"][chunk_indptr_start:chunk_indptr_end]
            chunk_indices = X["indices"][chunk_indptr_start:chunk_indptr_end]
            chunk_indptr = indptr[chunk_start:chunk_end + 1] - chunk_indptr_start
            
            # Create CSR matrix for this chunk
            chunk_csr = csr_matrix((chunk_data, chunk_indices, chunk_indptr), 
                                  shape=(chunk_n_rows, n_cols))
            del chunk_data, chunk_indices
            
            # Convert to CSC and reorder columns for this chunk
            chunk_csc = chunk_csr.tocsc()
            del chunk_csr
            chunk_aligned = _reorder_matrix_columns(chunk_csc, old_to_new_idx, n_new_cols)
            del chunk_csc
            gc.collect()
            
            # Yield this chunk
            yield chunk_aligned, chunk_n_rows, n_rows
            
            # Free chunk before next iteration
            del chunk_aligned
            gc.collect()
        
        # Clean up zarr references
        del indptr
        zarr_group = None
        gc.collect()
        
    except Exception as e:
        raise RuntimeError(f"Failed to process zarr file {zarr_path}: {e}") from e
        
    except Exception as e:
        raise RuntimeError(f"Failed to process zarr file {zarr_path}: {e}") from e

def align_zarr_directory_to_mtx(zarr_dir, gene_list_path, output_dir, tmp_dir=None, chunk_size=131072):
    """
    Process a directory of zarr or h5ad files, align columns to standard gene list, and write as MTX files.
    Concatenates multiple files into single MTX files (max chunk_size rows each).
    Auto-detects file type based on extensions: .zarr (directories) or .h5/.hdf5 (h5ad files).
    
    Args:
        zarr_dir: Directory containing .zarr files (directories) or .h5/.hdf5 files (h5ad) to process
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
    
    # Auto-detect and collect files (zarr directories and h5ad files)
    zarr_files = sorted([f for f in zarr_dir_path.glob("*.zarr") if f.is_dir()])
    h5ad_files = sorted([f for f in zarr_dir_path.iterdir() 
                         if f.is_file() and (f.suffix in ['.h5', '.hdf5'] or f.name.endswith('.h5ad'))])
    
    all_files = []
    file_types = {}
    
    for f in zarr_files:
        all_files.append(f)
        file_types[f] = 'zarr'
    
    for f in h5ad_files:
        all_files.append(f)
        file_types[f] = 'h5ad'
    
    if not all_files:
        raise ValueError(f"No .zarr files (directories) or .h5/.hdf5 files (h5ad) found in {zarr_dir}")
    
    print(f"Found {len(zarr_files)} zarr file(s) and {len(h5ad_files)} h5ad file(s) to process")
    print(f"Total files: {len(all_files)}")
    
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
    
    def write_mtx_chunk(chunk_rows, chunk_zarrs_info, row_start, mtx_output_dir, n_new_cols):
        """Write an MTX chunk and return updated tracking information.
        
        Args:
            chunk_rows: List of CSR matrices to combine and write
            chunk_zarrs_info: List of metadata about source files
            row_start: Starting row index for this chunk
            mtx_output_dir: Directory to write MTX files
            n_new_cols: Number of columns in the matrix
        
        Returns:
            tuple: (next_row_start, manifest_entry, column_nnz_chunk)
        """
        if not chunk_rows:
            return row_start, None, np.zeros(n_new_cols, dtype=np.uint32)
        
        combined_matrix = vstack(chunk_rows, format='csr')
        del chunk_rows
        gc.collect()
        
        row_end = row_start + combined_matrix.shape[0] - 1
        chunk_output_path = os.path.join(mtx_output_dir, f"rows_{row_start}_{row_end}.mtx")
        
        row_nnz = np.diff(combined_matrix.indptr).astype(np.uint32)
        column_nnz_chunk = combined_matrix.getnnz(axis=0).astype(np.uint32)
        
        print(f"  Writing MTX file: {os.path.basename(chunk_output_path)}")
        mmwrite(chunk_output_path, combined_matrix)
        print(f"  ✓ {combined_matrix.shape[0]} rows × {n_new_cols} cols, {combined_matrix.nnz} non-zeros")
        
        row_nnz_path = os.path.join(mtx_output_dir, f"rows_{row_start}_{row_end}_nnz.txt")
        np.savetxt(row_nnz_path, row_nnz, fmt='%u', delimiter='\n')
        
        manifest_entry = {
            'mtx_file': os.path.basename(chunk_output_path),
            'mtx_path': chunk_output_path,
            'row_start': row_start,
            'row_end': row_end,
            'n_rows': combined_matrix.shape[0],
            'source_files': chunk_zarrs_info.copy()
        }
        
        del combined_matrix
        gc.collect()
        
        return row_end + 1, manifest_entry, column_nnz_chunk
    
    current_chunk_rows = []
    current_chunk_zarrs = []  # Track which zarr files contributed to current chunk
    current_row_start = 0
    mtx_file_idx = 0
    
    for file_idx, file_path in enumerate(all_files):
        file_type = file_types[file_path]
        print(f"\n[{file_idx + 1}/{len(all_files)}] Processing {file_type.upper()}: {file_path.name}")
        
        # Get genes based on file type
        if file_type == 'zarr':
            zarr_group = zarr.open(str(file_path), mode='r')
            if 'var' not in zarr_group or 'gene' not in zarr_group['var']:
                raise ValueError(f"Zarr file {file_path.name} is missing required 'var/gene' array. All zarr files must have this structure.")
            file_genes = zarr_group['var']['gene'][:].tolist()
        elif file_type == 'h5ad':
            # Use full loading for gene list extraction (small operation)
            adata = ad.read_h5ad(file_path)
            # Try var_names (standard anndata) or var/gene (if exists)
            if hasattr(adata.var, 'index'):
                file_genes = adata.var.index.tolist()
            elif 'gene' in adata.var.columns:
                file_genes = adata.var['gene'].tolist()
            else:
                # Fallback to var_names
                file_genes = adata.var_names.tolist()
            # Close file immediately to free memory
            adata.file.close()
            del adata
            gc.collect()
        else:
            raise ValueError(f"Unknown file type for {file_path.name}")
        
        # Create gene mapping
        gene_to_old_idx = {gene: idx for idx, gene in enumerate(file_genes)}
        old_to_new_idx = {}
        for new_idx, gene in enumerate(gene_list):
            if gene in gene_to_old_idx:
                old_col_idx = gene_to_old_idx[gene]
                old_to_new_idx[old_col_idx] = new_idx
        
        # Process file in chunks matching MTX file size (131072 rows)
        if file_type == 'zarr':
            # Process zarr file in chunks using generator
            for chunk_matrix, chunk_n_rows, total_file_rows in process_zarr_file_chunks(
                str(file_path), gene_list_path, old_to_new_idx, n_new_cols, chunk_size
            ):
                # Add chunk to current accumulation
                current_chunk_rows.append(chunk_matrix)
                current_chunk_zarrs.append({
                    'file': file_path.name,
                    'file_path': str(file_path),
                    'file_type': file_type,
                    'rows_in_chunk': chunk_n_rows,
                    'row_start_in_chunk': sum(m.shape[0] for m in current_chunk_rows[:-1])
                })
                
                total_rows_in_chunk = sum(m.shape[0] for m in current_chunk_rows)
                
                # Write MTX file when we've accumulated enough rows
                if total_rows_in_chunk >= chunk_size:
                    if total_rows_in_chunk > chunk_size:
                        # Write exactly chunk_size rows
                        rows_to_write = []
                        rows_written = 0
                        zarr_info_to_write = []
                        
                        for i, matrix in enumerate(current_chunk_rows):
                            matrix_rows = matrix.shape[0]
                            if rows_written + matrix_rows <= chunk_size:
                                rows_to_write.append(matrix)
                                zarr_info_to_write.append(current_chunk_zarrs[i])
                                rows_written += matrix_rows
                            else:
                                # Partial matrix needed
                                rows_needed = chunk_size - rows_written
                                rows_to_write.append(matrix[:rows_needed])
                                zarr_info_to_write.append({
                                    **current_chunk_zarrs[i],
                                    'rows_in_chunk': rows_needed
                                })
                                
                                # Keep remainder for next chunk
                                remainder = matrix[rows_needed:]
                                current_chunk_rows = [remainder]
                                current_chunk_zarrs = [{
                                    **current_chunk_zarrs[i],
                                    'rows_in_chunk': matrix_rows - rows_needed,
                                    'row_start_in_chunk': 0
                                }]
                                break
                        
                        # Write the chunk
                        current_row_start, manifest_entry, col_nnz = write_mtx_chunk(
                            rows_to_write, zarr_info_to_write, current_row_start, 
                            mtx_output_dir, n_new_cols
                        )
                        output_files.append(manifest_entry['mtx_path'])
                        column_nnz_accumulator += col_nnz
                        manifest_data.append(manifest_entry)
                        mtx_file_idx += 1
                    else:
                        # Exactly chunk_size rows
                        current_row_start, manifest_entry, col_nnz = write_mtx_chunk(
                            current_chunk_rows, current_chunk_zarrs, current_row_start,
                            mtx_output_dir, n_new_cols
                        )
                        output_files.append(manifest_entry['mtx_path'])
                        column_nnz_accumulator += col_nnz
                        manifest_data.append(manifest_entry)
                        mtx_file_idx += 1
                        current_chunk_rows = []
                        current_chunk_zarrs = []
        
        elif file_type == 'h5ad':
            # For h5ad, process in chunks too (split large files)
            X_aligned, n_rows = process_h5ad_file(str(file_path), gene_list_path, old_to_new_idx, n_new_cols)
            
            if X_aligned is None:
                raise RuntimeError(f"Failed to process {file_type} file {file_path.name}.")
            
            print(f"  Processed {n_rows} rows from {file_path.name}")
            
            # Process h5ad file in chunks
            for chunk_start in range(0, n_rows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_rows)
                X_chunk = X_aligned[chunk_start:chunk_end]
                
                current_chunk_rows.append(X_chunk)
                current_chunk_zarrs.append({
                    'file': file_path.name,
                    'file_path': str(file_path),
                    'file_type': file_type,
                    'rows_in_chunk': X_chunk.shape[0],
                    'row_start_in_chunk': chunk_start
                })
                
                total_rows_in_chunk = sum(m.shape[0] for m in current_chunk_rows)
                
                # Write MTX file when we've accumulated enough rows
                if total_rows_in_chunk >= chunk_size:
                    if total_rows_in_chunk > chunk_size:
                        # Write exactly chunk_size, keep remainder
                        rows_to_write = []
                        rows_written = 0
                        zarr_info_to_write = []
                        
                        for i, matrix in enumerate(current_chunk_rows):
                            matrix_rows = matrix.shape[0]
                            if rows_written + matrix_rows <= chunk_size:
                                rows_to_write.append(matrix)
                                zarr_info_to_write.append(current_chunk_zarrs[i])
                                rows_written += matrix_rows
                            else:
                                rows_needed = chunk_size - rows_written
                                rows_to_write.append(matrix[:rows_needed])
                                zarr_info_to_write.append({
                                    **current_chunk_zarrs[i],
                                    'rows_in_chunk': rows_needed
                                })
                                remainder = matrix[rows_needed:]
                                current_chunk_rows = [remainder]
                                current_chunk_zarrs = [{
                                    **current_chunk_zarrs[i],
                                    'rows_in_chunk': matrix_rows - rows_needed,
                                    'row_start_in_chunk': current_chunk_zarrs[i]['row_start_in_chunk'] + rows_needed
                                }]
                                break
                        
                        current_row_start, manifest_entry, col_nnz = write_mtx_chunk(
                            rows_to_write, zarr_info_to_write, current_row_start,
                            mtx_output_dir, n_new_cols
                        )
                        output_files.append(manifest_entry['mtx_path'])
                        column_nnz_accumulator += col_nnz
                        manifest_data.append(manifest_entry)
                        mtx_file_idx += 1
                    else:
                        current_row_start, manifest_entry, col_nnz = write_mtx_chunk(
                            current_chunk_rows, current_chunk_zarrs, current_row_start,
                            mtx_output_dir, n_new_cols
                        )
                        output_files.append(manifest_entry['mtx_path'])
                        column_nnz_accumulator += col_nnz
                        manifest_data.append(manifest_entry)
                        mtx_file_idx += 1
                        current_chunk_rows = []
                        current_chunk_zarrs = []
                
                del X_chunk
                gc.collect()
            
            del X_aligned
            gc.collect()
    
    # Write final chunk if there are remaining rows
    if current_chunk_rows:
        print(f"\nWriting final chunk with {sum(m.shape[0] for m in current_chunk_rows)} rows")
        current_row_start, manifest_entry, col_nnz = write_mtx_chunk(
            current_chunk_rows, current_chunk_zarrs, current_row_start,
            mtx_output_dir, n_new_cols
        )
        output_files.append(manifest_entry['mtx_path'])
        column_nnz_accumulator += col_nnz
        manifest_data.append(manifest_entry)
        mtx_file_idx += 1
    
    # Write column nnz to file (accumulated across all MTX files)
    column_nnz_path = os.path.join(output_dir, "column_nnz.txt")
    np.savetxt(column_nnz_path, column_nnz_accumulator, fmt='%u', delimiter='\n')
    print(f"  ✓ Column nnz saved to {os.path.basename(column_nnz_path)}")
    
    # Write manifest file
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {
        'gene_list_file': str(gene_list_path),
        'n_genes': n_new_cols,
        'source_files_processed': [str(f) for f in all_files],
        'source_files_order': [f.name for f in all_files],
        'file_types': {str(f): file_types[f] for f in all_files},
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
        gc.collect()  # Force garbage collection after processing each MTX file
    
    print(f"  ✓ Created {sum(len(frags) for frags in fragment_map.values())} transposed fragments in tmp directory")
    
    # Step 2: Combine fragments covering the same column range
    # Process incrementally to avoid loading all fragments at once
    print(f"\n  Combining fragments by column range...")
    
    for frag_idx in range(num_fragments):
        if frag_idx not in fragment_map:
            continue
        
        frag_list = fragment_map[frag_idx]
        col_start = frag_idx * fragment_cols
        col_end = min(col_start + fragment_cols, n_cols)
        
        print(f"    Fragment {frag_idx}: columns {col_start}-{col_end-1} ({len(frag_list)} fragments to combine)...")
        
        # Process fragments incrementally: combine in batches to avoid memory bloat
        # For very large datasets, combine in smaller batches
        batch_size = max(10, min(50, len(frag_list)))  # Adaptive batch size
        
        if len(frag_list) <= batch_size:
            # Small enough to combine all at once
            combined_fragments = []
            for mtx_file, frag_path in frag_list:
                frag_matrix = mmread(frag_path)
                if not isinstance(frag_matrix, csr_matrix):
                    frag_matrix = frag_matrix.tocsr()
                combined_fragments.append(frag_matrix)
                del frag_matrix
            
            if combined_fragments:
                combined_matrix = hstack(combined_fragments, format='csr')
                del combined_fragments
        else:
            # Combine in batches to avoid memory issues
            print(f"      Combining {len(frag_list)} fragments in batches of {batch_size}...")
            combined_matrix = None
            
            for batch_start in range(0, len(frag_list), batch_size):
                batch_end = min(batch_start + batch_size, len(frag_list))
                batch_frags = []
                
                for mtx_file, frag_path in frag_list[batch_start:batch_end]:
                    frag_matrix = mmread(frag_path)
                    if not isinstance(frag_matrix, csr_matrix):
                        frag_matrix = frag_matrix.tocsr()
                    batch_frags.append(frag_matrix)
                    del frag_matrix
                
                if batch_frags:
                    batch_combined = hstack(batch_frags, format='csr')
                    del batch_frags
                    
                    if combined_matrix is None:
                        combined_matrix = batch_combined
                    else:
                        # Combine with previous batches
                        combined_matrix = hstack([combined_matrix, batch_combined], format='csr')
                        del batch_combined
                        gc.collect()
        
        if combined_matrix is not None:
            # Save combined fragment
            cm_filename = f"cols_{col_start}_{col_end-1}.mtx"
            cm_path = os.path.join(cm_mtx_dir, cm_filename)
            mmwrite(cm_path, combined_matrix)
            
            print(f"      ✓ Combined {len(frag_list)} fragments -> {combined_matrix.shape[0]} rows × {combined_matrix.shape[1]} cols")
            
            # Free memory immediately
            del combined_matrix
            gc.collect()
    
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
        description='Align zarr or h5ad file(s) columns to standard gene list and output as MTX format. '
                    'Processes a directory of .zarr files (directories) or .h5/.hdf5 files (h5ad), '
                    'concatenating them into MTX files (max 131072 rows each by default). '
                    'Auto-detects file type based on extensions.'
    )
    parser.add_argument(
        'zarr_input',
        type=str,
        help='Path to input directory containing .zarr files (directories) or .h5/.hdf5 files (h5ad format)'
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
        elif str(zarr_input_path).endswith('.zarr') or zarr_input_path.suffix in ['.h5', '.hdf5']:
            # Single file - treat parent directory as input (will only find this one file)
            input_dir = zarr_input_path.parent
            align_zarr_directory_to_mtx(
                str(input_dir),
                gene_list_path,
                args.output_dir,
                args.tmp_dir,
                args.chunk_size
            )
        else:
            print(f"ERROR: Input must be a directory containing .zarr files (directories) or .h5/.hdf5 files (h5ad format)")
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
