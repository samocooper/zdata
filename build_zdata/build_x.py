"""
Python wrapper for mtx_to_zdata.c that builds .zdata directories and adds metadata.
"""
import subprocess
import json
import os
import re
import tempfile
import shutil
from pathlib import Path

# Get the path to the mtx_to_zdata executable
_MODULE_DIR = Path(__file__).parent  # zdata/build_zdata/
_PROJECT_ROOT = _MODULE_DIR.parent   # zdata/
_MTX_TO_ZDATA = _PROJECT_ROOT / "ctools" / "mtx_to_zdata"

def _get_mtx_to_zdata_path():
    """Get the path to mtx_to_zdata executable, with validation."""
    bin_path = _MTX_TO_ZDATA.absolute()
    if not bin_path.exists():
        raise RuntimeError(
            f"mtx_to_zdata executable not found at {bin_path}. "
            f"Please ensure it is built in the ctools directory."
        )
    return str(bin_path)

def _sort_mtx_files_numerically(mtx_files):
    """
    Sort MTX files numerically by extracting the start index from the filename.
    
    Handles filenames like:
    - cols_START_END.mtx (e.g., cols_0_255.mtx, cols_35584_35803.mtx)
    - rows_START_END.mtx (e.g., rows_0_255.mtx)
    - Or any pattern with numbers separated by underscores
    
    Args:
        mtx_files: List of Path objects for MTX files
        
    Returns:
        Sorted list of Path objects
    """
    def extract_start_index(mtx_path):
        """Extract the first numeric value from filename for sorting."""
        name = mtx_path.stem  # Get filename without extension
        # Try to extract numbers from the filename
        # Pattern: prefix_START_END or similar
        parts = name.split('_')
        for part in parts:
            if part.isdigit():
                return int(part)
        # Fallback: try to find any number in the filename
        import re
        numbers = re.findall(r'\d+', name)
        if numbers:
            return int(numbers[0])
        # Last resort: return 0 to keep original order for files without numbers
        return 0
    
    return sorted(mtx_files, key=extract_start_index)

def build_zdata(mtx_file_or_dir, output_name, zstd_base=None, block_rows=16, block_columns=None, max_rows=8192, max_columns=256):
    """
    Build a .zdata directory from an MTX file or directory of MTX files.
    
    Args:
        mtx_file_or_dir: Path to a single MTX file or directory containing MTX files
        output_name: Output directory name (e.g., "andrews" -> "andrews/")
        zstd_base: Optional path to zstd library (for compilation if needed)
        block_rows: Number of rows per block for row-major (X_RM) files (default: 16)
        block_columns: Number of rows per block for column-major (X_CM) files (default: None, uses block_rows)
        max_rows: Maximum rows per chunk for row-major files (default: 8192)
        max_columns: Maximum rows per chunk for column-major files (default: 256)
    
    Returns:
        Path to the created zdata directory
    """
    input_path = Path(mtx_file_or_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {mtx_file_or_dir}")
    
    # Use block_rows for block_columns if not specified
    if block_columns is None:
        block_columns = block_rows
    
    # Validate parameters
    if block_rows < 1 or block_rows > 256:
        raise ValueError(f"block_rows must be between 1 and 256, got {block_rows}")
    if block_columns < 1 or block_columns > 256:
        raise ValueError(f"block_columns must be between 1 and 256, got {block_columns}")
    if max_rows < 1 or max_rows > 1000000:
        raise ValueError(f"max_rows must be between 1 and 1000000, got {max_rows}")
    if max_columns < 1 or max_columns > 1000000:
        raise ValueError(f"max_columns must be between 1 and 1000000, got {max_columns}")
    
    # Determine if input is a directory or single file
    if input_path.is_dir():
        # Check for rm_mtx_files and cm_mtx_files subdirectories
        rm_mtx_dir = input_path / "rm_mtx_files"
        cm_mtx_dir = input_path / "cm_mtx_files"
        
        # Clean up existing output directory to ensure clean rebuild
        zdata_dir = Path(output_name)
        if zdata_dir.exists():
            # Remove existing metadata and parquet files
            for old_file in zdata_dir.glob("*.json"):
                old_file.unlink()
            for old_file in zdata_dir.glob("*.parquet"):
                old_file.unlink()
            # Remove existing chunk directories
            for subdir_path in [zdata_dir / "X_RM", zdata_dir / "X_CM"]:
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)
        
        # Process row-major MTX files if they exist
        rm_metadata = None
        if rm_mtx_dir.exists() and rm_mtx_dir.is_dir():
            mtx_files = _sort_mtx_files_numerically(list(rm_mtx_dir.glob("*.mtx")))
            if mtx_files:
                print(f"Found {len(mtx_files)} row-major MTX files in rm_mtx_files")
                zdata_dir, rm_metadata = _build_zdata_from_multiple_files(mtx_files, output_name, block_rows, max_rows, subdir="X_RM", return_metadata=True)
        
        # Process column-major MTX files if they exist
        cm_metadata = None
        if cm_mtx_dir.exists() and cm_mtx_dir.is_dir():
            mtx_files = _sort_mtx_files_numerically(list(cm_mtx_dir.glob("*.mtx")))
            if mtx_files:
                print(f"\nFound {len(mtx_files)} column-major MTX files in cm_mtx_files")
                # Use existing zdata_dir if already created, otherwise create new one
                if zdata_dir is None:
                    zdata_dir = Path(output_name)
                    zdata_dir.mkdir(parents=True, exist_ok=True)
                # Use block_columns and max_columns for column-major files
                _, cm_metadata = _build_zdata_from_multiple_files(mtx_files, output_name, block_columns, max_columns, subdir="X_CM", return_metadata=True)
        
        # Combine metadata if both exist
        if rm_metadata is not None and cm_metadata is not None:
            # Merge metadata: use shape from RM (cells, genes), but include both chunk lists
            # X_RM shape: [cells, genes]
            # X_CM shape: [genes, cells] - but we want to use RM shape as the canonical shape
            # Build sorted ranges for CM chunks for fast binary search lookup
            cm_chunk_ranges = []
            chunks_by_file_cm = {}
            for chunk_info in cm_metadata['chunks_cm']:
                file_name = chunk_info['file']
                if file_name not in chunks_by_file_cm:
                    chunks_by_file_cm[file_name] = []
                chunks_by_file_cm[file_name].append(chunk_info)
            
            # Build ranges: group by file and create [start_row, end_row, chunk_num] lists
            for file_name, file_chunks in chunks_by_file_cm.items():
                chunk_num = file_chunks[0]['chunk_num']
                start_row = min(c['start_row'] for c in file_chunks)
                end_row = max(c['end_row'] for c in file_chunks)
                cm_chunk_ranges.append([start_row, end_row, chunk_num])
            
            # Sort by start_row for binary search
            cm_chunk_ranges.sort(key=lambda x: x[0])
            
            combined_metadata = {
                "version": 1,
                "format": "zdata",
                "shape": rm_metadata['shape'],  # [cells, genes] from X_RM
                "nnz_total": rm_metadata.get('nnz_total', 0) + cm_metadata.get('nnz_total', 0),
                "num_chunks_rm": rm_metadata['num_chunks_rm'],
                "num_chunks_cm": cm_metadata['num_chunks_cm'],
                "total_blocks_rm": rm_metadata['total_blocks_rm'],
                "total_blocks_cm": cm_metadata['total_blocks_cm'],
                "blocks_per_chunk": rm_metadata['blocks_per_chunk'],
                "block_rows": block_rows,
                "block_columns": block_columns,
                "max_rows_per_chunk": max_rows,
                "max_columns_per_chunk": max_columns,
                "chunks_rm": rm_metadata['chunks_rm'],  # Chunk ranges are cell indices (0 to nrows-1)
                "chunks_cm": cm_metadata['chunks_cm'],  # Chunk ranges are gene indices (0 to ncols-1)
                "cm_chunk_ranges": cm_chunk_ranges,  # Sorted list of [start_row, end_row, chunk_num] for binary search
                "source_files_rm": rm_metadata.get('source_files', []),
                "source_files_cm": cm_metadata.get('source_files', [])
            }
            metadata_file = zdata_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(combined_metadata, f, indent=2)
            print(f"\n✓ Combined metadata written to {metadata_file}")
        elif rm_metadata is not None:
            # Only RM metadata - already in new format
            metadata_file = zdata_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(rm_metadata, f, indent=2)
        elif cm_metadata is not None:
            # Only CM metadata - swap shape for CM-only: [genes, cells] -> [cells, genes]
            if 'shape' in cm_metadata:
                cm_shape = cm_metadata['shape']
                cm_metadata['shape'] = [cm_shape[1], cm_shape[0]]  # Swap to [cells, genes]
            
            # Build sorted ranges for CM chunks for fast binary search lookup
            cm_chunk_ranges = []
            chunks_by_file_cm = {}
            for chunk_info in cm_metadata['chunks_cm']:
                file_name = chunk_info['file']
                if file_name not in chunks_by_file_cm:
                    chunks_by_file_cm[file_name] = []
                chunks_by_file_cm[file_name].append(chunk_info)
            
            # Build ranges: group by file and create (start_row, end_row, chunk_num) tuples
            for file_name, file_chunks in chunks_by_file_cm.items():
                chunk_num = file_chunks[0]['chunk_num']
                start_row = min(c['start_row'] for c in file_chunks)
                end_row = max(c['end_row'] for c in file_chunks)
                cm_chunk_ranges.append([start_row, end_row, chunk_num])
            
            # Sort by start_row for binary search
            cm_chunk_ranges.sort(key=lambda x: x[0])
            cm_metadata['cm_chunk_ranges'] = cm_chunk_ranges
            cm_metadata['block_columns'] = block_columns
            cm_metadata['max_columns_per_chunk'] = max_columns
            
            metadata_file = zdata_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(cm_metadata, f, indent=2)
        
        # Fallback: if no subdirectories, look for MTX files directly in the directory
        if zdata_dir is None:
            mtx_files = _sort_mtx_files_numerically(list(input_path.glob("*.mtx")))
            if not mtx_files:
                raise ValueError(f"No .mtx files found in directory: {mtx_file_or_dir}")
            print(f"Found {len(mtx_files)} MTX files in directory")
            return _build_zdata_from_multiple_files(mtx_files, output_name, block_rows, max_rows, subdir="X_RM")
        
        return zdata_dir
    else:
        # Single file
        return _build_zdata_from_single_file(input_path, output_name, block_rows, max_rows)


def _build_zdata_from_single_file(mtx_path, output_name, block_rows, max_rows, row_offset=0, subdir="X_RM"):
    """Build zdata from a single MTX file.
    
    Args:
        mtx_path: Path to MTX file
        output_name: Output directory name
        block_rows: Number of rows per block
        max_rows: Maximum rows per chunk
        row_offset: Offset to add to row indices from MTX file (for globally contiguous numbering)
    """
    if not mtx_path.exists():
        raise FileNotFoundError(f"MTX file not found: {mtx_path}")
    
    # Call the C tool with optional parameters
    # Stream output in real-time while also capturing for metadata parsing
    bin_path = _get_mtx_to_zdata_path()
    cmd = [bin_path, str(mtx_path), output_name, str(block_rows), str(max_rows)]
    if row_offset > 0:
        cmd.append(str(row_offset))
    # Always pass subdir parameter (7th argument) if not default
    if subdir != "X_RM":
        if row_offset == 0:
            cmd.append("0")  # Pass 0 for row_offset if not provided
        cmd.append(subdir)
    
    # Use Popen to capture output (suppress verbose C tool output)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        universal_newlines=True
    )
    
    # Collect output lines (don't print - suppress verbose output)
    stdout_lines = []
    stdout_text, _ = process.communicate()
    if stdout_text:
        stdout_lines = stdout_text.split('\n')
    
    # Wait for process to complete
    returncode = process.returncode
    
    if returncode != 0:
        raise RuntimeError(
            f"mtx_to_zdata failed with return code {returncode}\n"
            f"STDOUT:\n{stdout_text}"
        )
    
    # Parse output to extract metadata
    zdata_dir = Path(output_name)
    if not zdata_dir.exists():
        raise RuntimeError(f"Output directory was not created: {zdata_dir}")
    
    # Read MTX header to get dimensions
    # Parse the output from mtx_to_zdata which prints: "Matrix: %lld rows, %lld cols, %lld nnz (global)\n"
    nrows = None
    ncols = None
    nnz_total = None
    
    for line in stdout_text.split('\n'):
        if 'Matrix:' in line and 'rows' in line:
            # Parse: "Matrix: 29432 rows, 99635 cols, 21820971 nnz (global)"
            match = re.search(r'(\d+)\s+rows,\s+(\d+)\s+cols,\s+(\d+)\s+nnz', line)
            if match:
                nrows = int(match.group(1))
                ncols = int(match.group(2))
                nnz_total = int(match.group(3))
                break
    
    # Fallback: read directly from MTX file if parsing stdout failed
    if nrows is None:
        with open(mtx_path, 'r') as f:
            for line in f:
                if line.startswith('%'):
                    continue
                # This should be the dimensions line
                parts = line.strip().split()
                if len(parts) >= 3:
                    nrows = int(parts[0])
                    ncols = int(parts[1])
                    nnz_total = int(parts[2])
                    break
            else:
                raise ValueError("Could not parse MTX file dimensions")
    
    # Count chunk files (in subdirectory: X_RM or X_CM)
    chunk_dir = zdata_dir / subdir
    if not chunk_dir.exists():
        chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_files = sorted(chunk_dir.glob("*.bin"))
    num_chunks = len(chunk_files)
    
    # Determine blocks per chunk (each chunk has max_rows rows,
    # each block has block_rows rows)
    blocks_per_chunk = max_rows // block_rows
    
    # Calculate total blocks
    total_blocks = 0
    chunk_metadata = []
    
    for chunk_file in chunk_files:
        chunk_num = int(chunk_file.stem)
        # For all chunks except the last, we have full blocks
        if chunk_num < num_chunks - 1:
            blocks_in_chunk = blocks_per_chunk
        else:
            # Last chunk: calculate based on remaining rows
            rows_in_last_chunk = nrows - (chunk_num * max_rows)
            blocks_in_last_chunk = (rows_in_last_chunk + block_rows - 1) // block_rows
            blocks_in_chunk = blocks_in_last_chunk
        
        chunk_metadata.append({
            "chunk_num": chunk_num,
            "file": chunk_file.name,
            "blocks": blocks_in_chunk,
            "start_row": chunk_num * max_rows,
            "end_row": min((chunk_num + 1) * max_rows, nrows)
        })
        total_blocks += blocks_in_chunk
    
    # Create metadata dictionary (always use new format with _rm or _cm suffix)
    # Determine suffix based on subdir
    suffix = "_rm" if subdir == "X_RM" else "_cm"
    metadata = {
        "version": 1,
        "format": "zdata",
        "shape": [nrows, ncols],
        "nnz_total": nnz_total,
        f"num_chunks{suffix}": num_chunks,
        f"total_blocks{suffix}": total_blocks,
        "blocks_per_chunk": blocks_per_chunk,
        "block_rows": block_rows,
        "max_rows_per_chunk": max_rows,
        f"chunks{suffix}": chunk_metadata
    }
    
    # Write metadata to JSON file in zdata directory
    metadata_file = zdata_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Suppress verbose output - only print if there's an issue
    # print(f"✓ Created {zdata_dir} with {num_chunks} chunks, {total_blocks} blocks")
    # print(f"✓ Metadata written to {metadata_file}")
    
    return zdata_dir


def _build_zdata_from_multiple_files(mtx_files, output_name, block_rows, max_rows, subdir="X_RM", return_metadata=False):
    """
    Build zdata from multiple MTX files, combining them into a single contiguous dataset.
    
    For X_RM: rows = cells, chunk ranges are based on cell indices (0 to nrows-1)
    For X_CM: rows = genes (transposed), chunk ranges are based on gene indices (0 to ncols-1)
    """
    zdata_dir = Path(output_name)
    
    # Create output directory if it doesn't exist
    zdata_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up existing chunk files in the subdirectory to ensure clean rebuild
    chunk_dir = zdata_dir / subdir
    if chunk_dir.exists():
        for old_chunk_file in chunk_dir.glob("*.bin"):
            old_chunk_file.unlink()
    
    # Track cumulative statistics
    # For X_RM: total_rows tracks cells
    # For X_CM: total_rows tracks genes (since rows in X_CM = genes)
    total_rows = 0
    total_nnz = 0
    ncols = None
    all_chunk_metadata = []
    total_blocks = 0
    blocks_per_chunk = max_rows // block_rows
    
    # Determine if we're processing X_CM (column-major) files
    is_cm = (subdir == "X_CM")
    
    # Reduced verbosity - only show progress for every 10 files or at milestones
    if len(mtx_files) > 1:
        print(f"Processing {len(mtx_files)} MTX files...", end='', flush=True)
    
    for mtx_idx, mtx_file in enumerate(mtx_files):
        # Only print progress every 10 files or at start/end
        if len(mtx_files) > 1 and (mtx_idx == 0 or (mtx_idx + 1) % 10 == 0 or mtx_idx == len(mtx_files) - 1):
            print(f" {mtx_idx + 1}/{len(mtx_files)}", end='', flush=True)
        
        # Read dimensions from this MTX file
        with open(mtx_file, 'r') as f:
            for line in f:
                if line.startswith('%'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    file_rows = int(parts[0])
                    file_cols = int(parts[1])
                    file_nnz = int(parts[2])
                    break
            else:
                raise ValueError(f"Could not parse MTX file dimensions: {mtx_file}")
        
        # For X_CM files: rows = genes, columns = cells (transposed)
        # For X_RM files: rows = cells, columns = genes (normal)
        if is_cm:
            # X_CM: file_rows = genes, file_cols = cells
            # We chunk by rows (genes), so validate gene count consistency
            # Note: X_CM files are created by fragmenting into 256-gene chunks,
            # so the last file may have fewer genes if total genes isn't divisible by 256
            if ncols is None:
                ncols = file_rows  # Number of genes (rows in X_CM) - use first file as reference
            elif ncols != file_rows:
                # Allow the last file to have fewer genes (it's a partial fragment)
                # Check if this is the last file and if it has fewer genes
                is_last_file = (mtx_idx == len(mtx_files) - 1)
                if is_last_file and file_rows < ncols:
                    # Last file with fewer genes is acceptable
                    pass
                else:
                    raise ValueError(
                        f"Gene count mismatch in X_CM: expected {ncols}, got {file_rows} in {mtx_file}"
                    )
        else:
            # X_RM: file_rows = cells, file_cols = genes
            # We chunk by rows (cells), so validate column (gene) count consistency
            if ncols is None:
                ncols = file_cols
            elif ncols != file_cols:
                raise ValueError(
                    f"Column count mismatch: expected {ncols}, got {file_cols} in {mtx_file}"
                )
        
        # Use a temporary output directory for this MTX file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output = Path(temp_dir) / "temp_zdata"
            
            # Process MTX file with row_offset to make row indices globally contiguous
            # MTX files use 1-based indexing, so total_rows (0-based) becomes the offset
            # The C code will add this offset to row indices from the MTX file
            # For X_CM: total_rows tracks genes (rows in X_CM)
            # For X_RM: total_rows tracks cells (rows in X_RM)
            _build_zdata_from_single_file(mtx_file, str(temp_output), block_rows, max_rows, row_offset=total_rows, subdir=subdir)
            
            # Get chunk files from temp directory (they're in subdirectory: X_RM or X_CM)
            temp_chunk_dir = temp_output / subdir
            temp_chunk_files = sorted(temp_chunk_dir.glob("*.bin")) if temp_chunk_dir.exists() else []
            num_chunks_copied = len(temp_chunk_files)
            
            # Copy and rename chunk files with correct global chunk numbering
            # Chunk numbers are calculated based on global row ranges to ensure sequential numbering
            # For X_CM: rows = genes, so chunk ranges are gene indices (0 to ncols-1)
            # For X_RM: rows = cells, so chunk ranges are cell indices (0 to nrows-1)
            chunk_nums_created = []
            for temp_chunk_file in temp_chunk_files:
                local_chunk_idx = int(temp_chunk_file.stem)
                
                # Calculate global row range for this chunk
                # For X_CM: this is gene index range
                # For X_RM: this is cell index range
                chunk_start_row = total_rows + (local_chunk_idx * max_rows)
                chunk_end_row = min(chunk_start_row + max_rows, total_rows + file_rows)
                
                # Calculate global chunk number based on start_row (ensures sequential numbering)
                # This matches the calculation used in zdata.py: chunk_num = global_row // max_rows_per_chunk
                new_chunk_num = chunk_start_row // max_rows
                
                # Create subdirectory if it doesn't exist (X_RM or X_CM)
                chunk_dir = zdata_dir / subdir
                chunk_dir.mkdir(parents=True, exist_ok=True)
                
                new_chunk_file = chunk_dir / f"{new_chunk_num}.bin"
                
                # Copy the file
                shutil.copy2(temp_chunk_file, new_chunk_file)
                chunk_nums_created.append(new_chunk_num)
                
                # Determine blocks in this chunk
                # Check if this is the last chunk of the entire dataset
                is_last_chunk = (mtx_idx == len(mtx_files) - 1 and 
                                local_chunk_idx == len(temp_chunk_files) - 1)
                
                if is_last_chunk:
                    # Last chunk: calculate based on remaining rows
                    rows_in_chunk = chunk_end_row - chunk_start_row
                    blocks_in_chunk = (rows_in_chunk + block_rows - 1) // block_rows
                else:
                    # Full chunk
                    blocks_in_chunk = blocks_per_chunk
                
                all_chunk_metadata.append({
                    "chunk_num": new_chunk_num,
                    "file": new_chunk_file.name,
                    "blocks": blocks_in_chunk,
                    "start_row": chunk_start_row,
                    "end_row": chunk_end_row,
                    "source_mtx": mtx_file.name
                })
                total_blocks += blocks_in_chunk
            
            # Update cumulative totals
            total_rows += file_rows
            total_nnz += file_nnz
            
            # Suppress per-file verbose output
            # print(f"  ✓ Processed {file_rows} rows, {file_nnz} nnz")
            # if chunk_nums_created:
            #     print(f"  ✓ Copied {num_chunks_copied} chunks (chunks {min(chunk_nums_created)}-{max(chunk_nums_created)})")
    
    # Sort chunks by chunk_num to ensure metadata is in order
    all_chunk_metadata.sort(key=lambda c: c['chunk_num'])
    
    # Create final metadata (always use new format with _rm or _cm suffix)
    num_chunks = len(all_chunk_metadata)
    suffix = "_rm" if subdir == "X_RM" else "_cm"
    metadata = {
        "version": 1,
        "format": "zdata",
        "shape": [total_rows, ncols],
        "nnz_total": total_nnz,
        f"num_chunks{suffix}": num_chunks,
        f"total_blocks{suffix}": total_blocks,
        "blocks_per_chunk": blocks_per_chunk,
        "block_rows": block_rows,
        "max_rows_per_chunk": max_rows,
        "source_files": [f.name for f in mtx_files],
        "num_source_files": len(mtx_files),
        f"chunks{suffix}": all_chunk_metadata
    }
    
    # Write metadata only if not returning it (for combination later)
    if not return_metadata:
        metadata_file = zdata_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        # Suppress verbose output
        # print(f"✓ Metadata written to {metadata_file}")
    
    # Reduced verbosity - single summary line
    if len(mtx_files) > 1:
        print()  # Newline after progress indicator
    print(f"✓ {num_chunks} chunks, {total_rows} rows, {ncols} cols")
    
    if return_metadata:
        return zdata_dir, metadata
    else:
        return zdata_dir

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print(f"Usage: {sys.argv[0]} <matrix.mtx> <output_name> [block_rows] [max_rows]")
        print(f"  Default: block_rows=16, max_rows=8192")
        sys.exit(1)
    
    block_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    max_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
    
    build_zdata(sys.argv[1], sys.argv[2], block_rows=block_rows, max_rows=max_rows)

