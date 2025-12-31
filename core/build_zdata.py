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
_MODULE_DIR = Path(__file__).parent  # zdata/core/
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

def build_zdata(mtx_file_or_dir, output_name, zstd_base=None, block_rows=16, max_rows=8192):
    """
    Build a .zdata directory from an MTX file or directory of MTX files.
    
    Args:
        mtx_file_or_dir: Path to a single MTX file or directory containing MTX files
        output_name: Output directory name (e.g., "andrews" -> "andrews/")
        zstd_base: Optional path to zstd library (for compilation if needed)
        block_rows: Number of rows per block (default: 16)
        max_rows: Maximum rows per chunk (default: 8192)
    
    Returns:
        Path to the created zdata directory
    """
    input_path = Path(mtx_file_or_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {mtx_file_or_dir}")
    
    # Validate parameters
    if block_rows < 1 or block_rows > 256:
        raise ValueError(f"block_rows must be between 1 and 256, got {block_rows}")
    if max_rows < 1 or max_rows > 1000000:
        raise ValueError(f"max_rows must be between 1 and 1000000, got {max_rows}")
    
    # Determine if input is a directory or single file
    if input_path.is_dir():
        # Get all .mtx files in directory, sorted
        mtx_files = sorted(input_path.glob("*.mtx"))
        if not mtx_files:
            raise ValueError(f"No .mtx files found in directory: {mtx_file_or_dir}")
        print(f"Found {len(mtx_files)} MTX files in directory")
        return _build_zdata_from_multiple_files(mtx_files, output_name, block_rows, max_rows)
    else:
        # Single file
        return _build_zdata_from_single_file(input_path, output_name, block_rows, max_rows)


def _build_zdata_from_single_file(mtx_path, output_name, block_rows, max_rows, row_offset=0):
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
    
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    # Collect output lines while streaming
    stdout_lines = []
    for line in process.stdout:
        print(line, end='', flush=True)  # Print immediately
        stdout_lines.append(line)
    
    # Wait for process to complete
    returncode = process.wait()
    stdout_text = ''.join(stdout_lines)
    
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
    
    # Count chunk files
    chunk_files = sorted(zdata_dir.glob("*.bin"))
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
    
    # Create metadata dictionary
    metadata = {
        "version": 1,
        "format": "zdata",
        "shape": [nrows, ncols],
        "nnz_total": nnz_total,
        "num_chunks": num_chunks,
        "total_blocks": total_blocks,
        "blocks_per_chunk": blocks_per_chunk,
        "block_rows": block_rows,
        "max_rows_per_chunk": max_rows,
        "chunks": chunk_metadata
    }
    
    # Write metadata to JSON file in zdata directory
    metadata_file = zdata_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created {zdata_dir} with {num_chunks} chunks, {total_blocks} blocks")
    print(f"✓ Metadata written to {metadata_file}")
    
    return zdata_dir


def _build_zdata_from_multiple_files(mtx_files, output_name, block_rows, max_rows):
    """Build zdata from multiple MTX files, combining them into a single contiguous dataset."""
    zdata_dir = Path(output_name)
    
    # Create output directory if it doesn't exist
    zdata_dir.mkdir(parents=True, exist_ok=True)
    
    # Track cumulative statistics
    total_rows = 0
    total_nnz = 0
    ncols = None
    all_chunk_metadata = []
    total_blocks = 0
    blocks_per_chunk = max_rows // block_rows
    
    print(f"\nProcessing {len(mtx_files)} MTX files into single zdata object...")
    print("=" * 70)
    
    for mtx_idx, mtx_file in enumerate(mtx_files):
        print(f"\nProcessing file {mtx_idx + 1}/{len(mtx_files)}: {mtx_file.name}")
        
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
        
        # Validate column count consistency
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
            _build_zdata_from_single_file(mtx_file, str(temp_output), block_rows, max_rows, row_offset=total_rows)
            
            # Get chunk files from temp directory
            temp_chunk_files = sorted(temp_output.glob("*.bin"))
            num_chunks_copied = len(temp_chunk_files)
            
            # Copy and rename chunk files with correct global chunk numbering
            # Chunk numbers are calculated based on global row ranges to ensure sequential numbering
            chunk_nums_created = []
            for temp_chunk_file in temp_chunk_files:
                local_chunk_idx = int(temp_chunk_file.stem)
                
                # Calculate global row range for this chunk
                chunk_start_row = total_rows + (local_chunk_idx * max_rows)
                chunk_end_row = min(chunk_start_row + max_rows, total_rows + file_rows)
                
                # Calculate global chunk number based on start_row (ensures sequential numbering)
                # This matches the calculation used in zdata.py: chunk_num = global_row // max_rows_per_chunk
                new_chunk_num = chunk_start_row // max_rows
                new_chunk_file = zdata_dir / f"{new_chunk_num}.bin"
                
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
            
            print(f"  ✓ Processed {file_rows} rows, {file_nnz} nnz")
            if chunk_nums_created:
                print(f"  ✓ Copied {num_chunks_copied} chunks (chunks {min(chunk_nums_created)}-{max(chunk_nums_created)})")
    
    # Sort chunks by chunk_num to ensure metadata is in order
    all_chunk_metadata.sort(key=lambda c: c['chunk_num'])
    
    # Create final metadata
    num_chunks = len(all_chunk_metadata)
    metadata = {
        "version": 1,
        "format": "zdata",
        "shape": [total_rows, ncols],
        "nnz_total": total_nnz,
        "num_chunks": num_chunks,
        "total_blocks": total_blocks,
        "blocks_per_chunk": blocks_per_chunk,
        "block_rows": block_rows,
        "max_rows_per_chunk": max_rows,
        "source_files": [f.name for f in mtx_files],
        "num_source_files": len(mtx_files),
        "chunks": all_chunk_metadata
    }
    
    # Write metadata
    metadata_file = zdata_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"✓ Created {zdata_dir} with {num_chunks} chunks, {total_blocks} blocks")
    print(f"✓ Total: {total_rows} rows, {ncols} cols, {total_nnz} nnz")
    print(f"✓ Metadata written to {metadata_file}")
    
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

