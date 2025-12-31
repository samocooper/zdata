"""
Python wrapper for mtx_to_zdata.c that builds .zdata directories and adds metadata.
"""
import subprocess
import json
import os
import re
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

def build_zdata(mtx_file, output_name, zstd_base=None):
    """
    Build a .zdata directory from an MTX file.
    
    Args:
        mtx_file: Path to the input Matrix Market (.mtx) file
        output_name: Base name for output (e.g., "andrews" -> "andrews.zdata/")
        zstd_base: Optional path to zstd library (for compilation if needed)
    
    Returns:
        Path to the created .zdata directory
    """
    mtx_path = Path(mtx_file)
    if not mtx_path.exists():
        raise FileNotFoundError(f"MTX file not found: {mtx_file}")
    
    # Call the C tool
    bin_path = _get_mtx_to_zdata_path()
    result = subprocess.run(
        [bin_path, str(mtx_path), output_name],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"mtx_to_zdata failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    
    # Parse output to extract metadata
    zdata_dir = Path(f"{output_name}.zdata")
    if not zdata_dir.exists():
        raise RuntimeError(f"Output directory was not created: {zdata_dir}")
    
    # Read MTX header to get dimensions
    # Parse the output from mtx_to_zdata which prints: "Matrix: %lld rows, %lld cols, %lld nnz (global)\n"
    nrows = None
    ncols = None
    nnz_total = None
    
    for line in result.stdout.split('\n'):
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
    
    # Determine blocks per chunk (each chunk has MAX_ROWS_PER_CHUNK rows,
    # each block has 16 rows, so max 256 blocks per chunk)
    MAX_ROWS_PER_CHUNK = 4096
    BLOCK_ROWS = 16
    blocks_per_chunk = MAX_ROWS_PER_CHUNK // BLOCK_ROWS  # 256
    
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
            rows_in_last_chunk = nrows - (chunk_num * MAX_ROWS_PER_CHUNK)
            blocks_in_last_chunk = (rows_in_last_chunk + BLOCK_ROWS - 1) // BLOCK_ROWS
            blocks_in_chunk = blocks_in_last_chunk
        
        chunk_metadata.append({
            "chunk_num": chunk_num,
            "file": chunk_file.name,
            "blocks": blocks_in_chunk,
            "start_row": chunk_num * MAX_ROWS_PER_CHUNK,
            "end_row": min((chunk_num + 1) * MAX_ROWS_PER_CHUNK, nrows)
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
        "block_rows": BLOCK_ROWS,
        "max_rows_per_chunk": MAX_ROWS_PER_CHUNK,
        "chunks": chunk_metadata
    }
    
    # Write metadata to JSON file in .zdata directory
    metadata_file = zdata_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created {zdata_dir} with {num_chunks} chunks, {total_blocks} blocks")
    print(f"✓ Metadata written to {metadata_file}")
    
    return zdata_dir

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <matrix.mtx> <output_name>")
        sys.exit(1)
    
    build_zdata(sys.argv[1], sys.argv[2])

