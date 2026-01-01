#!/usr/bin/env python3
"""
Test script for ZData - extracts a random set of 10 rows.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import zdata
# We need the parent of the zdata package directory, not the zdata directory itself
_test_dir = Path(__file__).parent
_project_root = _test_dir.parent  # This is /home/ubuntu/zdata_work/zdata
_parent_dir = _project_root.parent  # This is /home/ubuntu/zdata_work
sys.path.insert(0, str(_parent_dir))

from zdata.core.zdata import ZData
import numpy as np
import os

def test_random_rows(dir_name="andrews", n_rows=10, seed=42):
    """
    Test extracting random rows from a zdata directory.
    
    Args:
        dir_name: Name of the .zdata directory (default: "andrews")
        n_rows: Number of random rows to extract (default: 10)
        seed: Random seed for reproducibility (default: 42)
    """
    print(f"Testing ZData with directory: {dir_name}.zdata")
    print(f"Extracting {n_rows} random rows (seed={seed})")
    print("-" * 60)
    
    try:
        # Initialize reader
        print("Initializing reader...")
        # dir_name can be a full path or just a name
        reader = ZData(dir_name)
        print(f"✓ Reader initialized successfully")
        print(f"  - Number of columns: {reader.num_columns}")
        print(f"  - Number of rows: {reader.num_rows}")
        print(f"  - Available chunks: {sorted(reader.chunk_files.keys())}")
        print()
        
        # Get random row indices
        print(f"Generating {n_rows} random row indices...")
        random_rows = reader.get_random_rows(n_rows, seed=seed)
        print(f"✓ Random rows generated: {random_rows}")
        print()
        
        # Read the rows
        print("Reading rows from zdata files...")
        rows_data = reader.read_rows(random_rows)
        print(f"✓ Successfully read {len(rows_data)} rows")
        print()
        
        # Display summary for each row
        print("Row summary:")
        print("-" * 60)
        total_nnz = 0
        for i, (global_row, cols, vals) in enumerate(rows_data):
            nnz = len(cols)
            total_nnz += nnz
            print(f"  Row {i+1:2d}: Global index {global_row:6d} - {nnz:6d} non-zeros")
            if i < 3:  # Show details for first 3 rows
                print(f"           First 5 columns: {cols[:5].tolist()}")
                print(f"           First 5 values:  {vals[:5].tolist()}")
        print("-" * 60)
        print(f"Total non-zeros across all rows: {total_nnz}")
        print()
        
        # Convert to CSR matrix
        print("Converting to CSR matrix...")
        csr = reader.read_rows_csr(random_rows)
        print(f"✓ CSR matrix created")
        print(f"  - Shape: {csr.shape}")
        print(f"  - Total nnz: {csr.nnz}")
        print(f"  - Dtype: {csr.dtype}")
        print()
        
        # Verify CSR matches row data
        print("Verifying CSR matrix matches row data...")
        for i, (global_row, cols, vals) in enumerate(rows_data):
            csr_row = csr[i]
            if csr_row.nnz != len(cols):
                print(f"✗ Mismatch in row {i}: CSR nnz={csr_row.nnz}, row_data nnz={len(cols)}")
                return False
            # Check that columns match
            csr_cols = csr_row.indices
            if not np.array_equal(np.sort(csr_cols), np.sort(cols)):
                print(f"✗ Column mismatch in row {i}")
                return False
        print("✓ CSR matrix verification passed")
        print()
        
        print("=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)
        return True
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f"  Make sure the directory '{dir_name}.zdata' exists")
        return False
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        zdata_path = sys.argv[1]
        # Handle both full paths (e.g., /path/to/andrews.zdata) and directory names (e.g., andrews)
        zdata_path = os.path.abspath(zdata_path)
        
        if os.path.exists(zdata_path) and os.path.isdir(zdata_path):
            # Full path to .zdata directory provided - use it directly
            dir_name = zdata_path
            original_cwd = None
        else:
            # Just a directory name provided (relative path) or path doesn't exist yet
            # Remove .zdata suffix if present, ZData will add it
            dir_name = zdata_path.replace('.zdata', '') if zdata_path.endswith('.zdata') else zdata_path
            original_cwd = None
    else:
        # Default
        dir_name = "andrews"
        original_cwd = None
    
    # Parse additional arguments (n_rows and seed)
    # These come after the zdata path argument
    arg_idx = 2 if len(sys.argv) > 1 else 1
    n_rows = int(sys.argv[arg_idx]) if len(sys.argv) > arg_idx else 10
    seed = int(sys.argv[arg_idx + 1]) if len(sys.argv) > (arg_idx + 1) else 42
    
    try:
        success = test_random_rows(dir_name, n_rows, seed)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

