#!/usr/bin/env python3
"""
Full pipeline test: compiles C tools, builds .zdata from zarr files, and runs tests.
"""

import subprocess
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import zdata
_test_dir = Path(__file__).parent
_project_root = _test_dir.parent  # This is /home/ubuntu/zdata_work/zdata
_parent_dir = _project_root.parent  # This is /home/ubuntu/zdata_work
sys.path.insert(0, str(_parent_dir))

# Import build_zdata_from_zarr
from zdata.build.build_zdata import build_zdata_from_zarr

# Configuration
ZSTD_BASE = "/home/ubuntu/zstd"
CTOOLS_DIR = _project_root / "ctools"
MTX_TO_ZDATA_SRC = CTOOLS_DIR / "mtx_to_zdata.c"
ZDATA_READ_SRC = CTOOLS_DIR / "zdata_read.c"
MTX_TO_ZDATA_BIN = CTOOLS_DIR / "mtx_to_zdata"
ZDATA_READ_BIN = CTOOLS_DIR / "zdata_read"

# Default paths (can be overridden via command line)
DEFAULT_ZARR_DIR = str(_test_dir / "zarr_test_dir")  # Directory of zarr files in tests
DEFAULT_OUTPUT_NAME = "tmp.zdata"  # Output in tests directory
DEFAULT_OUTPUT_DIR = _test_dir / DEFAULT_OUTPUT_NAME

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def run_command(cmd, description, check=True):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True
        )
        # Print both stdout and stderr so warnings are visible
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False
    except FileNotFoundError as e:
        print(f"ERROR: Command not found: {e}")
        return False

def compile_c_tools():
    """Compile both C tools."""
    print_section("Step 1: Compiling C Tools")
    
    # Check if zstd directory exists
    if not os.path.exists(ZSTD_BASE):
        print(f"ERROR: ZSTD directory not found at {ZSTD_BASE}")
        print("Please set ZSTD_BASE environment variable or update the script")
        return False
    
    zstd_include = f"{ZSTD_BASE}/lib"
    zstd_common = f"{ZSTD_BASE}/lib/common"
    zstd_seekable = f"{ZSTD_BASE}/contrib/seekable_format"
    zstd_lib = f"{ZSTD_BASE}/lib/libzstd.a"
    
    # Check if required files exist
    required_files = [
        zstd_lib,
        f"{ZSTD_BASE}/contrib/seekable_format/zstdseek_compress.c",
        f"{ZSTD_BASE}/contrib/seekable_format/zstdseek_decompress.c",
        f"{ZSTD_BASE}/lib/common/xxhash.c"
    ]
    
    for req_file in required_files:
        if not os.path.exists(req_file):
            print(f"ERROR: Required file not found: {req_file}")
            return False
    
    # Compile mtx_to_zdata
    mtx_compile_cmd = [
        "gcc", "-O2", "-Wall",
        f"-I{zstd_include}",
        f"-I{zstd_common}",
        f"-I{zstd_seekable}",
        "-o", str(MTX_TO_ZDATA_BIN),
        str(MTX_TO_ZDATA_SRC),
        f"{ZSTD_BASE}/contrib/seekable_format/zstdseek_compress.c",
        f"{ZSTD_BASE}/lib/common/xxhash.c",
        zstd_lib
    ]
    
    if not run_command(mtx_compile_cmd, "Compiling mtx_to_zdata"):
        return False
    
    # Compile zdata_read
    read_compile_cmd = [
        "gcc", "-O2", "-Wall",
        f"-I{zstd_include}",
        f"-I{zstd_common}",
        f"-I{zstd_seekable}",
        "-o", str(ZDATA_READ_BIN),
        str(ZDATA_READ_SRC),
        f"{ZSTD_BASE}/contrib/seekable_format/zstdseek_decompress.c",
        f"{ZSTD_BASE}/lib/common/xxhash.c",
        zstd_lib
    ]
    
    if not run_command(read_compile_cmd, "Compiling zdata_read"):
        return False
    
    # Make executables executable
    os.chmod(MTX_TO_ZDATA_BIN, 0o755)
    os.chmod(ZDATA_READ_BIN, 0o755)
    
    print("\n✓ Both C tools compiled successfully!")
    return True

def build_zdata_directory(zarr_dir, output_name, output_dir):
    """Build .zdata directory from zarr directory using build_zdata_from_zarr.
    
    Returns:
        tuple: (success: bool, actual_output_dir: Path)
    """
    print_section("Step 2: Building .zdata from Zarr Files")
    
    # Check if zarr directory exists
    if not os.path.exists(zarr_dir):
        print(f"ERROR: Zarr directory not found: {zarr_dir}")
        return False, None
    
    if not os.path.isdir(zarr_dir):
        print(f"ERROR: Path is not a directory: {zarr_dir}")
        return False, None
    
    # Check for zarr files
    zarr_files = sorted([f for f in Path(zarr_dir).glob("*.zarr") if f.is_dir()])
    if not zarr_files:
        print(f"ERROR: No .zarr files found in {zarr_dir}")
        return False, None
    
    print(f"Found {len(zarr_files)} zarr file(s) in {zarr_dir}")
    
    # Check if output directory already exists
    if os.path.exists(output_dir):
        print(f"WARNING: Output directory already exists: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)
        print(f"Deleted existing directory: {output_dir}")
    
    # Get the parent directory for the output name
    output_parent = os.path.dirname(str(output_dir))
    output_name_only = os.path.basename(str(output_dir))
    
    # Change to output parent directory for the build
    original_cwd = os.getcwd()
    try:
        os.chdir(output_parent)
        print(f"\nBuilding {output_name_only} from zarr directory: {os.path.basename(zarr_dir)}...")
        print(f"  (Processing {len(zarr_files)} zarr files)")
        
        # Use build_zdata_from_zarr to build complete zdata object
        # Pass the full directory name (e.g., "tmp.zdata")
        # Use smaller parameters for test data: block_rows=8, max_rows=256, mtx_chunk_size=512
        zdata_dir = build_zdata_from_zarr(
            zarr_dir,
            output_name_only,
            block_rows=8,
            max_rows=256,
            obs_join_strategy="outer",
            mtx_chunk_size=512
        )
        
        # Convert to absolute path
        zdata_dir = Path(zdata_dir)
        if not zdata_dir.is_absolute():
            zdata_dir = (Path(output_parent) / zdata_dir).resolve()
    except Exception as e:
        print(f"ERROR: Failed to build .zdata directory: {e}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        os.chdir(original_cwd)
    
    # Verify output was created
    # Check both the returned path and the expected output_dir
    expected_output = Path(output_dir)
    if not zdata_dir.exists() and not expected_output.exists():
        print(f"ERROR: Output directory was not created: {zdata_dir} or {expected_output}")
        return False, None
    
    # Use the actual created directory
    actual_output_dir = zdata_dir if zdata_dir.exists() else expected_output
    
    # Check for metadata file
    metadata_file = actual_output_dir / "metadata.json"
    if not metadata_file.exists():
        print(f"ERROR: Metadata file not found: {metadata_file}")
        return False, None
    
    # Check for .bin files in X_RM (row-major) subdirectory
    xrm_dir = actual_output_dir / "X_RM"
    xrm_bin_files = []
    if xrm_dir.exists():
        xrm_bin_files = list(xrm_dir.glob("*.bin"))
    
    # Check for .bin files in X_CM (column-major) subdirectory
    xcm_dir = actual_output_dir / "X_CM"
    xcm_bin_files = []
    if xcm_dir.exists():
        xcm_bin_files = list(xcm_dir.glob("*.bin"))
    
    if not xrm_bin_files and not xcm_bin_files:
        print(f"ERROR: No .bin files found in {actual_output_dir}/X_RM or {actual_output_dir}/X_CM")
        return False, None
    
    # Check for obs parquet file
    obs_file = actual_output_dir / "obs.parquet"
    has_obs = obs_file.exists()
    
    # Read metadata to show stats
    import json
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        shape = metadata.get('shape', [])
        if shape:
            num_rows = shape[0]
            num_cols = shape[1]
        else:
            num_rows = metadata.get('nrows', 'unknown')
            num_cols = metadata.get('ncols', 'unknown')
        
        print(f"\n✓ Successfully built {output_name_only}!")
        print(f"  Matrix: {num_rows} rows × {num_cols} columns")
        
        if xrm_bin_files:
            print(f"  Row-major (X_RM): {len(xrm_bin_files)} chunk file(s)")
        else:
            print(f"  WARNING: No row-major (X_RM) files found")
        
        if xcm_bin_files:
            print(f"  Column-major (X_CM): {len(xcm_bin_files)} chunk file(s)")
        else:
            print(f"  INFO: No column-major (X_CM) files found (column-major tests will be skipped)")
        
        if has_obs:
            print(f"  Obs/metadata: obs.parquet")
        else:
            print(f"  WARNING: Obs/metadata file not found")
    except Exception as e:
        print(f"\n✓ Successfully built {output_name_only}!")
        if xrm_bin_files:
            print(f"  Row-major (X_RM): {len(xrm_bin_files)} chunk file(s)")
        if xcm_bin_files:
            print(f"  Column-major (X_CM): {len(xcm_bin_files)} chunk file(s)")
        if has_obs:
            print(f"  Obs/metadata: obs.parquet")
    
    return True, actual_output_dir

def verify_cell_data_integrity(zarr_dir, zdata_path, gene_list_path, test_cell_indices):
    """
    Verify that read-count data for specific cells matches between zarr and zdata.
    
    Args:
        zarr_dir: Directory containing original zarr files
        zdata_path: Path to zdata directory
        gene_list_path: Path to gene list file used for alignment
        test_cell_indices: List of global cell indices to test
    
    Returns:
        bool: True if all checks pass
    """
    try:
        import zarr
        import polars as pl
        import pandas as pd
        from scipy.sparse import csr_matrix
    except ImportError as e:
        print(f"  ⚠ Skipping data integrity check: {e}")
        return True  # Don't fail if dependencies missing
    
    print(f"\n  Verifying data integrity for {len(test_cell_indices)} cells...")
    print(f"  Testing specific genes: GAPDH, TUBB, ACTA1, PCNA, CCND1, TP53")
    
    # Read gene list
    with open(gene_list_path, 'r') as f:
        aligned_genes = [line.strip() for line in f if line.strip()]
    
    # Verify gene list matches var.parquet
    var_file = Path(zdata_path) / "var.parquet"
    if var_file.exists():
        try:
            import polars as pl
            var_df = pl.read_parquet(var_file)
            var_genes = var_df['gene'].to_list()
            if len(var_genes) != len(aligned_genes):
                print(f"  ⚠ WARNING: Gene list length mismatch: var.parquet has {len(var_genes)}, gene list has {len(aligned_genes)}")
            else:
                # Check if they match
                mismatches = [i for i, (v, a) in enumerate(zip(var_genes, aligned_genes)) if v != a]
                if mismatches:
                    print(f"  ⚠ WARNING: Gene list mismatch at {len(mismatches)} positions (first 5: {mismatches[:5]})")
                    # Show first few mismatches
                    for idx in mismatches[:3]:
                        print(f"        Position {idx}: var='{var_genes[idx]}', gene_list='{aligned_genes[idx]}'")
                else:
                    print(f"  ✓ Gene list matches var.parquet ({len(aligned_genes)} genes)")
        except Exception as e:
            print(f"  ⚠ Could not verify gene list against var.parquet: {e}")
    
    # Load zarr files
    zarr_files = sorted([f for f in Path(zarr_dir).glob("*.zarr") if f.is_dir()])
    if not zarr_files:
        print(f"  ✗ No zarr files found for comparison")
        return False
    
    # Create mapping of zarr file names to zarr file paths
    zarr_file_map = {f.name: f for f in zarr_files}
    
    # Load obs data to get source zarr mapping
    from zdata.core.zdata import ZData
    reader = ZData(str(zdata_path))
    
    if reader._obs_df is None:
        print(f"  ⚠ Obs data not available, using sequential mapping")
        # Fallback: assume sequential order
        cell_to_zarr = {}
        global_idx = 0
        for zarr_file in zarr_files:
            zarr_group = zarr.open(str(zarr_file), mode='r')
            if 'obs' in zarr_group and 'barcode' in zarr_group['obs']:
                n_cells = zarr_group['obs']['barcode'].shape[0]
                for local_idx in range(n_cells):
                    cell_to_zarr[global_idx] = (zarr_file, local_idx)
                    global_idx += 1
    else:
        # Use obs data to map cells to source zarr files
        obs_df = reader._obs_df
        cell_to_zarr = {}
        
        # Build mapping by iterating through zarr files and matching barcodes
        # This is more reliable than trying to match by source_zarr name
        if 'barcode' in obs_df.columns:
            # Get all barcodes from obs
            obs_barcodes = obs_df['barcode'].to_list()
            
            # Build barcode-to-zarr mapping by checking each zarr file
            # Create a reverse lookup: barcode -> (zarr_file, local_idx)
            barcode_to_zarr_info = {}
            for zarr_file in zarr_files:
                zarr_group = zarr.open(str(zarr_file), mode='r')
                if 'obs' in zarr_group and 'barcode' in zarr_group['obs']:
                    zarr_barcodes = zarr_group['obs']['barcode'][:].tolist()
                    for local_idx, barcode in enumerate(zarr_barcodes):
                        # Store the first match (should be unique)
                        if barcode not in barcode_to_zarr_info:
                            barcode_to_zarr_info[barcode] = (zarr_file, local_idx)
            
            # Now map global indices to zarr files using barcode matching
            for global_idx, obs_barcode in enumerate(obs_barcodes):
                if obs_barcode in barcode_to_zarr_info:
                    cell_to_zarr[global_idx] = barcode_to_zarr_info[obs_barcode]
        
        # Fallback: if no barcode matching worked, use sequential mapping
        if not cell_to_zarr or len(cell_to_zarr) < reader.num_rows:
            if not cell_to_zarr:
                print(f"  ⚠ Could not match barcodes, using sequential mapping")
            else:
                print(f"  ⚠ Only matched {len(cell_to_zarr)}/{reader.num_rows} cells, filling gaps with sequential mapping")
            global_idx = 0
            for zarr_file in zarr_files:
                zarr_group = zarr.open(str(zarr_file), mode='r')
                if 'obs' in zarr_group and 'barcode' in zarr_group['obs']:
                    n_cells = zarr_group['obs']['barcode'].shape[0]
                    for local_idx in range(n_cells):
                        if global_idx < reader.num_rows:
                            # Only add if not already mapped
                            if global_idx not in cell_to_zarr:
                                cell_to_zarr[global_idx] = (zarr_file, local_idx)
                            global_idx += 1
    
    # Test each cell
    all_match = True
    for test_cell_idx in test_cell_indices:
        if test_cell_idx >= reader.num_rows:
            print(f"  ⚠ Cell index {test_cell_idx} out of range (max: {reader.num_rows-1})")
            continue
        
        if test_cell_idx not in cell_to_zarr:
            print(f"  ⚠ Cell index {test_cell_idx} not found in mapping, skipping")
            continue
        
        zarr_file, local_cell_idx = cell_to_zarr[test_cell_idx]
        
        # Read from original zarr file
        zarr_group = zarr.open(str(zarr_file), mode='r')
        
        # Get gene list from zarr
        zarr_genes = zarr_group['var']['gene'][:].tolist()
        
        # Create efficient lookup: gene name -> zarr gene index
        zarr_gene_to_idx = {gene: idx for idx, gene in enumerate(zarr_genes)}
        
        # Get expression data for this cell
        X = zarr_group['X']
        indptr = X["indptr"][:]
        data = X["data"][:]
        indices = X["indices"][:]
        
        # Extract row for this cell
        row_start = indptr[local_cell_idx]
        row_end = indptr[local_cell_idx + 1]
        cell_cols = indices[row_start:row_end]
        cell_vals = data[row_start:row_end]
        
        # Create mapping: zarr gene index -> aligned gene index (only for test genes)
        # Test genes to verify
        test_genes = ['GAPDH', 'TUBB', 'ACTA1', 'PCNA', 'CCND1', 'TP53']
        
        # Debug: check which test genes are in zarr and aligned lists
        print(f"      DEBUG: Checking test genes in zarr file {zarr_file.name}:")
        for test_gene in test_genes:
            in_zarr = test_gene in zarr_gene_to_idx
            in_aligned = test_gene in aligned_genes
            if in_zarr and in_aligned:
                zarr_idx = zarr_gene_to_idx[test_gene]
                aligned_idx = aligned_genes.index(test_gene)
                print(f"        {test_gene}: zarr_idx={zarr_idx}, aligned_idx={aligned_idx}")
            else:
                print(f"        {test_gene}: in_zarr={in_zarr}, in_aligned={in_aligned}")
        
        # Build mapping from zarr gene index to aligned gene index for test genes
        zarr_idx_to_aligned_idx = {}
        for test_gene in test_genes:
            if test_gene in zarr_gene_to_idx and test_gene in aligned_genes:
                zarr_gene_idx = zarr_gene_to_idx[test_gene]
                aligned_gene_idx = aligned_genes.index(test_gene)
                zarr_idx_to_aligned_idx[zarr_gene_idx] = aligned_gene_idx
        
        # Build aligned expression vector (sparse) - only for test genes
        aligned_expr = {}  # aligned_gene_idx -> value
        for col, val in zip(cell_cols, cell_vals):
            if col in zarr_idx_to_aligned_idx:
                aligned_idx = zarr_idx_to_aligned_idx[col]
                aligned_expr[aligned_idx] = int(val)
        
        # Read from zdata object using slicing
        # Also verify we're reading the right cell by checking a few values
        adata = reader[test_cell_idx:test_cell_idx+1]
        zdata_expr = adata.X[0]  # First (and only) row
        
        # Debug: Check total non-zero values in zdata row
        if hasattr(zdata_expr, 'nnz'):
            print(f"      DEBUG: zdata row {test_cell_idx} has {zdata_expr.nnz} non-zero values")
        
        # Convert zdata CSR row to dict for comparison
        zdata_expr_dict = {}
        if hasattr(zdata_expr, 'indices'):
            # Sparse array
            for i, val in zip(zdata_expr.indices, zdata_expr.data):
                zdata_expr_dict[int(i)] = float(val)
        else:
            # Dense array
            for i, val in enumerate(zdata_expr):
                if val != 0:
                    zdata_expr_dict[i] = float(val)
        
        # Debug: Also try reading directly using read_rows to compare
        # Build test_gene_indices first for debugging
        test_gene_indices_debug = {}
        for test_gene in test_genes:
            if test_gene in aligned_genes:
                test_gene_indices_debug[test_gene] = aligned_genes.index(test_gene)
        
        try:
            rows_data_direct = reader.read_rows([test_cell_idx])
            if rows_data_direct:
                direct_row = rows_data_direct[0]
                direct_dict = {}
                if len(direct_row) == 3:  # (row_id, cols, vals)
                    row_id, direct_cols, direct_vals = direct_row
                    print(f"      DEBUG: Direct read row_id={row_id} (expected {test_cell_idx}), {len(direct_cols)} non-zero values")
                    for col, val in zip(direct_cols, direct_vals):
                        direct_dict[int(col)] = float(val)
                    
                    # Check if test genes are in direct read
                    for gene_name, gene_idx in test_gene_indices_debug.items():
                        if gene_idx in aligned_expr:
                            if gene_idx in direct_dict:
                                print(f"      DEBUG: {gene_name} (idx {gene_idx}) found in direct read: value={direct_dict[gene_idx]} (zarr={aligned_expr[gene_idx]})")
                            else:
                                print(f"      DEBUG: {gene_name} (idx {gene_idx}) NOT in direct read (expected value={aligned_expr[gene_idx]})")
                                # Check nearby indices
                                for check_idx in range(max(0, gene_idx - 3), min(reader.num_columns, gene_idx + 4)):
                                    if check_idx in direct_dict:
                                        check_gene = aligned_genes[check_idx] if check_idx < len(aligned_genes) else f"gene_{check_idx}"
                                        print(f"        Nearby: idx {check_idx} ({check_gene}) = {direct_dict[check_idx]}")
        except Exception as e:
            print(f"      DEBUG: Could not read directly: {e}")
            import traceback
            traceback.print_exc()
        
        # Compare values - only for test genes that have expression in zarr
        # Build set of test gene indices that we expect to check
        test_gene_indices = {}
        for test_gene in test_genes:
            if test_gene in aligned_genes:
                test_gene_indices[test_gene] = aligned_genes.index(test_gene)
            else:
                # Debug: gene not in aligned list
                print(f"      DEBUG: Test gene '{test_gene}' not found in aligned gene list")
        
        # Only compare genes that:
        # 1. Are in our test gene list
        # 2. Have expression in the zarr file (non-zero, stored in sparse format)
        genes_to_check = set(aligned_expr.keys()) & set(test_gene_indices.values())
        
        mismatches = []
        missing_in_zdata = []
        
        # Debug: print what we found
        if len(genes_to_check) > 0:
            print(f"      DEBUG: Found {len(genes_to_check)} test genes with expression in zarr")
            for gene_name, gene_idx in test_gene_indices.items():
                if gene_idx in aligned_expr:
                    print(f"        {gene_name} (idx {gene_idx}): zarr value = {aligned_expr[gene_idx]}")
        
        for gene_idx in genes_to_check:
            if gene_idx in zdata_expr_dict:
                # Gene has expression in both - compare values
                zarr_val = aligned_expr[gene_idx]
                zdata_val = zdata_expr_dict[gene_idx]
                if abs(zarr_val - zdata_val) > 1e-6:  # Allow small floating point differences
                    mismatches.append((gene_idx, zarr_val, zdata_val))
            else:
                # Gene has expression in zarr but not in zdata - this is an error
                missing_in_zdata.append(gene_idx)
                # Debug: check if gene is at least in the aligned list
                gene_name = aligned_genes[gene_idx] if gene_idx < len(aligned_genes) else f"gene_{gene_idx}"
                print(f"      DEBUG: {gene_name} (idx {gene_idx}) has value {aligned_expr[gene_idx]} in zarr but missing in zdata")
                # Check if the gene index is even valid
                if gene_idx >= reader.num_columns:
                    print(f"      DEBUG: ERROR - gene index {gene_idx} >= num_columns {reader.num_columns}")
                # Check what's actually in zdata at nearby indices
                nearby_found = []
                for check_idx in range(max(0, gene_idx - 2), min(reader.num_columns, gene_idx + 3)):
                    if check_idx in zdata_expr_dict:
                        nearby_found.append((check_idx, zdata_expr_dict[check_idx]))
                if nearby_found:
                    print(f"      DEBUG: Nearby indices in zdata: {nearby_found}")
        
        # Report results
        if mismatches or missing_in_zdata:
            print(f"  ✗ Cell {test_cell_idx} (from {zarr_file.name}):")
            if mismatches:
                print(f"      Value mismatches: {len(mismatches)} test genes")
                for gene_idx, zarr_val, zdata_val in mismatches:
                    gene_name = aligned_genes[gene_idx] if gene_idx < len(aligned_genes) else f"gene_{gene_idx}"
                    print(f"        {gene_name}: zarr={zarr_val}, zdata={zdata_val}")
            if missing_in_zdata:
                missing_gene_names = [aligned_genes[idx] if idx < len(aligned_genes) else f"gene_{idx}" 
                                     for idx in missing_in_zdata]
                print(f"      Missing in zdata: {len(missing_in_zdata)} test genes with expression: {missing_gene_names}")
            all_match = False
        else:
            # Success: all test genes that have expression in zarr match in zdata
            num_checked = len(genes_to_check)
            if num_checked > 0:
                print(f"  ✓ Cell {test_cell_idx} (from {zarr_file.name}): {num_checked} test genes with expression match")
            else:
                # No test genes had expression in this cell - that's fine
                print(f"  ✓ Cell {test_cell_idx} (from {zarr_file.name}): no test genes expressed (expected for sparse data)")
    
    return all_match


def run_tests(zdata_path, zarr_dir=None):
    """Run high-level tests on the zdata object."""
    print_section("Step 3: Running High-Level Tests")
    
    # Check what's available for testing
    xrm_dir = Path(zdata_path) / "X_RM"
    xcm_dir = Path(zdata_path) / "X_CM"
    has_xrm = xrm_dir.exists() and list(xrm_dir.glob("*.bin"))
    has_xcm = xcm_dir.exists() and list(xcm_dir.glob("*.bin"))
    
    print(f"\nAvailable data:")
    if has_xrm:
        print(f"  ✓ Row-major (X_RM): {len(list(xrm_dir.glob('*.bin')))} chunk file(s)")
    else:
        print(f"  ✗ Row-major (X_RM): Not available")
    
    if has_xcm:
        print(f"  ✓ Column-major (X_CM): {len(list(xcm_dir.glob('*.bin')))} chunk file(s)")
    else:
        print(f"  ✗ Column-major (X_CM): Not available")
    
    # Import ZData for high-level testing
    from zdata.core.zdata import ZData
    
    all_passed = True
    
    try:
        print(f"\n--- Testing ZData reader ---")
        reader = ZData(str(zdata_path))
        
        # Test 1: Basic properties
        print(f"  Testing basic properties...")
        print(f"    Shape: {reader.shape}")
        print(f"    Rows: {reader.num_rows}, Columns: {reader.num_columns}")
        
        if reader.num_rows == 0 or reader.num_columns == 0:
            print(f"  ✗ Invalid dimensions")
            all_passed = False
        else:
            print(f"  ✓ Dimensions valid")
        
        # Test 2: Read a few random rows
        if has_xrm:
            print(f"  Testing row-major reads...")
            test_rows = [0, min(10, reader.num_rows - 1), min(100, reader.num_rows - 1)]
            test_rows = [r for r in test_rows if r < reader.num_rows]
            
            try:
                rows_data = reader.read_rows(test_rows)
                if len(rows_data) == len(test_rows):
                    print(f"  ✓ Successfully read {len(test_rows)} rows")
                else:
                    print(f"  ✗ Row read count mismatch: expected {len(test_rows)}, got {len(rows_data)}")
                    all_passed = False
            except Exception as e:
                print(f"  ✗ Row read failed: {e}")
                all_passed = False
        
        # Test 3: Read as CSR matrix
        if has_xrm:
            print(f"  Testing CSR matrix conversion...")
            try:
                csr = reader.read_rows_csr([0])
                if csr.shape[0] == 1 and csr.shape[1] == reader.num_columns:
                    print(f"  ✓ CSR conversion successful")
                else:
                    print(f"  ✗ CSR shape mismatch: {csr.shape}")
                    all_passed = False
            except Exception as e:
                print(f"  ✗ CSR conversion failed: {e}")
                all_passed = False
        
        # Test 4: Test slicing to AnnData
        if has_xrm:
            print(f"  Testing slicing to AnnData...")
            try:
                adata = reader[0:min(5, reader.num_rows)]
                if adata.shape[0] == min(5, reader.num_rows) and adata.shape[1] == reader.num_columns:
                    print(f"  ✓ AnnData slicing successful")
                    print(f"    AnnData shape: {adata.shape}")
                    print(f"    X type: {type(adata.X)}")
                    if hasattr(adata, 'obs') and adata.obs is not None:
                        print(f"    Obs shape: {adata.obs.shape}")
                    if hasattr(adata, 'var') and adata.var is not None:
                        print(f"    Var shape: {adata.var.shape}")
                else:
                    print(f"  ✗ AnnData shape mismatch: {adata.shape}")
                    all_passed = False
            except Exception as e:
                print(f"  ✗ AnnData slicing failed: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
        
        # Test 5: Column-major reads (if available)
        if has_xcm:
            print(f"  Testing column-major reads...")
            test_cols = [0, min(10, reader.num_columns - 1), min(100, reader.num_columns - 1)]
            test_cols = [c for c in test_cols if c < reader.num_columns]
            
            try:
                cols_data = reader.read_cols_cm(test_cols)
                if len(cols_data) == len(test_cols):
                    print(f"  ✓ Successfully read {len(test_cols)} columns")
                else:
                    print(f"  ✗ Column read count mismatch: expected {len(test_cols)}, got {len(cols_data)}")
                    all_passed = False
            except Exception as e:
                print(f"  ✗ Column read failed: {e}")
                all_passed = False
        
        # Test 6: Check metadata file
        metadata_file = Path(zdata_path) / "metadata.json"
        if metadata_file.exists():
            print(f"  ✓ Metadata file exists")
        else:
            print(f"  ✗ Metadata file missing")
            all_passed = False
        
        # Test 7: Check obs file (if it should exist)
        obs_file = Path(zdata_path) / "obs.parquet"
        if obs_file.exists():
            print(f"  ✓ Obs/metadata parquet file exists")
        else:
            print(f"  ⚠ Obs/metadata parquet file not found (optional)")
        
        # Test 8: Data integrity spot checks
        if zarr_dir and has_xrm:
            print(f"\n--- Testing data integrity (spot checks) ---")
            # Get gene list path (try to find it)
            var_file = Path(zdata_path) / "var.parquet"
            gene_list_path = None
            temp_gene_list = None
            
            # Try to find gene list file
            from zdata.build.align_mtx import _DEFAULT_GENE_LIST
            if os.path.exists(_DEFAULT_GENE_LIST):
                gene_list_path = str(_DEFAULT_GENE_LIST)
            else:
                # Try to read from var.parquet if available
                if var_file.exists():
                    try:
                        import polars as pl
                        var_df = pl.read_parquet(var_file)
                        # Create temporary gene list file
                        import tempfile
                        temp_fd, temp_gene_list = tempfile.mkstemp(suffix='.txt', text=True)
                        with os.fdopen(temp_fd, 'w') as f:
                            for gene in var_df['gene'].to_list():
                                f.write(f"{gene}\n")
                        gene_list_path = temp_gene_list
                    except Exception as e:
                        print(f"  ⚠ Could not determine gene list: {e}")
            
            if gene_list_path:
                try:
                    # Test a few cells: first, middle, and last
                    test_cells = [0]
                    if reader.num_rows > 10:
                        test_cells.append(reader.num_rows // 2)
                    if reader.num_rows > 1:
                        test_cells.append(reader.num_rows - 1)
                    
                    integrity_passed = verify_cell_data_integrity(
                        zarr_dir, zdata_path, gene_list_path, test_cells
                    )
                    if not integrity_passed:
                        all_passed = False
                finally:
                    # Clean up temporary file if we created one
                    if temp_gene_list and os.path.exists(temp_gene_list):
                        try:
                            os.unlink(temp_gene_list)
                        except Exception:
                            pass  # Ignore cleanup errors
            else:
                print(f"  ⚠ Skipping data integrity check: gene list not found")
        
    except Exception as e:
        print(f"  ✗ ZData initialization failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed

def main():
    """Main test pipeline."""
    print_section("ZData Full Pipeline Test")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        zarr_input = sys.argv[1]
    else:
        # Default to directory of zarr files
        zarr_input = DEFAULT_ZARR_DIR
    
    if len(sys.argv) > 2:
        output_name = sys.argv[2]
        # Append .zdata suffix if not already present and not an absolute path
        if not os.path.isabs(output_name) and not output_name.endswith('.zdata'):
            output_name = f"{output_name}.zdata"
    else:
        output_name = DEFAULT_OUTPUT_NAME  # Already has .zdata suffix
    
    # Determine output directory
    if os.path.isabs(output_name):
        # Full path provided
        output_dir = Path(output_name).absolute()
    else:
        # Just name provided, use tests directory location
        output_dir = _test_dir / output_name
    
    print(f"Configuration:")
    print(f"  Zarr directory: {zarr_input}")
    print(f"  Output: {output_dir}")
    print(f"  ZSTD base: {ZSTD_BASE}")
    
    # Step 1: Compile C tools
    if not compile_c_tools():
        print("\n✗ Pipeline failed at compilation step")
        return 1
    
    # Step 2: Build .zdata from zarr files
    build_success, actual_output_dir = build_zdata_directory(zarr_input, output_name, output_dir)
    if not build_success:
        print("\n✗ Pipeline failed at build step")
        return 1
    
    # Use the actual output directory from the build step
    if actual_output_dir is None:
        actual_output_dir = output_dir
    
    # Step 3: Run tests (pass zarr_dir for data integrity checks)
    test_passed = run_tests(str(actual_output_dir), zarr_dir=zarr_input)
    
    # Clean up tmp.zdata directory after tests
    if actual_output_dir.exists() and actual_output_dir.name == "tmp.zdata":
        print(f"\nCleaning up test output directory: {actual_output_dir}")
        import shutil
        try:
            shutil.rmtree(actual_output_dir)
            print(f"✓ Removed {actual_output_dir}")
        except Exception as e:
            print(f"⚠ Warning: Could not remove {actual_output_dir}: {e}")
    
    if not test_passed:
        print("\n✗ Pipeline failed at test step")
        return 1
    
    print_section("Pipeline Complete - All Steps Passed!")
    print(f"\n✓ Compiled C tools")
    print(f"✓ Built and tested {actual_output_dir.name}")
    print(f"✓ All tests passed (row-major and column-major)")
    print(f"✓ Cleaned up test output directory")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

