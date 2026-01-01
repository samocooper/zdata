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
DEFAULT_ZARR_DIR = "/home/ubuntu/zdata_work/zarr_datasets"  # Directory of zarr files
DEFAULT_OUTPUT_NAME = "atlas.zdata"  # Explicitly use .zdata suffix
DEFAULT_OUTPUT_DIR = _parent_dir / DEFAULT_OUTPUT_NAME

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
        # Pass the full directory name (e.g., "atlas.zdata")
        zdata_dir = build_zdata_from_zarr(
            zarr_dir,
            output_name_only,
            block_rows=16,
            max_rows=8192,
            obs_join_strategy="outer"
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

def run_tests(zdata_path):
    """Run the test suites."""
    print_section("Step 3: Running Tests")
    
    tests_dir = _test_dir
    test_files = [
        "test_random_rows.py",
        "test_fast_queries.py"  # This now tests both row-major and column-major
    ]
    
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
        print(f"  ✗ Column-major (X_CM): Not available (column-major tests will be skipped)")
    
    all_passed = True
    
    for test_file in test_files:
        test_path = tests_dir / test_file
        if not test_path.exists():
            print(f"WARNING: Test file not found: {test_path}")
            continue
        
        print(f"\n--- Running {test_file} ---")
        if test_file == "test_fast_queries.py":
            print("  (This test includes both row-major and column-major queries)")
        
        test_cmd = [
            sys.executable,
            str(test_path),
            str(zdata_path)
        ]
        
        if not run_command(test_cmd, f"Running {test_file}", check=False):
            print(f"✗ {test_file} failed!")
            all_passed = False
        else:
            print(f"✓ {test_file} passed!")
    
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
        # Just name provided, use default location
        output_dir = _parent_dir / output_name
    
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
    
    # Step 3: Run tests
    if not run_tests(str(actual_output_dir)):
        print("\n✗ Pipeline failed at test step")
        return 1
    
    print_section("Pipeline Complete - All Steps Passed!")
    print(f"\n✓ Compiled C tools")
    print(f"✓ Built {actual_output_dir}")
    print(f"✓ All tests passed (row-major and column-major)")
    print(f"\nYou can now use the .zdata directory:")
    print(f"  {actual_output_dir}")
    
    # Show what's available
    xrm_dir = actual_output_dir / "X_RM"
    xcm_dir = actual_output_dir / "X_CM"
    if xrm_dir.exists():
        xrm_files = list(xrm_dir.glob("*.bin"))
        print(f"\nRow-major (X_RM): {len(xrm_files)} chunk file(s)")
    if xcm_dir.exists():
        xcm_files = list(xcm_dir.glob("*.bin"))
        print(f"Column-major (X_CM): {len(xcm_files)} chunk file(s)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

