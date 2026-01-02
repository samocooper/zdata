#!/usr/bin/env python3
"""
Full pipeline test: compiles C tools, builds .zdata from zarr or h5ad files, and runs tests.
"""

import subprocess
import sys
import os
import shutil
import atexit
import signal
from pathlib import Path

# Add the parent directory to the path so we can import zdata
_test_dir = Path(__file__).parent
_project_root = _test_dir.parent  # This is /home/ubuntu/zdata_work/zdata
_parent_dir = _project_root.parent  # This is /home/ubuntu/zdata_work
sys.path.insert(0, str(_parent_dir))

# Import build_zdata_from_zarr
from zdata.build_zdata.build_zdata import build_zdata_from_zarr

# Configuration
ZSTD_BASE = os.environ.get("ZSTD_BASE", "/usr/local")  # Default to common location, override with env var
CTOOLS_DIR = _project_root / "ctools"
MTX_TO_ZDATA_SRC = CTOOLS_DIR / "mtx_to_zdata.c"
ZDATA_READ_SRC = CTOOLS_DIR / "zdata_read.c"
MTX_TO_ZDATA_BIN = CTOOLS_DIR / "mtx_to_zdata"
ZDATA_READ_BIN = CTOOLS_DIR / "zdata_read"

# Default paths (can be overridden via command line)
DEFAULT_ZARR_DIR = os.environ.get("ZDATA_ZARR_DIR", str(_parent_dir / "zarr_datasets"))
DEFAULT_H5AD_DIR = os.environ.get("ZDATA_H5AD_DIR", str(_test_dir / "h5ad_test_dir"))
DEFAULT_OUTPUT_NAME = "atlas.zdata"
DEFAULT_OUTPUT_DIR = _parent_dir / DEFAULT_OUTPUT_NAME

# Track created directories for cleanup
_created_directories = []

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def register_cleanup(path: Path):
    """Register a path for cleanup on exit."""
    if path.exists():
        _created_directories.append(path)


def cleanup_created_files():
    """Clean up all registered test-created directories."""
    if not _created_directories:
        return
    
    print("\n" + "=" * 70)
    print("Cleaning up test-created files...")
    print("=" * 70)
    
    for path in _created_directories:
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"✓ Cleaned up: {path}")
            except Exception as e:
                print(f"⚠ Warning: Could not clean up {path}: {e}")


def signal_handler(signum, frame):
    """Handle interrupt signals and cleanup before exiting."""
    cleanup_created_files()
    sys.exit(1)


# Register cleanup function to run on exit
atexit.register(cleanup_created_files)
# Register signal handlers for cleanup on interrupt
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
    
    if not os.path.exists(ZSTD_BASE):
        print(f"ERROR: ZSTD directory not found at {ZSTD_BASE}")
        print("Please set ZSTD_BASE environment variable")
        return False
    
    zstd_include = f"{ZSTD_BASE}/lib"
    zstd_common = f"{ZSTD_BASE}/lib/common"
    zstd_seekable = f"{ZSTD_BASE}/contrib/seekable_format"
    zstd_lib = f"{ZSTD_BASE}/lib/libzstd.a"
    
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
    
    os.chmod(MTX_TO_ZDATA_BIN, 0o755)
    os.chmod(ZDATA_READ_BIN, 0o755)
    
    print("\n✓ Both C tools compiled successfully!")
    return True

def build_zdata_directory(input_dir, output_name, output_dir, file_type="auto"):
    """Build .zdata directory from zarr or h5ad directory using build_zdata_from_zarr.
    
    Args:
        input_dir: Directory containing .zarr files (directories) or .h5/.hdf5 files (h5ad)
        output_name: Output directory name
        output_dir: Full output directory path
        file_type: "zarr", "h5ad", or "auto" (auto-detect)
    
    Returns:
        tuple: (success: bool, actual_output_dir: Path)
    """
    if file_type == "auto":
        print_section("Step 2: Building .zdata from Files (auto-detecting type)")
    elif file_type == "zarr":
        print_section("Step 2: Building .zdata from Zarr Files")
    elif file_type == "h5ad":
        print_section("Step 2: Building .zdata from H5AD Files")
    else:
        print_section(f"Step 2: Building .zdata from {file_type} Files")
    
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return False, None
    
    if not os.path.isdir(input_dir):
        print(f"ERROR: Path is not a directory: {input_dir}")
        return False, None
    
    # Auto-detect file types
    zarr_files = sorted([f for f in Path(input_dir).glob("*.zarr") if f.is_dir()])
    h5ad_files = sorted([f for f in Path(input_dir).iterdir() 
                         if f.is_file() and (f.suffix in ['.h5', '.hdf5'] or f.name.endswith('.h5ad'))])
    
    if file_type == "zarr":
        if not zarr_files:
            print(f"ERROR: No .zarr files found in {input_dir}")
            return False, None
        print(f"Found {len(zarr_files)} zarr file(s) in {input_dir}")
    elif file_type == "h5ad":
        if not h5ad_files:
            print(f"ERROR: No .h5/.hdf5 files found in {input_dir}")
            return False, None
        print(f"Found {len(h5ad_files)} h5ad file(s) in {input_dir}")
    else:  # auto
        if not zarr_files and not h5ad_files:
            print(f"ERROR: No .zarr files (directories) or .h5/.hdf5 files (h5ad) found in {input_dir}")
            return False, None
        print(f"Found {len(zarr_files)} zarr file(s) and {len(h5ad_files)} h5ad file(s) in {input_dir}")
        print(f"Total files: {len(zarr_files) + len(h5ad_files)}")
    
    # Check if output directory already exists
    if os.path.exists(output_dir):
        print(f"WARNING: Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
        print(f"Deleted existing directory: {output_dir}")
    
    output_parent = os.path.dirname(str(output_dir))
    output_name_only = os.path.basename(str(output_dir))
    
    # Change to output parent directory for the build
    original_cwd = os.getcwd()
    try:
        os.chdir(output_parent)
        file_count = len(zarr_files) if file_type == "zarr" else (len(h5ad_files) if file_type == "h5ad" else len(zarr_files) + len(h5ad_files))
        file_type_str = file_type if file_type != "auto" else "zarr/h5ad"
        print(f"\nBuilding {output_name_only} from {file_type_str} directory: {os.path.basename(input_dir)}...")
        print(f"  (Processing {file_count} file(s))")
        
        # Use build_zdata_from_zarr to build complete zdata object
        # Pass the full directory name (e.g., "atlas.zdata")
        # Note: block_rows and max_rows match zdata.settings defaults
        # but are explicitly set here for reproducibility
        # The function auto-detects file types (zarr vs h5ad)
        zdata_dir = build_zdata_from_zarr(
            input_dir,
            output_name_only,
            block_rows=16,  # Matches zdata.settings.block_rows default
            max_rows=8192,  # Matches zdata.settings.max_rows_per_chunk default
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
    
    expected_output = Path(output_dir)
    if not zdata_dir.exists() and not expected_output.exists():
        print(f"ERROR: Output directory was not created: {zdata_dir} or {expected_output}")
        return False, None
    
    actual_output_dir = zdata_dir if zdata_dir.exists() else expected_output
    
    # Register for cleanup
    register_cleanup(actual_output_dir)
    
    metadata_file = actual_output_dir / "metadata.json"
    if not metadata_file.exists():
        print(f"ERROR: Metadata file not found: {metadata_file}")
        return False, None
    
    xrm_dir = actual_output_dir / "X_RM"
    xrm_bin_files = list(xrm_dir.glob("*.bin")) if xrm_dir.exists() else []
    
    xcm_dir = actual_output_dir / "X_CM"
    xcm_bin_files = list(xcm_dir.glob("*.bin")) if xcm_dir.exists() else []
    
    if not xrm_bin_files and not xcm_bin_files:
        print(f"ERROR: No .bin files found in {actual_output_dir}/X_RM or {actual_output_dir}/X_CM")
        return False, None
    
    obs_file = actual_output_dir / "obs.parquet"
    has_obs = obs_file.exists()
    
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
    
    # Check for --keep-output flag
    keep_output = "--keep-output" in sys.argv
    if keep_output:
        sys.argv.remove("--keep-output")
    
    # Check for --h5ad flag to use h5ad test directory
    use_h5ad = "--h5ad" in sys.argv
    if use_h5ad:
        sys.argv.remove("--h5ad")
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        file_type = "auto"  # Auto-detect
    else:
        if use_h5ad:
            input_dir = DEFAULT_H5AD_DIR
            file_type = "h5ad"
        else:
            input_dir = DEFAULT_ZARR_DIR
            file_type = "zarr"
    
    if len(sys.argv) > 2:
        output_name = sys.argv[2]
        if not os.path.isabs(output_name) and not output_name.endswith('.zdata'):
            output_name = f"{output_name}.zdata"
    else:
        output_name = DEFAULT_OUTPUT_NAME
    
    if os.path.isabs(output_name):
        output_dir = Path(output_name).absolute()
    else:
        output_dir = _parent_dir / output_name
    
    file_type_str = file_type if file_type != "auto" else "zarr/h5ad (auto)"
    print(f"Configuration:")
    print(f"  Input directory: {input_dir}")
    print(f"  File type: {file_type_str}")
    print(f"  Output: {output_dir}")
    print(f"  ZSTD base: {ZSTD_BASE}")
    if not keep_output:
        print(f"  Cleanup: enabled (use --keep-output to preserve files)")
    
    try:
        if not compile_c_tools():
            print("\n✗ Pipeline failed at compilation step")
            return 1
        
        build_success, actual_output_dir = build_zdata_directory(input_dir, output_name, output_dir, file_type)
        if not build_success:
            print("\n✗ Pipeline failed at build step")
            return 1
        
        if actual_output_dir is None:
            actual_output_dir = output_dir
        
        if not run_tests(str(actual_output_dir)):
            print("\n✗ Pipeline failed at test step")
            return 1
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        cleanup_created_files()
        return 1
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        cleanup_created_files()
        raise
    
    print_section("Pipeline Complete - All Steps Passed!")
    print(f"\n✓ Compiled C tools")
    print(f"✓ Built {actual_output_dir}")
    print(f"✓ All tests passed (row-major and column-major)")
    
    if keep_output:
        print(f"\nYou can now use the .zdata directory:")
        print(f"  {actual_output_dir}")
        # Unregister from cleanup if user wants to keep it
        if actual_output_dir in _created_directories:
            _created_directories.remove(actual_output_dir)
    else:
        print(f"\nNote: Output directory will be cleaned up on exit.")
        print(f"  Use --keep-output flag to preserve: {actual_output_dir}")
    
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

