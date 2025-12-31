#!/usr/bin/env python3
"""
Full pipeline test: compiles C tools, builds .zdata from MTX file, and runs tests.
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

# Import build_zdata
from zdata.core.build_zdata import build_zdata

# Configuration
ZSTD_BASE = "/home/ubuntu/zstd"
CTOOLS_DIR = _project_root / "ctools"
MTX_TO_ZDATA_SRC = CTOOLS_DIR / "mtx_to_zdata.c"
ZDATA_READ_SRC = CTOOLS_DIR / "zdata_read.c"
MTX_TO_ZDATA_BIN = CTOOLS_DIR / "mtx_to_zdata"
ZDATA_READ_BIN = CTOOLS_DIR / "zdata_read"

# Default paths (can be overridden via command line)
DEFAULT_MTX_FILE = "/home/ubuntu/zdata_work/mtx_files/external_andrews_hepatolcommun_2022_34792289.mtx"
DEFAULT_OUTPUT_NAME = "andrews"
DEFAULT_OUTPUT_DIR = _parent_dir / f"{DEFAULT_OUTPUT_NAME}.zdata"

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

def build_zdata_directory(mtx_file, output_name, output_dir):
    """Build .zdata directory from MTX file using build_zdata wrapper."""
    print_section("Step 2: Building .zdata from MTX File")
    
    # Check if MTX file exists
    if not os.path.exists(mtx_file):
        print(f"ERROR: MTX file not found: {mtx_file}")
        return False
    
    # Check if output directory already exists
    if os.path.exists(output_dir):
        print(f"WARNING: Output directory already exists: {output_dir}")
        response = input("Delete and rebuild? (y/N): ").strip().lower()
        if response == 'y':
            import shutil
            shutil.rmtree(output_dir)
            print(f"Deleted existing directory: {output_dir}")
        else:
            print("Skipping build step (using existing directory)")
            return True
    
    # Get the parent directory for the output name
    output_parent = os.path.dirname(str(output_dir))
    output_name_only = os.path.basename(str(output_dir)).replace('.zdata', '')
    
    # Change to output parent directory for the build
    original_cwd = os.getcwd()
    try:
        os.chdir(output_parent)
        print(f"\nBuilding {output_name}.zdata from {os.path.basename(mtx_file)}...")
        zdata_dir = build_zdata(mtx_file, output_name_only)
    except Exception as e:
        print(f"ERROR: Failed to build .zdata directory: {e}")
        return False
    finally:
        os.chdir(original_cwd)
    
    # Verify output was created
    if not os.path.exists(output_dir):
        print(f"ERROR: Output directory was not created: {output_dir}")
        return False
    
    # Check for metadata file
    metadata_file = Path(output_dir) / "metadata.json"
    if not metadata_file.exists():
        print(f"ERROR: Metadata file not found: {metadata_file}")
        return False
    
    # Check for .bin files
    bin_files = list(Path(output_dir).glob("*.bin"))
    if not bin_files:
        print(f"ERROR: No .bin files found in {output_dir}")
        return False
    
    print(f"\n✓ Successfully built {output_name}.zdata with {len(bin_files)} chunk files and metadata!")
    return True

def run_tests(zdata_path):
    """Run the test suites."""
    print_section("Step 3: Running Tests")
    
    tests_dir = _test_dir
    test_files = [
        "test_random_rows.py",
        "test_fast_queries.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        test_path = tests_dir / test_file
        if not test_path.exists():
            print(f"WARNING: Test file not found: {test_path}")
            continue
        
        print(f"\n--- Running {test_file} ---")
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
        mtx_file = sys.argv[1]
    else:
        mtx_file = DEFAULT_MTX_FILE
    
    if len(sys.argv) > 2:
        output_name = sys.argv[2]
    else:
        output_name = DEFAULT_OUTPUT_NAME
    
    # Determine output directory
    if os.path.isabs(output_name) or output_name.endswith('.zdata'):
        # Full path provided
        output_dir = Path(output_name).absolute()
        if not output_dir.name.endswith('.zdata'):
            output_dir = output_dir.parent / f"{output_dir.name}.zdata"
    else:
        # Just name provided, use default location
        output_dir = DEFAULT_OUTPUT_DIR
    
    print(f"Configuration:")
    print(f"  MTX file: {mtx_file}")
    print(f"  Output: {output_dir}")
    print(f"  ZSTD base: {ZSTD_BASE}")
    
    # Step 1: Compile C tools
    if not compile_c_tools():
        print("\n✗ Pipeline failed at compilation step")
        return 1
    
    # Step 2: Build .zdata
    if not build_zdata_directory(mtx_file, output_name, output_dir):
        print("\n✗ Pipeline failed at build step")
        return 1
    
    # Step 3: Run tests
    if not run_tests(str(output_dir)):
        print("\n✗ Pipeline failed at test step")
        return 1
    
    print_section("Pipeline Complete - All Steps Passed!")
    print(f"\n✓ Compiled C tools")
    print(f"✓ Built {output_dir}")
    print(f"✓ All tests passed")
    print(f"\nYou can now use the .zdata directory:")
    print(f"  {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

