"""Setup script for zdata package with C tools compilation."""

import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.install import install


def find_zstd_base():
    """Find ZSTD base directory from environment or common locations."""
    # Check environment variable first
    zstd_path = os.environ.get("ZSTD_BASE")
    if zstd_path and os.path.exists(zstd_path):
        return Path(zstd_path)
    
    # Try common locations
    common_locations = [
        "/usr/local",
        "/opt/zstd",
        Path.home() / "zstd",
        Path.home() / "zstd-build",
    ]
    
    for loc in common_locations:
        if isinstance(loc, str):
            loc = Path(loc)
        if loc.exists():
            # Check if it looks like a ZSTD source directory
            if (loc / "lib" / "libzstd.a").exists():
                return loc
    
    return None


def compile_c_tools(zstd_base: Path, ctools_dir: Path, bin_dir: Path, raise_on_error: bool = True):
    """Compile C tools.
    
    Parameters
    ----------
    zstd_base : Path
        Path to ZSTD source directory
    ctools_dir : Path
        Directory containing C source files
    bin_dir : Path
        Directory where binaries should be placed
    raise_on_error : bool
        If True, raise RuntimeError on failure. If False, return False.
    
    Returns
    -------
    bool
        True if compilation succeeded, False otherwise (only if raise_on_error=False)
    """
    if zstd_base is None:
        if raise_on_error:
            raise RuntimeError(
                "ZSTD base directory not found. C tools are required for this package.\n"
                "Set ZSTD_BASE environment variable to point to ZSTD source directory.\n"
                "The directory must contain: lib/libzstd.a, lib/common/xxhash.c,\n"
                "and contrib/seekable_format/ source files."
            )
        return False
    
    zstd_include = zstd_base / "lib"
    zstd_common = zstd_base / "lib" / "common"
    zstd_seekable = zstd_base / "contrib" / "seekable_format"
    zstd_lib = zstd_base / "lib" / "libzstd.a"
    
    # Check required files exist
    required_files = [
        zstd_lib,
        zstd_seekable / "zstdseek_compress.c",
        zstd_seekable / "zstdseek_decompress.c",
        zstd_base / "lib" / "common" / "xxhash.c",
    ]
    
    for req_file in required_files:
        if not req_file.exists():
            if raise_on_error:
                raise RuntimeError(
                    f"Required ZSTD file not found: {req_file}\n"
                    "C tools are required for this package. Please ensure ZSTD source code is properly installed."
                )
            return False
    
    # Ensure bin directory exists
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    # Platform-specific binary names
    if sys.platform == "win32":
        mtx_bin = bin_dir / "mtx_to_zdata.exe"
        read_bin = bin_dir / "zdata_read.exe"
    else:
        mtx_bin = bin_dir / "mtx_to_zdata"
        read_bin = bin_dir / "zdata_read"
    
    mtx_src = ctools_dir / "mtx_to_zdata.c"
    read_src = ctools_dir / "zdata_read.c"
    
    # Compile mtx_to_zdata
    mtx_cmd = [
        "gcc", "-O2", "-Wall",
        f"-I{zstd_include}",
        f"-I{zstd_common}",
        f"-I{zstd_seekable}",
        "-o", str(mtx_bin),
        str(mtx_src),
        str(zstd_seekable / "zstdseek_compress.c"),
        str(zstd_base / "lib" / "common" / "xxhash.c"),
        str(zstd_lib),
    ]
    
    # Compile zdata_read
    read_cmd = [
        "gcc", "-O2", "-Wall",
        f"-I{zstd_include}",
        f"-I{zstd_common}",
        f"-I{zstd_seekable}",
        "-o", str(read_bin),
        str(read_src),
        str(zstd_seekable / "zstdseek_decompress.c"),
        str(zstd_base / "lib" / "common" / "xxhash.c"),
        str(zstd_lib),
    ]
    
    try:
        print(f"Compiling mtx_to_zdata...")
        result = subprocess.run(mtx_cmd, capture_output=True, text=True, check=True)
        print(f"✓ mtx_to_zdata compiled successfully")
        
        print(f"Compiling zdata_read...")
        result = subprocess.run(read_cmd, capture_output=True, text=True, check=True)
        print(f"✓ zdata_read compiled successfully")
        
        # Verify binaries were created
        if not mtx_bin.exists():
            if raise_on_error:
                raise RuntimeError(f"mtx_to_zdata binary was not created at {mtx_bin}")
            return False
        if not read_bin.exists():
            if raise_on_error:
                raise RuntimeError(f"zdata_read binary was not created at {read_bin}")
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = "Failed to compile C tools.\n"
        if e.stdout:
            error_msg += f"STDOUT: {e.stdout}\n"
        if e.stderr:
            error_msg += f"STDERR: {e.stderr}\n"
        if raise_on_error:
            raise RuntimeError(error_msg)
        return False
    except FileNotFoundError:
        if raise_on_error:
            raise RuntimeError(
                "gcc compiler not found. C tools are required for this package.\n"
                "Please install GCC and ensure it is in your PATH."
            )
        return False


class BuildPyWithCTools(build_py):
    """Custom build command that compiles C tools."""
    
    def run(self):
        # Get paths
        project_root = Path(__file__).parent
        ctools_dir = project_root / "ctools"
        bin_dir = project_root / "ctools"
        
        # Compile C tools (required)
        zstd_base = find_zstd_base()
        if not zstd_base:
            raise RuntimeError(
                "ZSTD base directory not found. C tools are required for this package.\n"
                "Set ZSTD_BASE environment variable to point to ZSTD source directory."
            )
        
        print(f"Found ZSTD at: {zstd_base}")
        compile_c_tools(zstd_base, ctools_dir, bin_dir)
        
        # Run standard build
        super().run()


def find_precompiled_binaries(package_dir: Path) -> tuple[Path | None, Path | None]:
    """Find pre-compiled binaries in the package directory.
    
    Checks for platform-specific binary names (e.g., .exe on Windows).
    
    Returns
    -------
    tuple[Path | None, Path | None]
        Paths to mtx_to_zdata and zdata_read binaries, or None if not found.
    """
    ctools_dir = package_dir / "ctools"
    
    # Platform-specific binary extensions
    if sys.platform == "win32":
        mtx_name = "mtx_to_zdata.exe"
        read_name = "zdata_read.exe"
    else:
        mtx_name = "mtx_to_zdata"
        read_name = "zdata_read"
    
    mtx_bin = ctools_dir / mtx_name
    read_bin = ctools_dir / read_name
    
    mtx_exists = mtx_bin.exists() and mtx_bin.is_file()
    read_exists = read_bin.exists() and read_bin.is_file()
    
    return (mtx_bin if mtx_exists else None, read_bin if read_exists else None)


class InstallWithCTools(install):
    """Custom install command that compiles C tools or uses pre-compiled binaries."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._using_precompiled = False
    
    def run(self):
        # Run standard install first to get package files in place
        super().run()
        
        # Get paths
        project_root = Path(__file__).parent
        
        # Determine installation location for binaries
        # Check if this is an editable install (attribute may not exist during wheel building)
        is_editable = getattr(self, 'editable', False)
        if is_editable:
            bin_dir = project_root / "ctools"
            package_dir = project_root
        else:
            # For regular installs, install to package directory
            bin_dir = Path(self.install_lib) / "zdata" / "ctools"
            package_dir = Path(self.install_lib) / "zdata"
            bin_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to compile C tools
        zstd_base = find_zstd_base()
        compiled = False
        
        if zstd_base:
            print(f"Found ZSTD at: {zstd_base}")
            print("Compiling C tools from source...")
            compiled = compile_c_tools(zstd_base, project_root / "ctools", bin_dir, raise_on_error=False)
        
        # If compilation failed or ZSTD not found, try to use pre-compiled binaries
        if not compiled:
            print("ZSTD not found or compilation failed. Checking for pre-compiled binaries...")
            
            # Check multiple locations for pre-compiled binaries
            # 1. In the installed package directory
            mtx_precompiled, read_precompiled = find_precompiled_binaries(package_dir)
            
            # 2. In the source directory (for editable installs or if package_dir doesn't have them)
            if not (mtx_precompiled and read_precompiled):
                mtx_precompiled, read_precompiled = find_precompiled_binaries(project_root)
            
            if mtx_precompiled and read_precompiled:
                print("Found pre-compiled binaries. Copying to installation directory...")
                
                # Determine target binary names based on platform
                if sys.platform == "win32":
                    mtx_target = bin_dir / "mtx_to_zdata.exe"
                    read_target = bin_dir / "zdata_read.exe"
                else:
                    mtx_target = bin_dir / "mtx_to_zdata"
                    read_target = bin_dir / "zdata_read"
                
                shutil.copy2(mtx_precompiled, mtx_target)
                shutil.copy2(read_precompiled, read_target)
                
                # Make binaries executable (Unix-like systems only)
                if sys.platform != "win32":
                    mtx_target.chmod(0o755)
                    read_target.chmod(0o755)
                
                # Store flag to print warning at end
                self._using_precompiled = True
            else:
                # No pre-compiled binaries available
                raise RuntimeError(
                    "C tools are required but could not be compiled and no pre-compiled\n"
                    "binaries were found. Please either:\n"
                    "1. Install ZSTD source code and set ZSTD_BASE environment variable, or\n"
                    "2. Install from a distribution that includes pre-compiled binaries."
                )
        else:
            print("✓ C tools compiled successfully from source")
        
        # Print warning at end if using pre-compiled binaries
        if self._using_precompiled:
            print("\n" + "="*70)
            print("WARNING: Using pre-compiled C tools binaries.")
            print("="*70)
            print("The C tools were not compiled from source during installation.")
            print("Pre-compiled binaries are being used, which may have performance")
            print("implications or compatibility issues with your system.")
            print("")
            print("For optimal performance, install ZSTD source code and set ZSTD_BASE:")
            print("  export ZSTD_BASE=/path/to/zstd-source")
            print("  pip install --force-reinstall --no-cache-dir zdata")
            print("="*70 + "\n")


# Read version from __init__.py
def get_version():
    """Extract version from __init__.py."""
    init_file = Path(__file__).parent / "__init__.py"
    with open(init_file) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"


# Read long description from README
def get_long_description():
    """Read long description from README.md."""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        return readme_file.read_text()
    return ""


setup(
    name="zdata",
    version=get_version(),
    description="Efficient storage and access for large single-cell RNA datasets (supports Zarr and H5AD formats)",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Sam Cooper",
    author_email="",  # Add author email if desired
    url="",  # Add project URL if desired (e.g., GitHub repository)
    license="MIT",
    packages=["zdata", "zdata.core", "zdata.build_zdata", "zdata.ctools", "zdata.files"],
    package_dir={"zdata": "."},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "polars>=0.18.0",
        "pandas>=1.3.0",
        "anndata>=0.8.0",
        "zarr>=2.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
        ],
    },
    cmdclass={
        "build_py": BuildPyWithCTools,
        "install": InstallWithCTools,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",  # Update if different
    ],
)

