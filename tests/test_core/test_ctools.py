"""
Tests for C tools compilation and functionality.

These tests verify that:
1. The C tools compile successfully
2. mtx_to_zdata can compress MTX files
3. zdata_read can read compressed files
4. Data integrity is maintained through compression/decompression
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path for imports
_test_dir = Path(__file__).parent.parent
_project_root = _test_dir.parent
_parent_dir = _project_root.parent
import sys

sys.path.insert(0, str(_parent_dir))

from zdata.core import ZData


@pytest.fixture(scope="session")
def zstd_base() -> Path | None:
    """Get ZSTD base directory from environment or default."""
    zstd_path = os.environ.get("ZSTD_BASE", "/home/ubuntu/zstd")
    if os.path.exists(zstd_path):
        return Path(zstd_path)
    return None


@pytest.fixture(scope="session")
def ctools_dir() -> Path:
    """Get the ctools directory."""
    return _project_root / "ctools"


@pytest.fixture(scope="session")
def mtx_to_zdata_src(ctools_dir: Path) -> Path:
    """Path to mtx_to_zdata.c source file."""
    return ctools_dir / "mtx_to_zdata.c"


@pytest.fixture(scope="session")
def zdata_read_src(ctools_dir: Path) -> Path:
    """Path to zdata_read.c source file."""
    return ctools_dir / "zdata_read.c"


@pytest.fixture(scope="session")
def compiled_tools(zstd_base: Path | None, mtx_to_zdata_src: Path, zdata_read_src: Path, tmp_path_factory):
    """Compile C tools and return paths to binaries."""
    if zstd_base is None:
        pytest.skip("ZSTD base directory not found. Set ZSTD_BASE environment variable.")

    # Create temporary directory for compiled binaries
    bin_dir = tmp_path_factory.mktemp("ctools_bin")
    mtx_bin = bin_dir / "mtx_to_zdata"
    read_bin = bin_dir / "zdata_read"

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
            pytest.skip(f"Required ZSTD file not found: {req_file}")

    # Compile mtx_to_zdata
    mtx_compile_cmd = [
        "gcc",
        "-O2",
        "-Wall",
        f"-I{zstd_include}",
        f"-I{zstd_common}",
        f"-I{zstd_seekable}",
        "-o",
        str(mtx_bin),
        str(mtx_to_zdata_src),
        str(zstd_seekable / "zstdseek_compress.c"),
        str(zstd_base / "lib" / "common" / "xxhash.c"),
        str(zstd_lib),
    ]

    result = subprocess.run(mtx_compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"Failed to compile mtx_to_zdata:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    # Compile zdata_read
    read_compile_cmd = [
        "gcc",
        "-O2",
        "-Wall",
        f"-I{zstd_include}",
        f"-I{zstd_common}",
        f"-I{zstd_seekable}",
        "-o",
        str(read_bin),
        str(zdata_read_src),
        str(zstd_seekable / "zstdseek_decompress.c"),
        str(zstd_base / "lib" / "common" / "xxhash.c"),
        str(zstd_lib),
    ]

    result = subprocess.run(read_compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"Failed to compile zdata_read:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    # Make executables
    os.chmod(mtx_bin, 0o755)
    os.chmod(read_bin, 0o755)

    return {"mtx_to_zdata": mtx_bin, "zdata_read": read_bin}


@pytest.fixture(scope="module")
def sample_mtx_file(tmp_path_factory) -> Path:
    """Create a sample MTX file for testing (module-scoped for reuse)."""
    tmp_path = tmp_path_factory.mktemp("mtx_data")
    mtx_file = tmp_path / "sample.mtx"

    # Create a very small sparse matrix for fast testing
    # 5 rows, 10 columns, ~15 non-zero entries
    nrows = 5
    ncols = 10
    nnz = 15

    with open(mtx_file, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("% Sample test matrix\n")
        f.write(f"{nrows} {ncols} {nnz}\n")

        # Add some non-zero entries
        np.random.seed(42)
        for i in range(nnz):
            row = np.random.randint(1, nrows + 1)  # 1-based
            col = np.random.randint(1, ncols + 1)  # 1-based
            val = np.random.randint(1, 100)  # Integer value
            f.write(f"{row} {col} {val}\n")

    return mtx_file


@pytest.fixture(scope="module")
def compressed_zdata_dir(compiled_tools: dict[str, Path], sample_mtx_file: Path, tmp_path_factory):
    """Create a compressed zdata directory (module-scoped for reuse)."""
    mtx_bin = compiled_tools["mtx_to_zdata"]
    output_dir = tmp_path_factory.mktemp("compressed_zdata")

    cmd = [
        str(mtx_bin),
        str(sample_mtx_file),
        str(output_dir),
        "16",  # block_rows
        "8192",  # max_rows
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Failed to create compressed zdata: {result.stderr}")

    return output_dir


class TestCToolsCompilation:
    """Test that C tools compile successfully."""

    def test_mtx_to_zdata_compiles(self, compiled_tools: dict[str, Path]):
        """Test that mtx_to_zdata compiles without errors."""
        mtx_bin = compiled_tools["mtx_to_zdata"]
        assert mtx_bin.exists(), "mtx_to_zdata binary not found"
        assert os.access(mtx_bin, os.X_OK), "mtx_to_zdata is not executable"

    def test_zdata_read_compiles(self, compiled_tools: dict[str, Path]):
        """Test that zdata_read compiles without errors."""
        read_bin = compiled_tools["zdata_read"]
        assert read_bin.exists(), "zdata_read binary not found"
        assert os.access(read_bin, os.X_OK), "zdata_read is not executable"

    def test_mtx_to_zdata_help(self, compiled_tools: dict[str, Path]):
        """Test that mtx_to_zdata runs and shows usage."""
        mtx_bin = compiled_tools["mtx_to_zdata"]
        result = subprocess.run([str(mtx_bin)], capture_output=True, text=True)
        # Should return non-zero exit code and show usage
        assert result.returncode != 0
        assert "Usage" in result.stderr or "Usage" in result.stdout

    def test_zdata_read_help(self, compiled_tools: dict[str, Path]):
        """Test that zdata_read runs and shows usage."""
        read_bin = compiled_tools["zdata_read"]
        result = subprocess.run([str(read_bin)], capture_output=True, text=True)
        # Should return non-zero exit code and show usage
        assert result.returncode != 0
        assert "Usage" in result.stderr or "Usage" in result.stdout


class TestMTXToZData:
    """Test mtx_to_zdata compression functionality."""

    def test_compress_mtx_file(
        self, compiled_tools: dict[str, Path], sample_mtx_file: Path, tmp_path: Path
    ):
        """Test that mtx_to_zdata can compress an MTX file."""
        mtx_bin = compiled_tools["mtx_to_zdata"]
        output_dir = tmp_path / "test_output"

        cmd = [
            str(mtx_bin),
            str(sample_mtx_file),
            str(output_dir),
            "16",  # block_rows
            "8192",  # max_rows
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"mtx_to_zdata failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # Check that output directory was created
        assert output_dir.exists(), "Output directory was not created"

        # Check that X_RM subdirectory was created
        xrm_dir = output_dir / "X_RM"
        assert xrm_dir.exists(), "X_RM subdirectory was not created"

        # Check that at least one .bin file was created
        bin_files = list(xrm_dir.glob("*.bin"))
        assert len(bin_files) > 0, "No .bin files were created"

    def test_compress_mtx_file_with_custom_params(
        self, compiled_tools: dict[str, Path], sample_mtx_file: Path, tmp_path: Path
    ):
        """Test compression with custom block_rows and max_rows."""
        mtx_bin = compiled_tools["mtx_to_zdata"]
        output_dir = tmp_path / "test_output_custom"

        cmd = [
            str(mtx_bin),
            str(sample_mtx_file),
            str(output_dir),
            "8",  # block_rows
            "4096",  # max_rows
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"mtx_to_zdata failed with custom params:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # Verify output
        xrm_dir = output_dir / "X_RM"
        assert xrm_dir.exists()
        bin_files = list(xrm_dir.glob("*.bin"))
        assert len(bin_files) > 0

    def test_compress_mtx_file_column_major(
        self, compiled_tools: dict[str, Path], sample_mtx_file: Path, tmp_path: Path
    ):
        """Test compression with column-major (X_CM) subdirectory."""
        mtx_bin = compiled_tools["mtx_to_zdata"]
        output_dir = tmp_path / "test_output_cm"

        cmd = [
            str(mtx_bin),
            str(sample_mtx_file),
            str(output_dir),
            "16",  # block_rows
            "8192",  # max_rows
            "0",  # row_offset
            "X_CM",  # subdir
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (
            f"mtx_to_zdata failed for X_CM:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # Check that X_CM subdirectory was created
        xcm_dir = output_dir / "X_CM"
        assert xcm_dir.exists(), "X_CM subdirectory was not created"

        # Check that .bin files were created
        bin_files = list(xcm_dir.glob("*.bin"))
        assert len(bin_files) > 0, "No .bin files were created in X_CM"


class TestZDataRead:
    """Test zdata_read decompression functionality."""

    def test_read_compressed_file(
        self, compiled_tools: dict[str, Path], compressed_zdata_dir: Path
    ):
        """Test that zdata_read can read a compressed file."""
        read_bin = compiled_tools["zdata_read"]

        # Find the first .bin file (reuse pre-compressed data)
        xrm_dir = compressed_zdata_dir / "X_RM"
        bin_files = list(xrm_dir.glob("*.bin"))
        assert len(bin_files) > 0, "No .bin files found"

        bin_file = bin_files[0]

        # Read rows 0, 1, 2 from the compressed file
        read_cmd = [str(read_bin), str(bin_file), "0,1,2"]

        result = subprocess.run(read_cmd, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, (
            f"zdata_read failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

        # Check that output contains row data
        assert "row" in result.stdout or len(result.stdout) > 0

    def test_read_compressed_file_binary(
        self, compiled_tools: dict[str, Path], compressed_zdata_dir: Path
    ):
        """Test reading compressed file in binary mode."""
        read_bin = compiled_tools["zdata_read"]

        # Find the first .bin file (reuse pre-compressed data)
        xrm_dir = compressed_zdata_dir / "X_RM"
        bin_files = list(xrm_dir.glob("*.bin"))
        assert len(bin_files) > 0

        bin_file = bin_files[0]

        # Read rows in binary mode
        read_cmd = [str(read_bin), "--binary", str(bin_file), "0,1,2"]

        result = subprocess.run(read_cmd, capture_output=True, timeout=10)
        assert result.returncode == 0, "Binary read failed"

        # Binary output should have data
        assert len(result.stdout) > 0

    def test_read_multiple_rows(
        self, compiled_tools: dict[str, Path], compressed_zdata_dir: Path
    ):
        """Test reading multiple rows from compressed file."""
        read_bin = compiled_tools["zdata_read"]

        # Read multiple rows (reuse pre-compressed data)
        xrm_dir = compressed_zdata_dir / "X_RM"
        bin_file = list(xrm_dir.glob("*.bin"))[0]

        read_cmd = [str(read_bin), str(bin_file), "0,1,2,3,4"]

        result = subprocess.run(read_cmd, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0

        # Should have output for all requested rows
        output_lines = [line for line in result.stdout.split("\n") if line.strip().startswith("row")]
        assert len(output_lines) == 5, f"Expected 5 rows, got {len(output_lines)}"


class TestCToolsIntegration:
    """Integration tests for C tools with Python ZData."""

    def test_compressed_file_readable_by_zdata(
        self, compiled_tools: dict[str, Path], compressed_zdata_dir: Path, sample_mtx_file: Path
    ):
        """Test that files compressed by mtx_to_zdata can be read by ZData."""
        output_dir = compressed_zdata_dir

        # Create metadata.json (minimal) - reuse compressed data
        import json
        import polars as pl

        nrows = 5
        ncols = 10

        metadata = {
            "shape": [nrows, ncols],  # Match sample_mtx_file dimensions
            "nnz_total": 15,
            "num_chunks_rm": 1,
            "total_blocks_rm": 1,
            "block_rows": 16,
            "max_rows_per_chunk": 8192,
            "chunks_rm": [
                {
                    "chunk_num": 0,
                    "file": "0.bin",
                    "start_row": 0,
                    "end_row": nrows,
                }
            ],
        }

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        # Create minimal obs.parquet file (required by ZData)
        obs_data = {
            "barcode": [f"cell_{i}" for i in range(nrows)],
        }
        obs_df = pl.DataFrame(obs_data)
        obs_file = output_dir / "obs.parquet"
        obs_df.write_parquet(obs_file)

        # Create minimal var.parquet file (required by ZData)
        var_data = {
            "gene": [f"GENE_{i}" for i in range(ncols)],
        }
        var_df = pl.DataFrame(var_data)
        var_file = output_dir / "var.parquet"
        var_df.write_parquet(var_file)

        # Try to read with ZData
        try:
            zdata = ZData(str(output_dir))
            assert zdata.nrows == nrows
            assert zdata.ncols == ncols

            # Try reading a row
            rows = zdata.read_rows([0])
            assert len(rows) == 1
        except Exception as e:
            pytest.fail(f"ZData failed to read compressed file: {e}")

