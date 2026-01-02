"""
Pytest configuration and fixtures for zdata tests.

This module provides reusable fixtures for testing zdata functionality.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import sys
from pathlib import Path

# Add parent directory to path for imports before other imports
_test_dir = Path(__file__).parent
_project_root = _test_dir.parent
_parent_dir = _project_root.parent
sys.path.insert(0, str(_parent_dir))

import json
import os
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.sparse import csr_matrix

from zdata.core import ZData


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory path."""
    return _test_dir / "zarr_test_dir"


@pytest.fixture(scope="session")
def zarr_test_dir(test_data_dir: Path) -> Path:
    """Get the zarr test directory if it exists."""
    if test_data_dir.exists():
        return test_data_dir
    pytest.skip("zarr_test_dir not found - skipping tests that require it")


@pytest.fixture
def tmp_zdata_dir(tmp_path: Path) -> Path:
    """Create a temporary zdata directory structure for testing."""
    zdata_dir = tmp_path / "test_zdata"
    zdata_dir.mkdir()
    
    # Create subdirectories
    (zdata_dir / "X_RM").mkdir()
    (zdata_dir / "X_CM").mkdir()
    
    return zdata_dir


@pytest.fixture
def sample_metadata(tmp_zdata_dir: Path) -> dict:
    """Create sample metadata for testing."""
    metadata = {
        "shape": [1000, 5000],
        "nnz_total": 500000,
        "num_chunks_rm": 1,
        "total_blocks_rm": 63,
        "block_rows": 16,
        "max_rows_per_chunk": 8192,
        "chunks_rm": [
            {
                "chunk_num": 0,
                "file": "0.bin",
                "start_row": 0,
                "end_row": 1000,
            }
        ],
        "chunks_cm": [
            {
                "chunk_num": 0,
                "file": "0.bin",
                "start_row": 0,
                "end_row": 5000,
            }
        ],
    }
    
    # Write metadata file
    metadata_file = tmp_zdata_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)
    
    return metadata


@pytest.fixture
def sample_obs_data(tmp_zdata_dir: Path) -> pl.DataFrame:
    """Create sample obs (observation) data for testing."""
    n_rows = 1000
    obs_data = {
        "barcode": [f"cell_{i}" for i in range(n_rows)],
        "cell_type": np.random.choice(["T-cell", "B-cell", "NK-cell"], size=n_rows),
        "batch": np.random.choice(["batch1", "batch2"], size=n_rows),
    }
    
    obs_df = pl.DataFrame(obs_data)
    obs_file = tmp_zdata_dir / "obs.parquet"
    obs_df.write_parquet(obs_file)
    
    return obs_df


@pytest.fixture
def sample_var_data(tmp_zdata_dir: Path) -> pl.DataFrame:
    """Create sample var (variable/gene) data for testing."""
    n_cols = 5000
    var_data = {
        "gene": [f"GENE_{i}" for i in range(n_cols)],
        "highly_variable": np.random.choice([True, False], size=n_cols),
    }
    
    var_df = pl.DataFrame(var_data)
    var_file = tmp_zdata_dir / "var.parquet"
    var_df.write_parquet(var_file)
    
    return var_df


@pytest.fixture
def minimal_zdata_dir(
    tmp_zdata_dir: Path,
    sample_metadata: dict,
    sample_obs_data: pl.DataFrame,
    sample_var_data: pl.DataFrame,
) -> Path:
    """Create a minimal zdata directory with metadata and parquet files.
    
    Note: This creates the directory structure but not the actual .bin files.
    For tests that need actual data, use a real zdata directory or create
    mock .bin files.
    """
    return tmp_zdata_dir


@pytest.fixture(scope="session")
def zdata_instance(zarr_test_dir: Path, tmp_path_factory) -> ZData:
    """Create a ZData instance from test zarr files (session-scoped for speed).
    
    This fixture builds a zdata directory from the test zarr files
    and returns a ZData instance. The zdata directory is created once
    per test session and reused across all tests for better performance.
    
    Note: This requires:
    - Zarr test files in tests/zarr_test_dir/
    - C tools to be compiled (mtx_to_zdata, zdata_read)
    - ZSTD library available
    """
    # Check if we have zarr test files
    zarr_files = sorted(zarr_test_dir.glob("*.zarr"))
    if not zarr_files:
        pytest.skip(
            f"No zarr test files found in {zarr_test_dir}. "
            f"Tests that require zdata_instance will be skipped."
        )
    
    # Check if C tools are available (required for build)
    from pathlib import Path as PathLib
    
    project_root = PathLib(__file__).parent.parent
    mtx_bin = project_root / "ctools" / "mtx_to_zdata"
    read_bin = project_root / "ctools" / "zdata_read"
    
    if not mtx_bin.exists() or not read_bin.exists():
        pytest.skip(
            f"C tools not found. Expected: {mtx_bin}, {read_bin}. "
            f"Please compile the C tools first. "
            f"Tests that require zdata_instance will be skipped."
        )
    
    # Build zdata from zarr files for testing
    from zdata.build.build_zdata import build_zdata_from_zarr
    
    # Use session-scoped temporary directory
    tmp_path = tmp_path_factory.mktemp("zdata_test")
    output_name = "test_zdata"
    output_dir = tmp_path / output_name
    
    # Build zdata directory (only once per session)
    try:
        # Change to tmp_path directory for build (build_zdata_from_zarr creates output in current directory)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            zdata_dir = build_zdata_from_zarr(
                str(zarr_test_dir),
                output_name,
                block_rows=16,
                max_rows=8192,
                obs_join_strategy="outer",
            )
        finally:
            os.chdir(original_cwd)
        
        # Convert to absolute path
        zdata_dir = Path(zdata_dir)
        if not zdata_dir.is_absolute():
            # The output is created in the current working directory (tmp_path)
            zdata_dir = (tmp_path / zdata_dir).resolve()
        else:
            # If it's already absolute, use it as-is
            zdata_dir = zdata_dir.resolve()
        
        # Also check if it exists as just the output_name in tmp_path (common case)
        if not zdata_dir.exists():
            candidate = tmp_path / output_name
            if candidate.exists():
                zdata_dir = candidate.resolve()
        
        # Verify the directory was created
        if not zdata_dir.exists():
            # List what's actually in tmp_path for debugging
            existing = list(tmp_path.iterdir()) if tmp_path.exists() else []
            pytest.skip(
                f"ZData directory was not created. "
                f"Expected at: {zdata_dir} or {tmp_path / output_name}. "
                f"Found in tmp_path: {[str(p) for p in existing]}. "
                f"Build may have failed silently. "
                f"Tests that require zdata_instance will be skipped."
            )
        
        # Create ZData instance
        return ZData(str(zdata_dir))
    except ImportError as e:
        pytest.skip(
            f"Failed to import build_zdata_from_zarr: {e}. "
            f"Tests that require zdata_instance will be skipped."
        )
    except Exception as e:
        import traceback
        error_msg = f"Failed to build test zdata: {type(e).__name__}: {e}"
        # Print full traceback for debugging
        print(f"\n{'='*70}")
        print(f"ERROR building zdata_instance fixture:")
        print(f"{error_msg}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*70}\n")
        pytest.skip(
            f"{error_msg}. "
            f"Tests that require zdata_instance will be skipped. "
            f"Check the error above for details."
        )


@pytest.fixture
def sample_row_indices() -> list[int]:
    """Generate sample row indices for testing."""
    return [0, 10, 50, 100, 500]


@pytest.fixture
def sample_gene_names() -> list[str]:
    """Generate sample gene names for testing."""
    return ["GENE_0", "GENE_10", "GENE_50", "GENE_100"]


@pytest.fixture
def sample_boolean_mask() -> np.ndarray:
    """Generate a sample boolean mask for testing."""
    mask = np.zeros(1000, dtype=bool)
    mask[0:100] = True  # First 100 rows
    mask[500:600] = True  # Rows 500-599
    return mask


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset zdata settings before each test."""
    from zdata import settings
    
    # Store original values
    original_values = {
        "max_rows_per_chunk": settings.max_rows_per_chunk,
        "block_rows": settings.block_rows,
        "warn_on_large_queries": settings.warn_on_large_queries,
        "large_query_threshold": settings.large_query_threshold,
    }
    
    yield
    
    # Restore original values
    for key, value in original_values.items():
        setattr(settings, key, value)

