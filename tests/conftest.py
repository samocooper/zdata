"""
Pytest configuration and fixtures for zdata tests.

This module provides reusable fixtures for testing zdata functionality.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.sparse import csr_matrix

# Add parent directory to path for imports
_test_dir = Path(__file__).parent
_project_root = _test_dir.parent
_parent_dir = _project_root.parent
sys.path.insert(0, str(_parent_dir))

from zdata.core import ZData


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory path."""
    return _test_dir / "zarr_test_dir"


@pytest.fixture(scope="session")
def zarr_test_dir(test_data_dir: Path) -> Path:
    """Get the zarr test directory if it exists."""
    if not test_data_dir.exists():
        pytest.fail(f"zarr_test_dir not found at {test_data_dir}. Tests require this directory to exist.")
    
    zarr_files = sorted([f for f in test_data_dir.glob("*.zarr") if f.is_dir()])
    if not zarr_files:
        pytest.fail(f"No .zarr files found in {test_data_dir}. Tests require at least one .zarr directory.")
    
    return test_data_dir


@pytest.fixture(scope="session")
def h5ad_test_dir() -> Path:
    """Get the h5ad test directory if it exists."""
    _test_dir = Path(__file__).parent
    h5ad_dir = _test_dir / "h5ad_test_dir"
    
    if not h5ad_dir.exists():
        pytest.fail(f"h5ad_test_dir not found at {h5ad_dir}. Tests require this directory to exist.")
    
    h5ad_files = sorted([f for f in h5ad_dir.iterdir() 
                         if f.is_file() and (f.suffix in ['.h5', '.hdf5'] or f.name.endswith('.h5ad'))])
    if not h5ad_files:
        pytest.fail(f"No .h5/.hdf5 files found in {h5ad_dir}. Tests require at least one h5ad file.")
    
    return h5ad_dir


@pytest.fixture
def tmp_zdata_dir(tmp_path: Path) -> Path:
    """Create a temporary zdata directory structure for testing."""
    zdata_dir = tmp_path / "test_zdata"
    zdata_dir.mkdir()
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
    # Check if we have zarr test files (already validated by zarr_test_dir fixture)
    zarr_files = sorted(zarr_test_dir.glob("*.zarr"))
    if not zarr_files:
        pytest.fail(
            f"No zarr test files found in {zarr_test_dir}. "
            f"This should have been caught by the zarr_test_dir fixture."
        )
    
    project_root = Path(__file__).parent.parent
    mtx_bin = project_root / "ctools" / "mtx_to_zdata"
    read_bin = project_root / "ctools" / "zdata_read"
    
    if not mtx_bin.exists() or not read_bin.exists():
        pytest.fail(
            f"C tools not found. Expected: {mtx_bin}, {read_bin}. "
            f"Please compile the C tools first. "
            f"Run: cd {project_root} && export ZSTD_BASE=/path/to/zstd && python setup.py build_py"
        )
    
    from zdata.build_zdata.build_zdata import build_zdata_from_zarr
    
    tmp_path = tmp_path_factory.mktemp("zdata_test")
    output_name = "test_zdata"
    output_dir = tmp_path / output_name
    
    try:
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
        
        zdata_dir = Path(zdata_dir)
        if not zdata_dir.is_absolute():
            zdata_dir = (tmp_path / zdata_dir).resolve()
        else:
            zdata_dir = zdata_dir.resolve()
        
        if not zdata_dir.exists():
            candidate = tmp_path / output_name
            if candidate.exists():
                zdata_dir = candidate.resolve()
        
        if not zdata_dir.exists():
            existing = list(tmp_path.iterdir()) if tmp_path.exists() else []
            pytest.fail(
                f"ZData directory was not created. "
                f"Expected at: {zdata_dir} or {tmp_path / output_name}. "
                f"Found in tmp_path: {[str(p) for p in existing]}. "
                f"Build may have failed silently. Check the error output above."
            )
        
        return ZData(str(zdata_dir))
    except ImportError as e:
        pytest.fail(
            f"Failed to import build_zdata_from_zarr: {e}. "
            f"This indicates a problem with the package installation."
        )
    except Exception as e:
        import traceback
        error_msg = f"Failed to build test zdata: {type(e).__name__}: {e}"
        print(f"\n{'='*70}")
        print(f"ERROR building zdata_instance fixture:")
        print(f"{error_msg}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*70}\n")
        pytest.fail(
            f"{error_msg}. "
            f"Check the error above for details."
        )


@pytest.fixture(scope="session")
def zdata_instance_h5ad(h5ad_test_dir: Path, tmp_path_factory) -> ZData:
    """Create a ZData instance from test h5ad files (session-scoped for speed).
    
    This fixture builds a zdata directory from the test h5ad files
    and returns a ZData instance. The zdata directory is created once
    per test session and reused across all tests for better performance.
    
    Note: This requires:
    - H5ad test files in tests/h5ad_test_dir/
    - C tools to be compiled (mtx_to_zdata, zdata_read)
    - ZSTD library available
    """
    # Check if we have h5ad test files (already validated by h5ad_test_dir fixture)
    h5ad_files = sorted([f for f in h5ad_test_dir.iterdir() 
                         if f.is_file() and (f.suffix in ['.h5', '.hdf5'] or f.name.endswith('.h5ad'))])
    if not h5ad_files:
        pytest.fail(
            f"No h5ad test files found in {h5ad_test_dir}. "
            f"This should have been caught by the h5ad_test_dir fixture."
        )
    
    project_root = Path(__file__).parent.parent
    mtx_bin = project_root / "ctools" / "mtx_to_zdata"
    read_bin = project_root / "ctools" / "zdata_read"
    
    if not mtx_bin.exists() or not read_bin.exists():
        pytest.fail(
            f"C tools not found. Expected: {mtx_bin}, {read_bin}. "
            f"Please compile the C tools first. "
            f"Run: cd {project_root} && export ZSTD_BASE=/path/to/zstd && python setup.py build_py"
        )
    
    from zdata.build_zdata.build_zdata import build_zdata_from_zarr
    
    tmp_path = tmp_path_factory.mktemp("zdata_test_h5ad")
    output_name = "test_zdata_h5ad"
    output_dir = tmp_path / output_name
    
    try:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            zdata_dir = build_zdata_from_zarr(
                str(h5ad_test_dir),
                output_name,
                block_rows=16,
                max_rows=8192,
                obs_join_strategy="outer",
            )
        finally:
            os.chdir(original_cwd)
        
        zdata_dir = Path(zdata_dir)
        if not zdata_dir.is_absolute():
            zdata_dir = (tmp_path / zdata_dir).resolve()
        else:
            zdata_dir = zdata_dir.resolve()
        
        if not zdata_dir.exists():
            candidate = tmp_path / output_name
            if candidate.exists():
                zdata_dir = candidate.resolve()
        
        if not zdata_dir.exists():
            existing = list(tmp_path.iterdir()) if tmp_path.exists() else []
            pytest.fail(
                f"ZData directory was not created from h5ad files. "
                f"Expected at: {zdata_dir} or {tmp_path / output_name}. "
                f"Found in tmp_path: {[str(p) for p in existing]}. "
                f"Build may have failed silently. Check the error output above."
            )
        
        return ZData(str(zdata_dir))
    except ImportError as e:
        pytest.fail(
            f"Failed to import build_zdata_from_zarr: {e}. "
            f"This indicates a problem with the package installation."
        )
    except Exception as e:
        import traceback
        error_msg = f"Failed to build test zdata from h5ad: {type(e).__name__}: {e}"
        print(f"\n{'='*70}")
        print(f"ERROR building zdata_instance_h5ad fixture:")
        print(f"{error_msg}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*70}\n")
        pytest.fail(
            f"{error_msg}. "
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
    
    original_values = {
        "max_rows_per_chunk": settings.max_rows_per_chunk,
        "block_rows": settings.block_rows,
        "warn_on_large_queries": settings.warn_on_large_queries,
        "large_query_threshold": settings.large_query_threshold,
    }
    
    yield
    
    for key, value in original_values.items():
        setattr(settings, key, value)

