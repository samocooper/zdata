"""
Tests for h5ad file processing and zdata building from h5ad files.
"""

import pytest
from pathlib import Path
from zdata.core import ZData


def test_h5ad_build_and_read(zdata_instance_h5ad):
    """Test that h5ad files can be built into zdata and read back."""
    reader = zdata_instance_h5ad
    
    # Basic checks
    assert reader.num_rows > 0, "Should have rows"
    assert reader.num_columns > 0, "Should have columns"
    
    # Test reading specific rows
    test_rows = [0, 10, 50]
    rows_data = reader.read_rows(test_rows)
    
    assert len(rows_data) == len(test_rows), "Should return data for all requested rows"
    
    for row_id, cols, vals in rows_data:
        assert row_id in test_rows, f"Row {row_id} should be in requested rows"
        assert len(cols) == len(vals), "Columns and values should have same length"
        assert len(cols) > 0, "Row should have at least some non-zero values"


def test_h5ad_obs_data(zdata_instance_h5ad):
    """Test that obs data from h5ad files is correctly loaded."""
    reader = zdata_instance_h5ad
    
    # Check obs wrapper exists
    assert hasattr(reader, 'obs'), "Reader should have obs attribute"
    
    obs = reader.obs
    assert obs is not None, "Obs should not be None"
    
    # Check that obs has data
    assert len(obs) > 0, "Obs should have rows"
    assert obs.shape[1] > 0, "Obs should have columns"
    
    # Check for source file column (from h5ad processing)
    if "_source_file" in obs.columns:
        # Verify all rows have source file info
        # Access underlying polars DataFrame to use polars methods
        source_files = obs.obs_df.select("_source_file").unique()
        assert source_files.height > 0, "Should have source file information"


def test_h5ad_csr_reading(zdata_instance_h5ad):
    """Test reading h5ad-derived zdata as CSR matrix."""
    reader = zdata_instance_h5ad
    
    test_rows = [0, 5, 10]
    csr = reader.read_rows_csr(test_rows)
    
    assert csr is not None, "CSR matrix should not be None"
    assert csr.shape[0] == len(test_rows), f"CSR should have {len(test_rows)} rows"
    assert csr.shape[1] == reader.num_columns, "CSR should have correct number of columns"
    assert csr.nnz > 0, "CSR should have non-zero values"


def test_h5ad_random_rows(zdata_instance_h5ad):
    """Test getting random rows from h5ad-derived zdata."""
    reader = zdata_instance_h5ad
    
    n_random = 5
    random_rows = reader.get_random_rows(n_random, seed=42)
    
    assert len(random_rows) == n_random, f"Should get {n_random} random rows"
    assert all(0 <= r < reader.num_rows for r in random_rows), "All row indices should be valid"
    
    # Get same random rows with same seed
    random_rows2 = reader.get_random_rows(n_random, seed=42)
    assert random_rows == random_rows2, "Same seed should produce same random rows"

