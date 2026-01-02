"""
Tests for ObsWrapper class.

These tests verify that ObsWrapper correctly wraps polars DataFrames
and provides pandas-compatible indexing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from zdata.core import ObsWrapper


class TestObsWrapper:
    """Test ObsWrapper functionality."""

    @pytest.fixture
    def sample_obs_df(self) -> pl.DataFrame:
        """Create a sample polars DataFrame for testing."""
        data = {
            "barcode": [f"cell_{i}" for i in range(100)],
            "cell_type": np.random.choice(["T-cell", "B-cell"], size=100),
            "batch": np.random.choice(["batch1", "batch2"], size=100),
        }
        return pl.DataFrame(data)

    @pytest.fixture
    def obs_wrapper(self, sample_obs_df: pl.DataFrame) -> ObsWrapper:
        """Create an ObsWrapper instance."""
        return ObsWrapper(sample_obs_df)

    def test_init(self, obs_wrapper: ObsWrapper):
        """Test initialization."""
        assert obs_wrapper is not None

    def test_len(self, obs_wrapper: ObsWrapper):
        """Test __len__ method."""
        assert len(obs_wrapper) == 100

    def test_shape(self, obs_wrapper: ObsWrapper):
        """Test shape property."""
        assert obs_wrapper.shape == (100, 3)

    def test_columns(self, obs_wrapper: ObsWrapper):
        """Test columns property."""
        assert "barcode" in obs_wrapper.columns
        assert "cell_type" in obs_wrapper.columns
        assert "batch" in obs_wrapper.columns

    def test_index_single_row(self, obs_wrapper: ObsWrapper):
        """Test indexing a single row."""
        result = obs_wrapper[5, :]
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_index_slice(self, obs_wrapper: ObsWrapper):
        """Test indexing with slice."""
        result = obs_wrapper[0:10, :]
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_index_invalid_format(self, obs_wrapper: ObsWrapper):
        """Test that invalid indexing format raises error."""
        with pytest.raises(ValueError, match="Obs indexing must be in format"):
            obs_wrapper[5]  # Missing column slice

    def test_repr(self, obs_wrapper: ObsWrapper):
        """Test __repr__ method."""
        repr_str = repr(obs_wrapper)
        assert "ObsWrapper" in repr_str
        assert "100" in repr_str  # Number of rows

