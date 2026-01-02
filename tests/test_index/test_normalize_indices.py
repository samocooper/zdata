"""
Tests for index normalization functions.

These tests verify that the index normalization functions correctly handle
various input types and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zdata.core.index import (
    normalize_column_indices,
    normalize_row_indices,
    validate_column_indices,
    validate_row_indices,
)


class TestNormalizeRowIndices:
    """Test normalize_row_indices function."""

    def test_single_integer(self):
        """Test with single integer index."""
        indices = normalize_row_indices(5, nrows=100)
        assert indices == [5]

    def test_negative_index(self):
        """Test with negative index."""
        indices = normalize_row_indices(-1, nrows=100)
        assert indices == [99]

    def test_slice(self):
        """Test with slice."""
        indices = normalize_row_indices(slice(0, 10), nrows=100)
        assert indices == list(range(10))

    def test_slice_with_negative(self):
        """Test slice with negative bounds."""
        indices = normalize_row_indices(slice(-10, None), nrows=100)
        assert indices == list(range(90, 100))

    def test_list_of_indices(self):
        """Test with list of indices."""
        indices = normalize_row_indices([0, 5, 10, 20], nrows=100)
        assert indices == [0, 5, 10, 20]

    def test_list_with_duplicates(self):
        """Test that duplicates are removed."""
        indices = normalize_row_indices([0, 5, 5, 10, 10], nrows=100)
        assert indices == [0, 5, 10]

    def test_list_with_negative(self):
        """Test list with negative indices."""
        indices = normalize_row_indices([0, -1, 10], nrows=100)
        assert indices == [0, 10, 99]

    def test_numpy_array(self):
        """Test with numpy array."""
        arr = np.array([0, 5, 10, 20])
        indices = normalize_row_indices(arr, nrows=100)
        assert indices == [0, 5, 10, 20]

    def test_boolean_mask(self):
        """Test with boolean mask."""
        mask = np.array([True] * 10 + [False] * 90)
        indices = normalize_row_indices(mask, nrows=100)
        assert indices == list(range(10))

    def test_boolean_mask_wrong_length(self):
        """Test boolean mask with wrong length raises error."""
        mask = np.array([True] * 50)
        with pytest.raises(ValueError, match="Boolean index length"):
            normalize_row_indices(mask, nrows=100)

    def test_out_of_bounds_positive(self):
        """Test out of bounds positive index."""
        with pytest.raises(IndexError):
            normalize_row_indices(100, nrows=100)

    def test_out_of_bounds_negative(self):
        """Test out of bounds negative index."""
        with pytest.raises(IndexError):
            normalize_row_indices(-101, nrows=100)

    def test_slice_with_step(self):
        """Test slice with step != 1 raises error."""
        with pytest.raises(ValueError, match="Step slicing"):
            normalize_row_indices(slice(0, 10, 2), nrows=100)

    def test_empty_slice(self):
        """Test empty slice."""
        indices = normalize_row_indices(slice(10, 5), nrows=100)
        assert indices == []

    def test_pandas_index(self):
        """Test with pandas Index."""
        idx = pd.Index([0, 5, 10, 20])
        indices = normalize_row_indices(idx, nrows=100)
        assert indices == [0, 5, 10, 20]


class TestNormalizeColumnIndices:
    """Test normalize_column_indices function."""

    def test_single_integer(self):
        """Test with single integer index."""
        indices = normalize_column_indices(5, ncols=100)
        assert indices == [5]

    def test_single_gene_name(self):
        """Test with single gene name."""
        gene_names = pd.Index(["GAPDH", "PCNA", "COL1A1", "ACTB"])
        indices = normalize_column_indices("GAPDH", ncols=4, gene_names=gene_names)
        assert indices == [0]

    def test_list_of_gene_names(self):
        """Test with list of gene names."""
        gene_names = pd.Index(["GAPDH", "PCNA", "COL1A1", "ACTB"])
        indices = normalize_column_indices(
            ["GAPDH", "PCNA"], ncols=4, gene_names=gene_names
        )
        assert indices == [0, 1]

    def test_gene_name_not_found(self):
        """Test with gene name that doesn't exist."""
        gene_names = pd.Index(["GAPDH", "PCNA", "COL1A1"])
        with pytest.raises(IndexError, match="not found"):
            normalize_column_indices("UNKNOWN", ncols=3, gene_names=gene_names)

    def test_gene_names_not_provided(self):
        """Test that gene name indexing requires gene_names."""
        with pytest.raises(ValueError, match="requires gene_names"):
            normalize_column_indices("GAPDH", ncols=100, gene_names=None)

    def test_slice_with_gene_names(self):
        """Test slice with gene name bounds."""
        gene_names = pd.Index(["GAPDH", "PCNA", "COL1A1", "ACTB"])
        indices = normalize_column_indices(
            slice("GAPDH", "COL1A1"), ncols=4, gene_names=gene_names
        )
        assert indices == [0, 1, 2]

    def test_list_with_duplicates(self):
        """Test that duplicates are removed."""
        indices = normalize_column_indices([0, 5, 5, 10], ncols=100)
        assert indices == [0, 5, 10]

    def test_boolean_mask(self):
        """Test with boolean mask."""
        mask = np.array([True] * 10 + [False] * 90)
        indices = normalize_column_indices(mask, ncols=100)
        assert indices == list(range(10))

    def test_boolean_mask_wrong_length(self):
        """Test boolean mask with wrong length."""
        mask = np.array([True] * 50)
        with pytest.raises(ValueError, match="Boolean index length"):
            normalize_column_indices(mask, ncols=100)

    def test_out_of_bounds(self):
        """Test out of bounds index."""
        with pytest.raises(IndexError):
            normalize_column_indices(100, ncols=100)


class TestValidateIndices:
    """Test validation functions."""

    def test_validate_row_indices_valid(self):
        """Test validation with valid row indices."""
        validate_row_indices([0, 5, 10], nrows=100)

    def test_validate_row_indices_invalid(self):
        """Test validation with invalid row indices."""
        with pytest.raises(IndexError):
            validate_row_indices([0, 5, 100], nrows=100)

    def test_validate_column_indices_valid(self):
        """Test validation with valid column indices."""
        validate_column_indices([0, 5, 10], ncols=100)

    def test_validate_column_indices_invalid(self):
        """Test validation with invalid column indices."""
        with pytest.raises(IndexError):
            validate_column_indices([0, 5, 100], ncols=100)

    def test_validate_empty_list(self):
        """Test validation with empty list."""
        validate_row_indices([], nrows=100)
        validate_column_indices([], ncols=100)

