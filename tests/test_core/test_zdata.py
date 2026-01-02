"""
Tests for ZData class core functionality.

These tests cover the main ZData class methods including initialization,
row reading, column reading, and indexing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from zdata.core import ZData


class TestZDataInitialization:
    """Test ZData initialization and basic properties."""

    def test_init_with_valid_directory(self, zdata_instance: ZData):
        """Test initialization with a valid zdata directory."""
        assert zdata_instance is not None
        assert zdata_instance.nrows > 0
        assert zdata_instance.ncols > 0
        assert zdata_instance.shape == (zdata_instance.nrows, zdata_instance.ncols)

    def test_init_with_invalid_directory(self, tmp_path):
        """Test initialization fails with invalid directory."""
        invalid_dir = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            ZData(str(invalid_dir))

    def test_properties(self, zdata_instance: ZData):
        """Test basic properties."""
        assert zdata_instance.num_rows == zdata_instance.nrows
        assert zdata_instance.num_columns == zdata_instance.ncols
        assert isinstance(zdata_instance.shape, tuple)
        assert len(zdata_instance.shape) == 2

    def test_obs_property(self, zdata_instance: ZData):
        """Test obs property access."""
        obs = zdata_instance.obs
        assert obs is not None
        assert len(obs) == zdata_instance.nrows


class TestReadRows:
    """Test row reading functionality."""

    def test_read_single_row(self, zdata_instance: ZData):
        """Test reading a single row."""
        test_row = min(0, zdata_instance.nrows - 1)
        rows = zdata_instance.read_rows(test_row)
        assert len(rows) == 1
        row_id, cols, vals = rows[0]
        assert row_id == test_row
        assert isinstance(cols, np.ndarray)
        assert isinstance(vals, np.ndarray)

    def test_read_multiple_rows(self, zdata_instance: ZData, sample_row_indices: list[int]):
        """Test reading multiple rows."""
        valid_indices = [i for i in sample_row_indices if i < zdata_instance.nrows]
        if not valid_indices:
            pytest.skip("No valid row indices for this dataset")
        
        rows = zdata_instance.read_rows(valid_indices)
        assert len(rows) == len(valid_indices)

    def test_read_rows_with_slice(self, zdata_instance: ZData):
        """Test reading rows with a slice."""
        n_rows = min(5, zdata_instance.nrows)
        rows = zdata_instance.read_rows(slice(0, n_rows))
        assert len(rows) == n_rows

    def test_read_rows_with_negative_index(self, zdata_instance: ZData):
        """Test reading rows with negative index."""
        rows = zdata_instance.read_rows(-1)
        assert len(rows) == 1
        row_id, _, _ = rows[0]
        assert row_id == zdata_instance.nrows - 1

    def test_read_rows_with_boolean_mask(
        self, zdata_instance: ZData, sample_boolean_mask: np.ndarray
    ):
        """Test reading rows with boolean mask."""
        if len(sample_boolean_mask) > zdata_instance.nrows:
            mask = sample_boolean_mask[: zdata_instance.nrows]
        else:
            mask = np.zeros(zdata_instance.nrows, dtype=bool)
            mask[: len(sample_boolean_mask)] = sample_boolean_mask

        rows = zdata_instance.read_rows(mask)
        assert len(rows) == np.sum(mask)

    def test_read_rows_out_of_bounds(self, zdata_instance: ZData):
        """Test reading rows with out-of-bounds index."""
        invalid_index = zdata_instance.nrows + 100
        with pytest.raises(IndexError):
            zdata_instance.read_rows(invalid_index)

    def test_read_rows_csr(self, zdata_instance: ZData):
        """Test reading rows as CSR matrix."""
        indices = [0, min(5, zdata_instance.nrows - 1)]
        valid_indices = [i for i in indices if i < zdata_instance.nrows]
        if not valid_indices:
            pytest.skip("No valid row indices for this dataset")

        csr = zdata_instance.read_rows_csr(valid_indices)
        assert isinstance(csr, csr_matrix)
        assert csr.shape[0] == len(valid_indices)
        assert csr.shape[1] == zdata_instance.ncols


class TestReadColumns:
    """Test column (gene) reading functionality."""

    def test_read_cols_by_index(self, zdata_instance: ZData):
        """Test reading columns by integer index."""
        indices = [0, min(5, zdata_instance.ncols - 1)]
        valid_indices = [i for i in indices if i < zdata_instance.ncols]
        if not valid_indices:
            pytest.skip("No valid column indices for this dataset")

        cols = zdata_instance.read_cols_cm(valid_indices)
        assert len(cols) == len(valid_indices)

    def test_read_cols_by_gene_name(self, zdata_instance: ZData):
        """Test reading columns by gene name."""
        if not hasattr(zdata_instance, "_var_df") or "gene" not in zdata_instance._var_df.columns:
            pytest.skip("Gene names not available in var.parquet")

        gene_names = zdata_instance._var_df["gene"].head(3).tolist()
        cols = zdata_instance.read_cols_cm(gene_names)
        assert len(cols) == len(gene_names)

    def test_read_cols_cm_csr(self, zdata_instance: ZData):
        """Test reading columns as CSR matrix."""
        indices = [0, min(5, zdata_instance.ncols - 1)]
        valid_indices = [i for i in indices if i < zdata_instance.ncols]
        if not valid_indices:
            pytest.skip("No valid column indices for this dataset")

        csr = zdata_instance.read_cols_cm_csr(valid_indices)
        assert isinstance(csr, csr_matrix)
        assert csr.shape[0] == len(valid_indices)
        assert csr.shape[1] == zdata_instance.nrows


class TestIndexing:
    """Test __getitem__ indexing functionality."""

    def test_index_single_row(self, zdata_instance: ZData):
        """Test indexing a single row."""
        adata = zdata_instance[0]
        assert adata.shape[0] == 1
        assert adata.shape[1] == zdata_instance.ncols

    def test_index_row_slice(self, zdata_instance: ZData):
        """Test indexing with row slice."""
        n_rows = min(5, zdata_instance.nrows)
        adata = zdata_instance[0:n_rows]
        assert adata.shape[0] == n_rows
        assert adata.shape[1] == zdata_instance.ncols

    def test_index_row_list(self, zdata_instance: ZData):
        """Test indexing with list of row indices."""
        indices = [0, min(5, zdata_instance.nrows - 1)]
        valid_indices = [i for i in indices if i < zdata_instance.nrows]
        if not valid_indices:
            pytest.skip("No valid row indices for this dataset")

        adata = zdata_instance[valid_indices]
        assert adata.shape[0] == len(valid_indices)

    def test_index_by_gene_names(self, zdata_instance: ZData):
        """Test indexing by gene names."""
        if not hasattr(zdata_instance, "_var_df") or "gene" not in zdata_instance._var_df.columns:
            pytest.skip("Gene names not available in var.parquet")

        gene_names = zdata_instance._var_df["gene"].head(3).tolist()
        matrix = zdata_instance[gene_names]
        assert isinstance(matrix, csc_matrix)
        assert matrix.shape[1] == len(gene_names)

    def test_index_by_single_gene_name(self, zdata_instance: ZData):
        """Test indexing by single gene name."""
        if not hasattr(zdata_instance, "_var_df") or "gene" not in zdata_instance._var_df.columns:
            pytest.skip("Gene names not available in var.parquet")

        first_gene = zdata_instance._var_df["gene"].iloc[0]
        matrix = zdata_instance[first_gene]
        assert isinstance(matrix, csc_matrix)
        assert matrix.shape[1] == 1

    def test_get_random_rows(self, zdata_instance: ZData):
        """Test get_random_rows method."""
        n_rows = min(5, zdata_instance.nrows)
        random_rows = zdata_instance.get_random_rows(n_rows, seed=42)
        assert len(random_rows) == n_rows
        assert all(0 <= r < zdata_instance.nrows for r in random_rows)

