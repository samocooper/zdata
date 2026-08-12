"""
Tests for building zdata from MTX+CSV directories.

Uses the test fixtures in tests/mtx_test_dir/ (generated from zarr_test_dir).
Verifies the full pipeline: MTX+CSV -> aligned -> compressed -> ZData readable.
Also cross-validates against the zarr-built ZData to ensure equivalence.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zdata.core import ZData
from zdata.build_zdata.build_from_mtx_csv import (
    build_zdata_from_mtx_csv,
    discover_mtx_csv_directories,
    read_gene_list_from_var_csv,
)


@pytest.fixture(scope="session")
def mtx_test_dir() -> Path:
    """Get the mtx_test_dir path."""
    _test_dir = Path(__file__).parent.parent
    mtx_dir = _test_dir / "mtx_test_dir"
    if not mtx_dir.exists():
        pytest.fail(f"mtx_test_dir not found at {mtx_dir}")
    return mtx_dir


@pytest.fixture(scope="session")
def zdata_instance_mtx_csv(mtx_test_dir: Path, tmp_path_factory) -> ZData:
    """Build a ZData instance from mtx_test_dir (session-scoped)."""
    project_root = Path(__file__).parent.parent.parent
    mtx_bin = project_root / "ctools" / "mtx_to_zdata"
    read_bin = project_root / "ctools" / "zdata_read"

    if not mtx_bin.exists() or not read_bin.exists():
        # Environment gap, not a test failure -- see conftest._ensure_ctools.
        pytest.skip(
            "C tools not built. Set ZSTD_BASE=/path/to/zstd and re-run "
            "(conftest compiles them automatically). See tests/README.md."
        )

    tmp_path = tmp_path_factory.mktemp("zdata_test_mtx_csv")
    output_name = "test_zdata_mtx_csv"

    try:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            zdata_dir = build_zdata_from_mtx_csv(
                str(mtx_test_dir),
                output_name,
                block_rows=16,
                max_rows=8192,
                obs_join_strategy="outer",
                min_nnz=None,
            )
        finally:
            os.chdir(original_cwd)

        zdata_dir = Path(zdata_dir)
        if not zdata_dir.is_absolute():
            zdata_dir = (tmp_path / zdata_dir).resolve()

        if not zdata_dir.exists():
            candidate = tmp_path / output_name
            if candidate.exists():
                zdata_dir = candidate.resolve()

        if not zdata_dir.exists():
            pytest.fail(f"ZData directory not created at {zdata_dir}")

        return ZData(str(zdata_dir))

    except Exception as e:
        import traceback
        traceback.print_exc()
        pytest.fail(f"Failed to build zdata from MTX+CSV: {e}")


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------
class TestDiscovery:
    """Test discovery of MTX+CSV subdirectories."""

    def test_discover_valid_directories(self, mtx_test_dir: Path):
        dirs = discover_mtx_csv_directories(str(mtx_test_dir))
        assert len(dirs) == 5

    def test_discover_directory_names(self, mtx_test_dir: Path):
        dirs = discover_mtx_csv_directories(str(mtx_test_dir))
        names = [d.name for d in dirs]
        assert "synthetic_00_test" in names
        assert "synthetic_04_test" in names

    def test_discover_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            discover_mtx_csv_directories(str(tmp_path / "nope"))

    def test_discover_empty_dir_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No valid"):
            discover_mtx_csv_directories(str(tmp_path))

    def test_read_gene_list_from_var_csv(self, mtx_test_dir: Path):
        dirs = discover_mtx_csv_directories(str(mtx_test_dir))
        genes = read_gene_list_from_var_csv(dirs[0] / "var.csv")
        assert len(genes) > 0
        assert isinstance(genes[0], str)


# ---------------------------------------------------------------------------
# ZData initialization tests
# ---------------------------------------------------------------------------
class TestZDataFromMTXCSV:
    """Test that ZData built from MTX+CSV loads and has correct properties."""

    def test_init(self, zdata_instance_mtx_csv: ZData):
        assert zdata_instance_mtx_csv is not None

    def test_nrows(self, zdata_instance_mtx_csv: ZData):
        # 5 datasets x 256 cells each
        assert zdata_instance_mtx_csv.nrows == 1280

    def test_ncols(self, zdata_instance_mtx_csv: ZData):
        # Aligned to standard gene list (35804 genes in 2ks10c_genes.txt, but
        # the file has 35803 lines; the actual count depends on the file)
        assert zdata_instance_mtx_csv.ncols > 30000

    def test_obs_shape(self, zdata_instance_mtx_csv: ZData):
        obs = zdata_instance_mtx_csv.obs
        assert len(obs) == 1280

    def test_var_exists(self, zdata_instance_mtx_csv: ZData):
        var = zdata_instance_mtx_csv.var
        assert var is not None
        assert len(var) == zdata_instance_mtx_csv.ncols


# ---------------------------------------------------------------------------
# Row reading tests
# ---------------------------------------------------------------------------
class TestReadRows:
    """Test reading rows from MTX+CSV-built ZData."""

    def test_read_single_row(self, zdata_instance_mtx_csv: ZData):
        rows = zdata_instance_mtx_csv.read_rows([0])
        assert len(rows) == 1
        row_id, cols, vals = rows[0]
        assert row_id == 0
        assert len(cols) == len(vals)
        assert len(cols) > 0  # Should have non-zero entries

    def test_read_multiple_rows(self, zdata_instance_mtx_csv: ZData):
        rows = zdata_instance_mtx_csv.read_rows([0, 100, 500, 1000])
        assert len(rows) == 4

    def test_read_rows_slice(self, zdata_instance_mtx_csv: ZData):
        rows = zdata_instance_mtx_csv.read_rows(slice(0, 10))
        assert len(rows) == 10

    def test_read_rows_negative_index(self, zdata_instance_mtx_csv: ZData):
        rows = zdata_instance_mtx_csv.read_rows([-1])
        assert len(rows) == 1
        assert rows[0][0] == zdata_instance_mtx_csv.nrows - 1

    def test_read_rows_csr(self, zdata_instance_mtx_csv: ZData):
        csr = zdata_instance_mtx_csv.read_rows_csr([0, 1, 2])
        assert csr.shape == (3, zdata_instance_mtx_csv.ncols)
        assert csr.nnz > 0

    def test_read_rows_out_of_bounds(self, zdata_instance_mtx_csv: ZData):
        with pytest.raises((IndexError, ValueError)):
            zdata_instance_mtx_csv.read_rows([zdata_instance_mtx_csv.nrows + 100])


# ---------------------------------------------------------------------------
# Column reading tests
# ---------------------------------------------------------------------------
class TestReadColumns:
    """Test reading columns from MTX+CSV-built ZData."""

    def test_read_cols_by_index(self, zdata_instance_mtx_csv: ZData):
        csr = zdata_instance_mtx_csv.read_cols_cm_csr([0, 1, 2])
        assert csr.shape[0] in (3, zdata_instance_mtx_csv.nrows)
        assert csr.nnz >= 0

    def test_read_cols_by_gene_name(self, zdata_instance_mtx_csv: ZData):
        gene_names = zdata_instance_mtx_csv.var["gene"].tolist()[:3]
        csc = zdata_instance_mtx_csv[gene_names]
        assert csc.shape[1] == 3


# ---------------------------------------------------------------------------
# Indexing tests
# ---------------------------------------------------------------------------
class TestIndexing:
    """Test __getitem__ indexing on MTX+CSV-built ZData."""

    def test_index_single_row(self, zdata_instance_mtx_csv: ZData):
        result = zdata_instance_mtx_csv[0]
        assert result.shape[0] == 1

    def test_index_row_slice(self, zdata_instance_mtx_csv: ZData):
        result = zdata_instance_mtx_csv[0:5]
        assert result.shape[0] == 5

    def test_index_row_list(self, zdata_instance_mtx_csv: ZData):
        result = zdata_instance_mtx_csv[[0, 10, 50]]
        assert result.shape[0] == 3

    def test_get_random_rows(self, zdata_instance_mtx_csv: ZData):
        rows = zdata_instance_mtx_csv.get_random_rows(5)
        assert len(rows) == 5


# ---------------------------------------------------------------------------
# Cross-validation: zarr vs mtx_csv
# ---------------------------------------------------------------------------
class TestCrossValidationZarr:
    """Verify MTX+CSV-built ZData matches zarr-built ZData."""

    def test_same_shape(self, zdata_instance: ZData, zdata_instance_mtx_csv: ZData):
        assert zdata_instance.nrows == zdata_instance_mtx_csv.nrows
        assert zdata_instance.ncols == zdata_instance_mtx_csv.ncols

    def test_same_row_data(self, zdata_instance: ZData, zdata_instance_mtx_csv: ZData):
        """Spot-check that the same rows have the same data."""
        test_rows = [0, 100, 500, 1000, 1279]
        zarr_rows = zdata_instance.read_rows(test_rows)
        mtx_rows = zdata_instance_mtx_csv.read_rows(test_rows)

        assert len(zarr_rows) == len(mtx_rows)

        for (z_id, z_cols, z_vals), (m_id, m_cols, m_vals) in zip(zarr_rows, mtx_rows):
            assert z_id == m_id, f"Row IDs differ: {z_id} vs {m_id}"
            np.testing.assert_array_equal(
                np.sort(z_cols), np.sort(m_cols),
                err_msg=f"Column indices differ for row {z_id}",
            )

    def test_same_csr_matrix(self, zdata_instance: ZData, zdata_instance_mtx_csv: ZData):
        """Compare CSR matrices for a slice of rows."""
        zarr_csr = zdata_instance.read_rows_csr(slice(0, 50))
        mtx_csr = zdata_instance_mtx_csv.read_rows_csr(slice(0, 50))

        assert zarr_csr.shape == mtx_csr.shape
        diff = zarr_csr - mtx_csr
        assert diff.nnz == 0, f"CSR matrices differ: {diff.nnz} differing elements"

    def test_obs_row_count_matches(self, zdata_instance: ZData, zdata_instance_mtx_csv: ZData):
        assert len(zdata_instance.obs) == len(zdata_instance_mtx_csv.obs)


# ---------------------------------------------------------------------------
# Cross-validation: h5ad vs mtx_csv
# ---------------------------------------------------------------------------
class TestCrossValidationH5AD:
    """Verify MTX+CSV-built ZData matches h5ad-built ZData."""

    def test_same_shape(self, zdata_instance_h5ad: ZData, zdata_instance_mtx_csv: ZData):
        assert zdata_instance_h5ad.nrows == zdata_instance_mtx_csv.nrows
        assert zdata_instance_h5ad.ncols == zdata_instance_mtx_csv.ncols

    def test_same_row_data(self, zdata_instance_h5ad: ZData, zdata_instance_mtx_csv: ZData):
        """Spot-check that the same rows produce the same sparse data."""
        test_rows = [0, 100, 500, 1000, 1279]
        h5ad_rows = zdata_instance_h5ad.read_rows(test_rows)
        mtx_rows = zdata_instance_mtx_csv.read_rows(test_rows)

        assert len(h5ad_rows) == len(mtx_rows)

        for (h_id, h_cols, h_vals), (m_id, m_cols, m_vals) in zip(h5ad_rows, mtx_rows):
            assert h_id == m_id, f"Row IDs differ: {h_id} vs {m_id}"
            np.testing.assert_array_equal(
                np.sort(h_cols), np.sort(m_cols),
                err_msg=f"Column indices differ for row {h_id}",
            )

    def test_same_csr_matrix(self, zdata_instance_h5ad: ZData, zdata_instance_mtx_csv: ZData):
        """Compare CSR matrices for a slice of rows."""
        h5ad_csr = zdata_instance_h5ad.read_rows_csr(slice(0, 50))
        mtx_csr = zdata_instance_mtx_csv.read_rows_csr(slice(0, 50))

        assert h5ad_csr.shape == mtx_csr.shape
        diff = h5ad_csr - mtx_csr
        assert diff.nnz == 0, f"CSR matrices differ: {diff.nnz} differing elements"

    def test_obs_row_count_matches(self, zdata_instance_h5ad: ZData, zdata_instance_mtx_csv: ZData):
        assert len(zdata_instance_h5ad.obs) == len(zdata_instance_mtx_csv.obs)


# ---------------------------------------------------------------------------
# Cross-validation: all three pipelines (zarr vs h5ad vs mtx_csv)
# ---------------------------------------------------------------------------
class TestCrossValidationAllThree:
    """Verify all three pipelines (zarr, h5ad, mtx_csv) produce identical results."""

    def test_all_same_shape(
        self, zdata_instance: ZData, zdata_instance_h5ad: ZData, zdata_instance_mtx_csv: ZData
    ):
        assert zdata_instance.nrows == zdata_instance_h5ad.nrows == zdata_instance_mtx_csv.nrows
        assert zdata_instance.ncols == zdata_instance_h5ad.ncols == zdata_instance_mtx_csv.ncols

    def test_all_same_csr_full(
        self, zdata_instance: ZData, zdata_instance_h5ad: ZData, zdata_instance_mtx_csv: ZData
    ):
        """Compare full CSR matrices across all three pipelines."""
        rows = slice(0, 200)
        zarr_csr = zdata_instance.read_rows_csr(rows)
        h5ad_csr = zdata_instance_h5ad.read_rows_csr(rows)
        mtx_csr = zdata_instance_mtx_csv.read_rows_csr(rows)

        # zarr vs h5ad
        diff_zh = zarr_csr - h5ad_csr
        assert diff_zh.nnz == 0, f"zarr vs h5ad differ: {diff_zh.nnz} elements"

        # zarr vs mtx_csv
        diff_zm = zarr_csr - mtx_csr
        assert diff_zm.nnz == 0, f"zarr vs mtx_csv differ: {diff_zm.nnz} elements"

        # h5ad vs mtx_csv (transitive, but explicit for clarity)
        diff_hm = h5ad_csr - mtx_csr
        assert diff_hm.nnz == 0, f"h5ad vs mtx_csv differ: {diff_hm.nnz} elements"

    def test_all_same_obs_count(
        self, zdata_instance: ZData, zdata_instance_h5ad: ZData, zdata_instance_mtx_csv: ZData
    ):
        assert len(zdata_instance.obs) == len(zdata_instance_h5ad.obs) == len(zdata_instance_mtx_csv.obs)
