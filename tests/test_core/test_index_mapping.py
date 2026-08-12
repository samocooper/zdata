"""
Tests for explicit obs_index_col / var_index_col mapping in ZData.

Verifies:
- Default (no mapping): obs/var must match matrix dimensions; indices go straight through.
- obs_index_col: obs can have fewer rows; the named column maps obs positions to matrix rows.
- var_index_col: var can have fewer rows; the named column maps var positions to matrix columns.
- Error cases: dimension mismatch without mapping, missing column, non-integer column.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.sparse import csr_matrix

from zdata.core import ZData


@pytest.fixture()
def zdata_with_filtered_obs(zdata_instance: ZData, tmp_path: Path) -> Path:
    """Create a zdata directory where obs has fewer rows than the matrix.

    Copies the existing zdata directory but rewrites obs.parquet to contain
    only a subset of rows, with a ``_row_index`` column that maps back to
    the original matrix row indices.
    """
    import shutil

    src = Path(zdata_instance._ZData__dir_path if hasattr(zdata_instance, '_ZData__dir_path') else zdata_instance.dir_path)
    dst = tmp_path / "filtered_zdata"
    shutil.copytree(src, dst)

    # Read original obs, keep every other row, add _row_index
    obs = pl.read_parquet(str(dst / "obs.parquet"))
    n = len(obs)
    keep = list(range(0, n, 2))  # keep even rows
    filtered = obs[keep]
    filtered = filtered.with_columns(pl.Series("_row_index", keep, dtype=pl.Int64))
    filtered.write_parquet(str(dst / "obs.parquet"))

    return dst


@pytest.fixture()
def zdata_with_filtered_var(zdata_instance: ZData, tmp_path: Path) -> Path:
    """Create a zdata directory where var has fewer rows than the matrix columns.

    Keeps every other gene with a ``_col_index`` mapping column.
    """
    import shutil

    src = Path(zdata_instance.dir_path)
    dst = tmp_path / "filtered_var_zdata"
    shutil.copytree(src, dst)

    var = pd.read_parquet(str(dst / "var.parquet"))
    n = len(var)
    keep = list(range(0, n, 2))
    filtered = var.iloc[keep].copy()
    filtered["_col_index"] = keep
    filtered.to_parquet(str(dst / "var.parquet"))

    return dst


# ---------------------------------------------------------------------------
# Default behaviour: dimensions must match
# ---------------------------------------------------------------------------
class TestDefaultDirectIndexing:
    """When no mapping column is specified, obs/var must match matrix dims."""

    def test_default_loads_ok(self, zdata_instance: ZData):
        """Default init with matching dims works."""
        assert zdata_instance.nrows == len(zdata_instance.obs)
        assert zdata_instance.ncols == len(zdata_instance.var)

    def test_mismatch_obs_raises(self, zdata_with_filtered_obs: Path):
        """Mismatched obs without obs_index_col raises ValueError."""
        with pytest.raises(ValueError, match="obs.parquet has .* rows but the expression matrix"):
            ZData(str(zdata_with_filtered_obs))

    def test_mismatch_var_raises(self, zdata_with_filtered_var: Path):
        """Mismatched var without var_index_col raises ValueError."""
        with pytest.raises(ValueError, match="var.parquet has .* rows but the expression matrix"):
            ZData(str(zdata_with_filtered_var))


# ---------------------------------------------------------------------------
# obs_index_col mapping
# ---------------------------------------------------------------------------
class TestObsIndexCol:
    """Test obs_index_col mapping from obs positions to matrix rows."""

    def test_loads_with_mapping(self, zdata_with_filtered_obs: Path):
        """Loading with obs_index_col succeeds."""
        zd = ZData(str(zdata_with_filtered_obs), obs_index_col="_row_index")
        assert zd._obs_row_index_map is not None
        assert len(zd.obs) < zd.nrows

    def test_read_rows_maps_correctly(self, zdata_instance: ZData, zdata_with_filtered_obs: Path):
        """read_rows with mapping returns the correct matrix rows."""
        zd_mapped = ZData(str(zdata_with_filtered_obs), obs_index_col="_row_index")

        # Query obs row 0 in the mapped dataset -> should be matrix row 0
        mapped_rows = zd_mapped.read_rows([0])
        direct_rows = zdata_instance.read_rows([0])

        assert mapped_rows[0][0] == direct_rows[0][0]  # same row_id
        np.testing.assert_array_equal(mapped_rows[0][1], direct_rows[0][1])  # same cols

    def test_read_rows_obs_row_2_maps_to_matrix_row_4(self, zdata_instance: ZData, zdata_with_filtered_obs: Path):
        """obs row 2 should map to matrix row 4 (every other row kept)."""
        zd_mapped = ZData(str(zdata_with_filtered_obs), obs_index_col="_row_index")

        mapped = zd_mapped.read_rows([2])
        direct = zdata_instance.read_rows([4])

        assert mapped[0][0] == direct[0][0]
        np.testing.assert_array_equal(mapped[0][1], direct[0][1])

    def test_read_rows_csr(self, zdata_with_filtered_obs: Path):
        """read_rows_csr works with obs_index_col."""
        zd = ZData(str(zdata_with_filtered_obs), obs_index_col="_row_index")
        csr = zd.read_rows_csr([0, 1, 2])
        assert csr.shape == (3, zd.ncols)

    def test_getitem_row(self, zdata_with_filtered_obs: Path):
        """__getitem__ row queries use obs positions."""
        zd = ZData(str(zdata_with_filtered_obs), obs_index_col="_row_index")
        adata = zd[0:5]
        assert adata.shape[0] == 5

    def test_invalid_column_name_raises(self, zdata_with_filtered_obs: Path):
        """Non-existent column name raises ValueError."""
        with pytest.raises(ValueError, match="not found in obs.parquet"):
            ZData(str(zdata_with_filtered_obs), obs_index_col="nonexistent")

    def test_non_integer_column_raises(self, zdata_with_filtered_obs: Path, tmp_path: Path):
        """Non-integer column raises ValueError."""
        import shutil
        dst = tmp_path / "bad_type"
        shutil.copytree(zdata_with_filtered_obs, dst)
        obs = pl.read_parquet(str(dst / "obs.parquet"))
        obs = obs.with_columns(pl.lit("hello").alias("str_col"))
        obs.write_parquet(str(dst / "obs.parquet"))

        with pytest.raises(ValueError, match="must be an integer column"):
            ZData(str(dst), obs_index_col="str_col")


# ---------------------------------------------------------------------------
# var_index_col mapping
# ---------------------------------------------------------------------------
class TestVarIndexCol:
    """Test var_index_col mapping from var positions to matrix columns."""

    def test_loads_with_mapping(self, zdata_with_filtered_var: Path):
        """Loading with var_index_col succeeds."""
        zd = ZData(str(zdata_with_filtered_var), var_index_col="_col_index")
        assert zd._var_col_index_map is not None
        assert len(zd.var) < zd.ncols

    def test_invalid_column_name_raises(self, zdata_with_filtered_var: Path):
        """Non-existent column name raises ValueError."""
        with pytest.raises(ValueError, match="not found in var.parquet"):
            ZData(str(zdata_with_filtered_var), var_index_col="nonexistent")


# ---------------------------------------------------------------------------
# Both mappings together
# ---------------------------------------------------------------------------
class TestCombinedMapping:
    """Test using both obs_index_col and var_index_col simultaneously."""

    def test_both_mappings_load(self, zdata_instance: ZData, tmp_path: Path):
        """Can load with both obs and var mappings set."""
        import shutil

        src = Path(zdata_instance.dir_path)
        dst = tmp_path / "both_mapped"
        shutil.copytree(src, dst)

        # Filter obs
        obs = pl.read_parquet(str(dst / "obs.parquet"))
        keep_obs = list(range(0, len(obs), 2))
        obs_f = obs[keep_obs].with_columns(pl.Series("_row_index", keep_obs, dtype=pl.Int64))
        obs_f.write_parquet(str(dst / "obs.parquet"))

        # Filter var
        var = pd.read_parquet(str(dst / "var.parquet"))
        keep_var = list(range(0, len(var), 2))
        var_f = var.iloc[keep_var].copy()
        var_f["_col_index"] = keep_var
        var_f.to_parquet(str(dst / "var.parquet"))

        zd = ZData(str(dst), obs_index_col="_row_index", var_index_col="_col_index")
        assert len(zd.obs) < zd.nrows
        assert len(zd.var) < zd.ncols

        # Row reads should work
        rows = zd.read_rows([0])
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# No inference: a mapping column is used only when explicitly named
# ---------------------------------------------------------------------------
@pytest.fixture()
def zdata_filtered_obs_no_map(zdata_instance: ZData, tmp_path: Path) -> Path:
    """Filtered obs with NO ``_row_index`` column (unrecoverable subset)."""
    import shutil

    src = Path(zdata_instance.dir_path)
    dst = tmp_path / "filtered_obs_no_map"
    shutil.copytree(src, dst)

    obs = pl.read_parquet(str(dst / "obs.parquet"))
    filtered = obs[list(range(0, len(obs), 2))]
    # The stock test obs carries a real _row_index column; drop it so this
    # fixture genuinely represents "subset obs with no way to recover the map".
    if "_row_index" in filtered.columns:
        filtered = filtered.drop("_row_index")
    filtered.write_parquet(str(dst / "obs.parquet"))
    return dst


class TestNoAutomaticInference:
    """The mapping column is never inferred -- naming it is the only way in."""

    def test_default_ignores_present_row_index(self, zdata_with_filtered_obs: Path):
        """Even with _row_index present, the default stays standard row value."""
        with pytest.raises(ValueError, match="obs.parquet has .* rows but the expression matrix"):
            ZData(str(zdata_with_filtered_obs))

    def test_named_column_is_used(self, zdata_with_filtered_obs: Path):
        """Naming the column opts in to the mapping."""
        zd = ZData(str(zdata_with_filtered_obs), obs_index_col="_row_index")
        assert zd._obs_row_index_map is not None
        assert len(zd.obs) < zd.nrows
        assert len(zd.read_rows([0])) == 1

    def test_missing_named_column_raises(self, zdata_filtered_obs_no_map: Path):
        """Naming a column that does not exist is an error, not a fallback."""
        with pytest.raises(ValueError, match="obs_index_col '_row_index' not found"):
            ZData(str(zdata_filtered_obs_no_map), obs_index_col="_row_index")

    def test_subset_without_column_raises(self, zdata_filtered_obs_no_map: Path):
        """A genuine mismatch with no mapping column still raises."""
        with pytest.raises(ValueError, match="obs.parquet has .* rows but the expression matrix"):
            ZData(str(zdata_filtered_obs_no_map))

    def test_var_named_column_is_used(self, zdata_with_filtered_var: Path):
        """Symmetry: var behaves identically with an explicit name."""
        zd = ZData(str(zdata_with_filtered_var), var_index_col="_col_index")
        assert zd._var_col_index_map is not None
        assert len(zd.var) < zd.ncols

    def test_var_default_ignores_present_col_index(self, zdata_with_filtered_var: Path):
        """var default is also standard column value, no inference."""
        with pytest.raises(ValueError, match="var.parquet has .* rows but the expression matrix"):
            ZData(str(zdata_with_filtered_var))
