"""Tests for build_zdata.sample_id.assign_global_sample_id."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from zdata.build_zdata.sample_id import assign_global_sample_id


def _write_obs(tmp_path, rows: dict) -> str:
    d = tmp_path / "zd"
    d.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_parquet(str(d / "obs.parquet"))
    return str(d)


def _read(zdir) -> pl.DataFrame:
    return pl.read_parquet(f"{zdir}/obs.parquet")


class TestEffectiveSampleFallback:
    """sample_name -> 'donor:'+donor_id -> 'sidx:'+sample_idx."""

    def test_uses_sample_name_when_present(self, tmp_path):
        z = _write_obs(tmp_path, {
            "study_name": ["s1", "s1"],
            "sample_name": ["A", "B"],
            "donor_id": ["d1", "d2"],
            "sample_idx": [0, 1],
        })
        assign_global_sample_id(z, verbose=False)
        assert _read(z)["sample_uid"].to_list() == ["A", "B"]

    @pytest.mark.parametrize("blank", [None, "", "  ", "nan", "NaN", "None", "NA"])
    def test_blank_sample_name_falls_back_to_donor(self, tmp_path, blank):
        z = _write_obs(tmp_path, {
            "study_name": ["s1"],
            "sample_name": [blank],
            "donor_id": ["d9"],
            "sample_idx": [7],
        })
        assign_global_sample_id(z, verbose=False)
        assert _read(z)["sample_uid"].to_list() == ["donor:d9"]

    def test_blank_name_and_donor_falls_back_to_idx(self, tmp_path):
        z = _write_obs(tmp_path, {
            "study_name": ["s1"],
            "sample_name": [None],
            "donor_id": [None],
            "sample_idx": [7],
        })
        assign_global_sample_id(z, verbose=False)
        assert _read(z)["sample_uid"].to_list() == ["sidx:7"]


class TestSampleIdProperties:
    """The id must be globally unique, monotonic, and contiguous per study."""

    def test_same_name_different_studies_gets_different_ids(self, tmp_path):
        """The core bug this module exists to fix."""
        z = _write_obs(tmp_path, {
            "study_name": ["external_a_scRNA", "external_b_snRNA"],
            "sample_name": ["PSC014", "PSC014"],
            "donor_id": [None, None],
            "sample_idx": [0, 0],
        })
        assign_global_sample_id(z, verbose=False)
        ids = _read(z)["sample_id"].to_list()
        assert ids[0] != ids[1], "identical names in different studies must not share an id"

    def test_ids_are_contiguous_per_study(self, tmp_path):
        z = _write_obs(tmp_path, {
            "study_name": ["s1", "s2", "s1", "s2", "s1"],
            "sample_name": ["a", "x", "b", "y", "a"],
            "donor_id": [None] * 5,
            "sample_idx": [0, 0, 1, 1, 0],
        })
        assign_global_sample_id(z, verbose=False)
        df = _read(z)
        for _, grp in df.group_by("study_name"):
            ids = sorted(set(grp["sample_id"].to_list()))
            assert ids == list(range(ids[0], ids[0] + len(ids))), "study ids not contiguous"

    def test_same_sample_same_id(self, tmp_path):
        z = _write_obs(tmp_path, {
            "study_name": ["s1"] * 4,
            "sample_name": ["a", "b", "a", "b"],
            "donor_id": [None] * 4,
            "sample_idx": [0, 1, 0, 1],
        })
        assign_global_sample_id(z, verbose=False)
        ids = _read(z)["sample_id"].to_list()
        assert ids[0] == ids[2] and ids[1] == ids[3]
        assert ids[0] != ids[1]

    def test_id_dtype_is_int32(self, tmp_path):
        z = _write_obs(tmp_path, {
            "study_name": ["s1"], "sample_name": ["a"],
            "donor_id": [None], "sample_idx": [0],
        })
        assign_global_sample_id(z, verbose=False)
        assert _read(z).schema["sample_id"] == pl.Int32


class TestUidDisambiguation:
    """Colliding bare names are prefixed with a study token (external only)."""

    def test_external_collision_gets_token_prefix(self, tmp_path):
        z = _write_obs(tmp_path, {
            "study_name": ["external_x_scRNA", "external_y_snRNA"],
            "sample_name": ["PSC014", "PSC014"],
            "donor_id": [None, None],
            "sample_idx": [0, 0],
        })
        assign_global_sample_id(z, verbose=False)
        uids = _read(z)["sample_uid"].to_list()
        assert set(uids) == {"sc-PSC014", "sn-PSC014"}
        assert len(set(uids)) == 2

    def test_non_colliding_names_are_untouched(self, tmp_path):
        z = _write_obs(tmp_path, {
            "study_name": ["external_x_scRNA", "external_y_snRNA"],
            "sample_name": ["A", "B"],
            "donor_id": [None, None],
            "sample_idx": [0, 0],
        })
        assign_global_sample_id(z, verbose=False)
        assert _read(z)["sample_uid"].to_list() == ["A", "B"]

    def test_internal_studies_not_prefixed(self, tmp_path):
        """Only external_ studies get the token, per the documented rule."""
        z = _write_obs(tmp_path, {
            "study_name": ["internal_a", "internal_b"],
            "sample_name": ["S1", "S1"],
            "donor_id": [None, None],
            "sample_idx": [0, 0],
        })
        assign_global_sample_id(z, verbose=False)
        assert _read(z)["sample_uid"].to_list() == ["S1", "S1"]


class TestRerunSafety:
    def test_rerun_raises_by_default(self, tmp_path):
        z = _write_obs(tmp_path, {
            "study_name": ["s1"], "sample_name": ["a"],
            "donor_id": [None], "sample_idx": [0],
        })
        assign_global_sample_id(z, verbose=False)
        with pytest.raises(ValueError, match="already present"):
            assign_global_sample_id(z, verbose=False)

    def test_overwrite_recomputes(self, tmp_path):
        z = _write_obs(tmp_path, {
            "study_name": ["s1", "s1"], "sample_name": ["a", "b"],
            "donor_id": [None, None], "sample_idx": [0, 1],
        })
        assign_global_sample_id(z, verbose=False)
        n = assign_global_sample_id(z, verbose=False, overwrite=True)
        df = _read(z)
        assert n == 2
        # no duplicated columns left behind by the rewrite
        assert df.columns.count("sample_id") == 1
        assert df.columns.count("sample_uid") == 1

    def test_missing_study_col_raises(self, tmp_path):
        z = _write_obs(tmp_path, {"sample_name": ["a"]})
        with pytest.raises(ValueError, match="study_col"):
            assign_global_sample_id(z, verbose=False)


class TestOtherColumnsPreserved:
    def test_existing_columns_and_order_survive(self, tmp_path):
        z = _write_obs(tmp_path, {
            "_row_index": [0, 1, 2],
            "study_name": ["s1", "s1", "s2"],
            "sample_name": ["a", "b", "a"],
            "donor_id": [None, None, None],
            "sample_idx": [0, 1, 0],
            "nnz": [10, 20, 30],
        })
        assign_global_sample_id(z, verbose=False)
        df = _read(z)
        assert df["_row_index"].to_list() == [0, 1, 2]
        assert df["nnz"].to_list() == [10, 20, 30]
        assert df.height == 3

    def test_batched_rewrite_preserves_all_rows(self, tmp_path):
        """Exercise the multi-batch path of the parquet rewrite."""
        n = 3000
        rng = np.random.default_rng(0)
        z = _write_obs(tmp_path, {
            "study_name": ["s%d" % (i % 7) for i in range(n)],
            "sample_name": ["smp%d" % (i % 53) for i in range(n)],
            "donor_id": [None] * n,
            "sample_idx": rng.integers(0, 10, n).tolist(),
            "nnz": rng.integers(1, 500, n).tolist(),
        })
        assign_global_sample_id(z, verbose=False)
        df = _read(z)
        assert df.height == n
        assert df["sample_id"].null_count() == 0
