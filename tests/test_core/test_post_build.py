"""Tests for build_zdata.post_build.run_post_build_steps failure policy."""

from __future__ import annotations

import polars as pl
import pytest

from zdata.build_zdata.post_build import PostBuildError, run_post_build_steps


@pytest.fixture()
def zdir(tmp_path):
    d = tmp_path / "zd"
    d.mkdir()
    pl.DataFrame({
        "study_name": ["s1", "s1", "s2"],
        "sample_name": ["a", "b", "a"],
        "donor_id": [None, None, None],
        "sample_idx": [0, 1, 0],
    }).write_parquet(str(d / "obs.parquet"))
    return d


class TestHappyPath:
    def test_sample_id_and_optimize_run(self, zdir):
        res = run_post_build_steps(zdir, generate_feature_presence=False, verbose=False)
        assert res["sample_id"] == "ok"
        assert res["optimize_obs"] == "ok"
        assert "sample_id" in pl.read_parquet_schema(str(zdir / "obs.parquet"))

    def test_disabled_steps_reported_as_skipped(self, zdir):
        res = run_post_build_steps(
            zdir, generate_sample_id=False, optimize_obs=False,
            generate_feature_presence=False, verbose=False)
        assert all(v.startswith("skipped") for v in res.values())


class TestFailurePolicy:
    def test_strict_raises_on_failure(self, tmp_path):
        """A missing study_name makes sample_id fail; strict must surface it."""
        d = tmp_path / "bad"
        d.mkdir()
        pl.DataFrame({"sample_name": ["a"]}).write_parquet(str(d / "obs.parquet"))
        with pytest.raises(PostBuildError, match="sample_id"):
            run_post_build_steps(d, generate_feature_presence=False,
                                 strict=True, verbose=False)

    def test_non_strict_records_failure_and_continues(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        pl.DataFrame({"sample_name": ["a"]}).write_parquet(str(d / "obs.parquet"))
        res = run_post_build_steps(d, generate_feature_presence=False,
                                   strict=False, verbose=False)
        assert res["sample_id"].startswith("failed")
        # later steps still ran rather than aborting the whole build
        assert res["optimize_obs"] == "ok"

    def test_feature_presence_skipped_when_sample_id_failed(self, tmp_path):
        """The cascade guard: no confusing second error for a dependent step."""
        d = tmp_path / "bad"
        d.mkdir()
        pl.DataFrame({"sample_name": ["a"]}).write_parquet(str(d / "obs.parquet"))
        res = run_post_build_steps(d, generate_feature_presence=True,
                                   feature_presence_var_dirs=[str(tmp_path)],
                                   strict=False, verbose=False)
        assert res["sample_id"].startswith("failed")
        assert res["feature_presence"] == "skipped: sample_id step failed"

    def test_feature_presence_skipped_without_var_dirs(self, zdir):
        res = run_post_build_steps(zdir, generate_feature_presence=True,
                                   feature_presence_var_dirs=None, verbose=False)
        assert res["feature_presence"] == "skipped: no feature_presence_var_dirs"
