"""Tests for build_zdata.feature_presence.build_feature_presence_matrix."""

from __future__ import annotations

import gzip
import os

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from zdata.build_zdata.feature_presence import build_feature_presence_matrix

REF_GENES = ["A", "B", "C", "D"]


@pytest.fixture()
def atlas(tmp_path):
    """Two studies: s1 measured A,B; s2 measured C,D. Two samples each."""
    z = tmp_path / "zd"
    z.mkdir()
    pl.DataFrame({"gene": REF_GENES}).write_parquet(str(z / "var.parquet"))
    pl.DataFrame({
        "study_name": ["s1", "s1", "s2", "s2"],
        "sample_id": pl.Series([0, 1, 2, 3], dtype=pl.Int32),
        "study_idx": pl.Series([0, 0, 1, 1], dtype=pl.Int32),
    }).write_parquet(str(z / "obs.parquet"))

    src = tmp_path / "mtx"
    for study, genes in (("s1", ["A", "B"]), ("s2", ["C", "D"])):
        d = src / study
        d.mkdir(parents=True)
        with gzip.open(d / "var.csv.gz", "wt") as f:
            f.write("gene\n" + "\n".join(genes) + "\n")
    return z, src


def _load(z, name="feature_presence_matrix.npz"):
    return sp.load_npz(os.path.join(str(z), name)).toarray()


class TestPresenceFromVarCsv:
    def test_sample_granularity_shape_and_content(self, atlas):
        z, src = atlas
        build_feature_presence_matrix(str(z), [str(src)], row_col="sample_id",
                                      verbose=False)
        m = _load(z)
        assert m.shape == (4, 4)
        np.testing.assert_array_equal(m[0], [1, 1, 0, 0])   # s1 sample
        np.testing.assert_array_equal(m[1], [1, 1, 0, 0])
        np.testing.assert_array_equal(m[2], [0, 0, 1, 1])   # s2 sample
        np.testing.assert_array_equal(m[3], [0, 0, 1, 1])

    def test_study_granularity_is_compact_equivalent(self, atlas):
        """study_idx rows carry the same patterns without the duplication."""
        z, src = atlas
        build_feature_presence_matrix(str(z), [str(src)], row_col="study_idx",
                                      verbose=False)
        m = _load(z)
        assert m.shape == (2, 4)
        np.testing.assert_array_equal(m[0], [1, 1, 0, 0])
        np.testing.assert_array_equal(m[1], [0, 0, 1, 1])

    def test_sample_rows_are_redundant_copies_of_study_rows(self, atlas):
        """Documents why study granularity is the compact form."""
        z, src = atlas
        build_feature_presence_matrix(str(z), [str(src)], row_col="sample_id",
                                      verbose=False)
        per_sample = _load(z)
        build_feature_presence_matrix(str(z), [str(src)], row_col="study_idx",
                                      verbose=False)
        per_study = _load(z)
        np.testing.assert_array_equal(per_sample[0], per_study[0])
        np.testing.assert_array_equal(per_sample[2], per_study[1])
        assert len(np.unique(per_sample, axis=0)) == per_study.shape[0]


class TestFallbackBehaviour:
    def test_unresolved_study_marked_fully_present_with_warning(self, tmp_path):
        """The all-ones fallback must be loud -- it silently disables masking."""
        z = tmp_path / "zd"
        z.mkdir()
        pl.DataFrame({"gene": REF_GENES}).write_parquet(str(z / "var.parquet"))
        pl.DataFrame({
            "study_name": ["ghost"],
            "sample_id": pl.Series([0], dtype=pl.Int32),
        }).write_parquet(str(z / "obs.parquet"))
        src = tmp_path / "mtx"
        src.mkdir()

        with pytest.warns(UserWarning, match="no matching var.csv"):
            build_feature_presence_matrix(str(z), [str(src)], row_col="sample_id",
                                          verbose=False)
        np.testing.assert_array_equal(_load(z)[0], [1, 1, 1, 1])

    def test_folder_overrides_resolve_nonconventional_names(self, tmp_path):
        """Folder name unrelated to study name -- no transform could guess it."""
        z = tmp_path / "zd"
        z.mkdir()
        pl.DataFrame({"gene": REF_GENES}).write_parquet(str(z / "var.parquet"))
        pl.DataFrame({
            "study_name": ["internal_project_x_2025"],
            "sample_id": pl.Series([0], dtype=pl.Int32),
        }).write_parquet(str(z / "obs.parquet"))
        src = tmp_path / "mtx"
        d = src / "internal_x_snRNA_2025"
        d.mkdir(parents=True)
        with gzip.open(d / "var.csv.gz", "wt") as f:
            f.write("gene\nA\nC\n")

        build_feature_presence_matrix(
            str(z), [str(src)], row_col="sample_id",
            folder_overrides={"internal_project_x_2025": "internal_x_snRNA_2025"},
            verbose=False)
        np.testing.assert_array_equal(_load(z)[0], [1, 0, 1, 0])

    def test_prefix_stripping_resolves_internal_and_external(self, tmp_path):
        z = tmp_path / "zd"
        z.mkdir()
        pl.DataFrame({"gene": REF_GENES}).write_parquet(str(z / "var.parquet"))
        pl.DataFrame({
            "study_name": ["external_foo", "internal_bar"],
            "sample_id": pl.Series([0, 1], dtype=pl.Int32),
        }).write_parquet(str(z / "obs.parquet"))
        src = tmp_path / "mtx"
        for name, genes in (("foo", ["A"]), ("bar", ["D"])):
            d = src / name
            d.mkdir(parents=True)
            with gzip.open(d / "var.csv.gz", "wt") as f:
                f.write("gene\n" + "\n".join(genes) + "\n")

        build_feature_presence_matrix(str(z), [str(src)], row_col="sample_id",
                                      verbose=False)
        m = _load(z)
        np.testing.assert_array_equal(m[0], [1, 0, 0, 0])
        np.testing.assert_array_equal(m[1], [0, 0, 0, 1])


class TestValidation:
    def test_non_integer_row_col_raises(self, atlas):
        z, src = atlas
        with pytest.raises(TypeError, match="must be integer"):
            build_feature_presence_matrix(str(z), [str(src)], row_col="study_name",
                                          verbose=False)

    def test_sample_col_alias_still_works(self, atlas):
        """Backward compatibility for callers passing the old name."""
        z, src = atlas
        build_feature_presence_matrix(str(z), [str(src)], sample_col="study_idx",
                                      verbose=False)
        assert _load(z).shape == (2, 4)
