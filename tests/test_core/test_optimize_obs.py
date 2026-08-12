"""Tests for build_zdata.optimize_obs."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from zdata.build_zdata.optimize_obs import optimize_obs_dtypes, optimize_obs_parquet


def _write(tmp_path, rows: dict, name="obs.parquet") -> str:
    p = tmp_path / name
    pl.DataFrame(rows).write_parquet(str(p))
    return str(p)


class TestIntDowncast:
    @pytest.mark.parametrize("values,expected", [
        ([0, 1, 255], pl.UInt8),
        ([0, 256, 65_535], pl.UInt16),
        ([0, 70_000], pl.UInt32),
        ([-1, 0, 127], pl.Int8),
        ([-129, 0, 200], pl.Int16),
        ([-40_000, 40_000], pl.Int32),
    ])
    def test_smallest_type_chosen(self, tmp_path, values, expected):
        p = _write(tmp_path, {"v": pl.Series(values, dtype=pl.Int64)})
        optimize_obs_parquet(p, verbose=False)
        assert pl.read_parquet_schema(p)["v"] == expected

    def test_values_are_preserved(self, tmp_path):
        vals = [-129, 0, 200, 32_767]
        p = _write(tmp_path, {"v": pl.Series(vals, dtype=pl.Int64)})
        optimize_obs_parquet(p, verbose=False)
        assert pl.read_parquet(p)["v"].to_list() == vals

    def test_all_null_column_untouched(self, tmp_path):
        p = _write(tmp_path, {"v": pl.Series([None, None], dtype=pl.Int64)})
        optimize_obs_parquet(p, verbose=False)
        assert pl.read_parquet_schema(p)["v"] == pl.Int64


class TestEnumConversion:
    def test_low_cardinality_becomes_enum(self, tmp_path):
        p = _write(tmp_path, {"s": ["a", "b", "a", "c"] * 50})
        optimize_obs_parquet(p, verbose=False)
        assert isinstance(pl.read_parquet_schema(p)["s"], pl.Enum)

    def test_enum_values_round_trip(self, tmp_path):
        vals = ["a", "b", "a", "c"] * 50
        p = _write(tmp_path, {"s": vals})
        optimize_obs_parquet(p, verbose=False)
        assert pl.read_parquet(p)["s"].cast(pl.String).to_list() == vals

    def test_high_cardinality_stays_string(self, tmp_path):
        """Per-cell barcodes must not be turned into a 20k-category Enum."""
        vals = [f"bc{i}" for i in range(5000)]
        p = _write(tmp_path, {"s": vals})
        optimize_obs_parquet(p, enum_max_cardinality=100, verbose=False)
        assert pl.read_parquet_schema(p)["s"] == pl.String

    def test_nulls_preserved_in_enum(self, tmp_path):
        vals = ["a", None, "b", None]
        p = _write(tmp_path, {"s": vals})
        optimize_obs_parquet(p, verbose=False)
        out = pl.read_parquet(p)["s"]
        assert out.null_count() == 2
        assert out.cast(pl.String).to_list() == vals


class TestFloatDowncast:
    def test_float64_to_float32(self, tmp_path):
        p = _write(tmp_path, {"f": pl.Series([1.5, 2.25], dtype=pl.Float64)})
        optimize_obs_parquet(p, verbose=False)
        assert pl.read_parquet_schema(p)["f"] == pl.Float32

    def test_can_be_disabled(self, tmp_path):
        p = _write(tmp_path, {"f": pl.Series([1.5], dtype=pl.Float64)})
        optimize_obs_parquet(p, float64_to_float32=False, verbose=False)
        assert pl.read_parquet_schema(p)["f"] == pl.Float64


class TestExcludeAndNoop:
    def test_excluded_column_untouched(self, tmp_path):
        p = _write(tmp_path, {"keep": pl.Series([1, 2], dtype=pl.Int64),
                              "shrink": pl.Series([1, 2], dtype=pl.Int64)})
        optimize_obs_parquet(p, exclude={"keep"}, verbose=False)
        s = pl.read_parquet_schema(p)
        assert s["keep"] == pl.Int64
        assert s["shrink"] == pl.UInt8

    def test_noop_leaves_file_intact(self, tmp_path):
        p = _write(tmp_path, {"v": pl.Series([1, 2], dtype=pl.UInt8)})
        before = pl.read_parquet(p)
        optimize_obs_parquet(p, verbose=False)
        assert pl.read_parquet(p).equals(before)


class TestRowIntegrity:
    def test_all_rows_and_columns_survive(self, tmp_path):
        n = 5000
        rng = np.random.default_rng(0)
        p = _write(tmp_path, {
            "_row_index": pl.Series(np.arange(n), dtype=pl.Int64),
            "study": [f"s{i%7}" for i in range(n)],
            "barcode": [f"bc{i}" for i in range(n)],
            "nnz": pl.Series(rng.integers(1, 500, n), dtype=pl.Int64),
            "score": pl.Series(rng.random(n), dtype=pl.Float64),
        })
        optimize_obs_parquet(p, enum_max_cardinality=100, verbose=False)
        df = pl.read_parquet(p)
        assert df.height == n
        assert set(df.columns) == {"_row_index", "study", "barcode", "nnz", "score"}
        assert df["_row_index"].to_list() == list(range(n))
        # low-cardinality -> Enum, high-cardinality -> String
        s = pl.read_parquet_schema(p)
        assert isinstance(s["study"], pl.Enum)
        assert s["barcode"] == pl.String

    def test_no_temp_file_left_behind(self, tmp_path):
        p = _write(tmp_path, {"v": pl.Series([1, 2], dtype=pl.Int64)})
        optimize_obs_parquet(p, verbose=False)
        assert not (tmp_path / "obs.parquet.opt.tmp").exists()


class TestInMemoryApi:
    def test_dataframe_api_still_works(self):
        df = pl.DataFrame({
            "s": ["a", "b", "a"],
            "i": pl.Series([1, 2, 3], dtype=pl.Int64),
            "f": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
        })
        out = optimize_obs_dtypes(df)
        assert isinstance(out.schema["s"], pl.Enum)
        assert out.schema["i"] == pl.UInt8
        assert out.schema["f"] == pl.Float32
        assert out.height == 3
