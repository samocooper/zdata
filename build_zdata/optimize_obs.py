"""Dynamic obs dtype optimisation for compact storage + low-memory reads.

Chooses dtypes from each column's *actual* contents (scales to any dataset):

  * String, low-cardinality -> ``pl.Enum`` (native physical width adapts:
    UInt8 for <=256 categories, UInt16 for <=65 536, else falls back to String).
    High-cardinality strings (e.g. per-cell barcodes) stay String.
  * Integer columns -> the smallest native signed/unsigned int that fits the
    observed [min, max] (Int8/16/32/64 or UInt8/16/32/64).
  * Float64 -> Float32 (optional).

The win is in **decoded (in-memory) size**, which is what dominates when a large
obs is loaded for training. Measured on a ~2e7-row, 32-column obs, six
representative columns shrank from 1608 MB to 210 MB (7.6x); a low-cardinality
label column repeated across every row shrank ~36x.

On-disk size may be flat or slightly *worse* (~8% larger on that same file):
parquet already dictionary-encodes and compresses String columns well, so
re-encoding as Enum trades a little file size for a large decode-time saving.
Do not use this expecting smaller files.

Two entry points:

  * :func:`optimize_obs_dtypes` -- operates on an in-memory ``pl.DataFrame``.
  * :func:`optimize_obs_parquet` -- rewrites an obs.parquet **without ever
    materialising it**, using a lazy stats pass and a streaming sink. This
    matters at atlas scale: a wide obs of tens of millions of rows expands to
    tens of GB once decoded, so a read-all implementation would defeat the
    purpose of the optimisation (and can OOM the machine doing it).
"""
from __future__ import annotations

import os

import polars as pl

# (signed) and (unsigned) candidate dtypes, smallest first, with their ranges.
_UINT = [(pl.UInt8, 0, 255), (pl.UInt16, 0, 65_535),
         (pl.UInt32, 0, 4_294_967_295), (pl.UInt64, 0, 2**64 - 1)]
_SINT = [(pl.Int8, -128, 127), (pl.Int16, -32_768, 32_767),
         (pl.Int32, -2_147_483_648, 2_147_483_647), (pl.Int64, -(2**63), 2**63 - 1)]
_INT_TYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
              pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}


def _smallest_int_dtype(mn, mx):
    if mn is None:                      # all-null column: leave as-is
        return None
    table = _UINT if mn >= 0 else _SINT
    for dt, lo, hi in table:
        if lo <= mn and mx <= hi:
            return dt
    return pl.Int64


def optimize_obs_dtypes(
    df: pl.DataFrame,
    enum_max_cardinality: int = 65_535,
    downcast_ints: bool = True,
    float64_to_float32: bool = True,
    exclude: "set[str] | tuple" = (),
    verbose: bool = False,
) -> pl.DataFrame:
    """Return ``df`` with dynamically chosen compact dtypes.

    Parameters
    ----------
    enum_max_cardinality
        String columns with <= this many distinct values become ``pl.Enum``.
    downcast_ints
        Downcast integer columns to the smallest native int that fits.
    float64_to_float32
        Downcast Float64 columns to Float32.
    exclude
        Column names to leave untouched.
    """
    exprs = []
    for name, dt in df.schema.items():
        if name in exclude:
            continue
        try:
            if dt == pl.String:
                card = df[name].n_unique()
                if card <= enum_max_cardinality:
                    cats = df[name].drop_nulls().unique().sort().to_list()
                    exprs.append(pl.col(name).cast(pl.Enum(cats)))
                    if verbose:
                        print(f"  {name}: String -> Enum ({card} cats)")
            elif downcast_ints and dt in _INT_TYPES:
                mn, mx = df[name].min(), df[name].max()
                tgt = _smallest_int_dtype(mn, mx)
                if tgt is not None and tgt != dt:
                    exprs.append(pl.col(name).cast(tgt))
                    if verbose:
                        print(f"  {name}: {dt} -> {tgt} (range {mn}..{mx})")
            elif float64_to_float32 and dt == pl.Float64:
                exprs.append(pl.col(name).cast(pl.Float32))
                if verbose:
                    print(f"  {name}: Float64 -> Float32")
        except Exception as e:  # never let optimisation break a build
            if verbose:
                print(f"  {name}: skipped ({e})")
    return df.with_columns(exprs) if exprs else df


def _plan_from_scan(
    lf: pl.LazyFrame,
    schema: dict,
    enum_max_cardinality: int,
    downcast_ints: bool,
    float64_to_float32: bool,
    exclude,
    verbose: bool,
) -> list[pl.Expr]:
    """Work out the cast expressions using only aggregate scans of the file."""
    str_cols = [n for n, d in schema.items() if d == pl.String and n not in exclude]
    int_cols = ([n for n, d in schema.items() if d in _INT_TYPES and n not in exclude]
                if downcast_ints else [])
    f64_cols = ([n for n, d in schema.items() if d == pl.Float64 and n not in exclude]
                if float64_to_float32 else [])

    # One pass for cheap aggregates. approx_n_unique keeps memory bounded on
    # high-cardinality columns (per-cell barcodes) that we do not want to Enum
    # anyway; exact uniques are only collected for columns that pass the screen.
    aggs = [pl.col(c).approx_n_unique().alias(f"~n~{c}") for c in str_cols]
    for c in int_cols:
        aggs.append(pl.col(c).min().alias(f"~mn~{c}"))
        aggs.append(pl.col(c).max().alias(f"~mx~{c}"))
    stats = lf.select(aggs).collect() if aggs else None

    exprs: list[pl.Expr] = []

    # Columns whose approximate cardinality is small enough to Enum. Their exact
    # category sets are gathered in ONE further pass -- collecting them one
    # column at a time meant re-scanning the file per column, which on a wide
    # atlas obs is both slow and the dominant source of peak memory.
    enum_cands = [c for c in str_cols
                  if stats[f"~n~{c}"][0] is not None
                  and stats[f"~n~{c}"][0] <= enum_max_cardinality]
    if enum_cands:
        cat_row = lf.select([
            pl.col(c).drop_nulls().unique().sort().implode().alias(c)
            for c in enum_cands
        ]).collect()
        for c in enum_cands:
            cats = cat_row[c][0].to_list()
            # approx_n_unique can under-report; re-check against the exact set.
            if len(cats) > enum_max_cardinality:
                continue
            exprs.append(pl.col(c).cast(pl.Enum(cats)))
            if verbose:
                print(f"  {c}: String -> Enum ({len(cats)} cats)")

    for c in int_cols:
        mn, mx = stats[f"~mn~{c}"][0], stats[f"~mx~{c}"][0]
        tgt = _smallest_int_dtype(mn, mx)
        if tgt is not None and tgt != schema[c]:
            exprs.append(pl.col(c).cast(tgt))
            if verbose:
                print(f"  {c}: {schema[c]} -> {tgt} (range {mn}..{mx})")

    for c in f64_cols:
        exprs.append(pl.col(c).cast(pl.Float32))
        if verbose:
            print(f"  {c}: Float64 -> Float32")

    return exprs


def optimize_obs_parquet(
    path: str,
    enum_max_cardinality: int = 65_535,
    downcast_ints: bool = True,
    float64_to_float32: bool = True,
    exclude: "set[str] | tuple" = (),
    verbose: bool = True,
) -> str:
    """Optimise an obs.parquet's dtypes in place, without loading it whole.

    Uses a lazy aggregate pass to choose dtypes, then streams the rewrite via
    ``sink_parquet``. Peak memory is governed by the streaming engine and the
    Enum category sets, not by the file size.
    """
    path = str(path)
    lf = pl.scan_parquet(path)
    schema = dict(lf.collect_schema())

    exprs = _plan_from_scan(lf, schema, enum_max_cardinality, downcast_ints,
                            float64_to_float32, exclude, verbose)
    if not exprs:
        if verbose:
            print("optimize_obs: no columns to change")
        return path

    before_bytes = os.path.getsize(path)
    tmp = path + ".opt.tmp"
    try:
        lf.with_columns(exprs).sink_parquet(tmp, compression="zstd")
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)

    after_bytes = os.path.getsize(path)
    if verbose:
        print(f"optimize_obs: {len(exprs)} column(s) recast; on-disk "
              f"{before_bytes/1e6:.1f} -> {after_bytes/1e6:.1f} MB "
              f"({before_bytes/max(after_bytes,1):.2f}x)")
    return path
