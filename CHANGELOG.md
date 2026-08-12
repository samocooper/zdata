# Changelog

## 0.2.0

### Breaking

- **`ZData` no longer infers a row/column mapping column.** `obs_index_col` and
  `var_index_col` are now two-valued: `None` (the default) means obs/var rows
  align 1-to-1 with the matrix and a dimension mismatch raises, and a string
  names the mapping column explicitly. Code that relied on `_row_index` being
  picked up automatically must now pass `obs_index_col="_row_index"`.

  Nothing is selected implicitly, and naming a column that does not exist is an
  error rather than a silent fallback — a silent mismatch produces wrong data,
  where an error produces a stack trace.

- The compiled C tools (`ctools/mtx_to_zdata`, `ctools/zdata_read`) are no
  longer tracked in git. They are build artifacts; `setup.py` compiles them on
  install, and the test-suite compiles them on demand when `ZSTD_BASE` is set.

### Added

- **`float16`** support (on-disk version 12), giving full `float16`/`float32`/
  `float64` coverage alongside all eight integer widths. The C implementation
  uses native `_Float16` where available with a portable bit-level fallback;
  both produce identical on-disk bytes.
- **`zdata.dtypes`** — a single source of truth for the on-disk dtype table.
  `ctools/dtype_table.h` is generated from it and included by both C tools, so
  the table can no longer drift between the C and Python layers.
- **Post-build steps for multi-study atlases**, shared by both builders:
  - `assign_global_sample_id` — globally-unique, monotonic `sample_id` and a
    readable `sample_uid`, since `sample_name` is only unique within a study.
  - `optimize_obs_parquet` — compacts obs dtypes (`Enum`, smallest int width)
    via a streaming rewrite; the saving is in decoded size, not on disk.
  - `build_feature_presence_matrix` — per-study gene-presence mask derived from
    each study's `var.csv`, for masking structural zeros out of a
    reconstruction loss.
  - `run_post_build_steps` with `strict_post_build=` to raise on first failure
    rather than warn and continue.
- `obs_columns=` on `ZData`, to load a subset of obs columns.

### Fixed

- `MANIFEST.in` now ships `ctools/*.h`; without it the generated header was
  missing from an sdist and compilation failed.
- Python 3.11 compatibility: replaced PEP 695 generic syntax, which is 3.12+
  only, while the package declares `python_requires=">=3.11"`.

### Internal

- Test fixtures are generated (`tests/make_fixtures.py`) rather than committed,
  and are entirely synthetic. Repository history was rewritten to drop the
  previously-committed fixture data: ~37 MB → ~0.7 MB.
