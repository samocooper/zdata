# zdata Test Suite

This directory contains the test suite for zdata.

## Test Organization

Tests are organized into the following structure:

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_core/               # Core functionality tests
│   ├── test_zdata.py        # ZData class tests
│   ├── test_obs_wrapper.py  # ObsWrapper tests
│   └── test_ctools.py       # C tools compilation and functionality tests
├── test_index/              # Indexing tests
│   └── test_normalize_indices.py  # Index normalization tests
├── test_full_pipeline_at_scale.py  # Large-scale pipeline tests (external data)
├── zarr_test_dir/           # Test zarr data files
└── h5ad_test_dir/           # Test h5ad data files (.h5/.hdf5)
```

## H5AD File Support

The test suite now supports both zarr and h5ad file formats:

- **Zarr files**: Tested via `zarr_test_dir/` and `zdata_instance` fixture
- **H5AD files**: Tested via `h5ad_test_dir/` and `zdata_instance_h5ad` fixture

### Running H5AD Tests

```bash
# Run h5ad-specific tests
pytest tests/test_core/test_h5ad.py

# Run full pipeline test with h5ad files
python tests/test_full_pipeline_at_scale.py --h5ad

# Or specify h5ad directory explicitly
python tests/test_full_pipeline_at_scale.py /path/to/h5ad_test_dir
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test module
```bash
pytest tests/test_core/test_zdata.py
```

### Run with coverage
```bash
pytest tests/ --cov=zdata --cov-report=html
```

### Run in parallel (recommended for speed)
```bash
# Install pytest-xdist first: pip install pytest-xdist
pytest tests/ -n auto

# Or specify number of workers
pytest tests/ -n 4
```

### Skip slow tests
```bash
# Skip tests marked as slow
pytest tests/ -m "not slow"
```

## Test Fixtures

The `conftest.py` file provides reusable fixtures:

- `zdata_instance`: Creates a ZData instance from test zarr files (session-scoped)
- `zdata_instance_h5ad`: Creates a ZData instance from test h5ad files (session-scoped)
- `h5ad_test_dir`: Provides path to h5ad test directory
- `tmp_zdata_dir`: Creates a temporary zdata directory structure
- `sample_metadata`: Creates sample metadata for testing
- `sample_obs_data`: Creates sample observation data
- `sample_var_data`: Creates sample variable (gene) data
- `sample_row_indices`: Sample row indices for testing
- `sample_gene_names`: Sample gene names for testing
- `sample_boolean_mask`: Sample boolean mask for testing

The `test_ctools.py` file provides additional fixtures:

- `zstd_base`: ZSTD base directory path
- `ctools_dir`: Path to ctools directory
- `mtx_to_zdata_src`: Path to mtx_to_zdata.c source
- `zdata_read_src`: Path to zdata_read.c source
- `compiled_tools`: Compiled C tool binaries (session-scoped)
- `sample_mtx_file`: Sample MTX file for testing

## Writing New Tests

When writing new tests:

1. Use pytest fixtures from `conftest.py` when possible
2. Follow the naming convention: `test_<functionality>`
3. Group related tests into classes (e.g., `TestReadRows`)
4. Use descriptive test names that explain what is being tested
5. Add docstrings to test functions explaining the test purpose

Example:
```python
def test_read_rows_with_slice(zdata_instance: ZData):
    """Test reading rows with a slice."""
    n_rows = min(10, zdata_instance.nrows)
    rows = zdata_instance.read_rows(slice(0, n_rows))
    assert len(rows) == n_rows
```

## Test Data

- Small test data is created on-the-fly using fixtures
- Larger tests use the `zarr_test_dir/` or `h5ad_test_dir/` directories
- `test_full_pipeline_at_scale.py` requires external data and is not run by default
- C tools tests create sample MTX files dynamically

## Why Are Some Tests Skipped?

Some tests may be skipped if required dependencies are not available:

1. **`test_zdata.py` tests**: These require:
   - Zarr test files in `tests/zarr_test_dir/` or h5ad files in `tests/h5ad_test_dir/`
   - C tools compiled (`ctools/mtx_to_zdata`, `ctools/zdata_read`)
   - ZSTD library available
   - Successful build of zdata from zarr files

   If any of these are missing, the `zdata_instance` fixture will skip, causing all dependent tests to skip.

2. **To see why tests are skipped**, run with verbose output:
   ```bash
   pytest tests/ -v -rs
   ```
   The `-rs` flag shows skip reasons.

3. **To ensure all tests run**, make sure:
   - Zarr test files exist in `tests/zarr_test_dir/` or h5ad files in `tests/h5ad_test_dir/`
   - C tools are compiled (run the test suite once to compile them)
   - ZSTD library is available

## C Tools Tests

The `test_ctools.py` module tests the C tools (`mtx_to_zdata.c` and `zdata_read.c`):

### Requirements
- ZSTD library installed (set `ZSTD_BASE` environment variable or use default `/home/ubuntu/zstd`)
- GCC compiler available
- ZSTD source files in expected locations

### Test Coverage
- **Compilation**: Verifies both C tools compile without errors
- **MTX Compression**: Tests `mtx_to_zdata` can compress MTX files
- **File Reading**: Tests `zdata_read` can read compressed files
- **Data Integrity**: Verifies compressed files can be read by Python ZData
- **Parameter Handling**: Tests custom block_rows, max_rows, and subdirectory options

### Running C Tools Tests
```bash
# Run all C tools tests
pytest tests/test_core/test_ctools.py

# Run specific test class
pytest tests/test_core/test_ctools.py::TestCToolsCompilation

# Run with ZSTD_BASE environment variable
ZSTD_BASE=/path/to/zstd pytest tests/test_core/test_ctools.py
```

## Continuous Integration

Tests should be runnable in CI environments. Ensure:
- All required dependencies are in `pyproject.toml`
- Tests don't require external data unless explicitly marked
- Tests are deterministic (use seeds for random operations)
