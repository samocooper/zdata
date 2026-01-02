# zdata

Efficient sparse matrix storage and retrieval for large-scale transcriptomics datasets using seekable zstd compression.

## Overview

`zdata` is a high-performance library for storing and querying large sparse matrices (e.g., single-cell RNA-seq data) with efficient random row access. It uses a custom format based on:

- **Block-compressed sparse row (CSR) format** - Organized in 16-row blocks for efficient access
- **Zstd seekable compression** - Enables random access to compressed data without full decompression
- **Chunked storage** - Large matrices are split into 4096-row chunks stored as separate `.bin` files

This approach provides excellent compression ratios while maintaining fast random row retrieval performance, making it ideal for querying subsets of large datasets.

## Features

- **Fast random row access** - Retrieve arbitrary rows without loading the entire dataset
- **Efficient compression** - Zstd compression with seekable format for space savings
- **Scalable** - Handles datasets with millions of rows and columns
- **Python API** - Simple, intuitive interface for data access
- **C-based backend** - High-performance C implementation for core operations

## Installation

### Prerequisites

**Required:**
- **Python 3.8+**
- **GCC compiler** (for compiling C tools)
- **ZSTD source code** (not just the library) - The ZSTD source directory must contain:
  - `lib/libzstd.a` (static library)
  - `lib/common/xxhash.c` (source file)
  - `contrib/seekable_format/zstdseek_compress.c` (source file)
  - `contrib/seekable_format/zstdseek_decompress.c` (source file)

**Note:** C tools compilation is **required** for the package to work. The installation will fail if ZSTD is not found or if compilation fails.

### Setting up ZSTD

If you don't have ZSTD source code, clone and build it:

```bash
git clone https://github.com/facebook/zstd.git
cd zstd
make
export ZSTD_BASE=$(pwd)
```

### From PyPI

```bash
# Set ZSTD_BASE before installation
export ZSTD_BASE=/path/to/zstd-source
pip install zdata
```

The C tools will be automatically compiled during installation.

### From Source

1. Clone the repository:
```bash
git clone <repository-url>
cd zdata
```

2. Set ZSTD_BASE:
```bash
export ZSTD_BASE=/path/to/zstd-source
```

3. Install in development mode:
```bash
pip install -e .
```

Or install normally:
```bash
pip install .
```

The C tools will be automatically compiled during installation.

## Usage

### Converting MTX Files to zdata Format

```bash
./ctools/mtx_to_zdata matrix.mtx output_name
```

This creates a directory `output_name.zdata/` containing numbered `.bin` files (0.bin, 1.bin, etc.), each containing up to 4096 rows.

### Python API

```python
from zdata.core import ZData

# Initialize reader
reader = ZData("andrews")  # Looks for andrews.zdata/

# Get dataset info
print(f"Rows: {reader.num_rows}, Columns: {reader.num_columns}")

# Read specific rows
rows_data = reader.read_rows([100, 200, 300])
for row_id, cols, vals in rows_data:
    print(f"Row {row_id}: {len(cols)} non-zeros")

# Read rows as CSR matrix
csr = reader.read_rows_csr([100, 200, 300])

# Get random rows
random_rows = reader.get_random_rows(10, seed=42)
data = reader.read_rows(random_rows)
```

### Command-Line Tools

**Convert MTX to zdata:**
```bash
./ctools/mtx_to_zdata input.mtx output_name
```

**Read rows from zdata (binary output):**
```bash
./ctools/zdata_read --binary output_name.zdata/0.bin "100,200,300"
```

## Project Structure

```
zdata/
├── core/              # Python core module
│   ├── zdata.py      # ZData class implementation
│   └── __init__.py
├── build/             # Build and preprocessing utilities
│   ├── build_x.py     # Build zdata from MTX files
│   ├── align_mtx.py   # Align zarr files to standard gene list
│   ├── check_directory.py  # Check zarr directory structure
│   └── concat_obs.py  # Concatenate obs/metadata from zarr files
├── ctools/            # C command-line tools
│   ├── mtx_to_zdata.c    # MTX to zdata converter
│   ├── zdata_read.c      # Row reader
│   ├── mtx_to_zdata      # Compiled binary
│   └── zdata_read        # Compiled binary
├── files/             # Package data files
│   └── 2ks10c_genes.txt  # Default gene list for alignment (required)
└── tests/             # Test suite
    ├── test_random_rows.py      # Random row extraction test
    ├── test_fast_queries.py     # Performance benchmark
    └── test_full_pipeline.py    # Full pipeline test
```

### Default Gene List

The package includes a default gene list (`files/2ks10c_genes.txt`) that is used as the standard gene set for aligning zarr files. This file is:
- **Required**: Must be included in the package distribution
- **Default**: Used automatically when building zdata from zarr files
- **Overridable**: Can be replaced with `--gene-list` parameter if needed

## Testing

Run the full pipeline test (compiles, builds, and tests):

```bash
python tests/test_full_pipeline.py [mtx_file] [output_name]
```

Run individual tests:

```bash
# Test random row extraction
python tests/test_random_rows.py [zdata_directory] [n_rows] [seed]

# Performance benchmark
python tests/test_fast_queries.py [zdata_directory]
```

## Performance

The zdata format is optimized for:
- **Random row queries** - Fast retrieval of arbitrary row subsets
- **Compression** - Significant space savings compared to uncompressed formats
- **Scalability** - Efficient handling of datasets with millions of cells/genes

Benchmark results can be obtained by running `test_fast_queries.py`.

## Development

### Building for PyPI

```bash
# Install build tools
pip install build twine

# Build distribution packages
python -m build

# Test locally
pip install dist/zdata-*.whl

# Upload to PyPI
twine upload dist/*
```

### Cross-Platform Wheel Building

The project uses `cibuildwheel` to build platform-specific wheels. See `.github/workflows/build_wheels.yml` for the CI configuration.

**Building wheels locally:**
```bash
pip install cibuildwheel
export ZSTD_BASE=/path/to/zstd-source
cibuildwheel --output-dir wheelhouse
```

**Note:** C tools compilation is required. The setup script will compile C tools during installation if ZSTD is available, or use pre-compiled binaries from the wheel if available.

## License

See LICENSE file for details.
