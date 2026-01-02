# zdata

Efficient sparse matrix storage and retrieval for large-scale transcriptomics datasets using seekable zstd compression.

## Overview

`zdata` is a high-performance library for storing and querying large sparse matrices (e.g., single-cell RNA-seq data) with efficient random row access. It uses a custom format based on:

- **Block-compressed sparse row (CSR) format** - Organized into e.g. 16-row blocks for efficient access
- **Zstd seekable compression** - Enables random access to compressed data without full decompression
- **Chunked storage** - Large matrices are split into e.g. 8192-row chunks stored as separate `.bin` files

This approach provides excellent compression ratios while maintaining fast random row retrieval performance, making it ideal for querying subsets of large datasets.

## Features

- **Fast random row access** - Retrieve arbitrary rows without loading the entire dataset
- **Efficient compression** - Zstd compression with seekable format for space savings
- **Scalable** - Handles datasets with millions of rows and columns
- **Python API** - Simple, intuitive interface for data access
- **C-based backend** - High-performance C implementation for core operations
- **Multiple input formats** - Supports both Zarr and H5AD (AnnData) file formats
- **Auto-detection** - Automatically detects and processes mixed file types in a directory

## Quick Start

### Building zdata from Zarr or H5AD Files

The easiest way to create a zdata object is from a directory of zarr files or h5ad files:

First check that the directory of zarr files / h5ad files is correct:

```bash
python -m zdata.build_zdata.check_directory <path to directory>
```
Next build the zdata object

```python
from zdata import build_zdata_from_zarr

# Build zdata from a directory containing .zarr files or .h5/.hdf5/.h5ad files
# The function auto-detects file types based on extensions
zdata_dir = build_zdata_from_zarr(
    zarr_dir='/path/to/data/directory',   # Directory containing .zarr or .h5/.hdf5/.h5ad files
    output_name='my_dataset.zdata',        # Output zdata directory name
    block_rows=16,                         # Rows per block (default: 16)
    max_rows=8192,                         # Max rows per chunk (default: 8192)
    obs_join_strategy="outer"              # How to join obs metadata: "inner", "outer", or "columns"
)

# The function returns the path to the created zdata directory
print(f"Created zdata directory at: {zdata_dir}")
```

This single function:
1. Auto-detects file types (`.zarr` directories or `.h5`/`.hdf5`/`.h5ad` files)
2. Aligns all files to a standard gene list
3. Converts them to zdata format with efficient compression
4. Concatenates observation metadata from all files
5. Creates a complete `.zdata/` directory ready for querying

**Supported Input Formats:**
- **Zarr**: Directories ending in `.zarr` (e.g., `data.zarr/`)
- **H5AD**: Files with extensions `.h5`, `.hdf5`, or `.h5ad` (e.g., `data.h5ad`)

### Reading from zdata

```python
from zdata import ZData

# Open the zdata directory
reader = ZData("my_dataset.zdata")

# Query specific rows
rows_data = reader.read_rows([100, 200, 300])
for row_id, cols, vals in rows_data:
    print(f"Row {row_id}: {len(cols)} non-zero values")
```

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
├── build_zdata/        # Build and preprocessing utilities
│   ├── build_x.py     # Build zdata from MTX files
│   ├── build_zdata.py # Main build function for zarr/h5ad directories
│   ├── align_mtx.py   # Align zarr/h5ad files to standard gene list
│   ├── check_directory.py  # Check zarr directory structure
│   └── concat_obs.py  # Concatenate obs/metadata from zarr/h5ad files
├── ctools/            # C command-line tools
│   ├── mtx_to_zdata.c    # MTX to zdata converter
│   ├── zdata_read.c      # Row reader
│   ├── mtx_to_zdata      # Compiled binary (generated during install)
│   └── zdata_read        # Compiled binary (generated during install)
├── files/             # Package data files
│   └── 2ks10c_genes.txt  # Default gene list for alignment (required)
└── tests/             # Test suite
    ├── test_random_rows.py      # Random row extraction test
    ├── test_fast_queries.py     # Performance benchmark
    └── test_full_pipeline.py    # Full pipeline test
```

### Default Gene List

The package includes a default gene list (`files/2ks10c_genes.txt`) that is used as the standard gene set for aligning zarr and h5ad files. This file is:
- **Required**: Must be included in the package distribution
- **Default**: Used automatically when building zdata from zarr or h5ad files
- **Overridable**: Can be replaced with a custom gene list path if needed

## Testing

Run all tests with pytest:

```bash
pytest tests/
```

Run the full pipeline test (compiles, builds, and tests):

```bash
# With zarr files (default)
python tests/test_full_pipeline_at_scale.py [zarr_directory] [output_name]

# With h5ad files
python tests/test_full_pipeline_at_scale.py --h5ad [h5ad_directory] [output_name]
```

Run specific test modules:

```bash
# Test core functionality
pytest tests/test_core/

# Test h5ad support
pytest tests/test_core/test_h5ad.py

# Test with coverage
pytest tests/ --cov=zdata --cov-report=html
```

## Performance

The zdata format is optimized for:
- **Random row queries** - Fast retrieval of arbitrary row subsets
- **Compression** - Significant space savings compared to uncompressed formats
- **Scalability** - Efficient handling of datasets with millions of cells/genes

Benchmark results can be obtained by running `test_fast_queries.py`.

## License

See LICENSE file for details.
