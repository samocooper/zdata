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

- Python 3.7+
- GCC compiler
- Zstd library (with seekable format support)
- NumPy and SciPy

### Building

The C tools need to be compiled before use. The test script can handle this automatically:

```bash
python tests/test_full_pipeline.py
```

Or manually:

```bash
cd ctools
gcc -O2 -Wall -I/path/to/zstd/lib -I/path/to/zstd/lib/common \
    -I/path/to/zstd/contrib/seekable_format \
    -o mtx_to_zdata mtx_to_zdata.c \
    /path/to/zstd/contrib/seekable_format/zstdseek_compress.c \
    /path/to/zstd/lib/common/xxhash.c /path/to/zstd/lib/libzstd.a

gcc -O2 -Wall -I/path/to/zstd/lib -I/path/to/zstd/lib/common \
    -I/path/to/zstd/contrib/seekable_format \
    -o zdata_read zdata_read.c \
    /path/to/zstd/contrib/seekable_format/zstdseek_decompress.c \
    /path/to/zstd/lib/common/xxhash.c /path/to/zstd/lib/libzstd.a
```

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
├── ctools/            # C command-line tools
│   ├── mtx_to_zdata.c    # MTX to zdata converter
│   ├── zdata_read.c      # Row reader
│   ├── mtx_to_zdata      # Compiled binary
│   └── zdata_read        # Compiled binary
└── tests/             # Test suite
    ├── test_random_rows.py      # Random row extraction test
    ├── test_fast_queries.py     # Performance benchmark
    └── test_full_pipeline.py    # Full pipeline test
```

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

## License

See LICENSE file for details.
