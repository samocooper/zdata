# zdata

Efficient sparse matrix storage and retrieval for large-scale datasets using **seekable Zstandard compression**.

---

## What is zdata?

`zdata` is a Python library for storing and querying large sparse numerical matrices. While designed for single-cell RNA-seq data, it works with any sparse matrix that fits the `(observations x features)` pattern.

- **Store large datasets efficiently** using block-based seekable Zstandard compression
- **Query specific rows or columns quickly** without loading entire matrices into memory
- **Convert existing datasets** from Zarr, H5AD (AnnData), or MTX+CSV into a fast, queryable format
- **Store any numerical type**: uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64

---

## Overview

`zdata` uses a custom on-disk format optimized for sparse data:

- **Block-compressed CSR layout** -- data are grouped into small row blocks (default 16 rows per block) for fast access
- **Seekable Zstandard compression** -- random access without decompressing entire files
- **Chunked storage** -- large datasets split into multiple chunk files (default 8192 rows per file), scaling to millions of rows

---

## Installation

### From PyPI

```bash
pip install zdata-py
```

### From source

Requires the [Zstandard](https://github.com/facebook/zstd) library for compiling the C tools:

```bash
git clone https://github.com/facebook/zstd.git
cd zstd && make lib && cd ..

git clone <repository-url> zdata
cd zdata
export ZSTD_BASE=/path/to/zstd
pip install -e .
```

The C tools (`mtx_to_zdata`, `zdata_read`) are compiled automatically during installation when `ZSTD_BASE` is set. Pre-compiled binaries are included in PyPI wheels.

---

## Quick Start

### Building a dataset

`zdata` can build from three input formats:

#### From Zarr or H5AD directories

```python
from zdata import build_zdata_from_zarr

build_zdata_from_zarr(
    input_dir="/path/to/zarr_or_h5ad_files",
    output_name="my_dataset.zdata",
)
```

The input directory can contain `.zarr` directories, `.h5`/`.hdf5`/`.h5ad` files, or a mix of both.

#### From MTX+CSV directories

Each subdirectory should contain `matrix.mtx`, `obs.csv`, and `var.csv`:

```python
from zdata import build_zdata_from_mtx_csv

build_zdata_from_mtx_csv(
    input_dir="/path/to/mtx_csv_dirs",
    output_name="my_dataset.zdata",
)
```

#### Specifying a data type

By default, values are stored as `uint16` (suitable for raw scRNA-seq counts). For other data types:

```python
build_zdata_from_zarr(
    input_dir="/path/to/data",
    output_name="my_dataset.zdata",
    dtype="float32",   # or: uint8, int32, float64, etc.
)
```

Supported dtypes: `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.

---

### Opening a dataset

```python
from zdata import ZData

ds = ZData("my_dataset.zdata")
print(ds.shape)       # (n_cells, n_genes)
print(ds.obs.shape)   # observation metadata
print(ds.var.shape)   # variable metadata
```

---

### Querying data

#### Row-based (cells)

```python
# AnnData-like indexing
adata = ds[0:100]
adata = ds[[0, 10, 20]]

# Raw sparse matrix
csr = ds.read_rows_csr([0, 100, 200])

# Random sample
rows = ds.get_random_rows(50)
```

#### Column-based (genes)

Requires column-major (`X_CM`) data (built automatically):

```python
matrix = ds["GAPDH"]
matrix = ds[["GAPDH", "PCNA", "TP53"]]
```

---

## Gene list alignment

During dataset construction, genes are aligned to a standard reference list (default: `files/2ks10c_genes.txt`). To use your own gene list:

```python
build_zdata_from_zarr(
    input_dir="/path/to/data",
    output_name="my_dataset.zdata",
    gene_list_path="/path/to/my_genes.txt",  # one gene name per line
)
```

---

## Configuration

Settings can be changed at runtime, via environment variables, or with a context manager:

```python
from zdata import settings

# Runtime
settings.block_rows = 32
settings.warn_on_large_queries = False

# Context manager
with settings.override(override_memory_check=True):
    ds.read_rows_csr(range(100000))
```

| Setting | Default | Env var | Description |
|---------|---------|---------|-------------|
| `max_rows_per_chunk` | 8192 | `ZDATA_MAX_ROWS_PER_CHUNK` | Max rows per chunk file |
| `block_rows` | 16 | `ZDATA_BLOCK_ROWS` | Rows per compressed block |
| `warn_on_large_queries` | True | `ZDATA_WARN_ON_LARGE_QUERIES` | Warn when querying >50k rows |
| `large_query_threshold` | 50000 | `ZDATA_LARGE_QUERY_THRESHOLD` | Threshold for large query warning |
| `override_memory_check` | False | `ZDATA_OVERRIDE_MEMORY_CHECK` | Bypass memory safety checks |
| `max_workers` | None (auto) | `ZDATA_MAX_WORKERS` | Thread pool size for parallel reads |

---

## Command-Line Tools

After installation, these commands are available:

```bash
# Build from zarr/h5ad
zdata-build /path/to/data my_dataset.zdata

# Build from mtx+csv directories
zdata-build-mtx /path/to/mtx_dirs my_dataset.zdata

# Validate input directory
zdata-check /path/to/data

# Align genes and produce intermediate MTX files
zdata-align /path/to/data output_dir --gene-list genes.txt
```

All commands support `--help` for full options.

---

## Memory estimation

```python
estimate = ds.estimate_memory_requirements(row_indices=[0, 1, 2])
print(estimate["estimated_memory_gb"])
```

---

## Output format

A `.zdata` directory contains:

```
my_dataset.zdata/
  metadata.json       # shape, chunking info, dtype
  obs.parquet         # observation metadata (cells)
  var.parquet         # variable metadata (genes)
  X_RM/               # row-major compressed chunks
    0.bin, 1.bin, ...
  X_CM/               # column-major compressed chunks (optional)
    0.bin, 1.bin, ...
```

---

## Troubleshooting

**C tools not found**: Set `ZSTD_BASE` to your zstd source directory and reinstall:
```bash
export ZSTD_BASE=/path/to/zstd
pip install -e . --force-reinstall
```

**Gene list not found**: Pass `gene_list_path=` explicitly, or ensure `files/2ks10c_genes.txt` is present in the package.

**Memory errors on large queries**: Increase available memory, reduce query size, or set `settings.override_memory_check = True`.

---

## Testing

```bash
pip install pytest pytest-xdist
export ZSTD_BASE=/path/to/zstd
pytest tests/ -v
```

---

## License

MIT. See the [LICENSE](LICENSE) file for details.
