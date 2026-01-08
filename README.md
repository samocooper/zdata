# zdata

Efficient sparse matrix storage and retrieval for large-scale transcriptomics datasets using **seekable Zstandard compression**.

---

## What is zdata?

`zdata` is a Python library designed for bioinformaticians working with large single-cell RNA-seq datasets. It allows you to:

- **Store large datasets efficiently** using block-based compression
- **Query specific cells or genes quickly** without loading entire matrices into memory
- **Convert existing datasets** from Zarr or H5AD (AnnData) into a fast, queryable format

Think of `zdata` as a **compressed, random-access backend** for single-cell expression data.

---

## Overview

`zdata` uses a custom on-disk format optimized for sparse single-cell data:

- **Block-compressed CSR layout**  
  Expression data are grouped into small row blocks (e.g. 16 cells per block) for fast access.

- **Seekable Zstandard compression**  
  Enables random access without decompressing entire files.

- **Chunked storage**  
  Large datasets are split into multiple chunk files (e.g. 8192 cells per file), enabling scalability to millions of cells.

This design provides excellent compression ratios while maintaining fast random access.

---

## Features

- Fast random access to individual cells
- Efficient disk compression
- Scales to millions of cells and genes
- Simple Python API
- Supports Zarr and H5AD (AnnData) inputs
- Automatic detection of mixed input formats
- Optional column-major (gene-based) queries

---

## Installation

### Recommended: Install from PyPI

`zdata` can be installed directly from PyPI and **does not require manual compilation of ZSTD**.

```bash
pip install zdata-py
```

📦 PyPI package: http://pypi.org/project/zdata-py/

---

### Installing from source (developers only)

```bash
git clone <repository-url>
cd zdata
pip install -e .
```

---

## Quick Start Guide

### Building a zdata Dataset

`zdata` builds a compressed dataset from a directory containing:

- Zarr directories (`*.zarr/`)
- H5AD / HDF5 files (`*.h5`, `*.hdf5`, `*.h5ad`)

Mixed formats are supported.

#### Validate input data

```bash
python -m zdata.build_zdata.check_directory /path/to/data
```

---

#### Build the dataset

```python
from zdata import build_zdata_from_zarr

zdata_dir = build_zdata_from_zarr(
    zarr_dir="/path/to/data",
    output_name="my_dataset.zdata",
    block_rows=16,
    max_rows=8192,
    obs_join_strategy="outer"
)
```

---

## Opening a Dataset

```python
from zdata import ZData

reader = ZData("my_dataset.zdata")
print(reader.shape)
```

---

## Querying Data

### Row-based (cells)

```python
adata = reader[0:100]
adata = reader[[0, 10, 20]]
```

Raw sparse matrix:

```python
csr = reader.read_rows_csr([0, 100, 200])
```

---

### Column-based (genes)

> Requires column-major (`X_CM`) support.

```python
matrix = reader["GAPDH"]
matrix = reader[["GAPDH", "PCNA"]]
```

---

## Memory Estimation

```python
estimate = reader.estimate_memory_requirements(row_indices=[0, 1, 2])
print(estimate["estimated_memory_gb"])
```

---

## Command-Line Tools

Convert MTX to zdata:

```bash
./ctools/mtx_to_zdata matrix.mtx output_name
```

Read rows:

```bash
./ctools/zdata_read --binary output_name.zdata/0.bin "100,200,300"
```

---

## Project Structure

```
zdata/
├── core/
├── build_zdata/
├── ctools/
├── files/
└── tests/
```

---

## Testing

```bash
pytest tests/
```

---

## Performance

Optimized for random-access cell queries, compression efficiency, and scalability to millions of cells.

---

## License

See the LICENSE file for details.
