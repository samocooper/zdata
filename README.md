# zdata

Efficient sparse matrix storage and retrieval for large-scale transcriptomics datasets using **seekable Zstandard compression**.

---

## What is zdata?

`zdata` is a Python library for working with **large single-cell RNA-seq datasets** efficiently. It allows you to:

- **Store large datasets compactly** using block-based compression  
- **Query specific cells or genes quickly** without loading the full matrix into memory  
- **Convert existing datasets** from Zarr or H5AD (AnnData) into a fast, queryable format  

Think of `zdata` as a **compressed, random-access backend** for single-cell expression matrices.

---

## Overview

`zdata` uses a custom on-disk format optimized for sparse single-cell data:

- **Block-compressed CSR layout**  
  Expression data are grouped into small row blocks (e.g. 16 cells) for efficient access.

- **Seekable Zstandard compression**  
  Enables fast random reads without decompressing entire files.

- **Chunked storage**  
  Large datasets are split into multiple chunk files (e.g. 8192 cells per file), enabling scalability to millions of cells.

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

```bash
pip install zdata-py
```

📦 PyPI: http://pypi.org/project/zdata-py/

---

## Quick Start

```python
from zdata import ZData
reader = ZData("my_dataset.zdata")
print(reader.shape)
```

---

## License

See LICENSE file for details.
