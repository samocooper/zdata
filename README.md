# zdata

Efficient sparse matrix storage and retrieval for large-scale transcriptomics datasets using seekable zstd compression.

## What is zdata?

`zdata` is a Python library designed for bioinformaticians working with large single-cell RNA-seq datasets. It allows you to:

- **Store large datasets efficiently** - Compress your data to save disk space
- **Query specific cells or genes quickly** - Access subsets of your data without loading everything into memory
- **Work with multiple file formats** - Convert from Zarr or H5AD (AnnData) files to zdata format

Think of zdata as a compressed, queryable version of your single-cell data that you can access quickly without loading the entire dataset into memory.

## Overview

`zdata` uses a custom storage format optimized for single-cell RNA-seq data:

- **Block-compressed sparse row (CSR) format** - Data is organized into small blocks (e.g., 16 cells per block) for fast access
- **Zstd seekable compression** - Data is compressed to save space, but you can still access specific parts without decompressing everything
- **Chunked storage** - Large datasets are split into manageable chunks (e.g., 8192 cells per chunk) stored as separate files

This approach provides excellent compression ratios while maintaining fast random access, making it ideal for querying subsets of large datasets.

## Features

- **Fast random cell access** - Retrieve specific cells without loading the entire dataset
- **Efficient compression** - Significant space savings compared to uncompressed formats
- **Scalable** - Handles datasets with millions of cells and genes
- **Simple Python API** - Easy-to-use interface for data access
- **Multiple input formats** - Supports both Zarr and H5AD (AnnData) file formats
- **Auto-detection** - Automatically detects and processes mixed file types in a directory

## Installation

### Step 1: Install Prerequisites

Before installing zdata, you need:

1. **Python 3.8 or higher** - Check your version:
   ```bash
   python --version
   ```

2. **GCC compiler** - Usually already installed on Linux/Mac. On Windows, you may need to install it separately.

3. **ZSTD source code** - This is required for compression. You need the full source code, not just the library.

### Step 2: Set Up ZSTD

If you don't have ZSTD source code, download and build it:

```bash
# Download ZSTD
git clone https://github.com/facebook/zstd.git
cd zstd

# Build it
make

# Set an environment variable so zdata knows where to find it
export ZSTD_BASE=$(pwd)
```

**Important:** Remember the path where you built ZSTD. You'll need to set `ZSTD_BASE` to this path every time you want to use zdata, or add it to your shell configuration file (e.g., `~/.bashrc`).

### Step 3: Install zdata

**From PyPI (recommended):**
```bash
# Make sure ZSTD_BASE is set
export ZSTD_BASE=/path/to/zstd-source

# Install zdata
pip install zdata
```

**From source:**
```bash
# Clone the repository
git clone <repository-url>
cd zdata

# Set ZSTD_BASE
export ZSTD_BASE=/path/to/zstd-source

# Install
pip install -e .
```

The C tools will be automatically compiled during installation. If installation fails, check that:
- `ZSTD_BASE` is set correctly
- GCC compiler is available
- ZSTD was built successfully

## Quick Start Guide

This guide will walk you through building a zdata object from your data files and then querying it.

### Part 1: Building a zdata Object from Your Data

**What you need:** A directory containing your data files. zdata supports:
- **Zarr files**: Directories ending in `.zarr` (e.g., `data.zarr/`)
- **H5AD files**: Files with extensions `.h5`, `.hdf5`, or `.h5ad` (e.g., `data.h5ad`)

You can have a mix of both types in the same directory - zdata will automatically detect and process them.

#### Step 1: Check Your Data Directory

Before building, it's a good idea to verify that your directory structure is correct:

```bash
python -m zdata.build_zdata.check_directory /path/to/your/data/directory
```

This command will:
- Check that files are in the expected format
- Report any issues it finds
- Show you what files will be processed

**What this means:** This step validates that your data files are in a format zdata can understand. If there are errors, fix them before proceeding.

#### Step 2: Build the zdata Object

Open Python (or a Jupyter notebook) and run:

```python
from zdata import build_zdata_from_zarr

# Build zdata from your data directory
zdata_dir = build_zdata_from_zarr(
    zarr_dir='/path/to/your/data/directory',  # Replace with your actual path
    output_name='my_dataset.zdata',            # Name for the output directory
    block_rows=16,                             # Cells per block (default: 16, usually fine)
    max_rows=8192,                             # Cells per chunk file (default: 8192, usually fine)
    obs_join_strategy="outer"                  # How to combine metadata: "inner", "outer", or "columns"
)
```

**What each parameter means:**

- `zarr_dir`: The path to the directory containing your `.zarr` or `.h5ad` files
- `output_name`: The name of the output directory that will be created (will have `.zdata` appended)
- `block_rows`: Number of cells grouped together in each block (16 is usually optimal)
- `max_rows`: Maximum number of cells stored in each chunk file (8192 is usually fine)
- `obs_join_strategy`: How to combine cell metadata from different files:
  - `"outer"`: Keep all columns from all files (recommended)
  - `"inner"`: Keep only columns present in all files
  - `"columns"`: Keep all columns, fill missing values with NaN

**What happens during building:**

1. zdata automatically detects file types (`.zarr` directories or `.h5`/`.hdf5`/`.h5ad` files)
2. All files are aligned to a standard gene list (ensures genes match across files)
3. Files are converted to zdata format with efficient compression
4. Cell metadata from all files is combined
5. A complete `.zdata/` directory is created, ready for querying

**How long does it take?** This depends on the size of your data. For large datasets, this can take several minutes to hours. The function will print progress updates.

**Output:** The function returns the path to the created zdata directory. You'll use this path to open and query your data.

```python
# The function returns the path
print(f"Created zdata directory at: {zdata_dir}")
# Example output: /path/to/my_dataset.zdata
```

### Part 2: Opening and Querying Your zdata Object

Once you've built your zdata object, you can open it and start querying.

#### Step 1: Open Your zdata Object

```python
from zdata import ZData

# Open the zdata directory you created
reader = ZData("my_dataset.zdata")
# Or use the full path:
# reader = ZData("/full/path/to/my_dataset.zdata")

# Check basic information about your dataset
print(f"Number of cells: {reader.num_rows}")
print(f"Number of genes: {reader.num_columns}")
print(f"Dataset shape: {reader.shape}")  # (cells, genes)
```

**What this does:** Opens your zdata directory and loads metadata. The actual expression data stays on disk until you query it.

#### Step 2: Query Specific Cells (Row-Based Queries)

**What are row-based queries?** These let you retrieve expression data for specific cells across all genes. This is the most common type of query and is very fast.

**Method 1: Using indexing syntax (easiest, returns AnnData)**

```python
# Get cells 0 through 99 (first 100 cells)
adata = reader[0:100]

# Get specific cells by their indices
adata = reader[[0, 100, 200, 300]]

# Get a single cell
adata = reader[5]

# Get the last cell
adata = reader[-1]
```

**What you get:** An AnnData object (same format as scanpy/anndata uses) containing:
- `adata.X`: Expression matrix (cells × genes)
- `adata.obs`: Cell metadata
- `adata.var`: Gene metadata

**Method 2: Using read_rows_csr() (returns a matrix)**

```python
# Get cells as a sparse matrix
csr_matrix = reader.read_rows_csr([0, 100, 200])
print(f"Matrix shape: {csr_matrix.shape}")  # (3, n_genes)
```

**What you get:** A scipy sparse matrix (CSR format) with expression values. Use this if you just need the expression data without metadata.

**Method 3: Using read_rows() (returns raw tuples)**

```python
# Get cells as raw data
rows_data = reader.read_rows([100, 200, 300])
for row_id, cols, vals in rows_data:
    print(f"Cell {row_id}: {len(cols)} genes with non-zero expression")
```

**What you get:** A list of tuples, each containing:
- `row_id`: The cell index
- `cols`: Array of gene indices with non-zero expression
- `vals`: Array of expression values

**Note:** Results from `read_rows()` and `read_rows_csr()` are returned in sorted order (not the order you requested). The indexing syntax (`reader[...]`) preserves your query order.

#### Step 3: Query Specific Genes (Column-Based Queries)

**What are column-based queries?** These let you retrieve expression data for specific genes across all cells. This requires column-major files (X_CM directory) which may not be available for all datasets.

**Method 1: Using indexing syntax with gene names (easiest)**

```python
# Get a single gene by name
matrix = reader['GAPDH']

# Get multiple genes by name
matrix = reader[['GAPDH', 'PCNA', 'COL1A1']]

# Get a range of genes (inclusive)
matrix = reader['GAPDH':'PCNA']
```

**What you get:** A sparse matrix (CSC format) with shape (n_cells, n_genes). Each column is a gene, each row is a cell.

**Method 2: Using read_cols_cm_csr() (returns a matrix)**

```python
# Get genes as a sparse matrix
csr_matrix = reader.read_cols_cm_csr(['GAPDH', 'PCNA'])
# Note: This returns (n_genes, n_cells), so transpose if needed
csc_matrix = csr_matrix.T.tocsc()  # Now (n_cells, n_genes)
```

**Method 3: Using read_cols_cm() (returns raw tuples)**

```python
# Get genes as raw data
cols_data = reader.read_cols_cm('GAPDH')
for col_id, rows, vals in cols_data:
    print(f"Gene {col_id}: {len(rows)} cells with non-zero expression")
```

**Important:** Column-based queries require the X_CM directory to exist. If you get an error saying "X_CM directory not found", your dataset was built without column-major support. You'll need to rebuild it with column-major support enabled (check build options).

### Common Query Patterns

**Pattern 1: Get a random sample of cells**

```python
# Get 100 random cells
random_cells = reader.get_random_rows(100, seed=42)
adata = reader[random_cells]
```

**Pattern 2: Get cells matching a condition (using boolean mask)**

```python
import numpy as np

# Create a boolean mask (True for cells you want, False for others)
# Example: Get first 1000 cells
mask = np.array([True] * 1000 + [False] * (reader.num_rows - 1000))
adata = reader[mask]
```

**Pattern 3: Get specific genes for specific cells**

```python
# Step 1: Get the cells you want
adata = reader[0:100]  # First 100 cells

# Step 2: Filter to specific genes in memory
gene_names = ['GAPDH', 'PCNA', 'COL1A1']
gene_indices = [reader._var_df.index[reader._var_df['gene'] == g].tolist()[0] 
                for g in gene_names]
filtered_adata = adata[:, gene_indices]
```

**Note:** You cannot query both cells and genes simultaneously in one step. Query cells first, then filter genes in memory (or vice versa).

### Understanding Query Results

**Result Ordering:**
- `reader[rows]` (indexing syntax): Results are in the **same order** as your query
- `read_rows()` and `read_rows_csr()`: Results are in **sorted order** (not your query order)
- All column query methods: Results are in **sorted order**

**Memory Considerations:**
Large queries can use a lot of memory. Check memory requirements before querying:

```python
# Estimate memory for a query
estimate = reader.estimate_memory_requirements(row_indices=[0, 100, 200])
print(f"Estimated memory: {estimate['estimated_memory_gb']:.2f} GB")
```

### Key Differences: Row vs Column Queries

| Feature | Row Queries (Cells) | Column Queries (Genes) |
|---------|-------------------|----------------------|
| **What you get** | Expression for specific cells | Expression for specific genes |
| **Speed** | Very fast (optimized) | May be slower |
| **Availability** | Always available | Requires X_CM directory |
| **Query by** | Cell indices | Gene names or indices |
| **Best for** | Getting cell subsets | Getting gene expression across all cells |

## Advanced Usage

### Converting MTX Files to zdata Format

If you have MTX (Matrix Market) format files:

```bash
./ctools/mtx_to_zdata matrix.mtx output_name
```

This creates a directory `output_name.zdata/` containing compressed `.bin` files.

### Command-Line Tools

**Convert MTX to zdata:**
```bash
./ctools/mtx_to_zdata input.mtx output_name
```

**Read rows from zdata (binary output):**
```bash
./ctools/zdata_read --binary output_name.zdata/0.bin "100,200,300"
```

## Troubleshooting

**Problem: "zdata_read executable not found"**
- Solution: Make sure zdata was installed correctly and C tools were compiled. Try reinstalling.

**Problem: "X_CM directory not found" when querying genes**
- Solution: Your dataset was built without column-major support. Rebuild with column-major support enabled, or use row queries and filter genes in memory.

**Problem: "Memory error" when querying**
- Solution: Your query is too large. Try querying smaller batches, or check memory requirements first using `estimate_memory_requirements()`.

**Problem: "Gene name not found"**
- Solution: Check that the gene name is spelled correctly and exists in your dataset. You can check available genes with `reader._var_df['gene']`.

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
```

### Default Gene List

The package includes a default gene list (`files/2ks10c_genes.txt`) that is used as the standard gene set for aligning zarr and h5ad files. This ensures all files have the same genes in the same order.

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

## Performance

The zdata format is optimized for:
- **Random cell queries** - Fast retrieval of arbitrary cell subsets
- **Compression** - Significant space savings compared to uncompressed formats
- **Scalability** - Efficient handling of datasets with millions of cells/genes

Benchmark results can be obtained by running `test_fast_queries.py`.

## License

See LICENSE file for details.
