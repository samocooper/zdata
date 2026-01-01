# Indexing Improvements for zdata

## Overview

This document describes the improved indexing system for zdata, which provides robust support for row-major and column-major queries on disk-based data.

## Key Design Principles

Unlike in-memory data structures (like anndata), zdata uses disk-based storage and can be massive. Therefore:

1. **Row-major queries only**: Query rows using integers, slices, boolean arrays, etc.
2. **Column-major queries only**: Query columns (genes) using names, integers, slices, boolean arrays, etc.
3. **No arbitrary 2D indexing**: Cannot efficiently support queries like `zdata[rows, cols]` simultaneously.

## New Indexing Module: `core/index.py`

### Functions

#### `normalize_row_indices(index, nrows) -> list[int]`

Normalizes row indices to a sorted, deduplicated list of integers.

**Supported input types:**
- `int` - Single row index (supports negative indices)
- `slice` - Row slice (e.g., `slice(0, 100)`)
- `list[int]` - List of row indices
- `numpy.ndarray[int]` - Array of row indices
- `numpy.ndarray[bool]` - Boolean mask (length must match nrows)
- `pandas.Index` / `pandas.Series` - Pandas index objects

**Features:**
- Handles negative indices (e.g., `-1` for last row)
- Validates bounds and provides clear error messages
- Automatically sorts and deduplicates indices
- Supports boolean masking

**Example:**
```python
# Single row
indices = normalize_row_indices(5, nrows=1000)  # [5]

# Negative index
indices = normalize_row_indices(-1, nrows=1000)  # [999]

# Slice
indices = normalize_row_indices(slice(0, 100), nrows=1000)  # [0, 1, ..., 99]

# Boolean mask
mask = np.array([True] * 100 + [False] * 900)
indices = normalize_row_indices(mask, nrows=1000)  # [0, 1, ..., 99]

# List of indices
indices = normalize_row_indices([0, 5, 10, -1], nrows=1000)  # [0, 5, 10, 999]
```

#### `normalize_column_indices(index, ncols, gene_names=None) -> list[int]`

Normalizes column (gene) indices to a sorted, deduplicated list of integers.

**Supported input types:**
- `int` - Single column index (supports negative indices)
- `str` - Single gene name (requires `gene_names`)
- `slice` - Column slice (supports integer or string bounds)
- `list[int]` - List of column indices
- `list[str]` - List of gene names (requires `gene_names`)
- `numpy.ndarray[int]` - Array of column indices
- `numpy.ndarray[bool]` - Boolean mask (length must match ncols)
- `pandas.Index` / `pandas.Series` - Pandas index objects

**Features:**
- Handles negative indices
- Supports gene name lookups when `gene_names` is provided
- Validates bounds and provides clear error messages
- Automatically sorts and deduplicates indices
- Supports boolean masking

**Example:**
```python
# Single column index
indices = normalize_column_indices(5, ncols=20000)  # [5]

# Single gene name
gene_names = pd.Index(['GAPDH', 'PCNA', 'COL1A1', ...])
indices = normalize_column_indices('GAPDH', ncols=20000, gene_names=gene_names)  # [0]

# Slice with gene names
indices = normalize_column_indices(
    slice('GAPDH', 'PCNA'), 
    ncols=20000, 
    gene_names=gene_names
)  # [0, 1]

# List of gene names
indices = normalize_column_indices(
    ['GAPDH', 'PCNA', 'COL1A1'], 
    ncols=20000, 
    gene_names=gene_names
)  # [0, 1, 2]

# Boolean mask
mask = np.array([True] * 100 + [False] * 19900)
indices = normalize_column_indices(mask, ncols=20000)  # [0, 1, ..., 99]
```

## Updated ZData Methods

### Row-Major Methods

All row-major methods now support the enhanced indexing:

#### `read_rows(global_rows)`

**Enhanced signature:**
```python
def read_rows(
    self, 
    global_rows: int | np.integer | Sequence[int] | NDArray[np.integer] | NDArray[np.bool_] | slice
) -> list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]
```

**Examples:**
```python
# Single row
data = zdata.read_rows(5)

# Slice
data = zdata.read_rows(slice(0, 100))

# List of indices
data = zdata.read_rows([0, 5, 10, 100])

# Boolean mask
mask = zdata.obs['cell_type'] == 'T-cell'
data = zdata.read_rows(mask.values)

# Negative index
data = zdata.read_rows(-1)  # Last row
```

#### `read_rows_csr(global_rows)`

Same enhanced indexing support, returns CSR matrix.

#### `read_rows_rm(global_rows)` and `read_rows_rm_csr(global_rows)`

Aliases with same enhanced indexing support.

### Column-Major Methods

All column-major methods now support the enhanced indexing:

#### `read_cols_cm(global_cols)`

**Enhanced signature:**
```python
def read_cols_cm(
    self, 
    global_cols: int | np.integer | Sequence[int] | Sequence[str] | NDArray[np.integer] | NDArray[np.bool_] | slice | str
) -> list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]
```

**Examples:**
```python
# Single gene name
data = zdata.read_cols_cm('GAPDH')

# List of gene names
data = zdata.read_cols_cm(['GAPDH', 'PCNA', 'COL1A1'])

# Slice with gene names
data = zdata.read_cols_cm(slice('GAPDH', 'PCNA'))

# Integer indices
data = zdata.read_cols_cm([0, 5, 10])

# Boolean mask
mask = zdata.var['highly_variable'] == True
data = zdata.read_cols_cm(mask.values)
```

#### `read_cols_cm_csr(global_cols)`

Same enhanced indexing support, returns CSR matrix.

## Enhanced `__getitem__` Method

The `__getitem__` method now intelligently determines whether the query is row-major or column-major:

### Row-Major Queries (Returns AnnData)

```python
# Single row
adata = zdata[5]

# Slice
adata = zdata[5:10]

# List of indices
adata = zdata[[0, 5, 10, 100]]

# Boolean mask
mask = zdata.obs['cell_type'] == 'T-cell'
adata = zdata[mask.values]

# Negative index
adata = zdata[-1]  # Last row
```

### Column-Major Queries (Returns CSC Matrix)

```python
# Single gene name
matrix = zdata['GAPDH']

# List of gene names
matrix = zdata[['GAPDH', 'PCNA', 'COL1A1']]

# Slice with gene names
matrix = zdata['GAPDH':'PCNA']

# Note: Integer indices default to row queries
# To query columns by index, use read_cols_cm() directly
```

## Error Handling

The new indexing system provides clear, actionable error messages:

### Index Out of Bounds
```python
# Clear error message with valid range
zdata.read_rows(10000)  # IndexError: Row index 10000 out of range [0, 5000)
```

### Boolean Mask Length Mismatch
```python
# Clear error about length mismatch
mask = np.array([True] * 100)  # Wrong length
zdata.read_rows(mask)  # ValueError: Boolean index length 100 does not match number of rows 5000
```

### Missing Gene Names
```python
# Clear error when gene names not available
zdata.read_cols_cm('GAPDH')  # ValueError: String-based column indexing requires gene_names to be provided
```

### Gene Not Found
```python
# Clear error with available information
zdata.read_cols_cm('UNKNOWN_GENE')  # IndexError: Gene name 'UNKNOWN_GENE' not found in dataset
```

## Benefits

1. **More Robust**: Handles edge cases and provides clear error messages
2. **More Flexible**: Supports multiple indexing patterns (integers, slices, boolean masks, names)
3. **Better Performance**: Automatically sorts and deduplicates indices
4. **Type Safe**: Full type hints for better IDE support
5. **Consistent**: Same indexing behavior across all methods
6. **Disk-Aware**: Designed for efficient disk-based queries

## Migration Guide

### Old Code
```python
# Old: Only supported simple integer lists
rows = [0, 5, 10]
data = zdata.read_rows(rows)
```

### New Code
```python
# New: Supports all indexing patterns
# All of these work:
data = zdata.read_rows(5)                    # Single row
data = zdata.read_rows(slice(0, 100))        # Slice
data = zdata.read_rows([0, 5, 10])           # List
data = zdata.read_rows(mask)                 # Boolean mask
data = zdata.read_rows(-1)                   # Negative index
```

The old `_normalize_indices()` function is still available for backward compatibility but is deprecated. New code should use `normalize_row_indices()` or `normalize_column_indices()`.

