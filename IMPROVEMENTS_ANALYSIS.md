# Key Improvements for zdata Based on anndata Architecture

This document outlines key architectural improvements that could be made to the zdata module based on how anndata is structured and implemented.

## 1. Module Organization & Structure

### Current State (zdata)
- Single main file (`core/zdata.py`) with all functionality
- Minimal separation of concerns
- No clear separation between core logic, I/O, and utilities

### Improvement: Modular Architecture
**Recommendation**: Organize code into logical modules similar to anndata:

```
zdata/
├── _core/
│   ├── __init__.py
│   ├── zdata.py          # Main ZData class
│   ├── storage.py        # Storage abstractions
│   ├── views.py          # View support (if needed)
│   ├── index.py          # Index normalization
│   └── access.py         # Access patterns
├── _io/
│   ├── __init__.py
│   ├── read.py           # Reading operations
│   ├── write.py          # Writing operations
│   └── utils.py          # I/O utilities
├── _settings.py          # Settings management
├── _warnings.py          # Custom warnings
└── utils.py              # General utilities
```

**Benefits**:
- Better code organization
- Easier to maintain and test
- Clear separation of concerns
- More scalable architecture

## 2. Settings Management System

### Current State (zdata)
- No centralized settings
- Hard-coded values scattered throughout code
- No way to configure behavior

### Improvement: Settings Manager
**Recommendation**: Implement a settings system similar to anndata's `_settings.py`:

```python
# _settings.py
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

@dataclass
class SettingsManager:
    _config: dict[str, object] = field(default_factory=dict)
    
    def register(self, option: str, default_value: Any, description: str):
        """Register a new setting."""
        self._config[option] = default_value
    
    def __getattr__(self, option: str):
        return self._config.get(option)
    
    def __setattr__(self, option: str, val: object):
        if option in self._config:
            self._config[option] = val
        else:
            super().__setattr__(option, val)

settings = SettingsManager()

# Register settings
settings.register("max_rows_per_chunk", 8192, "Maximum rows per chunk file")
settings.register("block_rows", 16, "Number of rows per block")
settings.register("chunk_cache_size", 10, "Number of chunks to cache in memory")
```

**Benefits**:
- Centralized configuration
- Environment variable support
- Runtime configuration changes
- Better documentation of options

## 3. Type Hints & Type Safety

### Current State (zdata)
- Minimal type hints
- No `TYPE_CHECKING` blocks
- Limited type safety

### Improvement: Comprehensive Type Hints
**Recommendation**: Add extensive type hints like anndata:

```python
from __future__ import annotations
from typing import TYPE_CHECKING, overload
from pathlib import Path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numpy.typing import NDArray
    from scipy.sparse import csr_matrix, csc_matrix

class ZData:
    def read_rows(
        self, 
        global_rows: int | Sequence[int] | NDArray[np.integer]
    ) -> list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]:
        """Read rows with proper type hints."""
        ...
    
    @overload
    def __getitem__(self, key: slice) -> ad.AnnData: ...
    
    @overload
    def __getitem__(self, key: list[str]) -> csc_matrix: ...
```

**Benefits**:
- Better IDE support
- Catch errors at development time
- Self-documenting code
- Better tooling support (mypy, pylance)

## 4. Custom Warnings System

### Current State (zdata)
- Uses standard Python warnings
- No custom warning classes
- Limited warning context

### Improvement: Custom Warning Classes
**Recommendation**: Create custom warnings like anndata:

```python
# _warnings.py
class ZDataWarning(UserWarning):
    """Base class for zdata warnings."""
    pass

class DeprecatedFormatWarning(ZDataWarning):
    """Warning for deprecated file formats."""
    pass

class PerformanceWarning(ZDataWarning):
    """Warning about performance issues."""
    pass

def warn(message: str, category: type[Warning] = ZDataWarning):
    """Issue a warning with proper formatting."""
    import warnings
    warnings.warn(message, category, stacklevel=2)
```

**Benefits**:
- Better warning categorization
- Filterable warnings
- More informative error messages
- Better user experience

## 5. Index Normalization & Validation

### Current State (zdata)
- Basic index normalization in `_normalize_indices()`
- Limited validation
- No support for complex indexing patterns

### Improvement: Comprehensive Index Handling
**Recommendation**: Implement robust index normalization like anndata:

```python
# _core/index.py
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Union
    Index1D = Union[int, slice, list[int], np.ndarray, pd.Index]

def normalize_indices(
    index: Index1D,
    length: int,
    names: pd.Index | None = None
) -> np.ndarray | int | slice:
    """
    Normalize indices to handle:
    - Integer indices
    - Slices
    - Boolean arrays
    - String names (if names provided)
    - Negative indices
    """
    if isinstance(index, int):
        if index < 0:
            index = length + index
        if not 0 <= index < length:
            raise IndexError(f"Index {index} out of range [0, {length})")
        return index
    
    if isinstance(index, slice):
        start, stop, step = index.indices(length)
        return slice(start, stop, step)
    
    if isinstance(index, (list, np.ndarray)):
        index = np.asarray(index)
        if index.dtype == bool:
            if len(index) != length:
                raise IndexError(f"Boolean index length {len(index)} != {length}")
            return index
        # Handle negative indices
        index = np.where(index < 0, length + index, index)
        if np.any((index < 0) | (index >= length)):
            raise IndexError(f"Indices out of range")
        return index
    
    if isinstance(index, str) and names is not None:
        return names.get_loc(index)
    
    raise TypeError(f"Unsupported index type: {type(index)}")
```

**Benefits**:
- More robust indexing
- Better error messages
- Support for more indexing patterns
- Consistent behavior

## 6. File Backing Abstraction

### Current State (zdata)
- Direct file access
- No abstraction layer
- Hard to test or mock

### Improvement: File Manager Abstraction
**Recommendation**: Create a file manager similar to anndata's `AnnDataFileManager`:

```python
# _core/file_backing.py
from pathlib import Path
from typing import TYPE_CHECKING
import weakref

if TYPE_CHECKING:
    from .zdata import ZData

class ZDataFileManager:
    """Manages file operations for ZData."""
    
    def __init__(self, zdata: ZData, dir_path: Path | str):
        self._zdata_ref = weakref.ref(zdata)
        self.dir_path = Path(dir_path)
        self._metadata_cache = None
        self._chunk_cache = {}
    
    @property
    def metadata_path(self) -> Path:
        return self.dir_path / "metadata.json"
    
    @property
    def obs_path(self) -> Path:
        return self.dir_path / "obs.parquet"
    
    @property
    def var_path(self) -> Path:
        return self.dir_path / "var.parquet"
    
    def get_chunk_path(self, chunk_num: int, orientation: str = "RM") -> Path:
        """Get path to chunk file."""
        subdir = f"X_{orientation}"
        # Implementation...
    
    def load_metadata(self) -> dict:
        """Load and cache metadata."""
        if self._metadata_cache is None:
            import json
            with open(self.metadata_path) as f:
                self._metadata_cache = json.load(f)
        return self._metadata_cache
    
    def clear_cache(self):
        """Clear file caches."""
        self._chunk_cache.clear()
        self._metadata_cache = None
```

**Benefits**:
- Better abstraction
- Easier testing
- Caching support
- Cleaner code organization

## 7. View Support (Optional but Powerful)

### Current State (zdata)
- No view support
- Every operation creates new objects
- Memory inefficient for large datasets

### Improvement: Lazy Views
**Recommendation**: Implement view support for subsetting (if needed):

```python
# _core/views.py
from typing import TYPE_CHECKING
import weakref

if TYPE_CHECKING:
    from .zdata import ZData

class ZDataView:
    """A view into a ZData object that doesn't copy data."""
    
    def __init__(self, parent: ZData, row_indices: list[int]):
        self._parent_ref = weakref.ref(parent)
        self._row_indices = row_indices
        self._is_view = True
    
    @property
    def parent(self) -> ZData:
        return self._parent_ref()
    
    def read_rows(self, global_rows):
        """Read rows using parent's method with view indices."""
        # Map view indices to actual indices
        actual_indices = [self._row_indices[i] for i in global_rows]
        return self.parent.read_rows(actual_indices)
    
    def copy(self) -> ZData:
        """Convert view to actual ZData object."""
        # Implementation to create actual copy
        ...
```

**Benefits**:
- Memory efficient subsetting
- Lazy evaluation
- Copy-on-modify semantics
- Better performance for large datasets

## 8. Better Error Messages

### Current State (zdata)
- Basic error messages
- Limited context
- No suggestions for fixes

### Improvement: Contextual Error Messages
**Recommendation**: Provide more helpful error messages:

```python
# utils.py
def raise_file_not_found(path: Path, suggestion: str | None = None):
    """Raise FileNotFoundError with helpful message."""
    msg = f"File not found: {path}"
    if suggestion:
        msg += f"\nSuggestion: {suggestion}"
    if path.exists() and path.is_dir():
        msg += f"\nNote: Path exists but is a directory, not a file."
    raise FileNotFoundError(msg)

def raise_metadata_error(missing_field: str, available_fields: list[str]):
    """Raise error for missing metadata field with suggestions."""
    msg = f"Metadata missing required field: {missing_field}"
    if available_fields:
        msg += f"\nAvailable fields: {', '.join(available_fields)}"
        # Suggest closest match
        import difflib
        matches = difflib.get_close_matches(missing_field, available_fields, n=1)
        if matches:
            msg += f"\nDid you mean: {matches[0]}?"
    raise ValueError(msg)
```

**Benefits**:
- Better user experience
- Faster debugging
- More actionable errors
- Reduced support burden

## 9. Documentation & Docstrings

### Current State (zdata)
- Some docstrings present
- Inconsistent formatting
- Missing parameter/return documentation

### Improvement: Comprehensive Docstrings
**Recommendation**: Use NumPy-style docstrings like anndata:

```python
def read_rows(self, global_rows):
    """\
    Read rows using global indices that span across multiple .bin files.
    
    Parameters
    ----------
    global_rows
        List or array of global row indices (0-based, relative to full MTX file).
        Can be a single integer, list of integers, or numpy array.
    
    Returns
    -------
    List of (global_row_id, cols, vals) tuples in the same order as global_rows,
    where cols and vals are numpy arrays of column indices and values respectively.
    
    Raises
    ------
    IndexError
        If any row index is out of bounds.
    ValueError
        If indices are invalid (e.g., negative).
    
    Examples
    --------
    >>> zdata = ZData("dataset")
    >>> rows = zdata.read_rows([0, 100, 200])
    >>> for row_id, cols, vals in rows:
    ...     print(f"Row {row_id}: {len(cols)} non-zero values")
    """
```

**Benefits**:
- Better IDE support
- Auto-generated documentation
- Clearer API understanding
- Better examples

## 10. Testing Infrastructure

### Current State (zdata)
- Basic test files
- No test fixtures
- Limited test organization

### Improvement: Comprehensive Test Setup
**Recommendation**: Organize tests better:

```python
# tests/conftest.py
import pytest
from pathlib import Path
from zdata.core import ZData

@pytest.fixture
def sample_zdata_dir(tmp_path):
    """Create a sample zdata directory for testing."""
    # Setup test data
    ...
    return tmp_path / "test_data"

@pytest.fixture
def zdata_instance(sample_zdata_dir):
    """Create a ZData instance for testing."""
    return ZData(sample_zdata_dir)

# tests/test_core/test_zdata.py
def test_read_rows(zdata_instance):
    """Test reading rows."""
    ...

# tests/test_io/test_read.py
def test_read_metadata(zdata_instance):
    """Test reading metadata."""
    ...
```

**Benefits**:
- Better test organization
- Reusable fixtures
- Easier to maintain
- Better test coverage

## 11. Compatibility Layer

### Current State (zdata)
- Direct dependencies on specific libraries
- No abstraction for different array types

### Improvement: Array Type Abstraction
**Recommendation**: Support multiple array backends:

```python
# compat.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union
    import numpy as np
    from scipy.sparse import csr_matrix, csc_matrix
    
    ArrayLike = Union[
        np.ndarray,
        csr_matrix,
        csc_matrix,
        # Could add dask, cupy, etc.
    ]

def as_csr_matrix(x: ArrayLike) -> csr_matrix:
    """Convert various array types to CSR matrix."""
    if isinstance(x, csr_matrix):
        return x
    if isinstance(x, csc_matrix):
        return x.tocsr()
    if isinstance(x, np.ndarray):
        from scipy.sparse import csr_matrix
        return csr_matrix(x)
    raise TypeError(f"Cannot convert {type(x)} to CSR matrix")
```

**Benefits**:
- More flexible
- Better integration with other libraries
- Future-proof
- Easier to extend

## 12. Logging System

### Current State (zdata)
- No logging
- Print statements for debugging
- No log levels

### Improvement: Structured Logging
**Recommendation**: Add logging like anndata:

```python
# logging.py
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final

logger: Final = logging.getLogger("zdata")

def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f"zdata.{name}")

# Usage in zdata.py
from .logging import get_logger

_logger = get_logger(__name__)

class ZData:
    def read_rows(self, global_rows):
        _logger.debug(f"Reading {len(global_rows)} rows")
        # ...
        _logger.info(f"Successfully read {len(rows_data)} rows")
```

**Benefits**:
- Better debugging
- Configurable verbosity
- Production-ready
- Better observability

## Priority Recommendations

### High Priority (Immediate Impact)
1. **Type Hints** - Improves developer experience significantly
2. **Settings System** - Enables configuration without code changes
3. **Better Error Messages** - Improves user experience
4. **Module Organization** - Makes codebase more maintainable

### Medium Priority (Significant Value)
5. **Index Normalization** - Makes API more robust
6. **File Backing Abstraction** - Better code organization
7. **Documentation** - Better API understanding
8. **Logging** - Better debugging and monitoring

### Low Priority (Nice to Have)
9. **View Support** - Only if memory efficiency is critical
10. **Custom Warnings** - Nice for user experience
11. **Compatibility Layer** - Only if supporting multiple backends
12. **Testing Infrastructure** - Current setup may be sufficient

## Implementation Strategy

1. **Start with type hints** - Add gradually, file by file
2. **Create settings system** - Extract hard-coded values
3. **Reorganize modules** - Move code to appropriate modules
4. **Improve error messages** - Add context to existing errors
5. **Add documentation** - Improve docstrings incrementally

This approach allows incremental improvements without breaking existing functionality.

