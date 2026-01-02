"""
Index normalization and validation for zdata.

This module provides functions to normalize and validate indices for row-major
and column-major queries on disk-based data.

Unlike in-memory data structures, zdata can only efficiently support:
- Row-major queries: integer indices, slices, boolean arrays, lists of integers
- Column-major queries: gene names (strings), integer indices, slices, boolean arrays, lists of integers

Arbitrary 2D indexing is not supported due to disk-based storage constraints.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# Import NDArray at runtime - it's available from numpy.typing
try:
    from numpy.typing import NDArray
except ImportError:
    # Fallback for older numpy versions
    from typing import Any as NDArray


# Type aliases for clarity
RowIndex = int | slice | Sequence[int] | NDArray[np.integer] | NDArray[np.bool_]
ColumnIndex = (
    int
    | slice
    | str
    | Sequence[int]
    | Sequence[str]
    | NDArray[np.integer]
    | NDArray[np.bool_]
)


def normalize_row_indices(
    index: RowIndex,
    nrows: int,
) -> list[int]:
    """\
    Normalize row indices to a list of integer indices.

    This function handles various index types and converts them to a sorted,
    deduplicated list of integer indices suitable for efficient chunk-based access.

    Parameters
    ----------
    index
        Row index or indices to normalize. Supported types:
        - int: Single row index (supports negative indices, e.g., -1 for last row)
        - slice: Row slice (e.g., slice(0, 100) or 0:100)
        - list[int]: List of row indices
        - numpy.ndarray[int]: Array of row indices
        - numpy.ndarray[bool]: Boolean mask (length must match nrows)
        - pandas.Index or pandas.Series: Pandas index objects
    nrows
        Total number of rows in the dataset.

    Returns
    -------
    list[int]
        List of integer row indices (0-based, non-negative, sorted and deduplicated).
        The list is guaranteed to contain only valid indices in range [0, nrows).

    Raises
    ------
    IndexError
        If any index is out of bounds [0, nrows) or negative index is too large.
    ValueError
        If boolean array length doesn't match nrows or step slicing is not 1.
    TypeError
        If index type is not supported.

    Examples
    --------
    >>> normalize_row_indices(5, nrows=1000)
    [5]
    >>> normalize_row_indices(-1, nrows=1000)  # Last row
    [999]
    >>> normalize_row_indices(slice(0, 100), nrows=1000)
    [0, 1, 2, ..., 99]
    >>> normalize_row_indices([0, 5, 10, -1], nrows=1000)
    [0, 5, 10, 999]
    >>> mask = np.array([True] * 100 + [False] * 900)
    >>> normalize_row_indices(mask, nrows=1000)
    [0, 1, 2, ..., 99]
    """
    # Handle single integer
    if isinstance(index, (int, np.integer)):
        idx = int(index)
        if idx < 0:
            idx = nrows + idx
        if not 0 <= idx < nrows:
            raise IndexError(
                f"Row index {index} out of range [0, {nrows}). "
                f"Use negative indices like -1 for last row."
            )
        return [idx]

    # Handle slice
    if isinstance(index, slice):
        start, stop, step = index.indices(nrows)
        if step != 1:
            raise ValueError(
                f"Step slicing (step={step}) is not supported for row queries. "
                f"Use step=1 or convert to explicit index list."
            )
        if start >= stop:
            return []
        return list(range(start, stop))

    # Handle boolean array
    if isinstance(index, np.ndarray) and index.dtype == bool:
        if len(index) != nrows:
            raise ValueError(
                f"Boolean index length {len(index)} does not match number of rows {nrows}."
            )
        indices = np.where(index)[0].tolist()
        return sorted(set(indices))  # Sort and deduplicate

    # Handle sequences and arrays
    if isinstance(index, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        # Convert to numpy array for processing
        if isinstance(index, (pd.Index, pd.Series)):
            index_array = index.to_numpy()
        else:
            index_array = np.asarray(index)

        # Check for boolean array
        if index_array.dtype == bool:
            if len(index_array) != nrows:
                raise ValueError(
                    f"Boolean index length {len(index_array)} does not match "
                    f"number of rows {nrows}."
                )
            indices = np.where(index_array)[0].tolist()
            return sorted(set(indices))

        # Handle integer array
        if not np.issubdtype(index_array.dtype, np.integer):
            # Try to convert float indices (e.g., [1.0, 2.0])
            if np.issubdtype(index_array.dtype, np.floating):
                index_int = index_array.astype(int)
                if not np.allclose(index_array, index_int):
                    raise TypeError(
                        f"Row indices must be integers, got floating point values: {index_array}"
                    )
                index_array = index_int
            else:
                raise TypeError(
                    f"Row indices must be integers or booleans, got {index_array.dtype}"
                )

        # Handle negative indices
        indices = np.where(index_array < 0, nrows + index_array, index_array)

        # Validate bounds
        out_of_bounds = (indices < 0) | (indices >= nrows)
        if np.any(out_of_bounds):
            invalid = index_array[out_of_bounds]
            raise IndexError(
                f"Row indices out of range: {invalid.tolist()}. "
                f"Valid range is [0, {nrows}) or negative indices [-{nrows}, -1]."
            )

        # Convert to sorted, deduplicated list
        result = sorted(set(int(i) for i in indices))
        return result

    raise TypeError(
        f"Unsupported row index type: {type(index)}. "
        f"Supported types: int, slice, list[int], numpy array (int or bool), "
        f"pandas Index/Series."
    )


def normalize_column_indices(
    index: ColumnIndex,
    ncols: int,
    gene_names: pd.Index | None = None,
) -> list[int]:
    """\
    Normalize column (gene) indices to a list of integer indices.

    This function handles various index types including gene names (strings) and
    converts them to a sorted, deduplicated list of integer indices suitable for
    efficient chunk-based access.

    Parameters
    ----------
    index
        Column (gene) index or indices to normalize. Supported types:
        - int: Single column index (supports negative indices)
        - str: Single gene name (requires gene_names)
        - slice: Column slice (supports integer or string bounds)
        - list[int]: List of column indices
        - list[str]: List of gene names (requires gene_names)
        - numpy.ndarray[int]: Array of column indices
        - numpy.ndarray[bool]: Boolean mask (length must match ncols)
        - pandas.Index or pandas.Series: Pandas index objects
    ncols
        Total number of columns (genes) in the dataset.
    gene_names
        Optional pandas Index of gene names for string-based indexing.
        Required when index contains strings or string slices.

    Returns
    -------
    list[int]
        List of integer column indices (0-based, non-negative, sorted and deduplicated).
        The list is guaranteed to contain only valid indices in range [0, ncols).

    Raises
    ------
    IndexError
        If any index is out of bounds [0, ncols) or gene name not found.
    ValueError
        If boolean array length doesn't match ncols, gene_names not provided when
        needed, or step slicing is not 1.
    TypeError
        If index type is not supported.

    Examples
    --------
    >>> # Integer indexing
    >>> normalize_column_indices(5, ncols=20000)
    [5]
    >>> normalize_column_indices([0, 5, 10], ncols=20000)
    [0, 5, 10]
    >>> # Gene name indexing (requires gene_names)
    >>> gene_names = pd.Index(['GAPDH', 'PCNA', 'COL1A1', ...])
    >>> normalize_column_indices('GAPDH', ncols=20000, gene_names=gene_names)
    [0]
    >>> normalize_column_indices(['GAPDH', 'PCNA'], ncols=20000, gene_names=gene_names)
    [0, 1]
    >>> # Slice with gene names
    >>> normalize_column_indices(slice('GAPDH', 'PCNA'), ncols=20000, gene_names=gene_names)
    [0, 1]
    >>> # Boolean mask
    >>> mask = np.array([True] * 100 + [False] * 19900)
    >>> normalize_column_indices(mask, ncols=20000)
    [0, 1, 2, ..., 99]
    """
    # Handle single integer
    if isinstance(index, (int, np.integer)):
        idx = int(index)
        if idx < 0:
            idx = ncols + idx
        if not 0 <= idx < ncols:
            raise IndexError(
                f"Column index {index} out of range [0, {ncols}). "
                f"Use negative indices like -1 for last column."
            )
        return [idx]

    # Handle single string (gene name)
    if isinstance(index, str):
        if gene_names is None:
            raise ValueError(
                "String-based column indexing requires gene_names to be provided. "
                "Gene names are typically loaded from var.parquet."
            )
        try:
            idx = gene_names.get_loc(index)
            return [int(idx)]
        except KeyError:
            raise IndexError(f"Gene name '{index}' not found in dataset.") from None

    # Handle slice
    if isinstance(index, slice):
        # Check if slice uses string bounds (requires gene_names)
        if isinstance(index.start, str) or isinstance(index.stop, str):
            if gene_names is None:
                raise ValueError(
                    "String-based slice indexing requires gene_names to be provided."
                )
            # Convert string bounds to integer indices
            start = (
                gene_names.get_loc(index.start)
                if index.start is not None and isinstance(index.start, str)
                else (index.start if index.start is not None else 0)
            )
            stop = (
                gene_names.get_loc(index.stop)
                if index.stop is not None and isinstance(index.stop, str)
                else (index.stop if index.stop is not None else ncols)
            )
            # String slices are inclusive, so add 1 to stop
            if isinstance(index.stop, str):
                stop = stop + 1 if stop is not None else None
            step = index.step if index.step is not None else 1
            index = slice(start, stop, step)

        start, stop, step = index.indices(ncols)
        if step != 1:
            raise ValueError(
                f"Step slicing (step={step}) is not supported for column queries. "
                f"Use step=1 or convert to explicit index list."
            )
        if start >= stop:
            return []
        return list(range(start, stop))

    # Handle boolean array
    if isinstance(index, np.ndarray) and index.dtype == bool:
        if len(index) != ncols:
            raise ValueError(
                f"Boolean index length {len(index)} does not match number of columns {ncols}."
            )
        indices = np.where(index)[0].tolist()
        return sorted(set(indices))

    # Handle sequences and arrays
    if isinstance(index, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        # Check if it's a list of strings (gene names)
        if len(index) > 0 and isinstance(index[0], str):
            if gene_names is None:
                raise ValueError(
                    "String-based column indexing requires gene_names to be provided."
                )
            # Look up gene indices
            gene_indices: list[int] = []
            missing_genes: list[str] = []
            for gene_name in index:
                if gene_name in gene_names:
                    gene_indices.append(int(gene_names.get_loc(gene_name)))
                else:
                    missing_genes.append(gene_name)

            if missing_genes:
                raise IndexError(
                    f"Gene names not found: {missing_genes}. "
                    f"Available genes: {len(gene_names)} total."
                )

            return sorted(set(gene_indices))

        # Convert to numpy array for processing
        if isinstance(index, (pd.Index, pd.Series)):
            index_array = index.to_numpy()
        else:
            index_array = np.asarray(index)

        # Check for boolean array
        if index_array.dtype == bool:
            if len(index_array) != ncols:
                raise ValueError(
                    f"Boolean index length {len(index_array)} does not match "
                    f"number of columns {ncols}."
                )
            indices = np.where(index_array)[0].tolist()
            return sorted(set(indices))

        # Handle integer array
        if not np.issubdtype(index_array.dtype, np.integer):
            # Try to convert float indices
            if np.issubdtype(index_array.dtype, np.floating):
                index_int = index_array.astype(int)
                if not np.allclose(index_array, index_int):
                    raise TypeError(
                        f"Column indices must be integers, got floating point values: {index_array}"
                    )
                index_array = index_int
            else:
                raise TypeError(
                    f"Column indices must be integers, booleans, or strings, "
                    f"got {index_array.dtype}"
                )

        # Handle negative indices
        indices = np.where(index_array < 0, ncols + index_array, index_array)

        # Validate bounds
        out_of_bounds = (indices < 0) | (indices >= ncols)
        if np.any(out_of_bounds):
            invalid = index_array[out_of_bounds]
            raise IndexError(
                f"Column indices out of range: {invalid.tolist()}. "
                f"Valid range is [0, {ncols}) or negative indices [-{ncols}, -1]."
            )

        # Convert to sorted, deduplicated list
        result = sorted(set(int(i) for i in indices))
        return result

    raise TypeError(
        f"Unsupported column index type: {type(index)}. "
        f"Supported types: int, str, slice, list[int|str], numpy array (int or bool), "
        f"pandas Index/Series."
    )


def validate_row_indices(indices: list[int], nrows: int) -> None:
    """\
    Validate that row indices are within bounds.

    Parameters
    ----------
    indices
        List of row indices to validate.
    nrows
        Total number of rows.

    Raises
    ------
    IndexError
        If any index is out of bounds [0, nrows).

    Examples
    --------
    >>> validate_row_indices([0, 5, 10], nrows=100)  # OK
    >>> validate_row_indices([0, 5, 100], nrows=100)  # Raises IndexError
    """
    if not indices:
        return
    invalid = [i for i in indices if not 0 <= i < nrows]
    if invalid:
        raise IndexError(
            f"Row indices out of range: {invalid}. Valid range is [0, {nrows})."
        )


def validate_column_indices(indices: list[int], ncols: int) -> None:
    """\
    Validate that column indices are within bounds.

    Parameters
    ----------
    indices
        List of column indices to validate.
    ncols
        Total number of columns.

    Raises
    ------
    IndexError
        If any index is out of bounds [0, ncols).

    Examples
    --------
    >>> validate_column_indices([0, 5, 10], ncols=20000)  # OK
    >>> validate_column_indices([0, 5, 20000], ncols=20000)  # Raises IndexError
    """
    if not indices:
        return
    invalid = [i for i in indices if not 0 <= i < ncols]
    if invalid:
        raise IndexError(
            f"Column indices out of range: {invalid}. Valid range is [0, {ncols})."
        )

