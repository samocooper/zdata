from __future__ import annotations

import bisect
import json
import os
import struct
import subprocess
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, overload

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csc_matrix, csr_matrix

from .._settings import settings
from .index import normalize_column_indices, normalize_row_indices
from .utils import get_available_memory_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray
    from polars import DataFrame as PolarsDataFrame

# Get the path to the zdata_read executable
# This assumes the module structure: zdata/core/zdata.py and zdata/ctools/zdata_read
_MODULE_DIR = Path(__file__).parent  # zdata/core/
_PROJECT_ROOT = _MODULE_DIR.parent   # zdata/
_ZDATA_READ = _PROJECT_ROOT / "ctools" / "zdata_read"

def _get_zdata_read_path() -> str:
    """Get the path to zdata_read executable, with validation."""
    bin_path = _ZDATA_READ.absolute()
    if not bin_path.exists():
        raise RuntimeError(
            f"zdata_read executable not found at {bin_path}. "
            f"Please ensure it is built in the ctools directory."
        )
    return str(bin_path)


class ObsWrapper:
    """\
    Wrapper for polars obs DataFrame that supports indexing like obs[row_index, :].
    
    Returns pandas DataFrames for compatibility with anndata/AnnData.
    This wrapper allows the obs attribute to be indexed like a 2D array while
    maintaining compatibility with anndata's expected pandas DataFrame interface.
    
    Examples
    --------
    >>> zdata = ZData("dataset")
    >>> zdata.obs[5, :]  # Returns pandas DataFrame for row 5
    >>> zdata.obs[0:10, :]  # Returns pandas DataFrame for rows 0-9
    """
    
    def __init__(self, obs_df: PolarsDataFrame) -> None:
        """\
        Initialize the wrapper with a polars DataFrame.
        
        Parameters
        ----------
        obs_df
            Polars DataFrame containing observation metadata.
        """
        self.obs_df: PolarsDataFrame = obs_df
    
    def __getitem__(self, key: tuple[int | slice, slice]) -> pd.DataFrame:
        """\
        Support indexing like obs[row_index, :] or obs[slice, :].
        
        Parameters
        ----------
        key
            Tuple of (row_index or slice, column_slice).
            Currently only supports [row_index, :] or [slice, :].
        
        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with selected rows, indexed by row position.
        
        Raises
        ------
        ValueError
            If key format is not supported (must be [row_index, :] or [slice, :]).
        """
        if not isinstance(key, tuple) or len(key) != 2 or key[1] != slice(None):
            raise ValueError("Obs indexing must be in format [row_index, :] or [slice, :]")
        
        row_key = key[0]
        
        # Handle row indexing
        if isinstance(row_key, int):
            # Single row - get as polars DataFrame then convert to pandas
            polars_result = self.obs_df.slice(row_key, 1)
        elif isinstance(row_key, slice):
            # Slice of rows - get as polars DataFrame then convert to pandas
            polars_result = self.obs_df[row_key]
        else:
            raise ValueError(f"Row index must be int or slice, got {type(row_key)}")
        
        data_dict = polars_result.to_dict(as_series=False)
        pandas_df = pd.DataFrame(data_dict)
        
        if "_row_index" in pandas_df.columns:
            pandas_df = pandas_df.set_index("_row_index")
            pandas_df.index = pandas_df.index.astype(int)
        else:
            pandas_df.index = pd.RangeIndex(start=0, stop=len(pandas_df))
        
        return pandas_df
    
    def __len__(self) -> int:
        return len(self.obs_df)
    
    @property
    def shape(self) -> tuple[int, int]:
        return self.obs_df.shape
    
    @property
    def columns(self) -> list[str]:
        return self.obs_df.columns
    
    def __repr__(self) -> str:
        return f"ObsWrapper({self.obs_df.shape[0]} rows, {self.obs_df.shape[1]} columns)"

class ZData:
    """\
    Efficient reader for zdata directory structure containing .bin files.
    
    ZData provides methods to read random sets of rows or columns from compressed
    sparse matrix data stored in a disk-based format. The data is organized in
    chunked files with row-major (X_RM) and column-major (X_CM) orientations for
    efficient access patterns.
    
    The zdata format uses seekable zstd compression to enable random access
    without full decompression, making it ideal for querying subsets of large
    single-cell RNA-seq datasets.
    
    Parameters
    ----------
    dir_name
        Name or path of the zdata directory. Can be a relative or absolute path.
        The directory must contain:
        - metadata.json: Dataset metadata including shape and chunk information
        - obs.parquet: Observation (cell) metadata
        - var.parquet: Variable (gene) metadata
        - X_RM/: Row-major chunk files (.bin)
        - X_CM/: Column-major chunk files (.bin, optional)
    
    Attributes
    ----------
    nrows : int
        Number of rows (cells) in the dataset.
    ncols : int
        Number of columns (genes) in the dataset.
    shape : tuple[int, int]
        Shape of the dataset (nrows, ncols).
    obs : ObsWrapper
        Observation metadata wrapper supporting indexing.
    
    Examples
    --------
    >>> zdata = ZData("atlas.zdata")
    >>> print(f"Dataset shape: {zdata.shape}")
    >>> # Read specific rows
    >>> rows = zdata.read_rows([0, 100, 200])
    >>> # Read rows as CSR matrix
    >>> csr = zdata.read_rows_csr([0, 100, 200])
    >>> # Index by rows (returns AnnData)
    >>> adata = zdata[0:100]
    >>> # Index by gene names (returns CSC matrix)
    >>> matrix = zdata[['GAPDH', 'PCNA', 'COL1A1']]
    """
    
    def __init__(self, dir_name: str | Path) -> None:
        """\
        Initialize the reader for a zdata directory.
        
        Parameters
        ----------
        dir_name
            Name or path of the zdata directory.
            Can be a relative path (e.g., "atlas.zdata") or absolute path
            (e.g., "/path/to/atlas.zdata").
        
        Raises
        ------
        FileNotFoundError
            If the directory or required files (metadata.json, obs.parquet, var.parquet)
            are not found.
        ValueError
            If the path is not a directory or metadata is missing required fields.
        RuntimeError
            If parquet files cannot be loaded.
        """
        self.dir_path: str | Path = dir_name
        
        if not os.path.exists(self.dir_path):
            raise FileNotFoundError(f"Directory not found: {self.dir_path}")
        
        if not os.path.isdir(self.dir_path):
            raise ValueError(f"Path is not a directory: {self.dir_path}")
        
        metadata_file = os.path.join(self.dir_path, "metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}. "
                f"Please rebuild the zdata directory using build_zdata()"
            )
        
        with open(metadata_file, 'r') as f:
            self.metadata: dict[str, Any] = json.load(f)
        
        self.nrows: int = self.metadata['shape'][0]
        self.ncols: int = self.metadata['shape'][1]
        self.nnz_total: int | None = self.metadata.get('nnz_total', None)
        if 'num_chunks_rm' not in self.metadata:
            raise ValueError("Metadata missing 'num_chunks_rm' field (new format required)")
        self.num_chunks: int = self.metadata['num_chunks_rm']
        self.total_blocks: int | None = self.metadata.get('total_blocks_rm', None)
        
        self.block_rows: int = self.metadata.get('block_rows', settings.block_rows)
        self.block_columns: int = self.metadata.get('block_columns', self.block_rows)
        self.max_rows_per_chunk: int = self.metadata.get(
            'max_rows_per_chunk', settings.max_rows_per_chunk
        )
        
        # Using regular subprocess.check_output with optimized work distribution
        # ThreadPoolExecutor reuses threads efficiently, minimizing subprocess churn
        
        self.chunk_files: dict[int, str] = {}
        self.chunk_info: dict[int, dict[str, Any]] = {}
        self.file_to_chunk: dict[str, int] = {}
        
        if 'chunks_rm' not in self.metadata:
            raise ValueError("Metadata missing 'chunks_rm' field (new format required)")
        chunks_list = self.metadata['chunks_rm']
        subdir = "X_RM"
        
        for chunk_info in chunks_list:
            chunk_num = chunk_info['chunk_num']
            file_path = os.path.join(self.dir_path, subdir, chunk_info['file'])
            self.chunk_files[chunk_num] = file_path
            self.chunk_info[chunk_num] = chunk_info
            self.file_to_chunk[file_path] = chunk_num
        
        obs_file = os.path.join(self.dir_path, "obs.parquet")
        if not os.path.exists(obs_file):
            raise FileNotFoundError(
                f"obs.parquet not found: {obs_file}. "
                f"Please rebuild the zdata directory using build_zdata()"
            )
        var_file = os.path.join(self.dir_path, "var.parquet")
        if not os.path.exists(var_file):
            raise FileNotFoundError(
                f"var.parquet not found: {var_file}. "
                f"Please rebuild the zdata directory using build_zdata()"
            )
        
        try:
            self._obs_df: PolarsDataFrame = pl.read_parquet(obs_file)
            self._obs_wrapper: ObsWrapper = ObsWrapper(self._obs_df)
            
            # Cache var DataFrame for __getitem__ access
            var_polars = pl.read_parquet(var_file)
            var_dict = var_polars.to_dict(as_series=False)
            self._var_df: pd.DataFrame = pd.DataFrame(var_dict)
            self._var_df.index = pd.RangeIndex(start=0, stop=len(self._var_df))
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet files: {e}") from e
        
        # Cache for column-major chunk mappings (built lazily)
        self._cm_chunk_files: dict[int, str] | None = None
        self._cm_chunk_info: dict[int, dict[str, Any]] | None = None
        self._cm_file_to_chunk: dict[str, int] | None = None
        # Sorted list of (start_row, end_row, chunk_num) for binary search
        self._cm_chunk_ranges: list[tuple[int, int, int]] | None = None
    
    @property
    def obs(self) -> ObsWrapper:
        """
        Access obs/metadata DataFrame.
        
        Returns:
            ObsWrapper that supports indexing like obs[row_index, :]
        
        """
        return self._obs_wrapper
    
    def _check_memory_requirements(
        self,
        row_indices: list[int] | None = None,
        column_indices: list[int] | None = None,
    ) -> None:
        """\
        Check memory requirements for a query and raise/warn if needed.
        
        This is an internal helper method that performs memory estimation
        and validation for row or column queries.
        
        Parameters
        ----------
        row_indices
            Optional list of row indices to check.
        column_indices
            Optional list of column indices to check.
        
        Raises
        ------
        MemoryError
            If estimated memory exceeds 80% of available memory and
            override_memory_check is False.
        UserWarning
            If estimated memory exceeds 80% of available memory and
            override_memory_check is True, or if memory estimation fails.
        """
        try:
            memory_estimate = self.estimate_memory_requirements(
                row_indices=row_indices, column_indices=column_indices
            )
            estimated_memory_bytes = memory_estimate['estimated_memory_mb'] * 1024 * 1024
            available_memory_bytes = get_available_memory_bytes()
            memory_threshold = 0.8 * available_memory_bytes
            
            if estimated_memory_bytes > memory_threshold:
                error_message = (
                    f"Query would require {memory_estimate['estimated_memory_gb']:.2f} GB of memory, "
                    f"which exceeds 80% of available memory ({available_memory_bytes / (1024**3):.2f} GB). "
                    f"Available: {available_memory_bytes / (1024**3):.2f} GB, "
                    f"Threshold (80%): {memory_threshold / (1024**3):.2f} GB, "
                    f"Estimated: {estimated_memory_bytes / (1024**3):.2f} GB. "
                    f"Please reduce the query size or free up memory."
                )
                
                if settings.override_memory_check:
                    warnings.warn(
                        f"{error_message} "
                        f"Proceeding anyway because override_memory_check=True. "
                        f"This may cause the system to run out of memory.",
                        UserWarning
                    )
                else:
                    raise MemoryError(
                        f"{error_message} "
                        f"Set zdata.settings.override_memory_check = True to override this check."
                    )
        except ValueError as e:
            # If nnz values are missing, we can't estimate memory accurately
            # In this case, we'll skip the check but warn the user
            warnings.warn(
                f"Cannot estimate memory requirements: {e}. "
                f"Proceeding with query, but it may fail if insufficient memory is available.",
                UserWarning
            )
    
    def _read_rows_from_file(
        self, 
        file_path: str | Path, 
        local_rows: list[int],
        block_rows: int | None = None
    ) -> tuple[int, list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]]:
        """
        Read rows from a single .bin file. Rows are local indices within that file.
        
        Args:
            file_path: Path to the .bin file
            local_rows: List of local row indices (0-based within the chunk)
            block_rows: Block size to use. If None, uses self.block_rows from metadata.
        
        Returns:
            (ncols, results) where results is a list of (local_row_id, cols, vals) tuples
        """
        # Build CSV string (list comprehension for large lists, map for small)
        rows_csv = ",".join([str(r) for r in local_rows] if len(local_rows) > 50 else map(str, local_rows))
        
        # Build command: always pass block_rows from metadata (no auto-detect needed)
        # Cache zdata_read path to avoid repeated lookups
        if not hasattr(self, '_zdata_read_path'):
            self._zdata_read_path = _get_zdata_read_path()
        block_rows_val = block_rows if block_rows is not None else self.block_rows
        
        # Use regular subprocess.check_output (reliable and fast)
        # Work distribution is optimized at ThreadPoolExecutor level to minimize worker churn
        cmd_full = [self._zdata_read_path, "--binary", "--block-rows", str(block_rows_val), str(file_path), rows_csv]
        blob = subprocess.check_output(
            cmd_full,
            bufsize=262144,
            stderr=subprocess.DEVNULL
        )
        
        if len(blob) < 8:
            raise ValueError(f"Output too small: {len(blob)} bytes")
        
        # Unpack header and process rows
        nreq, ncols = struct.unpack_from("<II", blob, 0)
        out = [None] * nreq
        blob_mv = memoryview(blob)
        off = 8
        
        # Process rows - optimize by avoiding repeated array creation for empty rows
        empty_cols = np.array([], dtype=np.uint32)
        empty_vals = np.array([], dtype=np.uint16)
        
        for i in range(nreq):
            row_id, nnz = struct.unpack_from("<II", blob_mv, off)
            off += 8
            
            if nnz > 0:
                cols = np.frombuffer(blob_mv, dtype=np.uint32, count=nnz, offset=off).copy()
                off += nnz * 4
                vals = np.frombuffer(blob_mv, dtype=np.uint16, count=nnz, offset=off).copy()
                off += nnz * 2
                out[i] = (row_id, cols, vals)
            else:
                out[i] = (row_id, empty_cols, empty_vals)
        
        return ncols, out
    
    def _process_file_for_rows(
        self,
        file_path: str,
        row_info_list: list[tuple[int, int, int]],
        file_to_chunk: dict[str, int]
    ) -> tuple[str, int, list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]], int]:
        """
        Process a single file for row reading (used in parallel processing).
        
        Args:
            file_path: Path to the chunk file
            row_info_list: List of (local_row, orig_idx, global_row) tuples
            file_to_chunk: Mapping from file path to chunk number
        
        Returns:
            (file_path, chunk_num, file_results, file_ncols)
        """
        chunk_num = file_to_chunk[file_path]
        
        row_info_list_sorted = sorted(row_info_list, key=lambda x: x[0])
        local_rows = [info[0] for info in row_info_list_sorted]
        
        file_ncols, file_results = self._read_rows_from_file(file_path, local_rows, self.block_rows)
        
        return file_path, chunk_num, file_results, file_ncols
    
    def _process_file_for_cols(
        self,
        file_path: str,
        col_info_list: list[tuple[int, int, int]],
        file_to_chunk: dict[str, int]
    ) -> tuple[str, int, list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]], int, list[tuple[int, int, int]]]:
        """
        Process a single file for column reading (used in parallel processing).
        
        Args:
            file_path: Path to the chunk file
            col_info_list: List of (local_row, orig_idx, global_col) tuples
            file_to_chunk: Mapping from file path to chunk number
        
        Returns:
            (file_path, chunk_num, file_results, file_ncols, col_info_list_sorted)
        """
        chunk_num = file_to_chunk[file_path]
        
        # Sort once and reuse
        col_info_list_sorted = sorted(col_info_list, key=lambda x: x[0])
        local_rows = [info[0] for info in col_info_list_sorted]
        
        file_ncols, file_results = self._read_rows_from_file(file_path, local_rows, self.block_columns)
        
        return file_path, chunk_num, file_results, file_ncols, col_info_list_sorted
    
    def read_rows(
        self, 
        global_rows: int | np.integer | Sequence[int] | NDArray[np.integer] | NDArray[np.bool_] | slice
    ) -> list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]:
        """\
        Read rows using global indices that span across multiple .bin files.
        
        This method reads rows from the row-major (X_RM) chunk files. Indices are
        automatically normalized (sorted and deduplicated) for efficient chunk access.
        
        Parameters
        ----------
        global_rows
            Row index or indices (0-based, relative to full dataset).
            Supported types:
            - int: Single row index (supports negative indices, e.g., -1 for last row)
            - slice: Row slice (e.g., slice(0, 100) or 0:100)
            - list[int]: List of row indices
            - numpy.ndarray[int]: Array of row indices
            - numpy.ndarray[bool]: Boolean mask (length must match nrows)
        
        Returns
        -------
        list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]
            List of (global_row_id, cols, vals) tuples in normalized order.
            Each tuple contains:
            - global_row_id: The global row index
            - cols: numpy array of column indices (uint32)
            - vals: numpy array of values (uint16)
            Note: Results are in sorted order, not the original query order.
            Use read_rows_csr() if you need to preserve order or work with matrices.
        
        Raises
        ------
        IndexError
            If any row index is out of bounds [0, nrows).
        ValueError
            If boolean mask length doesn't match nrows.
        MemoryError
            If estimated memory requirements exceed 80% of available system memory.
        UserWarning
            If query size exceeds large_query_threshold (see zdata.settings).
            If nnz values are missing and memory estimation cannot be performed.
        
        Examples
        --------
        >>> zdata = ZData("dataset")
        >>> # Read single row
        >>> rows = zdata.read_rows(5)
        >>> # Read multiple rows
        >>> rows = zdata.read_rows([0, 100, 200])
        >>> # Read slice
        >>> rows = zdata.read_rows(slice(0, 100))
        >>> # Read with boolean mask
        >>> mask = np.array([True] * 1000 + [False] * (zdata.nrows - 1000))
        >>> rows = zdata.read_rows(mask)
        >>> # Access row data
        >>> for row_id, cols, vals in rows:
        ...     print(f"Row {row_id}: {len(cols)} non-zero values")
        """
        global_rows = normalize_row_indices(global_rows, self.nrows)
        self._check_memory_requirements(row_indices=global_rows)
        if settings.warn_on_large_queries and len(global_rows) > settings.large_query_threshold:
            warnings.warn(
                f"Querying {len(global_rows)} rows, which exceeds the threshold "
                f"of {settings.large_query_threshold}. This may be slow. "
                f"Consider using smaller batches or disable this warning with "
                f"zdata.settings.warn_on_large_queries = False",
                UserWarning
            )
        
        # Group rows by file: all rows from the same chunk file are grouped together
        # This ensures that rows from the same block are processed by a single worker
        rows_by_file = defaultdict(list)
        
        for idx, global_row in enumerate(global_rows):
            chunk_num = global_row // self.max_rows_per_chunk
            local_row = global_row % self.max_rows_per_chunk
            
            if chunk_num not in self.chunk_files:
                raise IndexError(f"Row {global_row} is beyond available data (chunk {chunk_num} not found)")
            
            # Group by file path - all rows from same file will be processed together
            rows_by_file[self.chunk_files[chunk_num]].append((local_row, idx, global_row))
        
        all_results = [None] * len(global_rows)
        
        num_files = len(rows_by_file)
        
        # Determine optimal number of workers
        max_workers = settings.max_workers
        if max_workers is None:
            max_workers = min(num_files, min(8, (os.cpu_count() or 1)))
        else:
            max_workers = min(max_workers, num_files)
        
        if num_files > 1 and max_workers > 1:
            # Batch files to reduce subprocess startup overhead
            # Each worker processes multiple files sequentially (up to 10 files per batch)
            files_per_batch = max(1, min(10, num_files // max_workers + 1))
            file_items = list(rows_by_file.items())
            file_batches = [file_items[i:i + files_per_batch] for i in range(0, len(file_items), files_per_batch)]
            
            def process_batch(batch: list[tuple[str, list[tuple[int, int, int]]]]) -> dict[str, tuple[int, list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]], int]]:
                """Process a batch of files sequentially in a single worker thread."""
                results = {}
                for file_path, row_info_list in batch:
                    _, chunk_num, file_results, file_ncols = self._process_file_for_rows(
                        file_path, row_info_list, self.file_to_chunk
                    )
                    results[file_path] = (chunk_num, file_results, file_ncols)
                return results
            
            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_batch, batch): batch for batch in file_batches}
                
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        for file_path, (chunk_num, file_results, file_ncols) in batch_results.items():
                            if self.ncols != file_ncols:
                                raise ValueError(f"Inconsistent ncols: {self.ncols} vs {file_ncols} in {file_path}")
                            
                            row_info_list_sorted = sorted(rows_by_file[file_path], key=lambda x: x[0])
                            for (local_row, orig_idx, global_row), (returned_local_row, cols, vals) in zip(row_info_list_sorted, file_results):
                                if returned_local_row != local_row:
                                    raise ValueError(f"Row mismatch: expected {local_row}, got {returned_local_row}")
                                all_results[orig_idx] = (global_row, cols, vals)
                    except Exception as e:
                        raise RuntimeError(f"Error processing batch: {e}") from e
        else:
            # Sequential processing (single file or max_workers <= 1)
            for file_path, row_info_list in rows_by_file.items():
                chunk_num = self.file_to_chunk[file_path]
                
                row_info_list_sorted = sorted(row_info_list, key=lambda x: x[0])
                local_rows = [info[0] for info in row_info_list_sorted]
                
                file_ncols, file_results = self._read_rows_from_file(file_path, local_rows, self.block_rows)
                if self.ncols != file_ncols:
                    raise ValueError(f"Inconsistent ncols: {self.ncols} vs {file_ncols} in {file_path}")
                
                for (local_row, orig_idx, global_row), (returned_local_row, cols, vals) in zip(row_info_list_sorted, file_results):
                    if returned_local_row != local_row:
                        raise ValueError(f"Row mismatch: expected local row {local_row}, got {returned_local_row}")
                    all_results[orig_idx] = (global_row, cols, vals)
        
        return all_results
    
    def read_rows_csr(
        self, 
        global_rows: int | np.integer | Sequence[int] | NDArray[np.integer] | NDArray[np.bool_] | slice
    ) -> csr_matrix:
        """\
        Read rows and return as a scipy.sparse.csr_matrix.
        
        This is a convenience method that calls read_rows() and converts the result
        to a CSR matrix. The matrix rows are in normalized (sorted) order.
        
        Parameters
        ----------
        global_rows
            Row index or indices. See read_rows() for supported types.
        
        Returns
        -------
        csr_matrix
            Compressed Sparse Row matrix of shape (n_selected_rows, ncols).
            Rows are in sorted order (not original query order).
            Dtype is float64.
        
        See Also
        --------
        read_rows : Read rows as raw tuples
        
        Examples
        --------
        >>> zdata = ZData("dataset")
        >>> # Read rows as CSR matrix
        >>> csr = zdata.read_rows_csr([0, 100, 200])
        >>> print(f"Matrix shape: {csr.shape}")
        >>> print(f"Non-zero elements: {csr.nnz}")
        """
        rows_data = self.read_rows(global_rows)
        return self._rows_to_csr(rows_data)
    
    def _build_cm_chunk_mapping(
        self
    ) -> tuple[dict[int, str], dict[int, dict[str, Any]], dict[str, int], list[tuple[int, int, int]]]:
        """\
        Build chunk mapping for X_CM (column-major) files from metadata.
        
        This is an internal method that constructs the mapping between column
        indices and chunk files for column-major access. Results are cached
        for subsequent calls.
        
        Returns
        -------
        tuple[dict[int, str], dict[int, dict[str, Any]], dict[str, int], list[tuple[int, int, int]]]
            Tuple containing:
            - cm_chunk_files: Mapping from chunk number to file path
            - cm_chunk_info: Mapping from chunk number to chunk metadata
            - cm_file_to_chunk: Reverse mapping from file path to chunk number
            - cm_chunk_ranges: Sorted list of (start_row, end_row, chunk_num) for binary search
        
        Raises
        ------
        FileNotFoundError
            If X_CM directory doesn't exist.
        ValueError
            If metadata is missing required 'chunks_cm' field.
        """
        # Return cached results if available
        if (self._cm_chunk_files is not None and 
            self._cm_chunk_info is not None and 
            self._cm_file_to_chunk is not None and
            self._cm_chunk_ranges is not None):
            return self._cm_chunk_files, self._cm_chunk_info, self._cm_file_to_chunk, self._cm_chunk_ranges
        
        xcm_dir = os.path.join(self.dir_path, "X_CM")
        if not os.path.exists(xcm_dir):
            raise FileNotFoundError(f"X_CM directory not found: {xcm_dir}")
        
        if 'chunks_cm' not in self.metadata:
            raise ValueError("Metadata missing 'chunks_cm' field (required for column-major access)")
        
        cm_chunk_files = {}
        cm_chunk_info = {}
        cm_file_to_chunk = {}
        
        chunks_list = self.metadata['chunks_cm']
        # Group chunks by file (multiple MTX files may map to same chunk file)
        chunks_by_file = {}
        for chunk_info in chunks_list:
            file_name = chunk_info['file']
            if file_name not in chunks_by_file:
                chunks_by_file[file_name] = []
            chunks_by_file[file_name].append(chunk_info)
        
        # Build mapping: use chunk_num from first entry for each file
        # With max_rows=256 for column-major files, each file maps to its own chunk
        for file_name, file_chunks in chunks_by_file.items():
            chunk_num = file_chunks[0]['chunk_num']
            file_path = os.path.join(self.dir_path, "X_CM", file_name)
            start_row = min(c['start_row'] for c in file_chunks)
            end_row = max(c['end_row'] for c in file_chunks)
            
            cm_chunk_files[chunk_num] = file_path
            cm_chunk_info[chunk_num] = {
                'chunk_num': chunk_num,
                'file': file_name,
                'start_row': start_row,
                'end_row': end_row
            }
            cm_file_to_chunk[file_path] = chunk_num
        
        # Use pre-computed ranges from metadata if available (faster, computed during build)
        # Otherwise compute them (for backward compatibility with old metadata)
        if 'cm_chunk_ranges' in self.metadata:
            # Read pre-computed sorted ranges from metadata (lists are stored as lists in JSON)
            cm_chunk_ranges = [tuple(r) for r in self.metadata['cm_chunk_ranges']]
        else:
            # Fallback: compute ranges (for old metadata files)
            cm_chunk_ranges = []
            for chunk_num, chunk_info in cm_chunk_info.items():
                cm_chunk_ranges.append((chunk_info['start_row'], chunk_info['end_row'], chunk_num))
            cm_chunk_ranges.sort(key=lambda x: x[0])
        
        # Cache the results
        self._cm_chunk_files = cm_chunk_files
        self._cm_chunk_info = cm_chunk_info
        self._cm_file_to_chunk = cm_file_to_chunk
        self._cm_chunk_ranges = cm_chunk_ranges
        
        return cm_chunk_files, cm_chunk_info, cm_file_to_chunk, cm_chunk_ranges
    
    def read_cols_cm(
        self, 
        global_cols: int | np.integer | Sequence[int] | Sequence[str] | NDArray[np.integer] | NDArray[np.bool_] | slice | str
    ) -> list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]:
        """\
        Read columns (genes) from X_CM (column-major) files.
        
        This method reads genes from the column-major (X_CM) chunk files. In X_CM
        files, rows represent genes (columns in the original matrix). Indices are
        automatically normalized (sorted and deduplicated) for efficient chunk access.
        
        Parameters
        ----------
        global_cols
            Column (gene) index or indices (0-based).
            Supported types:
            - int: Single column index (supports negative indices)
            - str: Single gene name (requires gene names in var.parquet)
            - slice: Column slice (supports integer or string bounds)
            - list[int]: List of column indices
            - list[str]: List of gene names
            - numpy.ndarray[int]: Array of column indices
            - numpy.ndarray[bool]: Boolean mask (length must match ncols)
        
        Returns
        -------
        list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]
            List of (global_col_id, rows, vals) tuples in normalized order.
            Each tuple contains:
            - global_col_id: The global column (gene) index
            - rows: numpy array of row (cell) indices (uint32)
            - vals: numpy array of values (uint16)
            Note: Results are in sorted order, not the original query order.
        
        Raises
        ------
        IndexError
            If any column index is out of bounds or gene name not found.
        ValueError
            If boolean mask length doesn't match ncols or gene names not available.
        MemoryError
            If estimated memory requirements exceed 80% of available system memory.
        FileNotFoundError
            If X_CM directory doesn't exist.
        UserWarning
            If nnz values are missing and memory estimation cannot be performed.
        
        Examples
        --------
        >>> zdata = ZData("dataset")
        >>> # Read by gene name
        >>> cols = zdata.read_cols_cm('GAPDH')
        >>> # Read multiple genes
        >>> cols = zdata.read_cols_cm(['GAPDH', 'PCNA', 'COL1A1'])
        >>> # Read by index
        >>> cols = zdata.read_cols_cm([0, 100, 200])
        >>> # Read slice of gene names
        >>> cols = zdata.read_cols_cm(slice('GAPDH', 'PCNA'))
        >>> # Access column data
        >>> for col_id, rows, vals in cols:
        ...     print(f"Gene {col_id}: {len(rows)} non-zero cells")
        """
        gene_names = None
        if hasattr(self, '_var_df') and 'gene' in self._var_df.columns:
            gene_names = pd.Index(self._var_df['gene'])
        
        global_cols = normalize_column_indices(global_cols, self.ncols, gene_names)
        
        if not global_cols:
            raise ValueError("Empty selection: no columns selected")
        
        self._check_memory_requirements(column_indices=global_cols)
        cm_chunk_files, cm_chunk_info, cm_file_to_chunk, cm_chunk_ranges = self._build_cm_chunk_mapping()
        
        # Group genes by file: all genes from the same chunk file are grouped together
        # This ensures that genes from the same block are processed by a single worker
        cols_by_file = defaultdict(list)
        
        # Use binary search for O(log n) chunk lookup
        # Optimize: cache float('inf') to avoid repeated creation
        inf_val = float('inf')
        for idx, global_col in enumerate(global_cols):
            # Binary search: find insertion point for (global_col, ...) in sorted list
            pos = bisect.bisect_right(cm_chunk_ranges, (global_col, inf_val, 0))
            if pos == 0:
                raise IndexError(f"Column {global_col} is beyond available data (no chunk found containing this gene)")
            
            # Check the range at pos-1 (the rightmost range with start_row <= global_col)
            start_row, end_row, cnum = cm_chunk_ranges[pos - 1]
            if global_col >= end_row:
                raise IndexError(f"Column {global_col} is beyond available data (no chunk found containing this gene)")
            
            # Group by file path - all genes from same file will be processed together
            cols_by_file[cm_chunk_files[cnum]].append((global_col - start_row, idx, global_col))
        
        all_results = [None] * len(global_cols)
        
        # Determine number of unique files that need to be processed
        num_files = len(cols_by_file)
        
        # Determine optimal number of workers
        max_workers = settings.max_workers
        if max_workers is None:
            max_workers = min(num_files, min(8, (os.cpu_count() or 1)))
        else:
            max_workers = min(max_workers, num_files)
        
        if num_files > 1 and max_workers > 1:
            # Batch files to reduce subprocess startup overhead
            # Each worker processes multiple files sequentially (up to 10 files per batch)
            files_per_batch = max(1, min(10, num_files // max_workers + 1))
            file_items = list(cols_by_file.items())
            file_batches = [file_items[i:i + files_per_batch] for i in range(0, len(file_items), files_per_batch)]
            
            def process_batch(batch: list[tuple[str, list[tuple[int, int, int]]]]) -> dict[str, tuple[int, list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]], int, list[tuple[int, int, int]]]]:
                """Process a batch of files sequentially in a single worker thread."""
                results = {}
                for file_path, col_info_list in batch:
                    _, chunk_num, file_results, file_ncols, col_info_list_sorted = self._process_file_for_cols(
                        file_path, col_info_list, cm_file_to_chunk
                    )
                    results[file_path] = (chunk_num, file_results, file_ncols, col_info_list_sorted)
                return results
            
            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_batch, batch): batch for batch in file_batches}
                
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        for file_path, (chunk_num, file_results, file_ncols, col_info_list_sorted) in batch_results.items():
                            for (local_row, orig_idx, global_col), (returned_local_row, rows, vals) in zip(col_info_list_sorted, file_results):
                                if returned_local_row != local_row:
                                    raise ValueError(f"Row mismatch: expected {local_row}, got {returned_local_row}")
                                all_results[orig_idx] = (global_col, rows, vals)
                    except Exception as e:
                        raise RuntimeError(f"Error processing batch: {e}") from e
        else:
            # Sequential processing (single file or max_workers <= 1)
            for file_path, col_info_list in cols_by_file.items():
                chunk_num = cm_file_to_chunk[file_path]
                
                col_info_list_sorted = sorted(col_info_list, key=lambda x: x[0])
                local_rows = [info[0] for info in col_info_list_sorted]
                
                file_ncols, file_results = self._read_rows_from_file(file_path, local_rows, self.block_columns)
                
                for (local_row, orig_idx, global_col), (returned_local_row, rows, vals) in zip(col_info_list_sorted, file_results):
                    if returned_local_row != local_row:
                        raise ValueError(f"Row mismatch: expected local row {local_row}, got {returned_local_row}")
                    all_results[orig_idx] = (global_col, rows, vals)
        
        return all_results
    
    def read_cols_cm_csr(
        self, 
        global_cols: int | np.integer | Sequence[int] | Sequence[str] | NDArray[np.integer] | NDArray[np.bool_] | slice | str
    ) -> csr_matrix:
        """\
        Read columns (genes) from X_CM files and return as CSR matrix.
        
        This is a convenience method that calls read_cols_cm() and converts the result
        to a CSR matrix. The matrix rows correspond to genes (columns in original matrix).
        
        Parameters
        ----------
        global_cols
            Column (gene) index or indices. See read_cols_cm() for supported types.
        
        Returns
        -------
        csr_matrix
            Compressed Sparse Row matrix of shape (n_selected_genes, nrows).
            Rows correspond to genes, columns to cells.
            Rows are in sorted order (not original query order).
            Dtype is float64.
            Note: To get (n_cells, n_genes) shape, transpose the result.
        
        See Also
        --------
        read_cols_cm : Read columns as raw tuples
        
        Examples
        --------
        >>> zdata = ZData("dataset")
        >>> # Read genes as CSR matrix
        >>> csr = zdata.read_cols_cm_csr(['GAPDH', 'PCNA'])
        >>> print(f"Matrix shape: {csr.shape}")  # (2, n_cells)
        >>> # Transpose to get (n_cells, n_genes) shape
        >>> csc = csr.T.tocsc()
        """
        cols_data = self.read_cols_cm(global_cols)
        
        row_indices: list[int] = []
        col_indices: list[int] = []
        values: list[float] = []
        for csr_row_idx, (col_id, rows, vals) in enumerate(cols_data):
            for row, val in zip(rows, vals):
                row_indices.append(csr_row_idx)
                col_indices.append(int(row))
                values.append(float(val))
        
        return csr_matrix((values, (row_indices, col_indices)), shape=(len(cols_data), self.nrows))
    
    def _rows_to_csr(
        self, 
        rows_data: list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]
    ) -> csr_matrix:
        """\
        Convert rows data into a scipy.sparse.csr_matrix.
        
        This is an internal method that converts the raw row data format
        (list of tuples) into a CSR matrix.
        
        Parameters
        ----------
        rows_data
            List of (row_id, cols, vals) tuples from read_rows().
        
        Returns
        -------
        csr_matrix
            Compressed Sparse Row matrix of shape (len(rows_data), ncols).
            Dtype is float64.
        """
        row_indices: list[int] = []
        col_indices: list[int] = []
        values: list[float] = []
        for csr_row_idx, (row_id, cols, vals) in enumerate(rows_data):
            for col, val in zip(cols, vals):
                row_indices.append(csr_row_idx)
                col_indices.append(int(col))
                values.append(float(val))
        return csr_matrix((values, (row_indices, col_indices)), shape=(len(rows_data), self.ncols))
    
    @property
    def num_columns(self) -> int:
        """\
        Number of columns (genes) in the dataset.
        
        Returns
        -------
        int
            Number of columns.
        """
        return self.ncols
    
    @property
    def num_rows(self) -> int:
        """\
        Number of rows (cells) in the dataset.
        
        Returns
        -------
        int
            Number of rows.
        """
        return self.nrows
    
    @property
    def shape(self) -> tuple[int, int]:
        """\
        Shape of the dataset.
        
        Returns
        -------
        tuple[int, int]
            Shape as (nrows, ncols).
        """
        return (self.nrows, self.ncols)
    
    def get_random_rows(self, n: int, seed: int | None = None) -> list[int]:
        """\
        Get n random row indices that are valid for this dataset.
        
        Parameters
        ----------
        n
            Number of random row indices to generate.
        seed
            Random seed for reproducibility. If None, uses current random state.
        
        Returns
        -------
        list[int]
            List of n random row indices in range [0, nrows).
        
        Examples
        --------
        >>> zdata = ZData("dataset")
        >>> random_rows = zdata.get_random_rows(10, seed=42)
        >>> data = zdata.read_rows(random_rows)
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.choice(self.nrows, size=min(n, self.nrows), replace=False).tolist()
    
    def estimate_memory_requirements(
        self,
        row_indices: list[int] | None = None,
        column_indices: list[int] | None = None,
    ) -> dict[str, float]:
        """\
        Estimate memory requirements for a query based on nnz values.
        
        Requires 'nnz' columns in obs (for rows) and var (for columns) DataFrames.
        These are calculated during zdata build and stored in the parquet files.
        
        Parameters
        ----------
        row_indices
            Optional list of row indices to query. If None, estimates for all rows.
        column_indices
            Optional list of column indices to query. If None, estimates for all columns.
        
        Returns
        -------
        dict[str, float]
            Dictionary with memory estimates:
            - 'estimated_nnz': Estimated number of non-zero values
            - 'estimated_memory_mb': Estimated memory in MB (assuming float64)
            - 'estimated_memory_gb': Estimated memory in GB
            - 'has_row_nnz': Whether row nnz values are available (always True if row query)
            - 'has_column_nnz': Whether column nnz values are available (always True if column query)
        
        Raises
        ------
        ValueError
            If nnz columns are missing from obs or var DataFrames, or if indices are invalid.
        
        Examples
        --------
        >>> zdata = ZData("dataset")
        >>> # Estimate for specific rows
        >>> estimate = zdata.estimate_memory_requirements(row_indices=[0, 100, 200])
        >>> print(f"Estimated memory: {estimate['estimated_memory_mb']:.2f} MB")
        >>> # Estimate for specific genes
        >>> estimate = zdata.estimate_memory_requirements(column_indices=[0, 10, 20])
        >>> print(f"Estimated memory: {estimate['estimated_memory_gb']:.2f} GB")
        """
        has_row_nnz = False
        has_column_nnz = False
        estimated_nnz = 0
        
        if row_indices is not None:
            if not (hasattr(self, '_obs_df') and 'nnz' in self._obs_df.columns):
                raise ValueError("Row nnz values are required but not found in obs DataFrame. Please rebuild zdata with nnz tracking enabled.")
            
            has_row_nnz = True
            obs_nnz = self._obs_df.select(['nnz']).to_numpy().flatten()
            if len(obs_nnz) == 0:
                raise ValueError("obs DataFrame has no nnz values. Please rebuild zdata with nnz tracking enabled.")
            
            valid_indices = [i for i in row_indices if 0 <= i < len(obs_nnz)]
            if not valid_indices:
                raise ValueError(f"All row indices are out of bounds. Valid range: [0, {len(obs_nnz)})")
            
            estimated_nnz = int(np.sum(obs_nnz[valid_indices]))
        
        elif column_indices is not None:
            if not (hasattr(self, '_var_df') and 'nnz' in self._var_df.columns):
                raise ValueError("Column nnz values are required but not found in var DataFrame. Please rebuild zdata with nnz tracking enabled.")
            
            has_column_nnz = True
            var_nnz = self._var_df['nnz'].values
            if len(var_nnz) == 0:
                raise ValueError("var DataFrame has no nnz values. Please rebuild zdata with nnz tracking enabled.")
            
            valid_indices = [i for i in column_indices if 0 <= i < len(var_nnz)]
            if not valid_indices:
                raise ValueError(f"All column indices are out of bounds. Valid range: [0, {len(var_nnz)})")
            
            estimated_nnz = int(np.sum(var_nnz[valid_indices]))
        
        else:
            if self.nnz_total is not None:
                estimated_nnz = self.nnz_total
            else:
                raise ValueError("Total nnz is not available. Please rebuild zdata with nnz tracking enabled.")
        
        # CSR format: data (float64), indices (int32), indptr (int32)
        bytes_per_nnz = 12  # 8 bytes (float64) + 4 bytes (int32 index)
        estimated_bytes = estimated_nnz * bytes_per_nnz
        
        if row_indices is not None:
            estimated_bytes += (len(row_indices) + 1) * 4
        elif column_indices is not None:
            estimated_bytes += (len(column_indices) + 1) * 4
        
        estimated_memory_mb = estimated_bytes / (1024 * 1024)
        estimated_memory_gb = estimated_memory_mb / 1024
        
        return {
            'estimated_nnz': estimated_nnz,
            'estimated_memory_mb': estimated_memory_mb,
            'estimated_memory_gb': estimated_memory_gb,
            'has_row_nnz': has_row_nnz,
            'has_column_nnz': has_column_nnz,
        }
    
    @overload
    def __getitem__(self, key: slice) -> ad.AnnData: ...
    
    @overload
    def __getitem__(self, key: int) -> ad.AnnData: ...
    
    @overload
    def __getitem__(self, key: list[int]) -> ad.AnnData: ...
    
    @overload
    def __getitem__(self, key: NDArray[np.integer] | NDArray[np.bool_]) -> ad.AnnData: ...
    
    @overload
    def __getitem__(self, key: list[str]) -> csc_matrix: ...
    
    @overload
    def __getitem__(self, key: str) -> csc_matrix: ...
    
    @overload
    def __getitem__(self, key: slice) -> csc_matrix: ...  # For gene name slices
    
    def __getitem__(
        self, 
        key: slice | int | list[int] | list[str] | str | NDArray[np.integer] | NDArray[np.bool_]
    ) -> ad.AnnData | csc_matrix:
        """\
        Support indexing by rows (returns AnnData) or columns/genes (returns CSC matrix).
        
        This method provides convenient indexing syntax for querying rows or columns.
        The method automatically determines whether the query is row-major or column-major
        based on the key type.
        
        Parameters
        ----------
        key
            Row or column index/indices. The type determines the query mode:
            
            **Row-major queries** (returns AnnData):
            - int: Single row index (e.g., zdata[5])
            - slice: Row slice (e.g., zdata[5:10])
            - list[int]: List of row indices (e.g., zdata[[0, 5, 10]])
            - numpy.ndarray[int]: Array of row indices
            - numpy.ndarray[bool]: Boolean mask (e.g., zdata[mask])
            
            **Column-major queries** (returns CSC matrix):
            - str: Single gene name (e.g., zdata['GAPDH'])
            - list[str]: List of gene names (e.g., zdata[['GAPDH', 'PCNA']])
            - slice with string bounds: Gene name slice (e.g., zdata['GAPDH':'PCNA'])
        
        Returns
        -------
        AnnData or csc_matrix
            - For row queries: AnnData object with:
              - X: CSR matrix of shape (n_selected_rows, ncols)
              - obs: Observation metadata for selected rows
              - var: Variable metadata (all genes)
            - For column queries: CSC matrix of shape (n_cells, n_selected_genes)
              with expression values for selected genes
        
        Raises
        ------
        IndexError
            If indices are out of bounds or gene names not found.
        ValueError
            If boolean mask length doesn't match or empty selection.
        TypeError
            If key type is not supported.
        
        Notes
        -----
        - zdata uses disk-based storage, so arbitrary 2D indexing is not supported.
          You can either query rows OR columns, not both simultaneously.
        - Row queries preserve the original query order (unlike read_rows() which sorts).
        - Column queries return results in sorted order.
        - Negative indices are supported for row queries (e.g., zdata[-1] for last row).
        
        Examples
        --------
        >>> zdata = ZData("dataset")
        >>> # Row queries (return AnnData)
        >>> adata = zdata[5:10]  # Rows 5-9
        >>> adata = zdata[[0, 100, 200]]  # Specific rows
        >>> adata = zdata[-1]  # Last row
        >>> # Column queries (return CSC matrix)
        >>> matrix = zdata['GAPDH']  # Single gene
        >>> matrix = zdata[['GAPDH', 'PCNA', 'COL1A1']]  # Multiple genes
        >>> matrix = zdata['GAPDH':'PCNA']  # Gene range
        """
        is_column_query = False
        
        if isinstance(key, str):
            is_column_query = True
        elif isinstance(key, list) and len(key) > 0 and isinstance(key[0], str):
            is_column_query = True
        elif isinstance(key, slice):
            if (isinstance(key.start, str) or isinstance(key.stop, str)):
                is_column_query = True
        
        if is_column_query:
            csr_result = self.read_cols_cm_csr(key)
            return csr_result.T.tocsc()
        
        row_indices = normalize_row_indices(key, self.nrows)
        
        if not row_indices:
            raise ValueError("Empty selection: no rows selected")
        
        X_csr = self.read_rows_csr(row_indices)
        
        # Get obs data for selected rows (already sorted and deduplicated)
        if len(row_indices) == 1:
            obs_df = self.obs[row_indices[0]:row_indices[0]+1, :]
        elif row_indices:
            # Use slice if indices are consecutive, otherwise get individually
            is_consecutive = (
                len(row_indices) == row_indices[-1] - row_indices[0] + 1
                and all(row_indices[i] == row_indices[0] + i for i in range(len(row_indices)))
            )
            if is_consecutive:
                obs_df = self.obs[row_indices[0]:row_indices[-1]+1, :]
            else:
                obs_dfs = [self.obs[idx:idx+1, :] for idx in row_indices]
                obs_df = pd.concat(obs_dfs, ignore_index=True)
        else:
            obs_df = self.obs[0:0, :].copy()
        
        obs_df.index = pd.RangeIndex(start=0, stop=len(obs_df))
        
        var_df = self._var_df.copy()
        
        with warnings.catch_warnings():
            try:
                from anndata._warnings import ImplicitModificationWarning as AnnDataWarning
                warnings.filterwarnings("ignore", category=AnnDataWarning)
            except ImportError:
                warnings.filterwarnings("ignore", message=".*Transforming to str index.*")
            
            adata = ad.AnnData(
                X=X_csr,
                obs=obs_df,
                var=var_df
            )
        
        return adata
    
    def close(self):
        """Explicitly close and cleanup resources (no-op for now, kept for API compatibility)."""
        pass
