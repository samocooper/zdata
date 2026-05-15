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
    
    def gather(self, row_indices: list[int]) -> pd.DataFrame:
        """\
        Efficiently select multiple (possibly non-consecutive) rows by index.
        
        Uses Polars row selection for a single fast gather instead of
        row-by-row slicing and concatenation.
        
        Parameters
        ----------
        row_indices
            List of integer row positions to select.
        
        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with the selected rows.
        """
        polars_result = self.obs_df[pl.Series(row_indices)]
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
    
    def __init__(
        self,
        dir_name: str | Path,
        obs_index_col: str | None = None,
        var_index_col: str | None = None,
    ) -> None:
        """\
        Initialize the reader for a zdata directory.

        Parameters
        ----------
        dir_name
            Name or path of the zdata directory.
        obs_index_col
            Optional column name (integer-typed) in obs.parquet to use as
            a mapping from obs row positions to expression-matrix row indices.
            When ``None`` (default), obs rows and matrix rows are assumed to
            be aligned 1-to-1, and ``len(obs)`` must equal ``nrows``.
            When set (e.g. ``"_row_index"``), queries index into obs first,
            then the named column translates to the actual matrix row.
        var_index_col
            Same concept for var.parquet → expression-matrix column indices.
            When ``None`` (default), var rows and matrix columns are aligned.
            When set, the named column translates var positions to matrix
            column indices.

        Raises
        ------
        FileNotFoundError
            If the directory or required files are not found.
        ValueError
            If the path is not a directory, metadata is missing required fields,
            or a requested index column does not exist / has wrong type.
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
        
        # Store user-specified index mapping columns
        self._obs_index_col: str | None = obs_index_col
        self._var_index_col: str | None = var_index_col
        
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

            var_polars = pl.read_parquet(var_file)
            var_dict = var_polars.to_dict(as_series=False)
            self._var_df: pd.DataFrame = pd.DataFrame(var_dict)
            self._var_df.index = pd.RangeIndex(start=0, stop=len(self._var_df))
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet files: {e}") from e

        # --- obs index mapping ---------------------------------------------------
        obs_nrows = len(self._obs_df)
        if obs_index_col is not None:
            if obs_index_col not in self._obs_df.columns:
                raise ValueError(
                    f"obs_index_col '{obs_index_col}' not found in obs.parquet. "
                    f"Available columns: {self._obs_df.columns}"
                )
            col_dtype = self._obs_df[obs_index_col].dtype
            if col_dtype not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                raise ValueError(
                    f"obs_index_col '{obs_index_col}' must be an integer column, "
                    f"got {col_dtype}"
                )
            self._obs_row_index_map: NDArray[np.integer] | None = (
                self._obs_df[obs_index_col].to_numpy()
            )
        else:
            if obs_nrows != self.nrows:
                raise ValueError(
                    f"obs.parquet has {obs_nrows} rows but the expression matrix "
                    f"has {self.nrows} rows. Dimensions must match when no "
                    f"obs_index_col is specified. To use a mapping column, pass "
                    f"obs_index_col='<column_name>' (e.g. obs_index_col='_row_index')."
                )
            self._obs_row_index_map = None

        # --- var index mapping ---------------------------------------------------
        var_nrows = len(self._var_df)
        if var_index_col is not None:
            if var_index_col not in self._var_df.columns:
                raise ValueError(
                    f"var_index_col '{var_index_col}' not found in var.parquet. "
                    f"Available columns: {list(self._var_df.columns)}"
                )
            col = self._var_df[var_index_col]
            if not np.issubdtype(col.dtype, np.integer):
                raise ValueError(
                    f"var_index_col '{var_index_col}' must be an integer column, "
                    f"got {col.dtype}"
                )
            self._var_col_index_map: NDArray[np.integer] | None = col.to_numpy()
        else:
            if var_nrows != self.ncols:
                raise ValueError(
                    f"var.parquet has {var_nrows} rows but the expression matrix "
                    f"has {self.ncols} columns. Dimensions must match when no "
                    f"var_index_col is specified. To use a mapping column, pass "
                    f"var_index_col='<column_name>'."
                )
            self._var_col_index_map = None
        
        # Cache zdata_read executable path (resolved once at init for thread safety)
        self._zdata_read_path: str = _get_zdata_read_path()
        
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
        rows_csv = ",".join(map(str, local_rows))
        
        block_rows_val = block_rows if block_rows is not None else self.block_rows
        
        # TODO: For even better performance, replace subprocess with ctypes/cffi
        # to call C decompression in-process (avoids ~5-20ms subprocess overhead per call).
        # This would require restructuring zdata_read.c into a shared library.
        cmd_full = [self._zdata_read_path, "--binary", "--block-rows", str(block_rows_val), str(file_path), rows_csv]
        blob = subprocess.check_output(
            cmd_full,
            bufsize=262144,
            stderr=subprocess.DEVNULL
        )
        
        if len(blob) < 12:
            raise ValueError(f"Output too small: {len(blob)} bytes")

        # Unpack header: nreq, ncols, version
        nreq, ncols, version = struct.unpack_from("<III", blob, 0)
        if nreq != len(local_rows):
            raise ValueError(
                f"zdata_read returned {nreq} rows but {len(local_rows)} were requested "
                f"for {file_path}. This usually indicates the C binary truncated the "
                f"row list — rebuild zdata_read with the latest source."
            )

        # Map version number to numpy dtype
        _VERSION_DTYPE = {
            2:  (np.uint16,  2),
            3:  (np.float32, 4),
            4:  (np.uint8,   1),
            5:  (np.uint32,  4),
            6:  (np.uint64,  8),
            7:  (np.int8,    1),
            8:  (np.int16,   2),
            9:  (np.int32,   4),
            10: (np.int64,   8),
            11: (np.float64, 8),
        }
        if version in _VERSION_DTYPE:
            val_dtype, val_bytes = _VERSION_DTYPE[version]
        else:
            val_dtype = np.uint16
            val_bytes = 2

        out = [None] * nreq
        blob_mv = memoryview(blob)
        off = 12

        # Process rows - optimize by avoiding repeated array creation for empty rows
        empty_cols = np.array([], dtype=np.uint32)
        empty_vals = np.array([], dtype=val_dtype)

        for i in range(nreq):
            row_id, nnz = struct.unpack_from("<II", blob_mv, off)
            off += 8

            if nnz > 0:
                cols = np.frombuffer(blob_mv, dtype=np.uint32, count=nnz, offset=off).copy()
                off += nnz * 4
                vals = np.frombuffer(blob_mv, dtype=val_dtype, count=nnz, offset=off).copy()
                off += nnz * val_bytes
                out[i] = (row_id, cols, vals)
            else:
                out[i] = (row_id, empty_cols, empty_vals)

        return ncols, out
    
    def _process_file(
        self,
        file_path: str,
        info_list: list[tuple[int, int, int]],
        file_to_chunk: dict[str, int],
        block_size: int,
    ) -> tuple[list[tuple[int, int, int]], list, int]:
        """
        Process a single chunk file: sort by local row, read, return results.

        Used by both row-major and column-major read paths.

        Returns:
            (info_list_sorted, file_results, file_ncols)
        """
        info_list_sorted = sorted(info_list, key=lambda x: x[0])
        local_rows = [t[0] for t in info_list_sorted]
        file_ncols, file_results = self._read_rows_from_file(file_path, local_rows, block_size)
        return info_list_sorted, file_results, file_ncols

    def _dispatch_file_reads(
        self,
        items_by_file: dict[str, list[tuple[int, int, int]]],
        file_to_chunk: dict[str, int],
        block_size: int,
        all_results: list,
        check_ncols: bool = True,
    ) -> None:
        """
        Shared parallel/sequential file dispatch for both read_rows and read_cols_cm.

        Reads from chunk files, fills *all_results* in-place.

        Parameters
        ----------
        items_by_file
            ``{file_path: [(local_row, orig_idx, global_id), ...]}``
        file_to_chunk
            Mapping from file path to chunk number.
        block_size
            ``block_rows`` or ``block_columns``.
        all_results
            Pre-allocated list; results written at ``orig_idx`` positions.
        check_ncols
            Whether to validate that file ncols matches self.ncols.
            Set to False for column-major reads (transposed semantics).
        """
        num_files = len(items_by_file)
        max_workers = settings.max_workers
        if max_workers is None:
            max_workers = min(num_files, min(8, (os.cpu_count() or 1)))
        else:
            max_workers = min(max_workers, num_files)

        def _assemble(info_sorted, file_results, file_ncols, file_path):
            if check_ncols and self.ncols != file_ncols:
                raise ValueError(f"Inconsistent ncols: {self.ncols} vs {file_ncols} in {file_path}")
            for info_tuple, (returned_local_row, data_cols, data_vals) in zip(info_sorted, file_results):
                local_row, orig_idx, global_id = info_tuple
                if returned_local_row != local_row:
                    raise ValueError(f"Row mismatch: expected {local_row}, got {returned_local_row}")
                all_results[orig_idx] = (global_id, data_cols, data_vals)

        if num_files > 1 and max_workers > 1:
            files_per_batch = max(1, min(10, num_files // max_workers + 1))
            file_items = list(items_by_file.items())
            file_batches = [file_items[i:i + files_per_batch] for i in range(0, len(file_items), files_per_batch)]

            def process_batch(batch):
                results = {}
                for fp, info_list in batch:
                    results[fp] = self._process_file(fp, info_list, file_to_chunk, block_size)
                return results

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_batch, b): b for b in file_batches}
                for future in as_completed(futures):
                    batch_results = future.result()
                    for fp, (info_sorted, file_results, file_ncols) in batch_results.items():
                        _assemble(info_sorted, file_results, file_ncols, fp)
        else:
            for fp, info_list in items_by_file.items():
                info_sorted, file_results, file_ncols = self._process_file(
                    fp, info_list, file_to_chunk, block_size
                )
                _assemble(info_sorted, file_results, file_ncols, fp)

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
        if self._obs_row_index_map is not None:
            # Indices are into obs; translate via the mapping column to matrix rows
            obs_indices = normalize_row_indices(global_rows, len(self._obs_df))
            self._check_memory_requirements(row_indices=obs_indices)
            global_rows = sorted(set(int(self._obs_row_index_map[i]) for i in obs_indices))
        else:
            # Direct: indices are matrix rows
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
        self._dispatch_file_reads(
            rows_by_file, self.file_to_chunk, self.block_rows, all_results
        )

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

        # When var_index_col is set, normalize against var length then map
        n_queryable_cols = len(self._var_df) if self._var_col_index_map is not None else self.ncols
        global_cols = normalize_column_indices(global_cols, n_queryable_cols, gene_names)

        if not global_cols:
            raise ValueError("Empty selection: no columns selected")

        # Translate var-positional indices to actual matrix column indices
        if self._var_col_index_map is not None:
            global_cols = sorted(set(int(self._var_col_index_map[c]) for c in global_cols))
        
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
        
        self._dispatch_file_reads(
            cols_by_file, cm_file_to_chunk, self.block_columns, all_results,
            check_ncols=False,
        )

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
        
        if not cols_data:
            return csr_matrix((0, self.nrows), dtype=np.float64)
        
        counts = np.array([len(rows) for _, rows, _ in cols_data], dtype=np.intp)
        all_rows = np.repeat(np.arange(len(cols_data), dtype=np.int32), counts)
        all_cols = np.concatenate([rows for _, rows, _ in cols_data]) if counts.sum() > 0 else np.array([], dtype=np.uint32)
        all_vals = np.concatenate([vals for _, _, vals in cols_data]).astype(np.float64) if counts.sum() > 0 else np.array([], dtype=np.float64)
        
        return csr_matrix((all_vals, (all_rows, all_cols)), shape=(len(cols_data), self.nrows))
    
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
        if not rows_data:
            return csr_matrix((0, self.ncols), dtype=np.float64)
        
        counts = np.array([len(cols) for _, cols, _ in rows_data], dtype=np.intp)
        all_rows = np.repeat(np.arange(len(rows_data), dtype=np.int32), counts)
        all_cols = np.concatenate([cols for _, cols, _ in rows_data]) if counts.sum() > 0 else np.array([], dtype=np.uint32)
        all_vals = np.concatenate([vals for _, _, vals in rows_data]).astype(np.float64) if counts.sum() > 0 else np.array([], dtype=np.float64)
        return csr_matrix((all_vals, (all_rows, all_cols)), shape=(len(rows_data), self.ncols))
    
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
        rng = np.random.default_rng(seed)
        return rng.choice(self.nrows, size=min(n, self.nrows), replace=False).tolist()
    
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
    
    def _resolve_gene_names_to_matrix_cols(self, names: Sequence[str]) -> list[int]:
        """Map gene names to matrix column indices, honouring var_index_col."""
        gene_to_var_pos = {g: i for i, g in enumerate(self._var_df["gene"])}
        try:
            var_positions = [gene_to_var_pos[n] for n in names]
        except KeyError as e:
            raise IndexError(f"Gene name not found in var: {e.args[0]}") from None
        if self._var_col_index_map is not None:
            return [int(self._var_col_index_map[p]) for p in var_positions]
        return var_positions

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
        - Column queries by gene name (or list of names) preserve the input order;
          duplicate gene names are returned as duplicate columns. Slice queries
          return columns in slice order.
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
            csc_result = csr_result.T

            # read_cols_cm_csr sorts and dedupes for efficient chunk access, so
            # the output columns are in sorted-by-matrix-index order. For
            # gene-name inputs the user expects columns back in the order they
            # asked, with duplicates preserved -- restore that here.
            if isinstance(key, str) or (
                isinstance(key, list) and len(key) > 0 and isinstance(key[0], str)
            ):
                names = [key] if isinstance(key, str) else key
                input_matrix_cols = self._resolve_gene_names_to_matrix_cols(names)
                sorted_unique = sorted(set(input_matrix_cols))
                sorted_pos = {c: i for i, c in enumerate(sorted_unique)}
                permutation = [sorted_pos[c] for c in input_matrix_cols]
                csc_result = csc_result[:, permutation]

            # If obs uses a mapping column, filter to only the obs-represented rows
            if self._obs_row_index_map is not None:
                csr_subset = csc_result.tocsr()[self._obs_row_index_map, :]
                csc_result = csr_subset.tocsc()

            return csc_result
        
        # When obs_index_col is set, indices are into obs (not matrix rows)
        n_queryable = len(self._obs_df) if self._obs_row_index_map is not None else self.nrows
        row_indices = normalize_row_indices(key, n_queryable)

        if not row_indices:
            raise ValueError("Empty selection: no rows selected")

        X_csr = self.read_rows_csr(row_indices)
        
        # Get obs data for selected rows (already sorted and deduplicated)
        if len(row_indices) == 1:
            obs_df = self.obs[row_indices[0]:row_indices[0]+1, :]
        elif row_indices:
            # Use slice if indices are consecutive, otherwise use gather
            is_consecutive = (
                len(row_indices) == row_indices[-1] - row_indices[0] + 1
                and all(row_indices[i] == row_indices[0] + i for i in range(len(row_indices)))
            )
            if is_consecutive:
                obs_df = self.obs[row_indices[0]:row_indices[-1]+1, :]
            else:
                obs_df = self.obs.gather(row_indices)
        else:
            obs_df = self.obs[0:0, :]
        
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
    
    def __enter__(self):
        """Support use as context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close resources when exiting context manager."""
        self.close()
        return False
    
    @property
    def var(self) -> pd.DataFrame:
        """\
        Access the variable (gene) metadata DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing gene metadata (gene names, nnz counts, etc.).
        """
        return self._var_df
    
    def __repr__(self) -> str:
        nnz_str = f", nnz={self.nnz_total}" if self.nnz_total is not None else ""
        return f"ZData('{self.dir_path}', shape={self.shape}{nnz_str})"
