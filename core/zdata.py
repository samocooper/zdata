from __future__ import annotations

import json
import os
import struct
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, overload

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csc_matrix, csr_matrix

from .._settings import settings
from .index import normalize_column_indices, normalize_row_indices

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

def _normalize_indices(
    indices: int | np.integer | Sequence[int] | NDArray[np.integer],
) -> list[int]:
    """
    Normalize indices to list of ints (handles int or list/array).
    
    This is a simple wrapper for backward compatibility.
    For new code, use normalize_row_indices() or normalize_column_indices().
    """
    if isinstance(indices, (int, np.integer)):
        return [int(indices)]
    return [int(i) for i in indices]


class ObsWrapper:
    """
    Wrapper for polars obs DataFrame that supports indexing like obs[row_index, :].
    Returns pandas DataFrames for compatibility with anndata/AnnData.
    
    Usage:
        sc_atlas.obs[5, :]  # Returns pandas DataFrame for row 5
        sc_atlas.obs[0:10, :]  # Returns pandas DataFrame for rows 0-9
    """
    
    def __init__(self, obs_df: PolarsDataFrame) -> None:
        """
        Initialize the wrapper with a polars DataFrame.
        
        Args:
            obs_df: polars DataFrame containing obs/metadata
        """
        self.obs_df: PolarsDataFrame = obs_df
    
    def __getitem__(self, key: tuple[int | slice, slice]) -> pd.DataFrame:
        """
        Support indexing like obs[row_index, :] or obs[slice, :].
        
        Args:
            key: Tuple of (row_index or slice, column_slice)
                 Currently only supports [row_index, :] or [slice, :]
        
        Returns:
            pandas DataFrame with selected rows
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
    """
    Efficient reader for zdata directory structure containing .bin files.
    
    Provides methods to read random sets of rows from the compressed sparse matrix data.
    """
    
    def __init__(self, dir_name: str | Path) -> None:
        """
        Initialize the reader for a zdata directory.
        
        Args:
            dir_name: Name or path of the zdata directory (e.g., "andrews" or "/path/to/andrews")
        """
        self.dir_name: str | Path = dir_name
        # Use dir_name directly as path (no .zdata suffix appended)
        self.dir_path: str | Path = dir_name
        
        if not os.path.exists(self.dir_path):
            raise FileNotFoundError(f"Directory not found: {self.dir_path}")
        
        if not os.path.isdir(self.dir_path):
            raise ValueError(f"Path is not a directory: {self.dir_path}")
        
        # Load metadata (required)
        metadata_file = os.path.join(self.dir_path, "metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}. "
                f"Please rebuild the zdata directory using build_zdata()"
            )
        
        with open(metadata_file, 'r') as f:
            self.metadata: dict[str, Any] = json.load(f)
        
        # Extract metadata (new format only)
        self.nrows: int = self.metadata['shape'][0]
        self.ncols: int = self.metadata['shape'][1]
        self.nnz_total: int | None = self.metadata.get('nnz_total', None)
        # Use RM chunks for row-major operations (required)
        if 'num_chunks_rm' not in self.metadata:
            raise ValueError("Metadata missing 'num_chunks_rm' field (new format required)")
        self.num_chunks: int = self.metadata['num_chunks_rm']
        self.total_blocks: int | None = self.metadata.get('total_blocks_rm', None)
        
        # Extract block configuration from metadata
        # Use settings defaults if not in metadata
        self.block_rows: int = self.metadata.get('block_rows', settings.block_rows)
        self.max_rows_per_chunk: int = self.metadata.get(
            'max_rows_per_chunk', settings.max_rows_per_chunk
        )
        
        # Build chunk_files dict and chunk info from metadata (new format only)
        self.chunk_files: dict[int, str] = {}
        self.chunk_info: dict[int, dict[str, Any]] = {}  # Store chunk metadata for validation
        self.file_to_chunk: dict[str, int] = {}  # Reverse mapping: file_path -> chunk_num (for O(1) lookup)
        
        # Use new format with separate RM and CM chunks (required)
        if 'chunks_rm' not in self.metadata:
            raise ValueError("Metadata missing 'chunks_rm' field (new format required)")
        chunks_list = self.metadata['chunks_rm']
        subdir = "X_RM"
        
        for chunk_info in chunks_list:
            chunk_num = chunk_info['chunk_num']
            # .bin files are stored in subdirectory (X_RM or X_CM)
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
            
            # Build gene name to index mapping for fast lookups
            if 'gene' in self._var_df.columns:
                self._gene_to_idx: dict[str, int] = {gene: idx for idx, gene in enumerate(self._var_df['gene'])}
            else:
                self._gene_to_idx: dict[str, int] = {}
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet files: {e}") from e
    
    @property
    def obs(self) -> ObsWrapper:
        """
        Access obs/metadata DataFrame.
        
        Returns:
            ObsWrapper that supports indexing like obs[row_index, :]
        
        """
        return self._obs_wrapper
    
    
    def _read_rows_from_file(
        self, 
        file_path: str | Path, 
        local_rows: list[int]
    ) -> tuple[int, list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]]:
        """
        Read rows from a single .bin file. Rows are local indices within that file.
        
        Args:
            file_path: Path to the .bin file
            local_rows: List of local row indices (0-based within the chunk)
        
        Returns:
            (ncols, results) where results is a list of (local_row_id, cols, vals) tuples
        """
        rows_csv = ",".join(map(str, local_rows))
        # Use the absolute path to zdata_read
        bin_path = _get_zdata_read_path()
        blob = subprocess.check_output([bin_path, "--binary", file_path, rows_csv])
        
        off = 0
        if len(blob) < 8:
            raise ValueError(f"Output too small: {len(blob)} bytes")
        
        nreq, ncols = struct.unpack_from("<II", blob, off); off += 8
        
        out = []
        for i in range(nreq):
            if off + 8 > len(blob):
                raise ValueError(f"Truncated before row header {i}: off={off}, len={len(blob)}")
            
            row_id, nnz = struct.unpack_from("<II", blob, off); off += 8
            
            need = off + nnz * 4 + nnz * 2  # indices (uint32) + data (uint16)
            if need > len(blob):
                raise ValueError(
                    f"Truncated row {i} (row_id={row_id}, nnz={nnz}). "
                    f"need={need}, len={len(blob)}, off={off}"
                )
            
            cols = np.frombuffer(blob, dtype=np.uint32, count=nnz, offset=off)
            off += nnz * 4
            vals = np.frombuffer(blob, dtype=np.uint16, count=nnz, offset=off)
            off += nnz * 2
            
            out.append((row_id, cols, vals))
        
        return ncols, out
    
    def read_rows(
        self, 
        global_rows: int | np.integer | Sequence[int] | NDArray[np.integer]
    ) -> list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]:
        """
        Read rows using global indices that span across multiple .bin files.
        
        Supports:
        - Integer indices (including negative)
        - Slices (e.g., slice(0, 100))
        - Lists/arrays of integers
        - Boolean arrays (length must match nrows)
        
        Args:
            global_rows: Row index or indices (0-based, relative to full MTX file).
                        Can be int, slice, list[int], numpy array (int or bool).
        
        Returns:
            List of (global_row_id, cols, vals) tuples in the same order as global_rows
            where cols and vals are numpy arrays
        """
        # Normalize indices using the new indexing system
        global_rows = normalize_row_indices(global_rows, self.nrows)
        
        # Warn on large queries if enabled
        if settings.warn_on_large_queries and len(global_rows) > settings.large_query_threshold:
            import warnings
            warnings.warn(
                f"Querying {len(global_rows)} rows, which exceeds the threshold "
                f"of {settings.large_query_threshold}. This may be slow. "
                f"Consider using smaller batches or disable this warning with "
                f"zdata.settings.warn_on_large_queries = False",
                UserWarning
            )
        
        # Group global rows by which chunk file they belong to
        rows_by_file = defaultdict(list)
        
        for idx, global_row in enumerate(global_rows):
            if global_row < 0:
                raise ValueError(f"Row index must be non-negative, got {global_row}")
            
            if global_row >= self.nrows:
                raise IndexError(f"Row {global_row} is beyond available data (max row: {self.nrows - 1})")
            
            chunk_num = global_row // self.max_rows_per_chunk
            local_row = global_row % self.max_rows_per_chunk
            
            if chunk_num not in self.chunk_files:
                raise IndexError(f"Row {global_row} is beyond available data (chunk {chunk_num} not found)")
            
            rows_by_file[self.chunk_files[chunk_num]].append((local_row, idx, global_row))
        
        # Read from each file and collect results
        all_results = [None] * len(global_rows)
        
        for file_path, row_info_list in rows_by_file.items():
            chunk_num = self.file_to_chunk[file_path]  # Guaranteed to exist
            
            row_info_list_sorted = sorted(row_info_list, key=lambda x: x[0])
            local_rows = [info[0] for info in row_info_list_sorted]
            
            file_ncols, file_results = self._read_rows_from_file(file_path, local_rows)
            if self.ncols != file_ncols:
                raise ValueError(f"Inconsistent ncols: {self.ncols} vs {file_ncols} in {file_path}")
            
            for (local_row, orig_idx, global_row), (returned_local_row, cols, vals) in zip(row_info_list_sorted, file_results):
                if returned_local_row != local_row:
                    raise ValueError(f"Row mismatch: expected local row {local_row}, got {returned_local_row}")
                all_results[orig_idx] = (global_row, cols, vals)
        
        return all_results
    
    def read_rows_csr(
        self, 
        global_rows: int | np.integer | Sequence[int] | NDArray[np.integer]
    ) -> csr_matrix:
        """
        Read rows and return as a scipy.sparse.csr_matrix.
        
        Args:
            global_rows: List or array of global row indices (0-based)
        
        Returns:
            scipy.sparse.csr_matrix of shape (len(global_rows), ncols)
        """
        rows_data = self.read_rows(global_rows)
        return self._rows_to_csr(rows_data)
    
    def read_rows_rm(
        self, 
        global_rows: int | np.integer | Sequence[int] | NDArray[np.integer] | NDArray[np.bool_] | slice
    ) -> list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]:
        """
        Read rows from X_RM (row-major) files.
        This is an alias for read_rows() for clarity.
        
        Args:
            global_rows: List or array of global row indices (0-based)
        
        Returns:
            List of (global_row_id, cols, vals) tuples
        """
        return self.read_rows(global_rows)
    
    def read_rows_rm_csr(
        self, 
        global_rows: int | np.integer | Sequence[int] | NDArray[np.integer] | NDArray[np.bool_] | slice
    ) -> csr_matrix:
        """
        Read rows from X_RM (row-major) files and return as CSR matrix.
        
        Args:
            global_rows: List or array of global row indices (0-based)
        
        Returns:
            scipy.sparse.csr_matrix of shape (len(global_rows), ncols)
        """
        return self.read_rows_csr(global_rows)
    
    def _build_cm_chunk_mapping(
        self
    ) -> tuple[dict[int, str], dict[int, dict[str, Any]], dict[str, int]]:
        """
        Build chunk mapping for X_CM (column-major) files from metadata.
        
        Returns:
            (cm_chunk_files, cm_chunk_info, cm_file_to_chunk) tuple
        """
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
            cm_chunk_files[chunk_num] = file_path
            cm_chunk_info[chunk_num] = {
                'chunk_num': chunk_num,
                'file': file_name,
                'start_row': min(c['start_row'] for c in file_chunks),
                'end_row': max(c['end_row'] for c in file_chunks)
            }
            cm_file_to_chunk[file_path] = chunk_num
        
        return cm_chunk_files, cm_chunk_info, cm_file_to_chunk
    
    def read_cols_cm(
        self, 
        global_cols: int | np.integer | Sequence[int] | Sequence[str] | NDArray[np.integer] | NDArray[np.bool_] | slice | str
    ) -> list[tuple[int, NDArray[np.uint32], NDArray[np.uint16]]]:
        """
        Read columns (genes) from X_CM (column-major) files.
        In X_CM files, rows represent genes (columns in original matrix).
        
        Supports:
        - Integer indices (including negative)
        - Gene names (strings) - requires gene names to be loaded
        - Slices (e.g., slice(0, 100) or slice('GAPDH', 'PCNA'))
        - Lists/arrays of integers or gene names
        - Boolean arrays (length must match ncols)
        
        Args:
            global_cols: Column (gene) index or indices (0-based).
                        Can be int, str, slice, list[int|str], numpy array (int or bool).
        
        Returns:
            List of (global_col_id, rows, vals) tuples where rows are cell indices
        """
        # Get gene names if available
        gene_names = None
        if hasattr(self, '_var_df') and 'gene' in self._var_df.columns:
            gene_names = pd.Index(self._var_df['gene'])
        
        # Normalize indices using the new indexing system
        global_cols = normalize_column_indices(global_cols, self.ncols, gene_names)
        
        # Build X_CM chunk mapping
        cm_chunk_files, cm_chunk_info, cm_file_to_chunk = self._build_cm_chunk_mapping()
        
        # In X_CM, rows = genes, so we treat column indices as row indices
        # The number of rows in X_CM equals the number of columns in the original matrix
        cm_nrows = self.ncols  # Number of genes (rows in X_CM)
        
        # Group columns (genes) by which chunk file they belong to
        cols_by_file = defaultdict(list)
        
        for idx, global_col in enumerate(global_cols):
            if global_col < 0:
                raise ValueError(f"Column index must be non-negative, got {global_col}")
            
            if global_col >= cm_nrows:
                raise IndexError(f"Column {global_col} is beyond available data (max column: {cm_nrows - 1})")
            
            # Find which chunk contains this gene by checking chunk ranges
            # With max_rows=256 for column-major files, each 256-gene MTX file maps to its own chunk
            chunk_num = None
            local_row = None
            for cnum, chunk_info in cm_chunk_info.items():
                chunk_start = chunk_info['start_row']  # Start of the chunk file
                chunk_end = chunk_info['end_row']      # End of the chunk file
                
                # Check if gene is within this chunk's range
                if chunk_start <= global_col < chunk_end:
                    chunk_num = cnum
                    # Calculate local row: offset from the start of the chunk file
                    local_row = global_col - chunk_start
                    break
            
            if chunk_num is None:
                raise IndexError(f"Column {global_col} is beyond available data (no chunk found containing this gene)")
            
            cols_by_file[cm_chunk_files[chunk_num]].append((local_row, idx, global_col))
        
        # Read from each file and collect results
        all_results = [None] * len(global_cols)
        
        for file_path, col_info_list in cols_by_file.items():
            chunk_num = cm_file_to_chunk[file_path]  # Guaranteed to exist
            
            col_info_list_sorted = sorted(col_info_list, key=lambda x: x[0])
            local_rows = [info[0] for info in col_info_list_sorted]
            
            file_ncols, file_results = self._read_rows_from_file(file_path, local_rows)
            
            for (local_row, orig_idx, global_col), (returned_local_row, rows, vals) in zip(col_info_list_sorted, file_results):
                if returned_local_row != local_row:
                    raise ValueError(f"Row mismatch: expected local row {local_row}, got {returned_local_row}")
                all_results[orig_idx] = (global_col, rows, vals)
        
        return all_results
    
    def read_cols_cm_csr(
        self, 
        global_cols: int | np.integer | Sequence[int] | Sequence[str] | NDArray[np.integer] | NDArray[np.bool_] | slice | str
    ) -> csr_matrix:
        """Read columns (genes) from X_CM files and return as CSR matrix."""
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
        """Convert rows data into a scipy.sparse.csr_matrix."""
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
        return self.ncols
    
    @property
    def num_rows(self) -> int:
        return self.nrows
    
    @property
    def shape(self) -> tuple[int, int]:
        return (self.nrows, self.ncols)
    
    def get_random_rows(self, n: int, seed: int | None = None) -> list[int]:
        """Get n random row indices that are valid for this dataset."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.randint(0, self.nrows, size=n).tolist()
    
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
        """
        Support indexing by rows (returns AnnData) or columns/genes (returns CSC matrix).
        
        Row-major queries (returns AnnData):
        - Integer: zdata[5] -> AnnData with single row
        - Slice: zdata[5:10] -> AnnData with rows 5-9
        - List of integers: zdata[[0, 5, 10]] -> AnnData with specified rows
        - Boolean array: zdata[mask] -> AnnData with rows where mask is True
        - Negative indices: zdata[-1] -> Last row
        
        Column-major queries (returns CSC matrix):
        - Gene name: zdata['GAPDH'] -> CSC matrix with single gene
        - List of gene names: zdata[['GAPDH', 'PCNA']] -> CSC matrix with specified genes
        - Slice of gene names: zdata['GAPDH':'PCNA'] -> CSC matrix with genes in range
        - Integer column index: zdata[0] when used as column query (not recommended)
        
        Note: zdata uses disk-based storage, so arbitrary 2D indexing is not supported.
        You can either query rows OR columns, not both simultaneously.
        
        Args:
            key: Row index/indices or column (gene) index/indices
        
        Returns:
            - For row queries: AnnData object with X (CSR), obs, and var
            - For column queries: CSC matrix of shape (n_cells, n_genes) with expression values
        """
        # Determine if this is a row query or column query
        # Strategy: if key is a string or list of strings, it's a column query
        # Otherwise, it's a row query
        
        is_column_query = False
        
        if isinstance(key, str):
            is_column_query = True
        elif isinstance(key, list) and len(key) > 0 and isinstance(key[0], str):
            is_column_query = True
        elif isinstance(key, slice):
            # For slices, check if start/stop are strings (gene names)
            if (isinstance(key.start, str) or isinstance(key.stop, str)):
                is_column_query = True
        
        if is_column_query:
            # Column-major query: return CSC matrix
            # Use read_cols_cm which handles all column indexing patterns
            csr_result = self.read_cols_cm_csr(key)
            return csr_result.T.tocsc()  # Transpose CSR to CSC
        
        # Row-major query: return AnnData object
        # Normalize row indices
        row_indices = normalize_row_indices(key, self.nrows)
        
        if not row_indices:
            raise ValueError("Empty selection: no rows selected")
        
        # Read rows and create AnnData
        X_csr = self.read_rows_csr(row_indices)
        
        # Get obs data for selected rows
        # Note: row_indices are already sorted and deduplicated
        obs_dfs = []
        for idx in row_indices:
            obs_slice = self.obs[idx:idx+1, :]
            obs_dfs.append(obs_slice)
        
        if obs_dfs:
            obs_df = pd.concat(obs_dfs, ignore_index=True)
        else:
            obs_df = self.obs[0:0, :].copy()  # Empty DataFrame with correct columns
        
        obs_df.index = pd.RangeIndex(start=0, stop=len(obs_df))
        
        var_df = self._var_df.copy()
        
        import warnings
        with warnings.catch_warnings():
            # Suppress ImplicitModificationWarning from anndata during AnnData construction
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
