import subprocess, struct
import numpy as np
import os
import json
from collections import defaultdict
from scipy.sparse import csr_matrix
from pathlib import Path
import polars as pl
import pandas as pd
import anndata as ad

# Get the path to the zdata_read executable
# This assumes the module structure: zdata/core/zdata.py and zdata/ctools/zdata_read
_MODULE_DIR = Path(__file__).parent  # zdata/core/
_PROJECT_ROOT = _MODULE_DIR.parent   # zdata/
_ZDATA_READ = _PROJECT_ROOT / "ctools" / "zdata_read"

def _get_zdata_read_path():
    """Get the path to zdata_read executable, with validation."""
    bin_path = _ZDATA_READ.absolute()
    if not bin_path.exists():
        raise RuntimeError(
            f"zdata_read executable not found at {bin_path}. "
            f"Please ensure it is built in the ctools directory."
        )
    return str(bin_path)

def _normalize_indices(indices):
    """Normalize indices to list of ints (handles int or list/array)."""
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
    
    def __init__(self, obs_df):
        """
        Initialize the wrapper with a polars DataFrame.
        
        Args:
            obs_df: polars DataFrame containing obs/metadata
        """
        self.obs_df = obs_df
    
    def __getitem__(self, key):
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
    
    def __len__(self):
        return len(self.obs_df)
    
    @property
    def shape(self):
        return self.obs_df.shape
    
    @property
    def columns(self):
        return self.obs_df.columns
    
    def __repr__(self):
        return f"ObsWrapper({self.obs_df.shape[0]} rows, {self.obs_df.shape[1]} columns)"

class ZData:
    """
    Efficient reader for zdata directory structure containing .bin files.
    
    Provides methods to read random sets of rows from the compressed sparse matrix data.
    """
    
    def __init__(self, dir_name):
        """
        Initialize the reader for a zdata directory.
        
        Args:
            dir_name: Name or path of the zdata directory (e.g., "andrews" or "/path/to/andrews")
        """
        self.dir_name = dir_name
        # Use dir_name directly as path (no .zdata suffix appended)
        self.dir_path = dir_name
        
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
            self.metadata = json.load(f)
        
        # Extract metadata (new format only)
        self.nrows = self.metadata['shape'][0]
        self.ncols = self.metadata['shape'][1]
        self.nnz_total = self.metadata.get('nnz_total', None)
        # Use RM chunks for row-major operations (required)
        if 'num_chunks_rm' not in self.metadata:
            raise ValueError("Metadata missing 'num_chunks_rm' field (new format required)")
        self.num_chunks = self.metadata['num_chunks_rm']
        self.total_blocks = self.metadata.get('total_blocks_rm', None)
        
        # Extract block configuration from metadata
        self.block_rows = self.metadata.get('block_rows', 16)
        self.max_rows_per_chunk = self.metadata.get('max_rows_per_chunk', 8192)
        
        # Build chunk_files dict and chunk info from metadata (new format only)
        self.chunk_files = {}
        self.chunk_info = {}  # Store chunk metadata for validation
        self.file_to_chunk = {}  # Reverse mapping: file_path -> chunk_num (for O(1) lookup)
        
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
            self._obs_df = pl.read_parquet(obs_file)
            self._obs_wrapper = ObsWrapper(self._obs_df)
            
            # Cache var DataFrame for __getitem__ access
            var_polars = pl.read_parquet(var_file)
            var_dict = var_polars.to_dict(as_series=False)
            self._var_df = pd.DataFrame(var_dict)
            self._var_df.index = pd.RangeIndex(start=0, stop=len(self._var_df))
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet files: {e}") from e
    
    @property
    def obs(self):
        """
        Access obs/metadata DataFrame.
        
        Returns:
            ObsWrapper that supports indexing like obs[row_index, :]
        
        """
        return self._obs_wrapper
    
    
    def _read_rows_from_file(self, file_path, local_rows):
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
    
    def read_rows(self, global_rows):
        """
        Read rows using global indices that span across multiple .bin files.
        
        Args:
            global_rows: List or array of global row indices (0-based, relative to full MTX file)
        
        Returns:
            List of (global_row_id, cols, vals) tuples in the same order as global_rows
            where cols and vals are numpy arrays
        """
        global_rows = _normalize_indices(global_rows)
        
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
    
    def read_rows_csr(self, global_rows):
        """
        Read rows and return as a scipy.sparse.csr_matrix.
        
        Args:
            global_rows: List or array of global row indices (0-based)
        
        Returns:
            scipy.sparse.csr_matrix of shape (len(global_rows), ncols)
        """
        rows_data = self.read_rows(global_rows)
        return self._rows_to_csr(rows_data)
    
    def read_rows_rm(self, global_rows):
        """
        Read rows from X_RM (row-major) files.
        This is an alias for read_rows() for clarity.
        
        Args:
            global_rows: List or array of global row indices (0-based)
        
        Returns:
            List of (global_row_id, cols, vals) tuples
        """
        return self.read_rows(global_rows)
    
    def read_rows_rm_csr(self, global_rows):
        """
        Read rows from X_RM (row-major) files and return as CSR matrix.
        
        Args:
            global_rows: List or array of global row indices (0-based)
        
        Returns:
            scipy.sparse.csr_matrix of shape (len(global_rows), ncols)
        """
        return self.read_rows_csr(global_rows)
    
    def _build_cm_chunk_mapping(self):
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
    
    def read_cols_cm(self, global_cols):
        """
        Read columns (genes) from X_CM (column-major) files.
        In X_CM files, rows represent genes (columns in original matrix).
        
        Args:
            global_cols: List or array of global column (gene) indices (0-based)
        
        Returns:
            List of (global_col_id, rows, vals) tuples where rows are cell indices
        """
        global_cols = _normalize_indices(global_cols)
        
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
    
    def read_cols_cm_csr(self, global_cols):
        """Read columns (genes) from X_CM files and return as CSR matrix."""
        cols_data = self.read_cols_cm(global_cols)
        
        row_indices, col_indices, values = [], [], []
        for csr_row_idx, (col_id, rows, vals) in enumerate(cols_data):
            for row, val in zip(rows, vals):
                row_indices.append(csr_row_idx)
                col_indices.append(int(row))
                values.append(float(val))
        
        return csr_matrix((values, (row_indices, col_indices)), shape=(len(cols_data), self.nrows))
    
    def _rows_to_csr(self, rows_data):
        """Convert rows data into a scipy.sparse.csr_matrix."""
        row_indices, col_indices, values = [], [], []
        for csr_row_idx, (row_id, cols, vals) in enumerate(rows_data):
            for col, val in zip(cols, vals):
                row_indices.append(csr_row_idx)
                col_indices.append(int(col))
                values.append(float(val))
        return csr_matrix((values, (row_indices, col_indices)), shape=(len(rows_data), self.ncols))
    
    @property
    def num_columns(self):
        return self.ncols
    
    @property
    def num_rows(self):
        return self.nrows
    
    @property
    def shape(self):
        return (self.nrows, self.ncols)
    
    def get_random_rows(self, n, seed=None):
        """Get n random row indices that are valid for this dataset."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.randint(0, self.nrows, size=n).tolist()
    
    def __getitem__(self, key):
        """
        Support slicing syntax to return AnnData object.
        
        Usage:
            sc_atlas[5:10]  # Returns AnnData object with rows 5-9
            sc_atlas[0:100]  # Returns AnnData object with rows 0-99
        
        Args:
            key: Slice object (e.g., slice(5, 10))
        
        Returns:
            AnnData object containing:
            - X: Sparse CSR matrix with expression data (from X_RM/row-major)
            - obs: pandas DataFrame with cell metadata for selected rows
            - var: pandas DataFrame with gene metadata (full var object)
        """
        if not isinstance(key, slice):
            raise TypeError(
                f"ZData slicing only supports slice objects, got {type(key)}. "
                f"Use syntax like sc_atlas[5:10]"
            )
        
        start, stop, step = key.indices(self.nrows)
        if step != 1:
            raise ValueError(f"Step slicing not yet supported. Use step=1")
        
        row_indices = list(range(start, stop))
        if not row_indices:
            raise ValueError(f"Empty slice: no rows selected")
        
        X_csr = self.read_rows_csr(row_indices)
        
        obs_slice = self.obs[start:stop, :]
        obs_df = obs_slice.copy()
        obs_df.index = pd.RangeIndex(start=0, stop=len(obs_df))
        
        var_df = self._var_df.copy()
        
        import warnings
        with warnings.catch_warnings():
            # Suppress ImplicitModificationWarning from anndata during AnnData construction
            # We've ensured explicit integer indices, but AnnData still triggers this warning internally
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
