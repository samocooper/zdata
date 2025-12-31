import subprocess, struct
import numpy as np
import os
import json
from collections import defaultdict
from scipy.sparse import csr_matrix
from pathlib import Path

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
        
        # Extract metadata
        self.nrows = self.metadata['shape'][0]
        self.ncols = self.metadata['shape'][1]
        self.nnz_total = self.metadata.get('nnz_total', None)
        self.num_chunks = self.metadata['num_chunks']
        self.total_blocks = self.metadata.get('total_blocks', None)
        
        # Extract block configuration from metadata (with defaults for backward compatibility)
        self.block_rows = self.metadata.get('block_rows', 16)
        self.max_rows_per_chunk = self.metadata.get('max_rows_per_chunk', 8192)
        
        # Build chunk_files dict and chunk info from metadata
        # Support both old format (chunks) and new format (chunks_rm, chunks_cm)
        self.chunk_files = {}
        self.chunk_info = {}  # Store chunk metadata for validation
        self.file_to_chunk = {}  # Reverse mapping: file_path -> chunk_num (for O(1) lookup)
        
        # Check for new format with separate RM and CM chunks
        if 'chunks_rm' in self.metadata:
            # New format: separate chunks for X_RM
            chunks_list = self.metadata['chunks_rm']
            subdir = "X_RM"
        elif 'chunks' in self.metadata:
            # Old format: single chunks list (assume X_RM)
            chunks_list = self.metadata['chunks']
            subdir = "X_RM"
        else:
            raise ValueError("Metadata missing 'chunks' or 'chunks_rm' field")
        
        for chunk_info in chunks_list:
            chunk_num = chunk_info['chunk_num']
            # .bin files are stored in subdirectory (X_RM or X_CM)
            file_path = os.path.join(self.dir_path, subdir, chunk_info['file'])
            self.chunk_files[chunk_num] = file_path
            self.chunk_info[chunk_num] = chunk_info
            self.file_to_chunk[file_path] = chunk_num
    
    
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
        if isinstance(global_rows, (int, np.integer)):
            global_rows = [int(global_rows)]
        else:
            global_rows = [int(r) for r in global_rows]
        
        # Group global rows by which chunk file they belong to
        rows_by_file = defaultdict(list)
        
        for idx, global_row in enumerate(global_rows):
            if global_row < 0:
                raise ValueError(f"Row index must be non-negative, got {global_row}")
            
            if global_row >= self.nrows:
                raise IndexError(f"Row {global_row} is beyond available data (max row: {self.nrows - 1})")
            
            # Calculate chunk number: each chunk contains max_rows_per_chunk rows
            # This matches how chunks are numbered in build_zdata.py
            chunk_num = global_row // self.max_rows_per_chunk
            local_row = global_row % self.max_rows_per_chunk
            
            if chunk_num not in self.chunk_files:
                raise IndexError(f"Row {global_row} is beyond available data (chunk {chunk_num} not found)")
            
            # Validate that the row is within this chunk's valid range (defensive check)
            chunk_info = self.chunk_info[chunk_num]
            chunk_start = chunk_info['start_row']
            chunk_end = chunk_info['end_row']
            
            if global_row < chunk_start or global_row >= chunk_end:
                raise IndexError(
                    f"Row {global_row} is out of range for chunk {chunk_num} "
                    f"(valid range: {chunk_start} to {chunk_end - 1})"
                )
            
            rows_by_file[self.chunk_files[chunk_num]].append((local_row, idx, global_row))
        
        # Read from each file and collect results
        all_results = [None] * len(global_rows)
        
        for file_path, row_info_list in rows_by_file.items():
            # Find which chunk this file belongs to (O(1) lookup using reverse mapping)
            chunk_num = self.file_to_chunk.get(file_path)
            if chunk_num is None:
                raise ValueError(f"Could not find chunk info for file: {file_path}")
            chunk_info = self.chunk_info[chunk_num]
            
            # Validate local rows are within chunk bounds
            chunk_start = chunk_info['start_row']
            chunk_end = chunk_info['end_row']
            max_local_row = chunk_end - chunk_start - 1
            
            for local_row, orig_idx, global_row in row_info_list:
                if local_row > max_local_row:
                    raise IndexError(
                        f"Local row {local_row} exceeds chunk {chunk_num} bounds "
                        f"(max local row: {max_local_row}, global row: {global_row})"
                    )
            
            # Sort by local_row to ensure ordered input to C tool (for better chunk locality)
            row_info_list_sorted = sorted(row_info_list, key=lambda x: x[0])  # Sort by local_row
            
            # Extract local rows in sorted order for C tool
            local_rows = [info[0] for info in row_info_list_sorted]
            
            # Read from this file (C tool returns rows in the order requested)
            file_ncols, file_results = self._read_rows_from_file(file_path, local_rows)
            
            if self.ncols is None:
                self.ncols = file_ncols
            elif self.ncols != file_ncols:
                raise ValueError(f"Inconsistent ncols: {self.ncols} vs {file_ncols} in {file_path}")
            
            # Map results back to original order using stored indices
            # file_results are in the same order as local_rows (sorted)
            # row_info_list_sorted is sorted, so we can zip them directly
            for (local_row, orig_idx, global_row), (returned_local_row, cols, vals) in zip(row_info_list_sorted, file_results):
                if returned_local_row != local_row:
                    raise ValueError(f"Row mismatch: expected local row {local_row}, got {returned_local_row}")
                # Store in original position (orig_idx) to maintain input order
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
        Build chunk mapping for X_CM (column-major) files.
        
        Returns:
            (cm_chunk_files, cm_chunk_info, cm_file_to_chunk) tuple
        """
        xcm_dir = os.path.join(self.dir_path, "X_CM")
        if not os.path.exists(xcm_dir):
            raise FileNotFoundError(f"X_CM directory not found: {xcm_dir}")
        
        xcm_path = Path(xcm_dir)
        cm_bin_files = sorted(xcm_path.glob("*.bin"))
        
        if not cm_bin_files:
            raise FileNotFoundError(f"No .bin files found in X_CM directory: {xcm_dir}")
        
        cm_chunk_files = {}
        cm_chunk_info = {}
        cm_file_to_chunk = {}
        
        # Build chunk mapping for X_CM files
        for chunk_idx, bin_file in enumerate(cm_bin_files):
            chunk_num = chunk_idx
            file_path = str(bin_file)
            cm_chunk_files[chunk_num] = file_path
            cm_chunk_info[chunk_num] = {
                'chunk_num': chunk_num,
                'file': bin_file.name,
                'start_row': chunk_num * self.max_rows_per_chunk,
                'end_row': (chunk_num + 1) * self.max_rows_per_chunk
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
        if isinstance(global_cols, (int, np.integer)):
            global_cols = [int(global_cols)]
        else:
            global_cols = [int(c) for c in global_cols]
        
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
            
            # Calculate chunk number: each chunk contains max_rows_per_chunk rows (genes)
            chunk_num = global_col // self.max_rows_per_chunk
            local_row = global_col % self.max_rows_per_chunk  # Local row in X_CM (which is a gene)
            
            if chunk_num not in cm_chunk_files:
                raise IndexError(f"Column {global_col} is beyond available data (chunk {chunk_num} not found)")
            
            # Validate that the column is within this chunk's valid range
            # Note: chunk_info['start_row'] and 'end_row' are gene indices (0 to ncols-1)
            chunk_info = cm_chunk_info[chunk_num]
            chunk_start = chunk_info['start_row']  # Gene index start
            chunk_end = chunk_info['end_row']      # Gene index end
            
            if global_col < chunk_start or global_col >= chunk_end:
                raise IndexError(
                    f"Column {global_col} is out of range for chunk {chunk_num} "
                    f"(valid range: {chunk_start} to {chunk_end - 1})"
                )
            
            cols_by_file[cm_chunk_files[chunk_num]].append((local_row, idx, global_col))
        
        # Read from each file and collect results
        all_results = [None] * len(global_cols)
        
        for file_path, col_info_list in cols_by_file.items():
            # Find which chunk this file belongs to
            chunk_num = cm_file_to_chunk.get(file_path)
            if chunk_num is None:
                raise ValueError(f"Could not find chunk info for file: {file_path}")
            chunk_info = cm_chunk_info[chunk_num]
            
            # Validate local rows are within chunk bounds
            chunk_start = chunk_info['start_row']
            chunk_end = chunk_info['end_row']
            max_local_row = chunk_end - chunk_start - 1
            
            for local_row, orig_idx, global_col in col_info_list:
                if local_row > max_local_row:
                    raise IndexError(
                        f"Local row {local_row} exceeds chunk {chunk_num} bounds "
                        f"(max local row: {max_local_row}, global column: {global_col})"
                    )
            
            # Sort by local_row for better chunk access locality
            col_info_list_sorted = sorted(col_info_list, key=lambda x: x[0])  # Sort by local_row
            
            # Extract local rows in sorted order for C tool
            local_rows = [info[0] for info in col_info_list_sorted]
            
            # Read from this file (C tool returns rows in the order requested)
            # In X_CM, ncols = nrows (original), so we get cells as columns
            file_ncols, file_results = self._read_rows_from_file(file_path, local_rows)
            
            # Map results back to original order
            for (local_row, orig_idx, global_col), (returned_local_row, rows, vals) in zip(col_info_list_sorted, file_results):
                if returned_local_row != local_row:
                    raise ValueError(f"Row mismatch: expected local row {local_row}, got {returned_local_row}")
                # Store in original position (orig_idx) to maintain input order
                # rows are cell indices, vals are values
                all_results[orig_idx] = (global_col, rows, vals)
        
        return all_results
    
    def read_cols_cm_csr(self, global_cols):
        """
        Read columns (genes) from X_CM (column-major) files and return as CSR matrix.
        The returned matrix has shape (len(global_cols), nrows) where each row is a gene
        and each column is a cell.
        
        Args:
            global_cols: List or array of global column (gene) indices (0-based)
        
        Returns:
            scipy.sparse.csr_matrix of shape (len(global_cols), nrows)
        """
        cols_data = self.read_cols_cm(global_cols)
        
        # Convert to CSR matrix
        # In the result, rows = genes, columns = cells
        ngenes = len(cols_data)
        
        row_indices = []
        col_indices = []
        values = []
        
        for csr_row_idx, (col_id, rows, vals) in enumerate(cols_data):
            for row, val in zip(rows, vals):
                row_indices.append(csr_row_idx)
                col_indices.append(int(row))
                values.append(float(val))
        
        # Create CSR matrix: shape is (ngenes, nrows) where nrows is number of cells
        csr = csr_matrix((values, (row_indices, col_indices)), shape=(ngenes, self.nrows))
        
        return csr
    
    def _rows_to_csr(self, rows_data):
        """
        Convert rows data into a scipy.sparse.csr_matrix.
        
        Args:
            rows_data: List of (row_id, cols, vals) tuples
        
        Returns:
            scipy.sparse.csr_matrix
        """
        nrows = len(rows_data)
        
        # Build arrays for CSR construction
        row_indices = []
        col_indices = []
        values = []
        
        for csr_row_idx, (row_id, cols, vals) in enumerate(rows_data):
            for col, val in zip(cols, vals):
                row_indices.append(csr_row_idx)
                col_indices.append(int(col))
                values.append(float(val))  # Convert uint16 to float for CSR matrix
        
        # Create CSR matrix
        csr = csr_matrix((values, (row_indices, col_indices)), shape=(nrows, self.ncols))
        
        return csr
    
    @property
    def num_columns(self):
        """Get the number of columns in the matrix."""
        return self.ncols
    
    @property
    def num_rows(self):
        """Get the actual number of rows in the dataset."""
        return self.nrows
    
    @property
    def shape(self):
        """Get the shape of the matrix as (nrows, ncols)."""
        return (self.nrows, self.ncols)
    
    def get_random_rows(self, n, seed=None):
        """
        Get n random row indices that are valid for this dataset.
        
        Args:
            n: Number of random rows to generate
            seed: Optional random seed for reproducibility
        
        Returns:
            List of random global row indices
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random rows using metadata
        random_rows = np.random.randint(0, self.nrows, size=n).tolist()
        return random_rows

# Example usage (commented out - see test script for actual usage)
if __name__ == "__main__":
    # Example: Basic usage
    # reader = ZData("andrews")
    # rows = reader.read_rows([300, 5096, 9000])
    # csr = reader.read_rows_csr([300, 5096, 9000])
    pass
