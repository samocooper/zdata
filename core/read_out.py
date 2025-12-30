import subprocess, struct
import numpy as np
import os
import glob
from collections import defaultdict
from scipy.sparse import csr_matrix
from pathlib import Path

MAX_ROWS_PER_CHUNK = 4096

# Get the path to the zdata_read executable
# This assumes the module structure: zdata/core/read_out.py and zdata/ctools/zdata_read
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
            dir_name: Name of the .zdata directory (e.g., "andrews" for andrews.zdata/)
        """
        self.dir_name = dir_name
        self.dir_path = f"{dir_name}.zdata"
        
        if not os.path.exists(self.dir_path):
            raise FileNotFoundError(f"Directory not found: {self.dir_path}")
        
        if not os.path.isdir(self.dir_path):
            raise ValueError(f"Path is not a directory: {self.dir_path}")
        
        # Discover available chunk files
        bin_files = sorted(glob.glob(os.path.join(self.dir_path, "*.bin")))
        if not bin_files:
            raise ValueError(f"No .bin files found in {self.dir_path}")
        
        # Extract chunk numbers and determine total rows
        self.chunk_files = {}
        max_chunk = -1
        for bin_file in bin_files:
            basename = os.path.basename(bin_file)
            try:
                chunk_num = int(basename.replace(".bin", ""))
                self.chunk_files[chunk_num] = bin_file
                max_chunk = max(max_chunk, chunk_num)
            except ValueError:
                continue
        
        if not self.chunk_files:
            raise ValueError(f"No valid chunk files found in {self.dir_path}")
        
        # Calculate total number of rows (assuming last chunk might be partial)
        # We'll determine this by reading the first file
        self.ncols = None
        self._initialize_metadata()
        
        # Calculate total rows
        self.nrows = (max_chunk + 1) * MAX_ROWS_PER_CHUNK
        # Note: Actual nrows might be less if last chunk is partial, but this is an upper bound
    
    def _initialize_metadata(self):
        """Initialize metadata by reading the first chunk file."""
        first_chunk = min(self.chunk_files.keys())
        first_file = self.chunk_files[first_chunk]
        
        # Read just one row to get ncols
        self.ncols, _ = self._read_rows_from_file(first_file, [0])
    
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
            
            need = off + nnz * 4 + nnz * 4
            if need > len(blob):
                raise ValueError(
                    f"Truncated row {i} (row_id={row_id}, nnz={nnz}). "
                    f"need={need}, len={len(blob)}, off={off}"
                )
            
            cols = np.frombuffer(blob, dtype=np.uint32, count=nnz, offset=off).astype(np.int32)
            off += nnz * 4
            vals = np.frombuffer(blob, dtype=np.float32, count=nnz, offset=off)
            off += nnz * 4
            
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
            
            chunk_num = global_row // MAX_ROWS_PER_CHUNK
            local_row = global_row % MAX_ROWS_PER_CHUNK
            
            if chunk_num not in self.chunk_files:
                raise IndexError(f"Row {global_row} is beyond available data (chunk {chunk_num} not found)")
            
            file_path = self.chunk_files[chunk_num]
            rows_by_file[file_path].append((local_row, idx, global_row))
        
        # Read from each file and collect results
        all_results = [None] * len(global_rows)
        
        for file_path, row_info_list in rows_by_file.items():
            # Extract local rows for this file
            local_rows = [info[0] for info in row_info_list]
            
            # Read from this file
            file_ncols, file_results = self._read_rows_from_file(file_path, local_rows)
            
            # Store ncols (should be same across all files)
            if self.ncols is None:
                self.ncols = file_ncols
            elif self.ncols != file_ncols:
                raise ValueError(f"Inconsistent ncols: {self.ncols} vs {file_ncols} in {file_path}")
            
            # Map results back to original global indices
            for (local_row, orig_idx, global_row), (returned_local_row, cols, vals) in zip(row_info_list, file_results):
                # Verify the returned row matches what we expect
                if returned_local_row != local_row:
                    raise ValueError(f"Row mismatch: expected local row {local_row}, got {returned_local_row}")
                
                # Store with global row ID
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
            # Add all non-zero entries for this row
            for col, val in zip(cols, vals):
                row_indices.append(csr_row_idx)
                col_indices.append(int(col))
                values.append(float(val))
        
        # Create CSR matrix
        csr = csr_matrix((values, (row_indices, col_indices)), shape=(nrows, self.ncols))
        
        return csr
    
    @property
    def num_columns(self):
        """Get the number of columns in the matrix."""
        if self.ncols is None:
            # Force initialization
            self._initialize_metadata()
        return self.ncols
    
    def _get_max_row(self):
        """
        Determine the actual maximum row number by checking the last chunk.
        Uses binary search to find the highest valid row.
        
        Returns:
            Maximum valid global row index
        """
        if not hasattr(self, '_cached_max_row'):
            max_chunk = max(self.chunk_files.keys())
            last_file = self.chunk_files[max_chunk]
            
            # Binary search for the maximum valid row in the last chunk
            # Start with the theoretical maximum
            chunk_start = max_chunk * MAX_ROWS_PER_CHUNK
            chunk_end = (max_chunk + 1) * MAX_ROWS_PER_CHUNK - 1
            
            low = 0  # Local row index within chunk
            high = MAX_ROWS_PER_CHUNK - 1  # Maximum possible local row
            
            # Binary search to find the highest valid local row in the last chunk
            while low <= high:
                mid = (low + high) // 2
                
                try:
                    # Try to read this row
                    self._read_rows_from_file(last_file, [mid])
                    # If successful, this row exists, try higher
                    low = mid + 1
                except Exception:
                    # If failed (any exception), this row doesn't exist, try lower
                    high = mid - 1
            
            # high is now the highest valid local row in the last chunk
            # Convert to global row index
            self._cached_max_row = chunk_start + high
        
        return self._cached_max_row
    
    @property
    def num_rows(self):
        """Get the actual number of rows in the dataset."""
        return self._get_max_row() + 1
    
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
        
        # Get the actual maximum row number
        max_row = self._get_max_row()
        
        # Generate random rows
        random_rows = np.random.randint(0, max_row + 1, size=n).tolist()
        return random_rows

# Example usage (commented out - see test script for actual usage)
if __name__ == "__main__":
    # Example: Basic usage
    # reader = ZData("andrews")
    # rows = reader.read_rows([300, 5096, 9000])
    # csr = reader.read_rows_csr([300, 5096, 9000])
    pass
