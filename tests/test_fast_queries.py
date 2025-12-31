#!/usr/bin/env python3
"""
Performance test for ZData - compares retrieval speed for random row queries.
Adapted from zarr-based test to use the new seekable zstd-based implementation.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import zdata
# We need the parent of the zdata package directory, not the zdata directory itself
_test_dir = Path(__file__).parent
_project_root = _test_dir.parent  # This is /home/ubuntu/zdata_work/zdata
_parent_dir = _project_root.parent  # This is /home/ubuntu/zdata_work
sys.path.insert(0, str(_parent_dir))

import numpy as np
import time
import random
import os
from zdata.core.zdata import ZData
from scipy.sparse import csr_matrix

# Parse command-line arguments
if len(sys.argv) > 1:
    zdata_input = sys.argv[1]
    
    # Convert to absolute path if it's not already
    if os.path.isabs(zdata_input):
        zdata_dir = zdata_input
    else:
        zdata_dir = os.path.abspath(zdata_input)
    
    # ZData accepts full paths directly, so we can pass it as-is
    # It will also handle relative paths and directory names
else:
    # Default
    zdata_dir = 'atlas'

# Configuration
num_queries = 20
min_query_size = 10
max_query_size = 5000

print(f"Opening ZData for directory: {zdata_dir}")
print("=" * 60)

try:
    reader = ZData(zdata_dir)
    n_rows = reader.num_rows
    n_cols = reader.num_columns
    
    print(f"Dataset shape: ({n_rows}, {n_cols})")
    print(f"Available chunks: {sorted(reader.chunk_files.keys())}")
    print()
    
except Exception as e:
    print(f"ERROR: Failed to initialize ZData: {e}")
    exit(1)

print(f"Running {num_queries} random row queries (query size: {min_query_size}-{max_query_size} rows)...")
print("NOTE: Row indices are SORTED before querying to improve chunk access locality")
print("=" * 60)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generate random query sizes for each query
query_sizes = np.random.randint(min_query_size, max_query_size + 1, size=num_queries)

query_times = []
sum_times = []
total_times = []
all_row_sums = []
query_lengths = []

for query_idx in range(num_queries):
    query_size = query_sizes[query_idx]
    query_lengths.append(query_size)
    
    # Generate random row indices (original order)
    original_rows = np.random.randint(0, n_rows, size=query_size)
    
    # Time the row query (retrieve all rows at once)
    # Includes sorting for better chunk access locality and reordering results
    start_query = time.perf_counter()
    
    # Sort row indices for better chunk access locality (groups rows from same chunk together)
    # Store the sort indices to restore original order later
    sort_indices = np.argsort(original_rows)
    sorted_rows = original_rows[sort_indices]
    
    # Perform the query with sorted rows and get CSR matrix directly
    csr_sorted = reader.read_rows_csr(sorted_rows.tolist())
    
    # Restore original query order using inverse permutation
    # Reorder CSR matrix rows to match original query order
    inverse_sort = np.argsort(sort_indices)
    csr_data = csr_sorted[inverse_sort]
    
    end_query = time.perf_counter()
    
    query_time = end_query - start_query
    query_times.append(query_time)
    
    # Time the sum calculation (sum each row)
    start_sum = time.perf_counter()
    # Sum along columns for each row (CSR format makes this efficient)
    row_sums = np.array(csr_data.sum(axis=1)).flatten()  # Sum along columns for each row
    end_sum = time.perf_counter()
    
    sum_time = end_sum - start_sum
    sum_times.append(sum_time)
    
    total_time = query_time + sum_time
    total_times.append(total_time)
    all_row_sums.extend(row_sums.tolist())
    
    # Calculate non-zero counts for the batch
    total_nonzero = csr_data.nnz
    avg_nonzero = total_nonzero / query_size
    avg_row_sum = np.mean(row_sums)
    
    print(f"Query {query_idx + 1:2d} ({query_size:4d} rows): "
          f"Query: {query_time*1000:.3f} ms, "
          f"Sum: {sum_time*1000:.3f} ms, "
          f"Total: {total_time*1000:.3f} ms - "
          f"Avg row sum: {avg_row_sum:.2f}, "
          f"Avg non-zero: {avg_nonzero:.1f}/{n_cols}")

print("=" * 60)

# Calculate on-disk size
def get_dir_size(path):
    """Calculate total size of directory in bytes"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total += os.path.getsize(filepath)
    return total

def format_size(size_bytes):
    """Format bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

# Calculate actual on-disk size
# zdata_dir might be a full path or relative path
if os.path.isdir(zdata_dir):
    disk_size = get_dir_size(zdata_dir)
elif os.path.isdir(zdata_dir + '.zdata'):
    # Try with .zdata suffix (backward compatibility)
    disk_size = get_dir_size(zdata_dir + '.zdata')
else:
    disk_size = 0

print(f"\nOn-disk size: {format_size(disk_size)}")

print(f"\nQuery times (row retrieval):")
print(f"  Mean: {np.mean(query_times)*1000:.3f} ms")
print(f"  Median: {np.median(query_times)*1000:.3f} ms")
print(f"  Min: {np.min(query_times)*1000:.3f} ms")
print(f"  Max: {np.max(query_times)*1000:.3f} ms")
print(f"  Std dev: {np.std(query_times)*1000:.3f} ms")

print(f"\nSum times (row sum calculation):")
print(f"  Mean: {np.mean(sum_times)*1000:.3f} ms")
print(f"  Median: {np.median(sum_times)*1000:.3f} ms")
print(f"  Min: {np.min(sum_times)*1000:.3f} ms")
print(f"  Max: {np.max(sum_times)*1000:.3f} ms")
print(f"  Std dev: {np.std(sum_times)*1000:.3f} ms")

print(f"\nTotal times (query + sum):")
print(f"  Mean: {np.mean(total_times)*1000:.3f} ms")
print(f"  Median: {np.median(total_times)*1000:.3f} ms")
print(f"  Min: {np.min(total_times)*1000:.3f} ms")
print(f"  Max: {np.max(total_times)*1000:.3f} ms")
print(f"  Std dev: {np.std(total_times)*1000:.3f} ms")
print(f"  Total: {sum(total_times)*1000:.3f} ms")

# Calculate throughput statistics
total_rows_queried = sum(query_lengths)
total_query_time = sum(query_times)
avg_rows_per_second = total_rows_queried / total_query_time if total_query_time > 0 else 0

print(f"\nThroughput:")
print(f"  Total rows queried: {total_rows_queried:,}")
print(f"  Total query time: {total_query_time:.3f} s")
print(f"  Average rows/second: {avg_rows_per_second:,.0f}")


