import zarr
import numpy as np
import time
import random
import os

# Open the zarr array
zarr_path = '/home/ubuntu/zarr_work/test.zdata'
g = zarr.open_group(zarr_path, mode='r')

if 'X_RM' not in g:
    print("ERROR: X_RM array not found in zarr group")
    exit(1)

X_RM = g['X_RM']

print(f"X_RM shape: {X_RM.shape}")
print(f"X_RM dtype: {X_RM.dtype}")
print(f"X_RM chunks: {X_RM.chunks}")

n_rows, n_cols = X_RM.shape
num_queries = 20
min_query_size = 10
max_query_size = 1000

print(f"\nProfiling {num_queries} random row queries (query size: {min_query_size}-{max_query_size} rows)...")
print("=" * 60)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generate random query sizes for each query (0 to 1000 rows)
query_sizes = np.random.randint(min_query_size, max_query_size + 1, size=num_queries)

query_times = []
sum_times = []
total_times = []
all_row_sums = []
query_lengths = []

for query_idx in range(num_queries):
    query_size = query_sizes[query_idx]
    query_lengths.append(query_size)
    
    # Generate random row indices
    random_rows = np.random.randint(0, n_rows, size=query_size)
    
    # Time the row query (retrieve all rows at once)
    start_query = time.perf_counter()
    rows_data = X_RM[random_rows, :]  # Shape: (query_size, n_cols)
    end_query = time.perf_counter()
    
    query_time = end_query - start_query
    query_times.append(query_time)
    
    # Time the sum calculation (sum each row)
    start_sum = time.perf_counter()
    row_sums = np.sum(rows_data, axis=1)  # Sum along columns for each row
    end_sum = time.perf_counter()
    
    sum_time = end_sum - start_sum
    sum_times.append(sum_time)
    
    total_time = query_time + sum_time
    total_times.append(total_time)
    all_row_sums.extend(row_sums.tolist())
    
    # Calculate non-zero counts for the batch
    total_nonzero = np.count_nonzero(rows_data)
    avg_nonzero = total_nonzero / query_size
    avg_row_sum = np.mean(row_sums)
    
    print(f"Query {query_idx + 1:2d} ({query_size:4d} rows): "
          f"Query: {query_time*1000:.3f} ms, "
          f"Sum: {sum_time*1000:.3f} ms, "
          f"Total: {total_time*1000:.3f} ms - "
          f"Avg row sum: {avg_row_sum:.2f}, "
          f"Avg non-zero: {avg_nonzero:.1f}/{n_cols}")

print("=" * 60)
total_rows_queried = sum(query_lengths)

# Calculate zarr object on-disk size
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
if os.path.isdir(zarr_path):
    disk_size = get_dir_size(zarr_path)
else:
    disk_size = os.path.getsize(zarr_path) if os.path.isfile(zarr_path) else 0

print(f"\nOn-disk size: {format_size(disk_size)}")

print(f"\nQuery times (row retrieval):")
print(f"  Mean: {np.mean(query_times)*1000:.3f} ms")
print(f"  Median: {np.median(query_times)*1000:.3f} ms")
print(f"  Min: {np.min(query_times)*1000:.3f} ms")
print(f"  Max: {np.max(query_times)*1000:.3f} ms")
print(f"  Std dev: {np.std(query_times)*1000:.3f} ms")

print(f"\nTotal times (query + sum):")
print(f"  Mean: {np.mean(total_times)*1000:.3f} ms")
print(f"  Median: {np.median(total_times)*1000:.3f} ms")
print(f"  Min: {np.min(total_times)*1000:.3f} ms")
print(f"  Max: {np.max(total_times)*1000:.3f} ms")
print(f"  Std dev: {np.std(total_times)*1000:.3f} ms")
print(f"  Total: {sum(total_times)*1000:.3f} ms")



