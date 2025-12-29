import zarr
import numpy as np
import time
import random

# Open the zarr array
zarr_path = '/home/ubuntu/zarr_work/tmp/tmp_rows.zarr'
g = zarr.open_group(zarr_path, mode='r')

if 'X_RM' not in g:
    print("ERROR: X_RM array not found in zarr group")
    exit(1)

X_RM = g['X_RM']

print(f"X_RM shape: {X_RM.shape}")
print(f"X_RM dtype: {X_RM.dtype}")
print(f"X_RM chunks: {X_RM.chunks}")

n_rows, n_cols = X_RM.shape
num_queries = 10
rows_per_query = 50

print(f"\nProfiling {num_queries} random row queries ({rows_per_query} rows each, {num_queries * rows_per_query} total rows)...")
print("=" * 60)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

query_times = []
sum_times = []
total_times = []
all_row_sums = []

for query_idx in range(num_queries):
    # Generate 50 random row indices
    random_rows = np.random.randint(0, n_rows, size=rows_per_query)
    
    # Time the row query (retrieve all 50 rows at once)
    start_query = time.perf_counter()
    rows_data = X_RM[random_rows, :]  # Shape: (50, n_cols)
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
    avg_nonzero = total_nonzero / rows_per_query
    
    print(f"Query {query_idx + 1:2d} ({rows_per_query} rows): "
          f"Query: {query_time*1000:.3f} ms, "
          f"Sum: {sum_time*1000:.3f} ms, "
          f"Total: {total_time*1000:.3f} ms - "
          f"Avg row sum: {np.mean(row_sums):.2f}, "
          f"Avg non-zero: {avg_nonzero:.1f}/{n_cols}")

print("=" * 60)
print(f"\nSummary:")
print(f"  Total queries: {num_queries}")
print(f"  Rows per query: {rows_per_query}")
print(f"  Total rows queried: {num_queries * rows_per_query}")
print(f"  Columns per row: {n_cols:,}")
print(f"  Total cells queried: {num_queries * rows_per_query * n_cols:,}")
print(f"\nQuery times (row retrieval):")
print(f"  Mean: {np.mean(query_times)*1000:.3f} ms")
print(f"  Median: {np.median(query_times)*1000:.3f} ms")
print(f"  Min: {np.min(query_times)*1000:.3f} ms")
print(f"  Max: {np.max(query_times)*1000:.3f} ms")
print(f"  Std dev: {np.std(query_times)*1000:.3f} ms")
print(f"\nSum calculation times:")
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
print(f"\nRow sum statistics (all {num_queries * rows_per_query} rows):")
print(f"  Mean row sum: {np.mean(all_row_sums):.2f}")
print(f"  Median row sum: {np.median(all_row_sums):.2f}")
print(f"  Min row sum: {np.min(all_row_sums):.2f}")
print(f"  Max row sum: {np.max(all_row_sums):.2f}")

