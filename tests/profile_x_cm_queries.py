import zarr
import numpy as np
import time
import random
import os

# Open the zarr group
zarr_path = '/home/ubuntu/zarr_work/test.zdata'
g = zarr.open_group(zarr_path, mode='r')

if 'X_CM' not in g:
    print("ERROR: X_CM array not found in zarr group")
    exit(1)

X_CM = g['X_CM']

print(f"X_CM shape: {X_CM.shape}")
print(f"X_CM dtype: {X_CM.dtype}")
print(f"X_CM chunks: {X_CM.chunks}")

n_rows, n_cols = X_CM.shape
num_queries = 20
min_query_size = 10
max_query_size = 100

print(f"\nProfiling {num_queries} random column queries (query size: {min_query_size}-{max_query_size} genes)...")
print("=" * 60)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generate random query sizes for each query (10 to 100 genes)
query_sizes = np.random.randint(min_query_size, max_query_size + 1, size=num_queries)

query_times = []
sum_times = []
total_times = []
all_column_sums = []
query_lengths = []

for query_idx in range(num_queries):
    query_size = query_sizes[query_idx]
    query_lengths.append(query_size)
    
    # Generate random column indices (genes)
    random_cols = np.sort(np.random.randint(0, n_cols, size=query_size))
    
    # Time the column query (retrieve all columns at once)
    start_query = time.perf_counter()
    cols_data = X_CM[:, random_cols]  # Shape: (n_rows, query_size)
    end_query = time.perf_counter()
    
    query_time = end_query - start_query
    query_times.append(query_time)
    
    # Time the sum calculation (sum each column)
    start_sum = time.perf_counter()
    col_sums = np.sum(cols_data, axis=0)  # Sum along rows for each column
    end_sum = time.perf_counter()
    
    sum_time = end_sum - start_sum
    sum_times.append(sum_time)
    
    total_time = query_time + sum_time
    total_times.append(total_time)
    all_column_sums.extend(col_sums.tolist())
    
    # Calculate non-zero counts for the batch
    total_nonzero = np.count_nonzero(cols_data)
    avg_nonzero = total_nonzero / query_size
    avg_col_sum = np.mean(col_sums)
    
    print(f"Query {query_idx + 1:2d} ({query_size:3d} genes): "
          f"Query: {query_time*1000:.3f} ms, "
          f"Sum: {sum_time*1000:.3f} ms, "
          f"Total: {total_time*1000:.3f} ms - "
          f"Avg column sum: {avg_col_sum:.2f}, "
          f"Avg non-zero: {avg_nonzero:.1f}/{n_rows}")

print("=" * 60)
total_genes_queried = sum(query_lengths)

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

# Calculate actual on-disk size for X_CM
# X_CM is stored in the same zarr group, so we need to calculate size of X_CM specifically
# For simplicity, we'll report the total zarr size (both X_RM and X_CM)
if os.path.isdir(zarr_path):
    # Try to get size of X_CM specifically by walking the X_CM subdirectory
    x_cm_path = os.path.join(zarr_path, 'X_CM')
    if os.path.isdir(x_cm_path):
        disk_size = get_dir_size(x_cm_path)
    else:
        # Fallback to total size
        disk_size = get_dir_size(zarr_path)
else:
    disk_size = os.path.getsize(zarr_path) if os.path.isfile(zarr_path) else 0

print(f"\nOn-disk size: {format_size(disk_size)}")

print(f"\nQuery times (column retrieval):")
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

# Query specific genes and print total counts
target_genes = ['RPL17','RPL7', 'GAPDH', 'TUBB', 'CD8A', 'COL1A1', 'TYROBP', 'FOXP3', 'FAP', 'CLEC11A','IL2RA']

print(f"\nTotal counts for specific genes:")
print("=" * 60)

# Load gene list to map gene names to column indices
gene_list_path = '/home/ubuntu/zarr_work/zdata/files/2ks10c_genes.txt'
with open(gene_list_path) as f:
    gene_list = [line.strip() for line in f if line.strip()]

# Create mapping from gene name to column index
gene_to_col = {gene: idx for idx, gene in enumerate(gene_list)}

# Find column indices for target genes and query them
found_genes = []
gene_col_indices = []

for gene in target_genes:
    if gene in gene_to_col:
        col_idx = gene_to_col[gene]
        found_genes.append(gene)
        gene_col_indices.append(col_idx)
    else:
        print(f"  {gene}: NOT FOUND in gene list")

if found_genes:
    # Query the columns for found genes
    gene_col_indices = np.array(gene_col_indices)
    gene_data = X_CM[:, gene_col_indices]  # Shape: (n_rows, len(found_genes))
    
    # Calculate total counts (sum across all rows) for each gene
    gene_totals = np.sum(gene_data, axis=0)
    
    # Print results
    for gene, total in zip(found_genes, gene_totals):
        print(f"  {gene}: {total:,.0f}")

