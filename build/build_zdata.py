import zarr
from scipy.sparse import csr_matrix, csc_matrix
import numpy as np

with open("/home/ubuntu/zarr_work/zdata/files/2ks10c_genes.txt") as f:
    gene_list = [line.strip() for line in f if line.strip()]

zarr_path = '/home/ubuntu/zarr_work/zarr_datasets/external_andrews_hepatolcommun_2022_34792289.zarr'
zarr_group = zarr.open(zarr_path, mode='r')

# number of columns
n_old_cols = zarr_group["var"]["gene"].shape[0]  # Original number of columns in zarr file
n_new_cols = len(gene_list)  # New number of columns after mapping
n_rows = zarr_group["obs"]["barcode"].shape[0]

out_path = '/home/ubuntu/zarr_work/test.zdata'


dtype = np.uint16
max_uint16 = np.iinfo(np.uint16).max  # 65535

g = zarr.open_group(out_path, mode="w")  # use mode="a" to append to existing store

rm_shard_size = 1024  
rm_chunk_size = 16 # Must be divisor of shard_size

shards = (rm_shard_size, n_new_cols)
chunks = (rm_chunk_size, n_new_cols)

row_major_kwargs = {
    'shape': (n_rows, n_new_cols),
    'shards': shards,
    'chunks': chunks,
    'dtype': dtype,
    'overwrite': True,
    'fill_value': 0,
    'config':{'order': 'C'},
}

X_RM = g.create_array("X_RM", **row_major_kwargs)

cm_shard_size = 1024  
cm_chunk_size = 16 # Must be divisor of shard_size

shards = (n_rows, cm_shard_size)
chunks = (n_rows, cm_chunk_size)

column_major_kwargs = {
    'shape': (n_rows, n_new_cols),
    'shards': shards,
    'chunks': chunks,
    'dtype': dtype,
    'overwrite': True,
    'fill_value': 0,
    'config':{'order': 'F'},
}

X_CM = g.create_array("X_CM", **column_major_kwargs)

X = zarr_group['X']
gene_array = zarr_group['var']['gene']
gene_values = gene_array[:].tolist()

# Create mapping: old_col_idx -> new_col_idx (same for all chunks)
gene_to_old_idx = {gene: idx for idx, gene in enumerate(gene_values)}
old_to_new_idx = {}
for new_idx, gene in enumerate(gene_list):
    if gene in gene_to_old_idx:
        old_to_new_idx[gene_to_old_idx[gene]] = new_idx

# Iterate over processing chunks
process_chunk_size = rm_shard_size  # Use row-major shard size for processing
for r0 in range(0, n_rows, process_chunk_size):
    r1 = min(r0 + process_chunk_size, n_rows)
    chunk_n_rows = r1 - r0
    
    print(f"Processing rows {r0} to {r1-1} ({chunk_n_rows} rows)")
    
    # Load chunk from zarr
    indptr = X["indptr"][r0 : r1 + 1]
    start = indptr[0]
    end = indptr[-1]
    
    data = X["data"][start:end]
    indices = X["indices"][start:end]
    
    # rebase indptr to start at zero
    indptr = indptr - start
    
    # Create matrix with original column count (n_old_cols) to match indices from zarr
    X_chunk = csr_matrix((data, indices, indptr), shape=(chunk_n_rows, n_old_cols))
    X_chunk = X_chunk.tocsc()
    
    # Get CSC arrays
    old_data = X_chunk.data
    old_indices = X_chunk.indices
    old_indptr = X_chunk.indptr
    
    # Initialize column storage
    new_col_data = [[] for _ in range(n_new_cols)]
    new_col_indices = [[] for _ in range(n_new_cols)]
    
    # Iterate through old columns and map to new positions
    for old_col in range(n_old_cols):
        if old_col in old_to_new_idx:
            new_col = old_to_new_idx[old_col]
            col_start = old_indptr[old_col]
            col_end = old_indptr[old_col + 1]
            new_col_data[new_col].extend(old_data[col_start:col_end])
            new_col_indices[new_col].extend(old_indices[col_start:col_end])
    
    # Build new CSC arrays
    new_data = []
    new_indices = []
    new_indptr = [0]
    
    for new_col in range(n_new_cols):
        new_data.extend(new_col_data[new_col])
        new_indices.extend(new_col_indices[new_col])
        new_indptr.append(len(new_data))
    
    # Create new CSC matrix, then convert to dense for efficient chunk-aligned writing
    X_chunk_reordered = csc_matrix((new_data, new_indices, new_indptr), shape=(chunk_n_rows, n_new_cols)).tocsc()
    
    # Convert to dense format for efficient chunk-aligned writing
    # This improves query performance by writing full chunks
    X_chunk_dense = X_chunk_reordered.toarray()
    
    # Clip values to uint16 range and convert to uint16
    X_chunk_dense = np.clip(X_chunk_dense, 0, max_uint16).astype(dtype)
    
    # Write entire dense chunk at once (chunk-aligned for optimal read performance)
    # This is more efficient than sparse writes for dense zarr arrays
    X_RM[r0:r1, :] = X_chunk_dense
    X_CM[r0:r1, :] = X_chunk_dense

print(f"Completed processing all {n_rows} rows")
