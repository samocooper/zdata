import zarr
from zarr.codecs import ShardingCodec
from scipy.sparse import csr_matrix, csc_matrix
import numpy as np

with open("/home/ubuntu/zarr_work/zdata/files/2ks10c_genes.txt") as f:
    gene_list = [line.strip() for line in f if line.strip()]

zarr_path = '/home/ubuntu/zarr_work/zarr_datasets/external_andrews_hepatolcommun_2022_34792289.zarr'
zarr_group = zarr.open(zarr_path, mode='r')

# number of columns (usually len(var))
n_cols = zarr_group["var"]["gene"].shape[0]
n_rows = zarr_group["obs"]["barcode"].shape[0]

out_path = '/home/ubuntu/zarr_work/tmp/tmp_rows.zarr'
name = "X_RM"

length, width = n_rows, n_cols
chunks = (500, n_cols)
dtype = np.float32
shard_length = 32

# Configure sharding codec
sharding_codec = ShardingCodec(chunks_per_shard=shard_length)

g = zarr.open_group(out_path, mode="w")  # use mode="a" to append to existing store

X_RM = g.create_array(name, shape=(length, width),
                     chunks=chunks, dtype=dtype, 
                     codec=sharding_codec,
                     overwrite=True, fill_value=0.0)

X = zarr_group['X']
gene_array = zarr_group['var']['gene']
gene_values = gene_array[:].tolist()

# Create mapping: old_col_idx -> new_col_idx (same for all chunks)
gene_to_old_idx = {gene: idx for idx, gene in enumerate(gene_values)}
old_to_new_idx = {}
for new_idx, gene in enumerate(gene_list):
    if gene in gene_to_old_idx:
        old_to_new_idx[gene_to_old_idx[gene]] = new_idx

n_new_cols = len(gene_list)
chunk_size = 500

# Iterate over 500-row chunks
for r0 in range(0, n_rows, chunk_size):
    r1 = min(r0 + chunk_size, n_rows)
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
    
    X_chunk = csr_matrix((data, indices, indptr), shape=(chunk_n_rows, n_cols))
    X_chunk = X_chunk.tocsc()
    
    # Get CSC arrays
    old_data = X_chunk.data
    old_indices = X_chunk.indices
    old_indptr = X_chunk.indptr
    n_old_cols = X_chunk.shape[1]
    
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
    
    # Create new CSC matrix
    X_chunk_reordered = csc_matrix((new_data, new_indices, new_indptr), shape=(chunk_n_rows, n_new_cols)).tocsc()
    
    # Stream sparse chunk into zarr array efficiently
    # Convert to COO format to get (row, col, data) tuples
    coo = X_chunk_reordered.tocoo()
    
    # Adjust row indices to global coordinates (r0 offset)
    rows = coo.row + r0
    cols = coo.col
    chunk_data = coo.data.astype(dtype)
    
    # Write non-zero values directly to zarr array
    # Using advanced indexing to write only sparse positions
    X_RM[rows, cols] = chunk_data

print(f"Completed processing all {n_rows} rows")
