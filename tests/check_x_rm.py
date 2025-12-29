import zarr
import numpy as np

# Open the output zarr array
zarr_path = '/home/ubuntu/zarr_work/tmp/tmp_rows.zarr'
g = zarr.open_group(zarr_path, mode='r')

if 'X_RM' not in g:
    print("ERROR: X_RM array not found in zarr group")
    exit(1)

X_RM = g['X_RM']

print(f"X_RM shape: {X_RM.shape}")
print(f"X_RM dtype: {X_RM.dtype}")
print(f"X_RM chunks: {X_RM.chunks}")

# Check if data has been written (non-zero values exist)
total_nonzero = np.count_nonzero(X_RM[:])
total_elements = X_RM.size
sparsity = 1.0 - (total_nonzero / total_elements) if total_elements > 0 else 0.0

print(f"\nData check:")
print(f"  Total elements: {total_elements:,}")
print(f"  Non-zero elements: {total_nonzero:,}")
print(f"  Sparsity: {sparsity:.4f}")

# Count non-zero values in input zarr file
input_zarr_path = '/home/ubuntu/zarr_work/zarr_datasets/external_andrews_hepatolcommun_2022_34792289.zarr'
input_zarr_group = zarr.open(input_zarr_path, mode='r')

if 'X' not in input_zarr_group:
    print("\nERROR: 'X' not found in input zarr file")
    exit(1)

X_input = input_zarr_group['X']
# The data array in sparse zarr format contains only non-zero values
input_nonzero = X_input['data'].shape[0]

print(f"\nInput zarr file non-zero count: {input_nonzero:,}")
print(f"X_RM non-zero count: {total_nonzero:,}")

if input_nonzero == total_nonzero:
    print("✓ PASS: Non-zero counts match!")
else:
    print(f"✗ FAIL: Non-zero counts do not match (difference: {abs(input_nonzero - total_nonzero):,})")

# Calculate and output row totals for first 10 rows
print(f"\nRow totals for first 10 rows:")
for i in range(min(10, X_RM.shape[0])):
    row_data = X_RM[i, :]
    row_total = np.sum(row_data)
    row_nonzero = np.count_nonzero(row_data)
    print(f"  Row {i}: total = {row_total:.2f}, non-zero elements = {row_nonzero}")

