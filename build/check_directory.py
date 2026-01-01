import os
import zarr
import csv

#### VAR CHECKS ####

# Identify the number of common genes across datasets    
# Flag any datasets where either no genes are detected,  
# or the number of common genes is below 10,000          

zarr_dir = '/home/ubuntu/zarr_work/zarr_datasets'
zarr_files = [f for f in os.listdir(zarr_dir) if os.path.isdir(os.path.join(zarr_dir, f)) and f.endswith('.zarr')]

common_genes = set()
for i, zarr_file in enumerate(zarr_files):
    zarr_path = os.path.join(zarr_dir, zarr_file)
    
    # Open zarr group in read-only mode for memory efficiency
    zarr_group = zarr.open(zarr_path, mode='r')
    
    # Access the var/gene array directly - zarr only loads data when accessed
    if 'var' in zarr_group and 'gene' in zarr_group['var']:
        gene_array = zarr_group['var']['gene']
        # Read the gene names - zarr arrays are lazy, only loads when indexed
        gene_values = gene_array[:]  # This reads the full array, but it's just gene names (small)        
        outstring = f"Checking {zarr_file} - Number of genes: {len(gene_values)}"
        
        gene_set = set(gene_values.tolist())
        if len(common_genes.intersection(gene_set)) < 10000 and i > 0:
            outstring += f" - Warning number of common genes appears low: {len(common_genes.intersection(gene_set))}"
        
        print(outstring)
        common_genes.update(gene_set)
    else:
        # List available keys in var if gene is not found
        if 'var' in zarr_group:
            var_keys = list(zarr_group['var'].keys())
            print(f"  WARNING: 'gene' column not found in {zarr_file}. Available var keys: {var_keys}")
        else:
            print(f"  WARNING: 'var' group not found in {zarr_file}")

print(f"Total number of common genes: {len(common_genes)}")

#### OBS CHECKS ####

# Identify the number of common observations across datasets
# Check that 'barcode' column exists in every dataset
# Create dictionary of column names for each dataset
# Generate summary and CSV report

print("\n" + "="*60)
print("OBS STRUCTURE CHECKS")
print("="*60)

# Dictionary to store column names for each dataset
obs_columns_dict = {}
missing_barcode_datasets = []

# First pass: collect all column names for each dataset
for zarr_file in zarr_files:
    zarr_path = os.path.join(zarr_dir, zarr_file)
    
    # Open zarr group in read-only mode for memory efficiency
    zarr_group = zarr.open(zarr_path, mode='r')
    
    # Access the obs group
    if 'obs' in zarr_group:
        # Get all keys in obs (these are the column names)
        # In zarr, column names are the top-level keys under obs
        # They can be either Arrays (simple data) or Groups (categorical data with categories/codes)
        obs_keys = list(zarr_group['obs'].keys())
        # Filter out _index as it's metadata, not a data column
        obs_columns = [key for key in obs_keys if key != '_index']
        obs_columns_dict[zarr_file] = set(obs_columns)
        
        # Check for barcode column
        if 'barcode' not in obs_columns:
            missing_barcode_datasets.append(zarr_file)
    else:
        obs_columns_dict[zarr_file] = set()
        missing_barcode_datasets.append(zarr_file)

# Check barcode presence
if missing_barcode_datasets:
    print(f"\nWARNING: 'barcode' column missing in {len(missing_barcode_datasets)} dataset(s):")
    for dataset in missing_barcode_datasets:
        print(f"  - {dataset}")
else:
    print("\n✓ 'barcode' column found in all datasets")

# Find common columns across all datasets
if obs_columns_dict:
    # Start with columns from first dataset
    common_columns = obs_columns_dict[list(obs_columns_dict.keys())[0]].copy()
    
    # Intersect with all other datasets
    for dataset, columns in obs_columns_dict.items():
        common_columns = common_columns.intersection(columns)
    
    # Sort for consistent output
    common_columns = sorted(list(common_columns))
    
    # Print summary to terminal
    print(f"\nSummary of OBS columns:")
    print(f"  Total datasets: {len(obs_columns_dict)}")
    print(f"  Common columns (present in all datasets): {len(common_columns)}")
    print(f"  Common column names: {', '.join(common_columns)}")
    
    # Generate CSV report
    csv_path = os.path.join(zarr_dir, 'obs_report.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['dataset_name', 'common_columns', 'unique_columns'])
        
        # Write data for each dataset
        for zarr_file in sorted(obs_columns_dict.keys()):
            dataset_columns = obs_columns_dict[zarr_file]
            unique_columns = sorted(list(dataset_columns - set(common_columns)))
            
            # Convert lists to string representations
            common_str = ', '.join(common_columns) if common_columns else ''
            unique_str = ', '.join(unique_columns) if unique_columns else ''
            
            writer.writerow([zarr_file, common_str, unique_str])
    
    print(f"\n✓ CSV report saved to: {csv_path}")
else:
    print("\nWARNING: No datasets found with 'obs' structure")