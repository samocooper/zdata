#!/usr/bin/env python3
"""
Check zarr/h5ad directory structure and gene/obs consistency.

This utility script checks:
- Common genes across datasets (supports both .zarr and .h5ad files)
- Obs column structure and consistency
- Generates CSV report of obs columns
"""

import os
import sys
import zarr
import csv
import argparse
from pathlib import Path
import anndata as ad


def check_zarr_directory(zarr_dir: str):
    """
    Check zarr/h5ad directory structure and gene/obs consistency.
    
    Args:
        zarr_dir: Path to directory containing .zarr and/or .h5ad files
    """
    zarr_dir_path = Path(zarr_dir)
    if not zarr_dir_path.exists():
        raise FileNotFoundError(f"Zarr directory not found: {zarr_dir}")
    
    if not zarr_dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {zarr_dir}")
    
    # Find both .zarr directories and .h5ad/.h5 files
    zarr_files = sorted([f for f in zarr_dir_path.iterdir() if f.is_dir() and f.name.endswith('.zarr')])
    h5ad_files = sorted([f for f in zarr_dir_path.iterdir() 
                        if f.is_file() and (f.suffix in ['.h5', '.hdf5'] or f.name.endswith('.h5ad'))])
    all_files = zarr_files + h5ad_files
    
    if not all_files:
        print(f"No .zarr or .h5ad/.h5 files found in {zarr_dir}")
        return
    
    print(f"Found {len(zarr_files)} .zarr file(s) and {len(h5ad_files)} .h5ad/.h5 file(s)\n")
    
    # Check common genes
    print("=" * 60)
    print("VAR/GENE CHECKS")
    print("=" * 60)
    
    common_genes = set()
    for i, data_file in enumerate(all_files):
        file_path = str(data_file)
        
        # Determine file type and read accordingly
        if data_file.suffix in ['.h5', '.hdf5'] or data_file.name.endswith('.h5ad'):
            # Read h5ad file with anndata
            try:
                adata = ad.read_h5ad(file_path, backed='r')
                if 'gene' in adata.var.columns:
                    gene_values = adata.var['gene'].values
                elif adata.var.index.name == 'gene' or 'gene' in adata.var.index.names:
                    gene_values = adata.var.index.values
                else:
                    print(f"  WARNING: 'gene' column not found in {data_file.name}. Available var columns: {list(adata.var.columns)}")
                    continue
                
                gene_set = set(gene_values.tolist())
                file_type = '.h5ad' if data_file.name.endswith('.h5ad') else data_file.suffix
                outstring = f"Checking {data_file.name} ({file_type}) - Number of genes: {len(gene_values)}"
                
            except Exception as e:
                print(f"  ERROR: Failed to read {data_file.name}: {e}")
                continue
        else:
            # Read zarr file
            zarr_group = zarr.open(file_path, mode='r')
            
            if 'var' in zarr_group and 'gene' in zarr_group['var']:
                gene_obj = zarr_group['var']['gene']
                
                # Handle both Array and Group (categorical) cases
                if isinstance(gene_obj, zarr.Array):
                    gene_values = gene_obj[:]
                elif isinstance(gene_obj, zarr.Group):
                    # For categorical, get the categories array
                    if 'categories' in gene_obj:
                        categories = gene_obj['categories']
                        gene_values = categories[:]
                    else:
                        print(f"  WARNING: 'gene' is a group but has no 'categories' in {data_file.name}")
                        continue
                else:
                    print(f"  WARNING: 'gene' has unexpected type {type(gene_obj)} in {data_file.name}")
                    continue
                
                gene_set = set(gene_values.tolist())
                outstring = f"Checking {data_file.name} (.zarr) - Number of genes: {len(gene_values)}"
            else:
                if 'var' in zarr_group:
                    var_keys = list(zarr_group['var'].keys())
                    print(f"  WARNING: 'gene' column not found in {data_file.name}. Available var keys: {var_keys}")
                else:
                    print(f"  WARNING: 'var' group not found in {data_file.name}")
                continue
        
        if len(common_genes.intersection(gene_set)) < 10000 and i > 0:
            outstring += f" - Warning: number of common genes appears low: {len(common_genes.intersection(gene_set))}"
        
        print(outstring)
        common_genes.update(gene_set)
    
    print(f"\nTotal number of common genes: {len(common_genes)}")
    
    # Check obs structure
    print("\n" + "=" * 60)
    print("OBS STRUCTURE CHECKS")
    print("=" * 60)
    
    obs_columns_dict = {}
    missing_barcode_datasets = []
    
    for data_file in all_files:
        file_path = str(data_file)
        
        # Determine file type and read accordingly
        if data_file.suffix in ['.h5', '.hdf5'] or data_file.name.endswith('.h5ad'):
            # Read h5ad file with anndata
            try:
                adata = ad.read_h5ad(file_path, backed='r')
                obs_columns = list(adata.obs.columns)
                obs_columns_dict[data_file.name] = set(obs_columns)
                
                if 'barcode' not in obs_columns:
                    missing_barcode_datasets.append(data_file.name)
            except Exception as e:
                print(f"  ERROR: Failed to read {data_file.name}: {e}")
                obs_columns_dict[data_file.name] = set()
                missing_barcode_datasets.append(data_file.name)
        else:
            # Read zarr file
            zarr_group = zarr.open(file_path, mode='r')
            
            if 'obs' in zarr_group:
                obs_keys = list(zarr_group['obs'].keys())
                obs_columns = [key for key in obs_keys if key != '_index']
                obs_columns_dict[data_file.name] = set(obs_columns)
                
                if 'barcode' not in obs_columns:
                    missing_barcode_datasets.append(data_file.name)
            else:
                obs_columns_dict[data_file.name] = set()
                missing_barcode_datasets.append(data_file.name)
    
    if missing_barcode_datasets:
        print(f"\nWARNING: 'barcode' column missing in {len(missing_barcode_datasets)} dataset(s):")
        for dataset in missing_barcode_datasets:
            print(f"  - {dataset}")
    else:
        print("\n✓ 'barcode' column found in all datasets")
    
    if obs_columns_dict:
        common_columns = obs_columns_dict[list(obs_columns_dict.keys())[0]].copy()
        
        for dataset, columns in obs_columns_dict.items():
            common_columns = common_columns.intersection(columns)
        
        common_columns = sorted(list(common_columns))
        
        print(f"\nSummary of OBS columns:")
        print(f"  Total datasets: {len(obs_columns_dict)}")
        print(f"  Common columns (present in all datasets): {len(common_columns)}")
        print(f"  Common column names: {', '.join(common_columns)}")
        
        csv_path = zarr_dir_path / 'obs_report.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset_name', 'common_columns', 'unique_columns'])
            
            for zarr_file_name in sorted(obs_columns_dict.keys()):
                dataset_columns = obs_columns_dict[zarr_file_name]
                unique_columns = sorted(list(dataset_columns - set(common_columns)))
                
                common_str = ', '.join(common_columns) if common_columns else ''
                unique_str = ', '.join(unique_columns) if unique_columns else ''
                
                writer.writerow([zarr_file_name, common_str, unique_str])
        
        print(f"\n✓ CSV report saved to: {csv_path}")
    else:
        print("\nWARNING: No datasets found with 'obs' structure")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Check zarr/h5ad directory structure and gene/obs consistency.'
    )
    parser.add_argument(
        'zarr_dir',
        type=str,
        help='Directory containing .zarr and/or .h5ad files to check'
    )
    
    args = parser.parse_args()
    
    try:
        check_zarr_directory(args.zarr_dir)
        return 0
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
