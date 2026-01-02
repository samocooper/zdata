#!/usr/bin/env python3
"""
Check zarr directory structure and gene/obs consistency.

This utility script checks:
- Common genes across datasets
- Obs column structure and consistency
- Generates CSV report of obs columns
"""

import os
import sys
import zarr
import csv
import argparse
from pathlib import Path


def check_zarr_directory(zarr_dir: str):
    """
    Check zarr directory structure and gene/obs consistency.
    
    Args:
        zarr_dir: Path to directory containing .zarr files
    """
    zarr_dir_path = Path(zarr_dir)
    if not zarr_dir_path.exists():
        raise FileNotFoundError(f"Zarr directory not found: {zarr_dir}")
    
    if not zarr_dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {zarr_dir}")
    
    zarr_files = sorted([f for f in zarr_dir_path.iterdir() if f.is_dir() and f.name.endswith('.zarr')])
    
    if not zarr_files:
        print(f"No .zarr files found in {zarr_dir}")
        return
    
    # Check common genes
    print("=" * 60)
    print("VAR/GENE CHECKS")
    print("=" * 60)
    
    common_genes = set()
    for i, zarr_file in enumerate(zarr_files):
        zarr_path = str(zarr_file)
        zarr_group = zarr.open(zarr_path, mode='r')
        
        if 'var' in zarr_group and 'gene' in zarr_group['var']:
            gene_array = zarr_group['var']['gene']
            gene_values = gene_array[:]
            outstring = f"Checking {zarr_file.name} - Number of genes: {len(gene_values)}"
            
            gene_set = set(gene_values.tolist())
            if len(common_genes.intersection(gene_set)) < 10000 and i > 0:
                outstring += f" - Warning: number of common genes appears low: {len(common_genes.intersection(gene_set))}"
            
            print(outstring)
            common_genes.update(gene_set)
        else:
            if 'var' in zarr_group:
                var_keys = list(zarr_group['var'].keys())
                print(f"  WARNING: 'gene' column not found in {zarr_file.name}. Available var keys: {var_keys}")
            else:
                print(f"  WARNING: 'var' group not found in {zarr_file.name}")
    
    print(f"\nTotal number of common genes: {len(common_genes)}")
    
    # Check obs structure
    print("\n" + "=" * 60)
    print("OBS STRUCTURE CHECKS")
    print("=" * 60)
    
    obs_columns_dict = {}
    missing_barcode_datasets = []
    
    for zarr_file in zarr_files:
        zarr_path = str(zarr_file)
        zarr_group = zarr.open(zarr_path, mode='r')
        
        if 'obs' in zarr_group:
            obs_keys = list(zarr_group['obs'].keys())
            obs_columns = [key for key in obs_keys if key != '_index']
            obs_columns_dict[zarr_file.name] = set(obs_columns)
            
            if 'barcode' not in obs_columns:
                missing_barcode_datasets.append(zarr_file.name)
        else:
            obs_columns_dict[zarr_file.name] = set()
            missing_barcode_datasets.append(zarr_file.name)
    
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
        description='Check zarr directory structure and gene/obs consistency.'
    )
    parser.add_argument(
        'zarr_dir',
        type=str,
        help='Directory containing .zarr files to check'
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
