"""
Tests for check_directory utility.

Tests verify that check_directory can process both zarr and h5ad files
and generate correct reports.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from zdata.build_zdata.check_directory import check_zarr_directory


class TestCheckDirectory:
    """Test check_directory functionality."""

    def test_check_zarr_directory(self, zarr_test_dir: Path, tmp_path: Path):
        """Test check_directory on zarr test directory."""
        # Run check_directory
        check_zarr_directory(str(zarr_test_dir))
        
        # Verify CSV report was created
        csv_path = zarr_test_dir / 'obs_report.csv'
        assert csv_path.exists(), "obs_report.csv should be created"
        
        # Verify CSV content
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) > 0, "CSV should contain at least one row"
        assert 'dataset_name' in rows[0], "CSV should have dataset_name column"
        assert 'common_columns' in rows[0], "CSV should have common_columns column"
        assert 'unique_columns' in rows[0], "CSV should have unique_columns column"
        
        # Verify all zarr files are in the report
        zarr_files = sorted([f.name for f in zarr_test_dir.iterdir() 
                            if f.is_dir() and f.name.endswith('.zarr')])
        csv_datasets = [row['dataset_name'] for row in rows]
        for zarr_file in zarr_files:
            assert zarr_file in csv_datasets, f"{zarr_file} should be in CSV report"

    def test_check_h5ad_directory(self, h5ad_test_dir: Path):
        """Test check_directory on h5ad test directory."""
        # Run check_directory
        check_zarr_directory(str(h5ad_test_dir))
        
        # Verify CSV report was created
        csv_path = h5ad_test_dir / 'obs_report.csv'
        assert csv_path.exists(), "obs_report.csv should be created"
        
        # Verify CSV content
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) > 0, "CSV should contain at least one row"
        
        # Verify all h5ad files are in the report
        h5ad_files = sorted([f.name for f in h5ad_test_dir.iterdir() 
                            if f.is_file() and (f.suffix in ['.h5', '.hdf5'] or f.name.endswith('.h5ad'))])
        csv_datasets = [row['dataset_name'] for row in rows]
        for h5ad_file in h5ad_files:
            assert h5ad_file in csv_datasets, f"{h5ad_file} should be in CSV report"

    def test_check_mixed_directory(self, zarr_test_dir: Path, h5ad_test_dir: Path, tmp_path: Path):
        """Test check_directory on a directory with both zarr and h5ad files."""
        # Create a temporary directory with both types
        mixed_dir = tmp_path / "mixed_test"
        mixed_dir.mkdir()
        
        # Copy a few zarr files (create symlinks to avoid copying)
        zarr_files = sorted([f for f in zarr_test_dir.iterdir() 
                            if f.is_dir() and f.name.endswith('.zarr')])[:2]
        for zarr_file in zarr_files:
            (mixed_dir / zarr_file.name).symlink_to(zarr_file)
        
        # Copy a few h5ad files (create symlinks)
        h5ad_files = sorted([f for f in h5ad_test_dir.iterdir() 
                           if f.is_file() and (f.suffix in ['.h5', '.hdf5'] or f.name.endswith('.h5ad'))])[:2]
        for h5ad_file in h5ad_files:
            (mixed_dir / h5ad_file.name).symlink_to(h5ad_file)
        
        # Run check_directory
        check_zarr_directory(str(mixed_dir))
        
        # Verify CSV report was created
        csv_path = mixed_dir / 'obs_report.csv'
        assert csv_path.exists(), "obs_report.csv should be created"
        
        # Verify both types are in the report
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        csv_datasets = [row['dataset_name'] for row in rows]
        assert len(csv_datasets) >= 4, "Should have at least 4 datasets (2 zarr + 2 h5ad)"
        
        # Verify zarr files are included
        zarr_names = [f.name for f in zarr_files]
        for name in zarr_names:
            assert name in csv_datasets, f"Zarr file {name} should be in report"
        
        # Verify h5ad files are included
        h5ad_names = [f.name for f in h5ad_files]
        for name in h5ad_names:
            assert name in csv_datasets, f"H5ad file {name} should be in report"

    def test_check_directory_nonexistent(self):
        """Test check_directory with nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            check_zarr_directory("/nonexistent/directory")

    def test_check_directory_not_a_directory(self, tmp_path: Path):
        """Test check_directory with a file instead of directory."""
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test")
        
        with pytest.raises(ValueError, match="Path is not a directory"):
            check_zarr_directory(str(test_file))

    def test_check_directory_empty(self, tmp_path: Path):
        """Test check_directory with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        # Should not raise, just print message
        check_zarr_directory(str(empty_dir))
        
        # CSV should not be created for empty directory
        csv_path = empty_dir / 'obs_report.csv'
        assert not csv_path.exists(), "CSV should not be created for empty directory"

