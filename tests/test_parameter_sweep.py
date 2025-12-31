#!/usr/bin/env python3
"""
Parameter sweep test: Tests all combinations of BLOCK_ROWS and MAX_ROWS values
and outputs performance metrics to a CSV file.
"""

import sys
import subprocess
import csv
import os
import time
from pathlib import Path

# Add the parent directory to the path so we can import zdata
_test_dir = Path(__file__).parent
_project_root = _test_dir.parent  # This is /home/ubuntu/zdata_work/zdata
_parent_dir = _project_root.parent  # This is /home/ubuntu/zdata_work
sys.path.insert(0, str(_parent_dir))

from zdata.core.build_zdata import build_zdata
import shutil

# Configuration
BLOCK_ROWS_VALUES = [2, 4, 8, 16, 32, 64, 128, 256]
MAX_ROWS_VALUES = [2048, 4096, 8192, 16384]

# Default MTX file (can be overridden via command line)
DEFAULT_MTX_FILE = "/home/ubuntu/zdata_work/mtx_files/external_andrews_hepatolcommun_2022_34792289.mtx"

def parse_mean_query_time(output_text):
    """
    Parse the mean query time from test_fast_queries.py output.
    Looks for line like: "  Mean: 1603.888 ms"
    """
    found_query_times_section = False
    for line in output_text.split('\n'):
        if 'Query times (row retrieval):' in line:
            found_query_times_section = True
            continue
        if found_query_times_section and 'Mean:' in line and 'ms' in line:
            # Extract the number before "ms"
            parts = line.split('Mean:')
            if len(parts) > 1:
                value_part = parts[1].strip().split()[0]
                try:
                    return float(value_part)
                except ValueError:
                    pass
    return None

def run_fast_queries_test(zdata_dir):
    """
    Run test_fast_queries.py on a zdata directory and return mean query time.
    """
    test_script = _test_dir / "test_fast_queries.py"
    
    # Convert to absolute path if it's not already
    if os.path.isabs(zdata_dir):
        zdata_path = zdata_dir
    else:
        zdata_path = os.path.abspath(zdata_dir)
    
    # Pass the full path directly to test_fast_queries.py
    result = subprocess.run(
        [sys.executable, str(test_script), zdata_path],
        capture_output=True,
        text=True,
        timeout=600  # 10 minute timeout per test
    )
        
        if result.returncode != 0:
            print(f"  WARNING: test_fast_queries.py failed with return code {result.returncode}")
            print(f"  STDERR: {result.stderr[:500]}")
            return None
        
        mean_time = parse_mean_query_time(result.stdout)
        return mean_time
        
    except subprocess.TimeoutExpired:
        print(f"  ERROR: test_fast_queries.py timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"  ERROR: Exception running test: {e}")
        return None

def test_parameter_combination(mtx_file, block_rows, max_rows, output_base_name):
    """
    Test a single parameter combination.
    Returns (block_rows, max_rows, mean_query_time_ms, success)
    """
    print(f"\n{'='*80}")
    print(f"Testing: BLOCK_ROWS={block_rows}, MAX_ROWS={max_rows}")
    print(f"{'='*80}")
    
    # Create unique output name for this combination
    output_name = f"{output_base_name}_b{block_rows}_m{max_rows}"
    output_dir = Path(_parent_dir) / f"{output_name}.zdata"
    
    # Clean up any existing directory
    if output_dir.exists():
        print(f"  Cleaning up existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    try:
        # Build zdata with these parameters
        print(f"  Building zdata with block_rows={block_rows}, max_rows={max_rows}...")
        start_build = time.time()
        
        original_cwd = os.getcwd()
        try:
            os.chdir(_parent_dir)
            build_zdata(mtx_file, output_name, block_rows=block_rows, max_rows=max_rows)
        finally:
            os.chdir(original_cwd)
        
        build_time = time.time() - start_build
        print(f"  Build completed in {build_time:.2f} seconds")
        
        # Run performance test
        print(f"  Running performance test...")
        start_test = time.time()
        mean_query_time = run_fast_queries_test(str(output_dir))
        test_time = time.time() - start_test
        
        if mean_query_time is not None:
            print(f"  Test completed in {test_time:.2f} seconds")
            print(f"  Mean query time: {mean_query_time:.3f} ms")
            return (block_rows, max_rows, mean_query_time, True)
        else:
            print(f"  Test failed or could not parse results")
            return (block_rows, max_rows, None, False)
            
    except Exception as e:
        print(f"  ERROR: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return (block_rows, max_rows, None, False)
    finally:
        # Clean up test directory to save space
        if output_dir.exists():
            print(f"  Cleaning up test directory: {output_dir}")
            shutil.rmtree(output_dir)

def main():
    """Run parameter sweep and output results to CSV."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parameter sweep test for BLOCK_ROWS and MAX_ROWS values'
    )
    parser.add_argument(
        '--mtx-file',
        type=str,
        default=DEFAULT_MTX_FILE,
        help=f'Path to MTX file (default: {DEFAULT_MTX_FILE})'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default='parameter_sweep_results.csv',
        help='Output CSV file name (default: parameter_sweep_results.csv)'
    )
    parser.add_argument(
        '--keep-dirs',
        action='store_true',
        help='Keep test directories after completion (default: delete them)'
    )
    
    args = parser.parse_args()
    
    mtx_file = args.mtx_file
    output_csv = args.output_csv
    
    # Check if MTX file exists
    if not os.path.exists(mtx_file):
        print(f"ERROR: MTX file not found: {mtx_file}")
        sys.exit(1)
    
    print("="*80)
    print("ZData Parameter Sweep Test")
    print("="*80)
    print(f"MTX file: {mtx_file}")
    print(f"BLOCK_ROWS values: {BLOCK_ROWS_VALUES}")
    print(f"MAX_ROWS values: {MAX_ROWS_VALUES}")
    print(f"Total combinations: {len(BLOCK_ROWS_VALUES) * len(MAX_ROWS_VALUES)}")
    print(f"Output CSV: {output_csv}")
    print("="*80)
    
    # Generate output base name from MTX file
    mtx_basename = Path(mtx_file).stem
    output_base_name = f"{mtx_basename}_sweep"
    
    # Prepare results list
    results = []
    
    # Test all combinations
    total_combinations = len(BLOCK_ROWS_VALUES) * len(MAX_ROWS_VALUES)
    current = 0
    
    for block_rows in BLOCK_ROWS_VALUES:
        for max_rows in MAX_ROWS_VALUES:
            current += 1
            print(f"\n[{current}/{total_combinations}] ", end="")
            
            block_rows_val, max_rows_val, mean_time, success = test_parameter_combination(
                mtx_file, block_rows, max_rows, output_base_name
            )
            
            results.append({
                'block_rows': block_rows_val,
                'max_rows': max_rows_val,
                'mean_query_time_ms': mean_time if success else 'FAILED',
                'success': success
            })
            
            # Write intermediate results to CSV after each test
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['block_rows', 'max_rows', 'mean_query_time_ms', 'success'])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"  Progress: {current}/{total_combinations} combinations tested")
    
    # Final summary
    print("\n" + "="*80)
    print("Parameter Sweep Complete")
    print("="*80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"Total combinations tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults written to: {output_csv}")
    
    # Print summary statistics
    successful_results = [r for r in results if r['success']]
    if successful_results:
        times = [r['mean_query_time_ms'] for r in successful_results]
        print(f"\nQuery time statistics (successful tests only):")
        print(f"  Min: {min(times):.3f} ms")
        print(f"  Max: {max(times):.3f} ms")
        print(f"  Mean: {sum(times)/len(times):.3f} ms")
        
        # Find best combination
        best = min(successful_results, key=lambda x: x['mean_query_time_ms'])
        print(f"\nBest combination:")
        print(f"  BLOCK_ROWS={best['block_rows']}, MAX_ROWS={best['max_rows']}")
        print(f"  Mean query time: {best['mean_query_time_ms']:.3f} ms")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

