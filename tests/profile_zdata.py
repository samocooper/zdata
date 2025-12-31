#!/usr/bin/env python3
"""
Profiling script for ZData to identify performance bottlenecks.
"""

import sys
import time
import cProfile
import pstats
import io
from pathlib import Path

# Add the parent directory to the path so we can import zdata
_test_dir = Path(__file__).parent
_project_root = _test_dir.parent
_parent_dir = _project_root.parent
sys.path.insert(0, str(_parent_dir))

import numpy as np
import subprocess
from zdata.core.zdata import ZData

def profile_read_rows(reader, rows):
    """Profile the read_rows method"""
    pr = cProfile.Profile()
    pr.enable()
    result = reader.read_rows(rows)
    pr.disable()
    return result, pr

def profile_subprocess_call(bin_path, file_path, rows_csv):
    """Profile a single subprocess call"""
    pr = cProfile.Profile()
    pr.enable()
    blob = subprocess.check_output([bin_path, "--binary", file_path, rows_csv])
    pr.disable()
    return blob, pr

def analyze_timing(reader, rows, num_iterations=5):
    """Detailed timing analysis"""
    print("=" * 80)
    print("DETAILED TIMING ANALYSIS")
    print("=" * 80)
    
    # Time subprocess calls separately
    rows_by_file = {}
    for global_row in rows[:100]:  # Sample first 100 rows
        chunk_num = global_row // 4096
        local_row = global_row % 4096
        file_path = reader.chunk_files[chunk_num]
        if file_path not in rows_by_file:
            rows_by_file[file_path] = []
        rows_by_file[file_path].append(local_row)
    
    print(f"\nQuery spans {len(rows_by_file)} files")
    print(f"Total rows to query: {len(rows)}")
    
    # Time subprocess overhead
    from zdata.core.zdata import _get_zdata_read_path
    bin_path = _get_zdata_read_path()
    
    subprocess_times = []
    for file_path, local_rows in list(rows_by_file.items())[:3]:  # Test first 3 files
        rows_csv = ",".join(map(str, local_rows[:50]))  # Test with 50 rows
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            blob = subprocess.check_output([bin_path, "--binary", file_path, rows_csv])
            end = time.perf_counter()
            times.append((end - start) * 1000)
        avg_time = np.mean(times)
        subprocess_times.append(avg_time)
        print(f"  File {file_path}: {avg_time:.2f} ms (avg over {num_iterations} runs, {len(local_rows[:50])} rows)")
    
    # Time full read_rows
    full_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = reader.read_rows(rows)
        end = time.perf_counter()
        full_times.append((end - start) * 1000)
    
    avg_full = np.mean(full_times)
    print(f"\nFull read_rows({len(rows)} rows): {avg_full:.2f} ms (avg over {num_iterations} runs)")
    print(f"  Per row: {avg_full/len(rows):.3f} ms")
    
    # Time Python overhead (parsing, etc.)
    if rows_by_file:
        file_path = list(rows_by_file.keys())[0]
        local_rows = rows_by_file[file_path][:100]
        rows_csv = ",".join(map(str, local_rows))
        
        # Time just subprocess
        subprocess_start = time.perf_counter()
        blob = subprocess.check_output([bin_path, "--binary", file_path, rows_csv])
        subprocess_time = (time.perf_counter() - subprocess_start) * 1000
        
        # Time parsing
        import struct
        parse_start = time.perf_counter()
        off = 0
        nreq, ncols = struct.unpack_from("<II", blob, off); off += 8
        out = []
        for i in range(nreq):
            row_id, nnz = struct.unpack_from("<II", blob, off); off += 8
            cols = np.frombuffer(blob, dtype=np.uint32, count=nnz, offset=off)
            off += nnz * 4
            vals = np.frombuffer(blob, dtype=np.uint16, count=nnz, offset=off)
            off += nnz * 2
            out.append((row_id, cols, vals))
        parse_time = (time.perf_counter() - parse_start) * 1000
        
        print(f"\nBreakdown for 100 rows from one file:")
        print(f"  Subprocess call: {subprocess_time:.2f} ms ({subprocess_time/(subprocess_time+parse_time)*100:.1f}%)")
        print(f"  Python parsing:  {parse_time:.2f} ms ({parse_time/(subprocess_time+parse_time)*100:.1f}%)")

def main():
    if len(sys.argv) < 2:
        print("Usage: profile_zdata.py <zdata_dir> [num_rows]")
        sys.exit(1)
    
    zdata_dir = sys.argv[1]
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    print(f"Profiling ZData with directory: {zdata_dir}")
    print(f"Query size: {num_rows} rows")
    print("=" * 80)
    
    # Initialize reader
    reader = ZData(zdata_dir)
    print(f"Dataset: {reader.shape}")
    print(f"Chunks: {sorted(reader.chunk_files.keys())}")
    
    # Generate random rows
    np.random.seed(42)
    rows = np.random.randint(0, reader.num_rows, size=num_rows).tolist()
    
    # Sort rows for better locality (as done in test_fast_queries.py)
    rows = sorted(rows)
    
    # Detailed timing analysis
    analyze_timing(reader, rows)
    
    # Full profiling
    print("\n" + "=" * 80)
    print("FULL PROFILING (cProfile)")
    print("=" * 80)
    
    result, pr = profile_read_rows(reader, rows)
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())
    
    # Save profile to file
    profile_file = f"profile_zdata_{num_rows}rows.prof"
    pr.dump_stats(profile_file)
    print(f"\nProfile saved to: {profile_file}")
    print("View with: python -m pstats", profile_file)

if __name__ == "__main__":
    main()

