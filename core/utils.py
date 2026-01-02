"""Utility functions for zdata package.

This module contains general-purpose utility functions that are not specific
to zdata's core functionality but are used throughout the package.
"""

from __future__ import annotations

import platform
import subprocess


def get_available_memory_bytes() -> int:
    """Get available system memory in bytes using cross-platform methods.
    
    Tries multiple approaches in order:
    1. psutil (if available) - most reliable and cross-platform
    2. Linux: /proc/meminfo
    3. macOS/BSD: sysctl hw.memsize and vm_stat
    4. Fallback: conservative default (32 GB)
    
    Returns
    -------
    int
        Available memory in bytes.
    
    Examples
    --------
    >>> mem_bytes = get_available_memory_bytes()
    >>> mem_gb = mem_bytes / (1024 ** 3)
    >>> print(f"Available memory: {mem_gb:.2f} GB")
    """
    # Method 1: Try psutil (best cross-platform solution)
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    
    # Method 2: Linux - read from /proc/meminfo
    if platform.system() == 'Linux':
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        # Format: "MemAvailable:   12345678 kB"
                        parts = line.split()
                        if len(parts) >= 2:
                            kb_available = int(parts[1])
                            return kb_available * 1024  # Convert KB to bytes
        except (FileNotFoundError, ValueError, OSError):
            pass
    
    # Method 3: macOS/BSD - use sysctl and vm_stat
    if platform.system() == 'Darwin' or platform.system().endswith('BSD'):
        try:
            # Get total memory using sysctl
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            total_memory = int(result.stdout.strip())
            
            # Get free memory using vm_stat
            # vm_stat shows pages, we need to get page size first
            page_size_result = subprocess.run(
                ['sysctl', '-n', 'vm.pagesize'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            page_size = int(page_size_result.stdout.strip())
            
            # Get free pages from vm_stat
            vm_stat_result = subprocess.run(
                ['vm_stat'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            
            # Parse vm_stat output for free pages
            # Look for "Pages free:" line
            free_pages = 0
            inactive_pages = 0
            for line in vm_stat_result.stdout.split('\n'):
                if 'Pages free:' in line:
                    # Format: "Pages free:                         12345678."
                    parts = line.split(':')
                    if len(parts) == 2:
                        free_pages = int(parts[1].strip().rstrip('.').replace(',', ''))
                elif 'Pages inactive:' in line:
                    # Format: "Pages inactive:                     12345678."
                    parts = line.split(':')
                    if len(parts) == 2:
                        inactive_pages = int(parts[1].strip().rstrip('.').replace(',', ''))
            
            # Available memory = free + inactive (macOS considers inactive as available)
            available_memory = (free_pages + inactive_pages) * page_size
            return available_memory
            
        except (subprocess.SubprocessError, ValueError, FileNotFoundError, OSError):
            pass
    
    # Method 4: Try sysctl on other Unix systems (FreeBSD, OpenBSD, etc.)
    if platform.system() != 'Windows':
        try:
            # Try to get available memory using sysctl (varies by system)
            # FreeBSD: vm.stats.vm.v_free_count
            # OpenBSD: vm.stats.vm.v_free
            # This is more complex and system-specific, so we'll try a generic approach
            result = subprocess.run(
                ['sysctl', '-a'],
                capture_output=True,
                text=True,
                check=False,
                timeout=5
            )
            # This is too complex to parse generically, so we'll skip it
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass
    
    # Fallback: return a conservative default (32 GB)
    # This ensures the code doesn't crash, but users should install psutil for accurate results
    return 32 * 1024 * 1024 * 1024

