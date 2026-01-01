"""
Example usage of zdata settings system.

This demonstrates how to configure zdata behavior through the settings system.
"""

import zdata

# View all available settings
print("Available settings:")
zdata.settings.describe()

# Access a setting
print(f"\nCurrent max_rows_per_chunk: {zdata.settings.max_rows_per_chunk}")
print(f"Current block_rows: {zdata.settings.block_rows}")

# Change a setting globally
zdata.settings.max_rows_per_chunk = 4096
print(f"\nAfter change, max_rows_per_chunk: {zdata.settings.max_rows_per_chunk}")

# Reset to default
zdata.settings.reset("max_rows_per_chunk")
print(f"After reset, max_rows_per_chunk: {zdata.settings.max_rows_per_chunk}")

# Use context manager for temporary override
print("\nUsing context manager for temporary override:")
with zdata.settings.override(max_rows_per_chunk=2048, block_rows=8):
    print(f"Inside context: max_rows_per_chunk = {zdata.settings.max_rows_per_chunk}")
    print(f"Inside context: block_rows = {zdata.settings.block_rows}")

print(f"Outside context: max_rows_per_chunk = {zdata.settings.max_rows_per_chunk}")
print(f"Outside context: block_rows = {zdata.settings.block_rows}")

# View specific setting
print("\nDescription of max_rows_per_chunk:")
zdata.settings.describe("max_rows_per_chunk")

# Environment variable example
print("\n" + "=" * 70)
print("To set via environment variable, use:")
print("  export ZDATA_MAX_ROWS_PER_CHUNK=4096")
print("  export ZDATA_BLOCK_ROWS=32")
print("  export ZDATA_WARN_ON_LARGE_QUERIES=0  # 0 for False, 1 for True")
print("=" * 70)

