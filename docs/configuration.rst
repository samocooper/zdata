Configuration
=============

``zdata`` provides a settings system that can be configured at runtime, via
environment variables, or with a context manager.

::

    from zdata import settings


Runtime settings
----------------

::

    settings.block_rows = 32
    settings.warn_on_large_queries = False


Environment variables
---------------------

All settings can be set via environment variables with the ``ZDATA_`` prefix::

    export ZDATA_MAX_WORKERS=4
    export ZDATA_WARN_ON_LARGE_QUERIES=0
    export ZDATA_OVERRIDE_MEMORY_CHECK=1


Context manager
---------------

Temporarily override settings for a block of code::

    with settings.override(override_memory_check=True, max_workers=1):
        csr = ds.read_rows_csr(range(100000))
    # settings are restored after the block


Available settings
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 10 20 45

   * - Setting
     - Default
     - Env var
     - Description
   * - ``max_rows_per_chunk``
     - 8192
     - ``ZDATA_MAX_ROWS_PER_CHUNK``
     - Maximum rows per chunk file
   * - ``block_rows``
     - 16
     - ``ZDATA_BLOCK_ROWS``
     - Rows per compressed block
   * - ``warn_on_large_queries``
     - ``True``
     - ``ZDATA_WARN_ON_LARGE_QUERIES``
     - Warn when querying more than ``large_query_threshold`` rows
   * - ``large_query_threshold``
     - 50000
     - ``ZDATA_LARGE_QUERY_THRESHOLD``
     - Row count that triggers the large query warning
   * - ``override_memory_check``
     - ``False``
     - ``ZDATA_OVERRIDE_MEMORY_CHECK``
     - Allow queries exceeding 80% of available memory (warns instead of
       raising ``MemoryError``)
   * - ``max_workers``
     - ``None``
     - ``ZDATA_MAX_WORKERS``
     - Maximum thread pool workers for parallel chunk reads. ``None`` means
       auto-detect based on CPU count (capped at 8).


Inspecting settings
-------------------

::

    settings.describe()          # print all settings with descriptions
    settings.describe("block_rows")  # print one setting
