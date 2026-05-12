On-Disk Format
==============

A ``.zdata`` directory contains the following structure::

    atlas.zdata/
      metadata.json       # dataset shape, chunking info, dtype
      obs.parquet         # observation (cell) metadata
      var.parquet         # variable (gene) metadata
      X_RM/               # row-major compressed chunks
        0.bin, 1.bin, ...
      X_CM/               # column-major compressed chunks
        0.bin, 1.bin, ...


metadata.json
-------------

Contains dataset dimensions, chunk layout, and compression parameters:

.. code-block:: javascript

    {
      "version": 1,
      "format": "zdata",
      "dtype": "uint16",
      "shape": [100000, 35804],
      "nnz_total": 50000000,
      "num_chunks_rm": 13,
      "num_chunks_cm": 140,
      "block_rows": 16,
      "block_columns": 16,
      "max_rows_per_chunk": 8192,
      "max_columns_per_chunk": 256,
      "chunks_rm": ["..."],
      "chunks_cm": ["..."]
    }


obs.parquet / var.parquet
-------------------------

Standard Parquet files readable by any Parquet library. ``obs.parquet`` contains
one row per cell with metadata columns (barcodes, cell types, batch info, QC
metrics). ``var.parquet`` contains one row per gene with names and per-gene
non-zero counts.

When cells are filtered during build (via ``min_nnz``), ``obs.parquet`` may have
fewer rows than the expression matrix. In that case it contains a ``_row_index``
column mapping obs positions back to matrix row indices. See the **Index mapping** section in :doc:`querying` for usage.


Compressed chunk files (.bin)
-----------------------------

Each ``.bin`` file is a `seekable Zstandard <https://github.com/facebook/zstd/tree/dev/contrib/seekable_format>`_
archive containing one frame per block. Each decompressed frame is a binary
CSR block:

.. code-block:: text

    u32  magic          0x5253435A ("ZCSR")
    u32  version        dtype version number (see below)
    u32  start_row      first row index in this block
    u32  nrows_block    number of rows in this block
    u32  ncols          total number of columns
    u32  nnz            non-zero count in this block
    u32[block_rows+1]   indptr (CSR row pointers)
    u32[nnz]            indices (column indices)
    T[nnz]              data (values, element size depends on version)

All integers are little-endian.


Version-to-dtype mapping
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 10 15 10

   * - Version
     - dtype
     - Bytes
   * - 2
     - uint16
     - 2
   * - 3
     - float32
     - 4
   * - 4
     - uint8
     - 1
   * - 5
     - uint32
     - 4
   * - 6
     - uint64
     - 8
   * - 7
     - int8
     - 1
   * - 8
     - int16
     - 2
   * - 9
     - int32
     - 4
   * - 10
     - int64
     - 8
   * - 11
     - float64
     - 8


Row-major vs column-major
--------------------------

**X_RM** stores cells as rows and genes as columns. Each chunk file covers a
contiguous range of cells (default 8192 per file). Row queries decompress only
the relevant blocks within the target chunk.

**X_CM** stores genes as rows and cells as columns (transposed). Each chunk file
covers a contiguous range of genes (default 256 per file). This enables efficient
per-gene queries without scanning all row-major chunks.
