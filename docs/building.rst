Building Datasets
=================

``zdata`` can build compressed datasets from three input formats. In all cases the
pipeline aligns genes to a standard reference list, compresses the expression matrix
into seekable Zstandard chunks, and writes observation/variable metadata as Parquet
files.


Input formats
-------------

Zarr / H5AD
^^^^^^^^^^^^

A directory containing ``.zarr`` directories, ``.h5`` / ``.hdf5`` / ``.h5ad`` files,
or a mix of both::

    from zdata import build_zdata_from_zarr

    build_zdata_from_zarr(
        input_dir="/path/to/data",
        output_name="atlas.zdata",
        obs_join_strategy="outer",
        min_nnz=300,
    )

File types are auto-detected from extensions.


MTX + CSV
^^^^^^^^^

A directory of subdirectories, each containing:

- ``matrix.mtx`` -- sparse expression matrix in MatrixMarket format
- ``obs.csv`` -- observation (cell) metadata
- ``var.csv`` -- variable (gene) metadata

::

    from zdata import build_zdata_from_mtx_csv

    build_zdata_from_mtx_csv(
        input_dir="/path/to/mtx_dirs",
        output_name="atlas.zdata",
    )


Data types
----------

The ``dtype`` parameter controls the numerical type used to store values. The default
is ``uint16``, which is appropriate for raw scRNA-seq integer counts.

Supported types: ``uint8``, ``uint16``, ``uint32``, ``uint64``, ``int8``, ``int16``,
``int32``, ``int64``, ``float32``, ``float64``.

::

    from zdata import build_zdata

    build_zdata("aligned_mtx/", "output.zdata", dtype="float32")

The full set of supported type names is available as :data:`zdata.SUPPORTED_DTYPES`.


Gene list alignment
-------------------

During construction, gene columns are reordered to match a standard reference list.
By default the package ships ``files/2ks10c_genes.txt``. To use a custom list::

    build_zdata_from_zarr(
        input_dir="/path/to/data",
        output_name="atlas.zdata",
        gene_list_path="/path/to/my_genes.txt",   # one gene name per line
    )


Observation metadata options
----------------------------

When building from multiple input files the observation metadata from each file is
concatenated. The ``obs_join_strategy`` parameter controls how columns are handled:

- ``"outer"`` (default) -- keep all columns from all files, fill missing with null
- ``"inner"`` -- keep only columns present in every file

The ``min_nnz`` parameter filters out cells (rows) with fewer than the given number
of non-zero values. Set to ``None`` to disable filtering::

    build_zdata_from_zarr(
        input_dir="/path/to/data",
        output_name="atlas.zdata",
        obs_join_strategy="inner",
        min_nnz=None,
    )


Build parameters
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Parameter
     - Default
     - Description
   * - ``block_rows``
     - 16
     - Rows per compressed block (row-major)
   * - ``block_columns``
     - 16
     - Rows per compressed block (column-major)
   * - ``max_rows``
     - 8192
     - Maximum rows per chunk file (row-major)
   * - ``max_columns``
     - 256
     - Maximum rows per chunk file (column-major)
   * - ``dtype``
     - ``"uint16"``
     - Numerical type for values
   * - ``obs_join_strategy``
     - ``"outer"``
     - How to join obs columns across files
   * - ``min_nnz``
     - 300
     - Minimum non-zeros for cell inclusion (None to disable)
   * - ``mtx_chunk_size``
     - 131072
     - Max rows per intermediate MTX file during alignment
   * - ``gene_list_path``
     - package default
     - Path to standard gene list
