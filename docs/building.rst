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

Post-build steps
----------------

After the matrix and metadata are written, ``build_zdata_from_zarr`` and
``build_zdata_from_mtx_csv`` run three optional steps. Each can be disabled
with its own flag; all are on by default.

``sample_id`` / ``sample_uid``
    A ``sample_name`` is only unique *within* a study -- the same identifier
    recurs across studies, and some studies leave it null. For batch correction
    you need one globally-unique key, so :func:`zdata.assign_global_sample_id`
    adds a monotonic integer ``sample_id`` (contiguous per study) and a readable
    ``sample_uid``, falling back ``sample_name`` -> ``donor_id`` ->
    ``sample_idx`` where values are missing.

obs dtype optimisation
    :func:`zdata.optimize_obs_parquet` recasts low-cardinality strings to
    ``Enum`` and integers to their smallest fitting width. The win is in
    **decoded** size -- on a large obs this is commonly 5-10x, which is what
    matters when the whole table is loaded for training. On-disk size may be
    flat or marginally worse, since parquet already compresses strings well.

feature-presence matrix
    In a multi-study atlas each study measures a different subset of genes.
    Genes a study never measured are *structural* zeros, not biological ones,
    and must be masked out of a reconstruction loss.
    :func:`zdata.build_feature_presence_matrix` derives the mask from each
    study's ``var.csv`` rather than from expression non-zeros -- a measured gene
    that simply isn't expressed would otherwise be wrongly marked absent.

    This step needs a per-study ``var.csv`` source. The MTX+CSV builder uses its
    own input directories; the zarr builder has no such source, so pass
    ``feature_presence_var_dirs=[...]`` to enable it.

Failure policy
~~~~~~~~~~~~~~

These steps are optional, but several are load-bearing downstream: the
feature-presence matrix is indexed by ``sample_id``, so a failure in the first
step invalidates the third. By default a failure is reported and the build
continues, with a summary at the end and dependent steps skipped rather than
producing a second confusing error.

Pass ``strict_post_build=True`` to raise on the first failure instead. That is
the right choice for automated pipelines, where an atlas that *looks* built but
is missing its batch key is worse than a loud error::

    build_zdata_from_mtx_csv(input_dir, output_dir, strict_post_build=True)
