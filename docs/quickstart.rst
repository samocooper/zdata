Quick Start
===========

Installation
------------

From PyPI::

    pip install zdata-py

From source (requires `Zstandard <https://github.com/facebook/zstd>`_ headers for
compiling the C tools)::

    git clone https://github.com/facebook/zstd.git
    cd zstd && make lib && cd ..

    git clone <repository-url> zdata
    cd zdata
    pip install -e .

The C tools (``mtx_to_zdata``, ``zdata_read``) are compiled automatically during
installation from the bundled Zstandard sources; only a C compiler is
required. Pre-compiled binaries are included in PyPI
wheels.


Building a dataset
------------------

From a directory of Zarr or H5AD files::

    from zdata import build_zdata_from_zarr

    build_zdata_from_zarr(
        input_dir="/path/to/data",
        output_name="my_dataset.zdata",
    )

From MTX+CSV directories (each subdirectory contains ``matrix.mtx``,
``obs.csv``, ``var.csv``)::

    from zdata import build_zdata_from_mtx_csv

    build_zdata_from_mtx_csv(
        input_dir="/path/to/mtx_csv_dirs",
        output_name="my_dataset.zdata",
    )


Opening a dataset
-----------------

::

    from zdata import ZData

    ds = ZData("my_dataset.zdata")
    print(ds.shape)       # (n_rows, n_cols)
    print(ds.obs.shape)   # observation metadata
    print(ds.var.shape)   # variable metadata


Querying rows
-------------

::

    # AnnData-style indexing (returns AnnData object)
    adata = ds[0:100]
    adata = ds[[0, 10, 20]]

    # Raw sparse matrix
    csr = ds.read_rows_csr([0, 100, 200])


Querying columns (genes)
-------------------------

Requires column-major (``X_CM``) data, which is built automatically::

    matrix = ds["GAPDH"]
    matrix = ds[["GAPDH", "PCNA", "TP53"]]
