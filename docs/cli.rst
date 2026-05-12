Command-Line Tools
==================

After installing ``zdata-py``, four CLI commands are available.


zdata-build
-----------

Build a ``.zdata`` directory from a directory of Zarr or H5AD files::

    zdata-build /path/to/data my_dataset.zdata

Options::

    zdata-build /path/to/data output.zdata \
        --gene-list /path/to/genes.txt \
        --block-rows 16 \
        --max-rows 8192 \
        --obs-join-strategy outer \
        --min-nnz 300

Run ``zdata-build --help`` for all options.


zdata-build-mtx
----------------

Build a ``.zdata`` directory from MTX+CSV subdirectories::

    zdata-build-mtx /path/to/mtx_dirs my_dataset.zdata

Each subdirectory must contain ``matrix.mtx``, ``obs.csv``, and ``var.csv``.

Options::

    zdata-build-mtx /path/to/mtx_dirs output.zdata \
        --gene-list /path/to/genes.txt \
        --dtype float32 \
        --min-nnz 0

Run ``zdata-build-mtx --help`` for all options.


zdata-check
-----------

Validate the structure of an input directory before building::

    zdata-check /path/to/data

Reports the number of Zarr/H5AD files found, gene consistency across files,
and observation column overlap.


zdata-align
-----------

Run only the gene alignment step (produces intermediate MTX files without
compressing)::

    zdata-align /path/to/data output_dir --gene-list genes.txt

This is useful for inspecting the intermediate representation or for
rerunning the compression step separately.


Low-level C tools
-----------------

The compiled C binaries are also available directly:

**mtx_to_zdata** -- compress an MTX file into seekable Zstandard format::

    ./ctools/mtx_to_zdata --dtype float32 matrix.mtx output_dir 16 8192

**zdata_read** -- read specific rows from a compressed chunk file::

    ./ctools/zdata_read --binary --block-rows 16 output_dir/X_RM/0.bin "0,1,100"

Run either tool without arguments to see full usage.
