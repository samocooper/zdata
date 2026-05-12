Querying Data
=============

Once a ``.zdata`` directory is built you open it with :class:`~zdata.ZData` and
query rows (cells) or columns (genes) without loading the full matrix into memory.


Opening a dataset
-----------------

::

    from zdata import ZData

    ds = ZData("atlas.zdata")
    print(ds.shape)        # (n_rows, n_cols)
    print(len(ds.obs))     # observation count
    print(len(ds.var))     # variable count


Row queries (cells)
-------------------

Row queries access the row-major (``X_RM``) chunk files.

**AnnData-style indexing** -- returns an ``anndata.AnnData`` object with ``X``,
``obs``, and ``var``::

    adata = ds[0]              # single row
    adata = ds[0:100]          # slice
    adata = ds[[0, 10, 20]]    # list of indices

**Raw sparse tuples** -- each element is ``(row_id, col_indices, values)``::

    rows = ds.read_rows([0, 100, 500])
    for row_id, cols, vals in rows:
        print(f"Row {row_id}: {len(cols)} non-zeros")

**CSR matrix** -- returns a ``scipy.sparse.csr_matrix``::

    csr = ds.read_rows_csr(slice(0, 1000))
    print(csr.shape)   # (1000, n_cols)

**Random sampling**::

    indices = ds.get_random_rows(50, seed=42)
    csr = ds.read_rows_csr(indices)

**Boolean masks**::

    import numpy as np
    mask = np.zeros(ds.nrows, dtype=bool)
    mask[0:500] = True
    rows = ds.read_rows(mask)


Column queries (genes)
----------------------

Column queries access the column-major (``X_CM``) chunk files. These are built
automatically during dataset construction.

**By gene name** -- returns a ``scipy.sparse.csc_matrix`` of shape
``(n_cells, n_selected_genes)``::

    matrix = ds["GAPDH"]
    matrix = ds[["GAPDH", "PCNA", "TP53"]]

**By column index**::

    csr = ds.read_cols_cm_csr([0, 100, 200])
    # shape: (n_selected_genes, n_cells)

**Raw tuples** -- each element is ``(col_id, row_indices, values)``::

    cols = ds.read_cols_cm([0, 1, 2])
    for col_id, rows, vals in cols:
        print(f"Gene {col_id}: {len(rows)} non-zero cells")


Index mapping
-------------

By default, ``obs`` and ``var`` dimensions must exactly match the expression matrix.
If your ``obs.parquet`` has been filtered (e.g. by ``min_nnz``) and contains a
mapping column, you can specify it at load time::

    # obs has fewer rows than the matrix; _row_index maps obs positions
    # to actual matrix row indices
    ds = ZData("atlas.zdata", obs_index_col="_row_index")

    # Now ds[i] indexes into obs, and the _row_index column translates
    # to the correct matrix row
    adata = ds[0:100]

The same mechanism works for variables::

    ds = ZData("atlas.zdata", var_index_col="_col_index")

If dimensions don't match and no mapping column is specified, a ``ValueError`` is
raised with an actionable error message.


Memory estimation
-----------------

Before running a large query you can estimate memory requirements::

    estimate = ds.estimate_memory_requirements(row_indices=[0, 1, 2])
    print(estimate["estimated_memory_mb"])
    print(estimate["estimated_memory_gb"])

By default, queries that would exceed 80% of available system memory raise a
``MemoryError``. Override this with::

    from zdata import settings
    settings.override_memory_check = True
