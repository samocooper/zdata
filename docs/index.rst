zdata-py
========

Efficient sparse matrix storage and retrieval using **seekable Zstandard compression**.

``zdata`` stores large sparse numerical matrices in a block-compressed, chunked format
that supports fast random access to individual rows or columns without decompressing
entire files. It is designed for single-cell RNA-seq data but works with any sparse
matrix.

Key features:

- Random access to rows (cells) and columns (genes) without full decompression
- Block-compressed CSR layout with seekable Zstandard compression
- Scales to millions of rows across chunked files
- Builds from **Zarr**, **H5AD**, or **MTX+CSV** input formats
- Stores any numerical type: uint8 through float64
- Simple Python API with AnnData-compatible indexing

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   building
   querying
   configuration
   cli
   api
   format


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
