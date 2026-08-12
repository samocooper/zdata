API Reference
=============

Core
----

.. autoclass:: zdata.ZData
   :members:
   :undoc-members: False
   :show-inheritance:

.. autoclass:: zdata.ObsWrapper
   :members:
   :undoc-members: False
   :show-inheritance:


Building datasets
-----------------

.. autofunction:: zdata.build_zdata_from_zarr

.. autofunction:: zdata.build_zdata_from_mtx_csv

.. autofunction:: zdata.build_zdata


Post-build steps
----------------

Optional steps run after a dataset is built. They are what turn a readable
zdata object into one that machine-learning pipelines can train on: a
globally-unique sample key, compact obs dtypes, and a structural-zero mask.

.. autofunction:: zdata.run_post_build_steps

.. autoexception:: zdata.PostBuildError

.. autofunction:: zdata.assign_global_sample_id

.. autofunction:: zdata.build_feature_presence_matrix

.. autofunction:: zdata.optimize_obs_parquet

.. autofunction:: zdata.optimize_obs_dtypes


Utilities
---------

.. autofunction:: zdata.check_zarr_directory

.. autofunction:: zdata.get_default_gene_list_path

.. autofunction:: zdata.align_zarr_directory_to_mtx

.. autofunction:: zdata.concat_obs_from_zarr_directory


Constants
---------

.. autodata:: zdata.SUPPORTED_DTYPES
   :annotation: = {"uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float32", "float64"}
