"""
zdata - Efficient sparse matrix storage and retrieval using seekable zstd compression.
"""

from __future__ import annotations

__version__ = "0.2.0"

from zdata._settings import settings
from zdata.core import ObsWrapper, ZData
from zdata.build_zdata.build_zdata import build_zdata_from_zarr
from zdata.build_zdata.build_from_mtx_csv import build_zdata_from_mtx_csv
from zdata.build_zdata.build_x import build_zdata, SUPPORTED_DTYPES
from zdata.build_zdata.align_mtx import align_zarr_directory_to_mtx, get_default_gene_list_path
from zdata.build_zdata.concat_obs import concat_obs_from_zarr_directory
from zdata.build_zdata.check_directory import check_zarr_directory
from zdata.build_zdata.feature_presence import build_feature_presence_matrix
from zdata.build_zdata.sample_id import assign_global_sample_id
from zdata.build_zdata.optimize_obs import optimize_obs_dtypes, optimize_obs_parquet
from zdata.build_zdata.post_build import run_post_build_steps, PostBuildError

__all__ = [
    "ObsWrapper",
    "ZData",
    "__version__",
    "settings",
    "build_zdata_from_zarr",
    "build_zdata_from_mtx_csv",
    "build_zdata",
    "SUPPORTED_DTYPES",
    "align_zarr_directory_to_mtx",
    "get_default_gene_list_path",
    "concat_obs_from_zarr_directory",
    "check_zarr_directory",
    "build_feature_presence_matrix",
    "assign_global_sample_id",
    "optimize_obs_dtypes",
    "optimize_obs_parquet",
    "run_post_build_steps",
    "PostBuildError",
]
