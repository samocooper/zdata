"""
zdata - Efficient sparse matrix storage and retrieval using seekable zstd compression.
"""

from __future__ import annotations

__version__ = "0.1.2"

from zdata._settings import settings
from zdata.core import ObsWrapper, ZData
from zdata.build_zdata.build_zdata import build_zdata_from_zarr
from zdata.build_zdata.build_x import build_zdata
from zdata.build_zdata.align_mtx import align_zarr_directory_to_mtx, get_default_gene_list_path
from zdata.build_zdata.concat_obs import concat_obs_from_zarr_directory
from zdata.build_zdata.check_directory import check_zarr_directory

__all__ = [
    "ObsWrapper",
    "ZData",
    "__version__",
    "settings",
    "build_zdata_from_zarr",
    "build_zdata",
    "align_zarr_directory_to_mtx",
    "get_default_gene_list_path",
    "concat_obs_from_zarr_directory",
    "check_zarr_directory",
]

