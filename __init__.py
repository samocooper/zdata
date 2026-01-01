"""
zdata - Efficient sparse matrix storage and retrieval using seekable zstd compression.
"""

from __future__ import annotations

__version__ = "0.1.0"

from zdata._settings import settings
from zdata.core import ObsWrapper, ZData

__all__ = ["ObsWrapper", "ZData", "__version__", "settings"]

