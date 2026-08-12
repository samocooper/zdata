"""Canonical dtype table for the zdata on-disk format.

This is the **single source of truth** for the (version, name, size, numpy type)
mapping used by every layer of the stack:

  * ``ctools/mtx_to_zdata.c``  -- writes the version into each block header
  * ``ctools/zdata_read.c``    -- decodes it back
  * ``core/zdata.py``          -- maps version -> numpy dtype when reading
  * ``build_zdata/build_x.py`` -- validates the requested dtype name
  * the test-suite

The C side does not import Python, so its table is **generated** from this one
into ``ctools/dtype_table.h`` by :func:`render_c_header`. ``tests`` asserts the
committed header still matches, so hand-editing either C table is caught rather
than silently diverging.

Adding a dtype: append one row here, run ``python -m zdata.dtypes --write-header``,
rebuild the C tools, and add a case to ``store_value``/``print_value``.

Version numbers are an on-disk format contract: never reuse or renumber one.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class DType(NamedTuple):
    """One entry of the on-disk dtype table."""

    version: int          # value stored in the block header
    name: str             # CLI / API name, e.g. "uint16"
    size: int             # bytes per element
    numpy: type           # corresponding numpy scalar type
    printf: str           # printf format used by zdata_read text output
    is_float: bool
    note: str = ""        # optional comment carried into the generated header


#: The canonical table. Order is the on-disk version order; do not renumber.
DTYPES: tuple[DType, ...] = (
    DType(2,  "uint16",  2, np.uint16,  "%u",     False, "backward-compatible default"),
    DType(3,  "float32", 4, np.float32, "%.6g",   True,  "backward-compatible"),
    DType(4,  "uint8",   1, np.uint8,   "%u",     False),
    DType(5,  "uint32",  4, np.uint32,  "%u",     False),
    DType(6,  "uint64",  8, np.uint64,  "%llu",   False),
    DType(7,  "int8",    1, np.int8,    "%d",     False),
    DType(8,  "int16",   2, np.int16,   "%d",     False),
    DType(9,  "int32",   4, np.int32,   "%d",     False),
    DType(10, "int64",   8, np.int64,   "%lld",   False),
    DType(11, "float64", 8, np.float64, "%.15g",  True),
    DType(12, "float16", 2, np.float16, "%.4g",   True,  "IEEE-754 binary16"),
)

#: Default when a block header carries an unknown version (legacy files).
DEFAULT_VERSION = 2

# --- derived lookups (import these rather than re-declaring the table) -------
BY_VERSION: dict[int, DType] = {d.version: d for d in DTYPES}
BY_NAME: dict[str, DType] = {d.name: d for d in DTYPES}
SUPPORTED_DTYPES = frozenset(d.name for d in DTYPES)
NAME_TO_NUMPY: dict[str, type] = {d.name: d.numpy for d in DTYPES}
VERSION_TO_NUMPY: dict[int, tuple[type, int]] = {
    d.version: (d.numpy, d.size) for d in DTYPES
}

C_HEADER_PATH = "ctools/dtype_table.h"


def render_c_header() -> str:
    """Render the generated C header holding both tables."""
    w_rows, r_rows = [], []
    for d in DTYPES:
        note = f"   /* {d.note} */" if d.note else ""
        w_rows.append(f'    {{ {d.version:2d}, {d.size}, "{d.name}" }},{note}')
        r_rows.append(
            f'    {{ {d.version:2d}, {d.size}, "{d.name}", "{d.printf}", '
            f'{1 if d.is_float else 0} }},'
        )
    nl = "\n"
    return f"""/* GENERATED FILE -- DO NOT EDIT BY HAND.
 *
 * Regenerate with:  python -m zdata.dtypes --write-header
 * Source of truth:  zdata/dtypes.py
 *
 * Editing this file directly will be caught by
 * tests/test_core/test_dtype_table_sync.py.
 */
#ifndef ZDATA_DTYPE_TABLE_H
#define ZDATA_DTYPE_TABLE_H

#include <stddef.h>
#include <stdint.h>

#define ZDATA_DEFAULT_DTYPE_VERSION {DEFAULT_VERSION}

/* Writer-side table (mtx_to_zdata.c): version <-> element size <-> name. */
typedef struct {{
    uint32_t    version;
    size_t      val_size;
    const char *name;
}} DTypeInfo;

static const DTypeInfo DTYPE_TABLE[] = {{
{nl.join(w_rows)}
}};

/* Reader-side table (zdata_read.c): adds a printf format and a float flag. */
typedef struct {{
    uint32_t    version;
    size_t      val_size;
    const char *name;
    const char *fmt;
    int         is_float;
}} DTypeRead;

static const DTypeRead DTYPE_READ_TABLE[] = {{
{nl.join(r_rows)}
}};

#define ZDATA_NUM_DTYPES (sizeof(DTYPE_TABLE) / sizeof(DTYPE_TABLE[0]))

#endif /* ZDATA_DTYPE_TABLE_H */
"""


def write_c_header(path: str | None = None) -> str:
    """Write the generated header to ``path`` (default ``ctools/dtype_table.h``)."""
    import os

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), C_HEADER_PATH)
    with open(path, "w") as f:
        f.write(render_c_header())
    return path


if __name__ == "__main__":
    import sys

    if "--write-header" in sys.argv:
        print(f"wrote {write_c_header()}")
    else:
        print(render_c_header())
