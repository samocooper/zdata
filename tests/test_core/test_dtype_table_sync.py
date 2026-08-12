"""The dtype table must stay in sync across Python, C, and the build layer.

Adding float16 previously required five synchronised edits in five files, and
missing one produced a ``KeyError`` at runtime. ``zdata/dtypes.py`` is now the
single source of truth and the C header is generated from it; these tests fail
if anything drifts back out of step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zdata import SUPPORTED_DTYPES
from zdata.build_zdata.build_x import _DTYPE_NP
from zdata.build_zdata.build_x import SUPPORTED_DTYPES as BUILD_X_SUPPORTED
from zdata.dtypes import (
    BY_NAME,
    BY_VERSION,
    DTYPES,
    VERSION_TO_NUMPY,
    render_c_header,
)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_HEADER = _PROJECT_ROOT / "ctools" / "dtype_table.h"


class TestGeneratedHeaderIsCurrent:
    def test_committed_header_matches_generator(self):
        """Catches hand-editing either C table instead of regenerating."""
        assert _HEADER.exists(), "ctools/dtype_table.h is missing; run: python -m zdata.dtypes --write-header"
        on_disk = _HEADER.read_text()
        assert on_disk == render_c_header(), (
            "ctools/dtype_table.h is stale or was hand-edited.\n"
            "Regenerate with: python -m zdata.dtypes --write-header"
        )

    def test_c_sources_include_the_header_not_their_own_table(self):
        """Neither .c file may redeclare the table locally."""
        for name in ("mtx_to_zdata.c", "zdata_read.c"):
            src = (_PROJECT_ROOT / "ctools" / name).read_text()
            assert '#include "dtype_table.h"' in src, f"{name} does not include the generated header"
            assert "static const DTypeInfo DTYPE_TABLE[]" not in src, f"{name} redeclares DTYPE_TABLE"
            assert "static const DTypeRead DTYPE_READ_TABLE[]" not in src, f"{name} redeclares DTYPE_READ_TABLE"

    def test_every_dtype_appears_in_header(self):
        header = _HEADER.read_text()
        for d in DTYPES:
            assert f'"{d.name}"' in header, f"{d.name} missing from generated header"


class TestPythonLayersAgree:
    def test_build_x_supported_matches_canonical(self):
        assert set(BUILD_X_SUPPORTED) == {d.name for d in DTYPES}

    def test_package_export_matches_canonical(self):
        assert set(SUPPORTED_DTYPES) == {d.name for d in DTYPES}

    def test_build_x_numpy_map_matches_canonical(self):
        assert {k: np.dtype(v) for k, v in _DTYPE_NP.items()} == \
               {d.name: np.dtype(d.numpy) for d in DTYPES}

    def test_core_reader_version_map_matches_canonical(self):
        """core/zdata.py decodes block headers using the same version numbers."""
        src = (_PROJECT_ROOT / "core" / "zdata.py").read_text()
        assert "VERSION_TO_NUMPY" in src, (
            "core/zdata.py should import the canonical table, not redeclare it"
        )
        assert "3:  (np.float32, 4)" not in src, (
            "core/zdata.py still contains a hand-written version->dtype table"
        )


class TestTableInvariants:
    def test_versions_are_unique(self):
        versions = [d.version for d in DTYPES]
        assert len(versions) == len(set(versions))

    def test_names_are_unique(self):
        names = [d.name for d in DTYPES]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("d", DTYPES, ids=[d.name for d in DTYPES])
    def test_declared_size_matches_numpy_itemsize(self, d):
        """A wrong size silently misreads every value of that dtype."""
        assert np.dtype(d.numpy).itemsize == d.size

    @pytest.mark.parametrize("d", DTYPES, ids=[d.name for d in DTYPES])
    def test_is_float_flag_matches_numpy_kind(self, d):
        assert d.is_float == (np.dtype(d.numpy).kind == "f")

    def test_lookups_are_consistent(self):
        for d in DTYPES:
            assert BY_VERSION[d.version] is d
            assert BY_NAME[d.name] is d
            assert VERSION_TO_NUMPY[d.version] == (d.numpy, d.size)

    def test_float_widths_all_present(self):
        """float16/32/64 are all supported."""
        assert {"float16", "float32", "float64"} <= {d.name for d in DTYPES}

    def test_all_numpy_integer_widths_present(self):
        expected = {f"{s}int{b}" for s in ("", "u") for b in (8, 16, 32, 64)}
        assert expected <= {d.name for d in DTYPES}
