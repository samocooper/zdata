"""Compile the bundled C tools.

zdata ships the Zstandard sources it needs under ``ctools/vendor/`` (the
official single-file amalgamation, plus the seekable-format code from zstd's
``contrib/``, which is not part of any system libzstd). Building therefore
needs only a C compiler -- no system packages, no ``ZSTD_BASE``, nothing to
install first.

``ZSTD_BASE`` is still honoured as an override for anyone who wants to build
against their own zstd tree, but it is no longer required.

This module is deliberately dependency-free so ``setup.py`` can import it
during a build, and the test-suite can reuse it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
CTOOLS_DIR = HERE / "ctools"
VENDOR_DIR = CTOOLS_DIR / "vendor"

#: (binary stem, tool source, seekable source) for each tool.
TOOLS = (
    ("mtx_to_zdata", "mtx_to_zdata.c", "zstdseek_compress.c"),
    ("zdata_read", "zdata_read.c", "zstdseek_decompress.c"),
)


def binary_name(stem: str) -> str:
    return f"{stem}.exe" if sys.platform == "win32" else stem


def _vendor_sources_present() -> bool:
    needed = ["zstd.c", "xxhash.c", "zstdseek_compress.c", "zstdseek_decompress.c"]
    return VENDOR_DIR.is_dir() and all((VENDOR_DIR / n).exists() for n in needed)


def _external_zstd_flags(zstd_base: Path) -> tuple[list[str], list[str]]:
    """Include flags and extra sources for building against an external zstd."""
    includes = [
        f"-I{zstd_base / 'lib'}",
        f"-I{zstd_base / 'lib' / 'common'}",
        f"-I{zstd_base / 'contrib' / 'seekable_format'}",
    ]
    extra = [
        str(zstd_base / "lib" / "common" / "xxhash.c"),
        str(zstd_base / "lib" / "libzstd.a"),
    ]
    return includes, extra


def compile_c_tools(bin_dir: Path, cc: str | None = None,
                    verbose: bool = True) -> bool:
    """Compile both tools into ``bin_dir``. Returns True on success.

    Uses the vendored zstd sources unless ``ZSTD_BASE`` points at a usable
    zstd tree, in which case that is used instead.
    """
    bin_dir = Path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    cc = cc or os.environ.get("CC") or "gcc"

    # zstd's amalgamation includes its thread pool (POOL_create/pthread_create).
    # glibc >= 2.34 folds pthread into libc so this links without a flag, but
    # older glibc (manylinux_2_28) and other libcs need it explicitly. Harmless
    # where it is already implied.
    thread_flags = [] if sys.platform == "win32" else ["-pthread"]

    zstd_base = os.environ.get("ZSTD_BASE")
    use_external = bool(zstd_base) and (Path(zstd_base) / "lib" / "libzstd.a").exists()

    if use_external:
        includes, common_extra = _external_zstd_flags(Path(zstd_base))
        seek_dir = Path(zstd_base) / "contrib" / "seekable_format"
        if verbose:
            print(f"Building C tools against external zstd at {zstd_base}")
    elif _vendor_sources_present():
        includes = [f"-I{CTOOLS_DIR}", f"-I{VENDOR_DIR}"]
        common_extra = [str(VENDOR_DIR / "zstd.c"), str(VENDOR_DIR / "xxhash.c")]
        seek_dir = VENDOR_DIR
        if verbose:
            print("Building C tools against bundled zstd sources")
    else:
        print("ERROR: neither bundled zstd sources nor a usable ZSTD_BASE found.",
              file=sys.stderr)
        return False

    for stem, tool_src, seek_src in TOOLS:
        out = bin_dir / binary_name(stem)
        cmd = ([cc, "-O2", "-Wall"] + thread_flags + includes + ["-o", str(out)]
               + [str(CTOOLS_DIR / tool_src), str(seek_dir / seek_src)]
               + common_extra)
        if verbose:
            print(f"Compiling {stem}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not out.exists():
            print(f"Failed to compile {stem}:\n{result.stderr[:2000]}", file=sys.stderr)
            return False
        if sys.platform != "win32":
            out.chmod(0o755)
    if verbose:
        print("C tools compiled successfully")
    return True


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else CTOOLS_DIR
    raise SystemExit(0 if compile_c_tools(target) else 1)
