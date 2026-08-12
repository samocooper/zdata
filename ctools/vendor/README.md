# Vendored Zstandard sources

zdata bundles the Zstandard sources it needs so that building requires only a C
compiler — no system packages and no `ZSTD_BASE`.

**Upstream:** https://github.com/facebook/zstd
**Version:** 1.6.0
**Licence:** BSD-3-Clause (dual BSD / GPLv2) — see `LICENSE-zstd`.

## Contents

| file(s) | origin |
|---|---|
| `zstd.c` | official single-file amalgamation (`build/single_file_libs/create_single_file_library.sh`) |
| `xxhash.c` | `lib/common/xxhash.c` — the amalgamation keeps its copy internal, but the seekable format needs the exported `ZSTD_XXH64_*` symbols |
| `zstdseek_compress.c`, `zstdseek_decompress.c`, `zstd_seekable.h` | `contrib/seekable_format/` — **not** part of any system libzstd, so this must be vendored regardless of how zstd itself is obtained |
| `zstd.h`, `zstd_errors.h`, `mem.h`, `xxhash.h`, `zstd_deps.h`, `compiler.h`, `debug.h`, `portability_macros.h` | headers the seekable code includes |

## Updating

1. Check out the desired zstd tag.
2. Run `build/single_file_libs/create_single_file_library.sh` and copy `zstd.c`.
3. Copy the files listed above, plus `LICENSE`, and update the version here.
4. Rebuild and run the suite: `python build_ctools.py && pytest tests/`

These are unmodified upstream sources. Do not patch them locally — carry any
change upstream instead, so the next update does not silently revert it.
