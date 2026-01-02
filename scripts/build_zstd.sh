#!/bin/bash
# Script to build ZSTD from source for CI environments
# This ensures we have the static library and source files needed for compilation

set -e

ZSTD_VERSION="${ZSTD_VERSION:-1.5.5}"
ZSTD_DIR="${ZSTD_BASE:-/tmp/zstd}"

echo "Building ZSTD from source..."
echo "Version: ${ZSTD_VERSION}"
echo "Directory: ${ZSTD_DIR}"

# Clone ZSTD if directory doesn't exist or is empty
if [ ! -d "${ZSTD_DIR}" ] || [ -z "$(ls -A ${ZSTD_DIR})" ]; then
    echo "Cloning ZSTD repository..."
    git clone --depth 1 --branch v${ZSTD_VERSION} https://github.com/facebook/zstd.git "${ZSTD_DIR}" || \
    git clone --depth 1 https://github.com/facebook/zstd.git "${ZSTD_DIR}"
fi

cd "${ZSTD_DIR}"

# Build ZSTD
echo "Building ZSTD..."
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    make -j$(sysctl -n hw.ncpu)
else
    # Linux
    make -j$(nproc)
fi

# Verify required files exist
echo "Verifying ZSTD build..."
REQUIRED_FILES=(
    "lib/libzstd.a"
    "lib/common/xxhash.c"
    "contrib/seekable_format/zstdseek_compress.c"
    "contrib/seekable_format/zstdseek_decompress.c"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${ZSTD_DIR}/${file}" ]; then
        echo "ERROR: Required file not found: ${file}"
        exit 1
    fi
done

echo "ZSTD build complete!"
echo "ZSTD_BASE=${ZSTD_DIR}"
export ZSTD_BASE="${ZSTD_DIR}"

