#!/bin/bash
# Quick script to publish zdata to PyPI
# Usage: ./publish.sh [testpypi|pypi]

set -e  # Exit on error

REPO="${1:-pypi}"  # Default to pypi if not specified

echo "=========================================="
echo "Publishing zdata to $REPO"
echo "=========================================="

# Step 1: Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

# Step 2: Build distribution packages
echo "Building distribution packages..."
python -m build

# Step 3: Check what was built
echo ""
echo "Built packages:"
ls -lh dist/

# Step 4: Upload
if [ "$REPO" = "testpypi" ]; then
    echo ""
    echo "Uploading to TestPyPI..."
    echo "Username: __token__"
    echo "Password: (enter your TestPyPI API token)"
    twine upload --repository testpypi dist/*
else
    echo ""
    echo "Uploading to PyPI..."
    echo "Username: __token__"
    echo "Password: (enter your PyPI API token)"
    twine upload dist/*
fi

echo ""
echo "=========================================="
echo "Upload complete!"
echo "=========================================="
echo ""
echo "Verify your package at:"
if [ "$REPO" = "testpypi" ]; then
    echo "  https://test.pypi.org/project/zdata/"
else
    echo "  https://pypi.org/project/zdata/"
fi

