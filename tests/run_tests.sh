#!/bin/bash
# Simple script to run pytest tests
# Usage: ./run_tests.sh [test_path]

set -e

cd "$(dirname "$0")/.."

# Check if pytest is installed
if ! python3 -m pytest --version > /dev/null 2>&1; then
    echo "Error: pytest is not installed"
    echo "Install it with: pip install pytest pytest-cov pytest-xdist"
    exit 1
fi

# Run tests
if [ $# -eq 0 ]; then
    # Run all tests
    python3 -m pytest tests/ -v
else
    # Run specific test path
    python3 -m pytest "$@" -v
fi

