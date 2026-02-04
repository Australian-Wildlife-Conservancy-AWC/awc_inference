#!/bin/bash
# Build and upload to PyPI - Linux/Mac
# Usage: ./publish.sh [conda_env_name]

set -e

ENV_NAME="${1:-awc}"

echo "Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Cleaning dist folder..."
rm -rf dist/

echo "Building package..."
python -m build

echo "Uploading to PyPI..."
python -m twine upload dist/*
