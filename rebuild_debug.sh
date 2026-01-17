#!/bin/bash
# Quick rebuild script for debugging

set -e

echo "========================================="
echo "Rebuilding flash-attention with debug..."
echo "========================================="

# Clean build artifacts
rm -rf build/
rm -rf vllm_flash_attn/*.so
rm -rf csrc/flash_attn/*.so

# Rebuild
python setup.py build_ext --inplace

echo "========================================="
echo "Build complete!"
echo "========================================="
echo ""
echo "Now run tests with:"
echo "  pytest tests/test_flexi_direct.py::test_flexi_direct_correctness -k 'dtype0-False-128-8-128-1-1' -x -s 2>&1 | tee debug_output.log"
