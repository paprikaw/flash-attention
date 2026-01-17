#!/bin/bash
# Activate the correct virtual environment
source /data/gpfs/projects/punim2715/vllm_workbench/flash-attention/.venv/bin/activate
export CMAKE_PREFIX_PATH="/home/bxb1/vllm_workbench/flash-attention/.venv:${CMAKE_PREFIX_PATH:-}"
export FLASH_ATTN_HEAD_SIZE=128
export FLASHATTENTION_ONLY_BF16=1
export FLASH_ATTN_DISABLE_FA3=TRUE
ccache -z
uv pip install -v --no-build-isolation -e   . > uv_build.log 2>&1