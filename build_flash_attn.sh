export FLASH_ATTN_HEAD_SIZE=128
export FLASHATTENTION_ONLY_BF16=1
export FLASH_ATTN_DISABLE_FA3=TRUE
ccache -z
uv pip install --no-build-isolation -v -e . > uv_build.log 2>&1