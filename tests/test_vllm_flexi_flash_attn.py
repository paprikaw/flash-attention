import math
import pytest
import torch
from einops import rearrange, repeat
from typing import List, Optional, Tuple

from vllm_flash_attn.flash_attn_interface import (
    flexi_flash_attn_varlen_func,
    flash_attn_varlen_func,
    is_fa_version_supported,
    prepare_flexi_kv_ptrs,
    free_flexi_kv_ptrs
)

# 使用绝对导入而不是相对导入
try:
    from .test_vllm_flash_attn import ref_paged_attn
except ImportError:
    from test_vllm_flash_attn import ref_paged_attn


NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
# NUM_HEADS = [(8, 2)]
HEAD_SIZES = [128, 256]
# HEAD_SIZES = [128]
BLOCK_SIZES = [16, 32]
# BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16, torch.float16]
# DTYPES = [torch.float16, torch.bfloat16]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768]
# NUM_BLOCKS = [2048]

# Check FA version support with better error reporting
VERSIONS = []
for fa_ver in [2, 3]:
    supported = is_fa_version_supported(fa_ver)
    if supported:
        VERSIONS.append(fa_ver)
    else:
        from vllm_flash_attn.flash_attn_interface import fa_version_unsupported_reason
        reason = fa_version_unsupported_reason(fa_ver)
        print(f"FA{fa_ver} not available: {reason}")

if not VERSIONS:
    import warnings
    warnings.warn("No FlashAttention versions available, tests will be skipped")
# -----------------------------------------------------------------------------
# Placeholder for your new kernel function
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
# @pytest.mark.parametrize("seq_lens", [[(100, 2000)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("soft_cap", [10.0])
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("fa_version", VERSIONS)
@torch.inference_mode()
def test_flexi_flash_attn_kv_split(
        seq_lens: List[Tuple[int, int]],
        num_heads: Tuple[int, int],
        head_size: int,
        sliding_window: Optional[int],
        dtype: torch.dtype,
        block_size: int,
        soft_cap: Optional[float],
        num_blocks: int,
        fa_version: int,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    print(f"num of kv heads: {num_kv_heads}")
    print(f"num of query heads: {num_query_heads}")
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = ((sliding_window,
                    sliding_window) if sliding_window is not None else
                   (-1, -1))
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    assert fa_version == 2
    scheduler_metadata = None
    # 3. Prepare "Split" Data (List of Tensors)
    # We split the large tensor into a list of smaller tensors (pages)
    # k_pages[i] corresponds to the i-th block in the original tensor
    k_pages = []
    v_pages = []
    for k_tensor in key_cache:
        copied_tensor = k_tensor.clone()
        k_pages.append(copied_tensor)
    for v_tensor in value_cache:
        copied_tensor = v_tensor.clone()
        v_pages.append(copied_tensor)
    print(f"block_tables 1: {block_tables}");
    k_ptrs, v_ptrs = prepare_flexi_kv_ptrs(k_pages, v_pages);
    # Warmup
    for _ in range(3):
        flexi_flash_attn_varlen_func(
            q=query,
            k=k_pages,
            v=v_pages,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            scheduler_metadata=scheduler_metadata,
            fa_version=fa_version,
            cached_k_ptrs=k_ptrs,
            cached_v_ptrs=v_ptrs
        )
        flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_tables,
            softcap=soft_cap if soft_cap is not None else 0,
            scheduler_metadata=scheduler_metadata,
            fa_version=fa_version
        )

    block_tables2 = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32) 
    print(f"start to perform formal test")
    # key_cache = torch.randn(num_blocks,
    #                     block_size,
    #                     num_kv_heads,
    #                     head_size,
    #                     dtype=dtype)
    # value_cache = torch.randn_like(key_cache)
    # k_pages = []
    # v_pages = []
    # for k_tensor in key_cache:
    #     copied_tensor = k_tensor.clone()
    #     k_pages.append(copied_tensor)
    # for v_tensor in value_cache:
    #     copied_tensor = v_tensor.clone()
    #     v_pages.append(copied_tensor)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    output1 = flexi_flash_attn_varlen_func(
        q=query,
        k=k_pages,
        v=v_pages,
        cu_seqlens_q=cu_query_lens,
        seqused_k=seqused_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables2,
        softcap=soft_cap if soft_cap is not None else 0,
        scheduler_metadata=scheduler_metadata,
        fa_version=fa_version,
        cached_k_ptrs=k_ptrs,
        cached_v_ptrs=v_ptrs
    )
    end_event.record()
    end_event.synchronize()
    flexi_time = start_event.elapsed_time(end_event)

    start_event.record()
    output2 = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=seqused_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables2,
        softcap=soft_cap if soft_cap is not None else 0,
        scheduler_metadata=scheduler_metadata,
        fa_version=fa_version
    )
    end_event.record()
    end_event.synchronize()
    flash_time = start_event.elapsed_time(end_event)

    print(f"\nFlexi time: {flexi_time:.3f} ms")
    print(f"Flash time: {flash_time:.3f} ms")
    print(f"Speedup: {flash_time / flexi_time:.2f}x")

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables2,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    torch.testing.assert_close(output1, ref_output, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output1 - ref_output))}"

    torch.testing.assert_close(output2, ref_output, atol=2e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output2 - ref_output))}"