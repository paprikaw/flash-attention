#!/usr/bin/env python3
"""Minimal test to reproduce spike in pytest environment."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
import time

from vllm_flash_attn.flash_attn_interface import (
    flexi_direct_flash_attn_varlen_func,
    block_table_to_ptr_tables,
    prepare_flexi_kv_ptrs,
)


def test_spike_minimal():
    """Minimal test to see if spike occurs."""
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 100
    seqlen_q = 10
    seqlen_k = 10
    num_heads = 8
    head_dim = 128
    block_size = 16
    
    kBlockN = 64
    seqlen_k_rounded = ((seqlen_k + kBlockN - 1) // kBlockN) * kBlockN
    max_num_blocks_per_seq = (seqlen_k_rounded // block_size) + 1
    num_blocks = batch_size * max_num_blocks_per_seq
    
    # Setup
    q = torch.randn(batch_size * seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
    cu_seqlens_q = torch.arange(0, batch_size * seqlen_q + 1, seqlen_q, device=device, dtype=torch.int32)
    seqused_k = torch.full((batch_size,), seqlen_k, device=device, dtype=torch.int32)
    
    # Create paged KV cache
    k_cache = [torch.randn(block_size, num_heads, head_dim, dtype=dtype, device=device) for _ in range(num_blocks)]
    v_cache = [torch.randn(block_size, num_heads, head_dim, dtype=dtype, device=device) for _ in range(num_blocks)]
    k_meta = k_cache[0]
    v_meta = v_cache[0]
    
    # Create packed K/V cache for regular flash attention (like original test)
    k_cache_packed = torch.stack(k_cache, dim=0)  # Extra memory allocation
    v_cache_packed = torch.stack(v_cache, dim=0)  # Extra memory allocation
    
    # This sync should make the memory allocation complete before warmup
    torch.cuda.synchronize()
    
    # Create block_table
    block_table = torch.zeros((batch_size, max_num_blocks_per_seq), device=device, dtype=torch.int32)
    for i in range(batch_size):
        for j in range(max_num_blocks_per_seq):
            block_table[i, j] = i * max_num_blocks_per_seq + j
    
    # Prepare pointers
    cached_k_ptrs, cached_v_ptrs = prepare_flexi_kv_ptrs(k_cache, v_cache)
    k_page_ptrs = [k.data_ptr() for k in k_cache]
    v_page_ptrs = [v.data_ptr() for v in v_cache]
    k_ptr_table, v_ptr_table = block_table_to_ptr_tables(block_table, k_page_ptrs, v_page_ptrs)
    
    torch.cuda.synchronize()
    
    num_warmup = 7
    num_benchmark = 5
    
    print("\n" + "=" * 60)
    print("Minimal pytest test - observing benchmark pattern")
    print("=" * 60)
    
    # Warmup
    for i in range(num_warmup):
        _ = flexi_direct_flash_attn_varlen_func(
            q=q, k_meta=k_meta, v_meta=v_meta, num_blocks=num_blocks,
            k_ptr_table=k_ptr_table, v_ptr_table=v_ptr_table,
            max_seqlen_q=seqlen_q, cu_seqlens_q=cu_seqlens_q, max_seqlen_k=seqlen_k,
            seqused_k=seqused_k, causal=False,
        )
    
    print("\nSync after warmup...")
    torch.cuda.synchronize()
    
    # Benchmark
    print("\nBenchmark phase:")
    for i in range(num_benchmark):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = flexi_direct_flash_attn_varlen_func(
            q=q, k_meta=k_meta, v_meta=v_meta, num_blocks=num_blocks,
            k_ptr_table=k_ptr_table, v_ptr_table=v_ptr_table,
            max_seqlen_q=seqlen_q, cu_seqlens_q=cu_seqlens_q, max_seqlen_k=seqlen_k,
            seqused_k=seqused_k, causal=False,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"  Benchmark {i+1}: Total: {elapsed*1000:.3f} ms")
    
    # Test passes - we're just observing
    assert True
