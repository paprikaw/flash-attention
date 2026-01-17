#!/usr/bin/env python3
"""
Test flexi_direct_flash_attn_varlen_func against flexi_flash_attn_varlen_func.
Verify that using direct pointer tables produces identical results to traditional two-level lookup.
"""

import torch
import pytest
import sys
import time
import os
from typing import List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vllm_flash_attn.flash_attn_interface import (
    flexi_flash_attn_varlen_func,
    flexi_direct_flash_attn_varlen_func,
    flash_attn_varlen_func,
    block_table_to_ptr_tables,
    prepare_flexi_kv_ptrs,
    free_flexi_kv_ptrs,
)
DTYPES = [torch.bfloat16]

def create_paged_kv_cache(num_blocks, block_size, num_heads_k, head_dim, dtype, device):
    """Create paged KV cache with random data."""
    k_cache = []
    v_cache = []
    for _ in range(num_blocks):
        k_cache.append(torch.randn(block_size, num_heads_k, head_dim, dtype=dtype, device=device))
        v_cache.append(torch.randn(block_size, num_heads_k, head_dim, dtype=dtype, device=device))
    return k_cache, v_cache


def create_block_table(batch_size, max_num_blocks_per_seq, num_blocks, device, random_seed=None):
    """Create block_table for paged attention.
    
    Args:
        batch_size: Number of sequences in batch
        max_num_blocks_per_seq: Maximum number of blocks per sequence
        num_blocks: Total number of available blocks
        device: Device to create tensor on
        random_seed: If None, assign consecutive blocks. If int, use as seed for random assignment.
    
    Returns:
        block_table: (batch_size, max_num_blocks_per_seq) tensor of block indices
    """
    block_table = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.int32, device=device)
    
    if random_seed is None:
        # Sequential assignment: each sequence gets consecutive blocks
        for i in range(batch_size):
            start_block = i * max_num_blocks_per_seq
            end_block = (i + 1) * max_num_blocks_per_seq
            block_table[i, :] = torch.arange(
                start_block, 
                end_block, 
                dtype=torch.int32, 
                device=device
            )
    else:
        # Random assignment: shuffle available blocks and assign to sequences
        generator = torch.Generator(device='cpu')
        generator.manual_seed(random_seed)
        
        # Create a pool of available block indices
        available_blocks = torch.randperm(num_blocks, generator=generator, dtype=torch.int32)
        
        # Assign blocks to each sequence
        for i in range(batch_size):
            start_idx = i * max_num_blocks_per_seq
            end_idx = (i + 1) * max_num_blocks_per_seq
            block_table[i, :] = available_blocks[start_idx:end_idx]
        
        block_table = block_table.to(device)
    
    return block_table


@pytest.mark.parametrize("batch_size", [20])
@pytest.mark.parametrize("seqlen_q", [256])
@pytest.mark.parametrize("seqlen_k", [256])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("block_size", [16])
def test_flexi_direct_correctness(batch_size, seqlen_q, seqlen_k, num_heads, head_dim, causal, dtype, block_size):
    """Test that flexi_direct produces identical results to original flexi."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = "cuda"
    num_heads_k = num_heads  # Can also test MQA/GQA
    kBlockN = 64  # Flash attention block size
    
    # Calculate number of blocks needed per sequence and total blocks
    # Must round seqlen_k up to kBlockN to ensure ptr_table has enough entries
    # for all virtual page indices that may be accessed by the kernel
    seqlen_k_rounded = ((seqlen_k + kBlockN - 1) // kBlockN) * kBlockN
    # CRITICAL FIX: ptr_table size must accommodate all possible virtual_page_idx accesses.
    # In kernel, virtual_page_idx = (n_block * kBlockN + block_row_offset) / page_block_size
    # Since n_block ranges from 0 to ceil(seqlen_k / kBlockN) - 1, and block_row_offset < kBlockN,
    # max global_row_offset can reach seqlen_k_rounded, requiring seqlen_k_rounded / block_size pages.
    # Adding +1 as safety margin for boundary cases.
    max_num_blocks_per_seq = (seqlen_k_rounded // block_size) + 1
    num_blocks = batch_size * max_num_blocks_per_seq  # Allocate blocks for all sequences
    
    # Create Q
    total_q = batch_size * seqlen_q
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    
    # Create paged KV cache
    k_cache, v_cache = create_paged_kv_cache(num_blocks, block_size, num_heads_k, head_dim, dtype, device)
    k_meta = k_cache[0]
    v_meta = v_cache[0]
    
    # Create cu_seqlens
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=device)
    seqused_k = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)
    
    # Create block_table: each sequence gets consecutive blocks
    block_table = create_block_table(batch_size, max_num_blocks_per_seq, num_blocks, device, random_seed=None)
    print(f"block_table shape: {block_table.shape}, num_blocks: {num_blocks}") 
    # Prepare cached pointers for original flexi
    cached_k_ptrs, cached_v_ptrs = prepare_flexi_kv_ptrs(k_cache, v_cache)
    
    # Test original flexi
    out_original = flexi_flash_attn_varlen_func(
        q=q,
        k_meta=k_meta,
        v_meta=v_meta,
        num_blocks=num_blocks,
        max_seqlen_q=seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
        block_table=block_table,
        cached_k_ptrs=cached_k_ptrs,
        cached_v_ptrs=cached_v_ptrs,
    )
    
    # Convert block_table to direct pointer tables
    # Note: We need to extract page pointers from cached_k_ptrs/cached_v_ptrs
    # For testing, we'll create them directly from k_cache/v_cache
    k_page_ptrs = [k.data_ptr() for k in k_cache]
    v_page_ptrs = [v.data_ptr() for v in v_cache]
    k_ptr_table, v_ptr_table = block_table_to_ptr_tables(block_table, k_page_ptrs, v_page_ptrs)
    print(f"k_ptr_table:{k_ptr_table}") 
    print(f"v_ptr_table:{v_ptr_table}") 
    # Test direct pointer version
    out_direct = flexi_direct_flash_attn_varlen_func(
        q=q,
        k_meta=k_meta,
        v_meta=v_meta,
        num_blocks=num_blocks,
        k_ptr_table=k_ptr_table,
        v_ptr_table=v_ptr_table,
        max_seqlen_q=seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=seqlen_k,
        seqused_k=seqused_k,
        causal=causal,
    )
    torch.cuda.synchronize() 
    # Compare outputs
    torch.testing.assert_close(out_direct, out_original, rtol=1e-3, atol=1e-3)
    
    # Cleanup
    free_flexi_kv_ptrs(cached_k_ptrs, cached_v_ptrs)
    
    print(f"✓ Test passed: batch_size={batch_size}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
          f"num_heads={num_heads}, head_dim={head_dim}, causal={causal}, dtype={dtype}")


@pytest.mark.parametrize("batch_size", [100])
@pytest.mark.parametrize("seqlen_q", [100])
@pytest.mark.parametrize("seqlen_k", [2000])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", DTYPES)
def test_flexi_direct_performance(batch_size, seqlen_q, seqlen_k, num_heads, head_dim, dtype):
    """Benchmark flexi_direct vs original flexi."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = "cuda"
    num_heads_k = num_heads
    block_size = 16
    kBlockN = 64  # Flash attention block size
    
    # Calculate number of blocks needed per sequence and total blocks
    # Must round seqlen_k up to kBlockN to ensure ptr_table has enough entries
    # for all virtual page indices that may be accessed by the kernel
    seqlen_k_rounded = ((seqlen_k + kBlockN - 1) // kBlockN) * kBlockN
    # CRITICAL FIX: ptr_table size must accommodate all possible virtual_page_idx accesses.
    # In kernel, virtual_page_idx = (n_block * kBlockN + block_row_offset) / page_block_size
    # Since n_block ranges from 0 to ceil(seqlen_k / kBlockN) - 1, and block_row_offset < kBlockN,
    # max global_row_offset can reach seqlen_k_rounded, requiring seqlen_k_rounded / block_size pages.
    # Adding +1 as safety margin for boundary cases.
    max_num_blocks_per_seq = (seqlen_k_rounded // block_size) + 1
    num_blocks = batch_size * max_num_blocks_per_seq  # Allocate blocks for all sequences
    print(f"num_blocks: {num_blocks}, max_num_blocks_per_seq: {max_num_blocks_per_seq}")
    
    # Create Q
    total_q = batch_size * seqlen_q
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    
    # Create paged KV cache
    k_cache, v_cache = create_paged_kv_cache(num_blocks, block_size, num_heads_k, head_dim, dtype, device)
    k_meta = k_cache[0]
    v_meta = v_cache[0]
    
    # Create packed K/V cache for regular flash attention
    k_cache_packed = torch.stack(k_cache, dim=0)  # (num_blocks, block_size, num_heads_k, head_dim)
    v_cache_packed = torch.stack(v_cache, dim=0)  # (num_blocks, block_size, num_heads_k, head_dim)
    
    # Create cu_seqlens
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=device)
    print(cu_seqlens_q)
    seqused_k = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=device)

    # Create block_table: each sequence gets consecutive blocks
    block_table = create_block_table(batch_size, max_num_blocks_per_seq, num_blocks, device, random_seed=None)
    # Prepare cached pointers
    cached_k_ptrs, cached_v_ptrs = prepare_flexi_kv_ptrs(k_cache, v_cache)
    k_page_ptrs = [k.data_ptr() for k in k_cache]
    v_page_ptrs = [v.data_ptr() for v in v_cache]
    k_ptr_table, v_ptr_table = block_table_to_ptr_tables(block_table, k_page_ptrs, v_page_ptrs)

    torch.cuda.synchronize()
    # Warmup direct
    for _ in range(3):
        _ = flexi_direct_flash_attn_varlen_func(
            q=q, k_meta=k_meta, v_meta=v_meta, num_blocks=num_blocks,
            k_ptr_table=k_ptr_table, v_ptr_table=v_ptr_table,
            max_seqlen_q=seqlen_q, cu_seqlens_q=cu_seqlens_q, max_seqlen_k=seqlen_k,
            seqused_k=seqused_k, causal=False,
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    output1 = flexi_direct_flash_attn_varlen_func(
        q=q, k_meta=k_meta, v_meta=v_meta, num_blocks=num_blocks,
        k_ptr_table=k_ptr_table, v_ptr_table=v_ptr_table,
        max_seqlen_q=seqlen_q, cu_seqlens_q=cu_seqlens_q, max_seqlen_k=seqlen_k,
        seqused_k=seqused_k, causal=False,
    )

    torch.cuda.synchronize()
    direct_time = (time.perf_counter() - start)

    # Warmup
    for _ in range(3):
        _ = flexi_flash_attn_varlen_func(
            q=q, k_meta=k_meta, v_meta=v_meta, num_blocks=num_blocks,
            max_seqlen_q=seqlen_q, cu_seqlens_q=cu_seqlens_q, max_seqlen_k=seqlen_k,
            seqused_k=seqused_k, causal=False, block_table=block_table,
            cached_k_ptrs=cached_k_ptrs, cached_v_ptrs=cached_v_ptrs,
        )
    # Benchmark original
    torch.cuda.synchronize()
    start = time.perf_counter()
    output2 = flexi_flash_attn_varlen_func(
        q=q, k_meta=k_meta, v_meta=v_meta, num_blocks=num_blocks,
        max_seqlen_q=seqlen_q, cu_seqlens_q=cu_seqlens_q, max_seqlen_k=seqlen_k,
        seqused_k=seqused_k, causal=False, block_table=block_table,
        cached_k_ptrs=cached_k_ptrs, cached_v_ptrs=cached_v_ptrs,
    )
    torch.cuda.synchronize()
    original_time = (time.perf_counter() - start)

    # Warmup regular flash attention
    for _ in range(3):
        _ = flash_attn_varlen_func(
            q=q, k=k_cache_packed, v=v_cache_packed,
            cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
            max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
            causal=False, block_table=block_table,
        )
    
    # Benchmark regular flash attention
    torch.cuda.synchronize()
    start = time.perf_counter()
    output3 = flash_attn_varlen_func(
        q=q, k=k_cache_packed, v=v_cache_packed,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
        max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
        causal=False, block_table=block_table,
    )
    torch.cuda.synchronize()
    regular_time = (time.perf_counter() - start)

    speedup_direct_vs_flexi = original_time / direct_time
    speedup_direct_vs_regular = regular_time / direct_time
    speedup_flexi_vs_regular = regular_time / original_time
    
    print(f"\n{'='*60}")
    print(f"Performance Comparison:")
    print(f"  Regular Flash:  {regular_time*1000:.3f} ms")
    print(f"  Original Flexi: {original_time*1000:.3f} ms")
    print(f"  Direct Flexi:   {direct_time*1000:.3f} ms")
    print(f"  ---")
    print(f"  Direct vs Original Flexi speedup: {speedup_direct_vs_flexi:.2f}x")
    print(f"  Direct vs Regular speedup:        {speedup_direct_vs_regular:.2f}x")
    print(f"  Flexi vs Regular speedup:         {speedup_flexi_vs_regular:.2f}x")
    print(f"{'='*60}\n")

    torch.testing.assert_close(output1, output2, atol=2e-2, rtol=1e-2), \
    f"{torch.max(torch.abs(output1 - output2))}"
    torch.testing.assert_close(output1, output3, atol=2e-2, rtol=1e-2), \
    f"{torch.max(torch.abs(output1 - output3))}"
    # Cleanup
    free_flexi_kv_ptrs(cached_k_ptrs, cached_v_ptrs)
    
    # Assert some speedup (should be at least a small improvement)
    # assert speedup > 0.95, f"Direct version should not be slower: speedup={speedup}"