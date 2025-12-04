"""
End-to-end test for flexi attention:
1. Write key/value into flexi KV cache using flexi_reshape_and_cache_flash
2. Run flexi_flash_attn_varlen_func using the cache
3. Compare with reference implementation

This test verifies the entire data path from writing to reading.
"""
import math
import pytest
import torch
from einops import rearrange
from typing import List, Optional, Tuple

from vllm_flash_attn.flash_attn_interface import (
    flexi_flash_attn_varlen_func,
    flash_attn_varlen_func,
    is_fa_version_supported,
    prepare_flexi_kv_ptrs,
    free_flexi_kv_ptrs
)

# Import vllm ops for flexi_reshape_and_cache_flash
import sys
sys.path.insert(0, '/home/student.unimelb.edu.au/bxb1/vllm_workbench/vllm')
try:
    from vllm._custom_ops import flexi_reshape_and_cache_flash
except ImportError:
    print("Warning: Could not import vllm._custom_ops, trying alternative import")
    from vllm import _custom_ops as ops
    flexi_reshape_and_cache_flash = ops.flexi_reshape_and_cache_flash

# Reference implementation
try:
    from .test_vllm_flash_attn import ref_paged_attn
except ImportError:
    from test_vllm_flash_attn import ref_paged_attn


NUM_HEADS = [(8, 2), (4, 4)]
HEAD_SIZES = [128]
BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16, torch.float16]


def create_flexi_kv_cache(num_blocks: int, block_size: int, num_kv_heads: int, 
                          head_size: int, dtype: torch.dtype) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Create flexi KV cache as a list of tensors."""
    k_pages = []
    v_pages = []
    for _ in range(num_blocks):
        # Shape: (block_size, num_kv_heads, head_size) - NHD layout
        k_page = torch.zeros(block_size, num_kv_heads, head_size, dtype=dtype, device='cuda')
        v_page = torch.zeros(block_size, num_kv_heads, head_size, dtype=dtype, device='cuda')
        k_pages.append(k_page)
        v_pages.append(v_page)
    return k_pages, v_pages


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_flexi_e2e_prefill(
        num_heads: Tuple[int, int],
        head_size: int,
        block_size: int,
        dtype: torch.dtype,
) -> None:
    """
    End-to-end test for prefill scenario:
    - Write all tokens to KV cache using flexi_reshape_and_cache_flash
    - Run attention using flexi_flash_attn_varlen_func
    - Compare with standard flash attention
    """
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(42)
    
    num_query_heads, num_kv_heads = num_heads
    scale = head_size ** -0.5
    
    # Test configuration
    num_seqs = 2
    seq_lens = [64, 128]  # KV lengths for each sequence
    query_lens = seq_lens  # For prefill, query_len == kv_len
    
    total_tokens = sum(seq_lens)
    max_kv_len = max(seq_lens)
    max_query_len = max(query_lens)
    
    # Calculate number of blocks needed
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks_per_seq + 10  # Extra blocks
    
    # Generate input tensors
    query = torch.randn(total_tokens, num_query_heads, head_size, dtype=dtype)
    key = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype)
    
    # Create block tables (mapping sequence positions to block indices)
    block_tables = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32)
    block_idx = 0
    for seq_idx in range(num_seqs):
        num_blocks_for_seq = (seq_lens[seq_idx] + block_size - 1) // block_size
        for b in range(num_blocks_for_seq):
            block_tables[seq_idx, b] = block_idx
            block_idx += 1
    
    # Create slot mapping for all tokens
    slot_mapping_list = []
    token_offset = 0
    for seq_idx in range(num_seqs):
        for pos in range(seq_lens[seq_idx]):
            block_idx_for_pos = block_tables[seq_idx, pos // block_size].item()
            block_offset = pos % block_size
            slot = block_idx_for_pos * block_size + block_offset
            slot_mapping_list.append(slot)
        token_offset += seq_lens[seq_idx]
    
    slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.long, device='cuda')
    
    # Create flexi KV cache
    k_pages, v_pages = create_flexi_kv_cache(num_blocks, block_size, num_kv_heads, head_size, dtype)
    
    # Prepare pointers
    k_ptrs, v_ptrs = prepare_flexi_kv_ptrs(k_pages, v_pages)
    
    # Also create standard (contiguous) KV cache for comparison
    key_cache_standard = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device='cuda')
    value_cache_standard = torch.zeros_like(key_cache_standard)
    
    # Write to flexi KV cache
    k_scale = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    v_scale = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    
    flexi_reshape_and_cache_flash(
        key, value, k_ptrs, v_ptrs,
        k_pages[0], v_pages[0],  # Meta tensors for shape info
        slot_mapping, "auto", k_scale, v_scale
    )
    
    # Write to standard KV cache (reference)
    for i, slot in enumerate(slot_mapping_list):
        block_idx = slot // block_size
        block_offset = slot % block_size
        key_cache_standard[block_idx, block_offset] = key[i]
        value_cache_standard[block_idx, block_offset] = value[i]
    
    # Verify that flexi cache contains the same data as standard cache
    for block_idx in range(num_blocks):
        torch.testing.assert_close(
            k_pages[block_idx], 
            key_cache_standard[block_idx],
            atol=1e-4, rtol=1e-4,
            msg=f"Key cache mismatch at block {block_idx}"
        )
        torch.testing.assert_close(
            v_pages[block_idx], 
            value_cache_standard[block_idx],
            atol=1e-4, rtol=1e-4,
            msg=f"Value cache mismatch at block {block_idx}"
        )
    
    print("✓ KV cache write verification passed")
    
    # Prepare attention inputs
    cu_seqlens_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(query_lens), dim=0).numpy()), 
                                 dtype=torch.int32, device='cuda')
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device='cuda')
    
    # Run flexi attention
    output_flexi = flexi_flash_attn_varlen_func(
        q=query,
        k=k_pages,
        v=v_pages,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        fa_version=2,
        cached_k_ptrs=k_ptrs,
        cached_v_ptrs=v_ptrs
    )
    
    # Run standard flash attention for comparison
    output_standard = flash_attn_varlen_func(
        q=query,
        k=key_cache_standard,
        v=value_cache_standard,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        fa_version=2
    )
    
    # Compare outputs
    torch.testing.assert_close(
        output_flexi, output_standard,
        atol=2e-2, rtol=1e-2,
        msg=f"Output mismatch! Max diff: {torch.max(torch.abs(output_flexi - output_standard))}"
    )
    
    print("✓ Attention output verification passed")
    print(f"Max output difference: {torch.max(torch.abs(output_flexi - output_standard)):.6f}")
    
    # Cleanup
    free_flexi_kv_ptrs(k_ptrs, v_ptrs)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_flexi_e2e_decode(
        num_heads: Tuple[int, int],
        head_size: int,
        block_size: int,
        dtype: torch.dtype,
) -> None:
    """
    End-to-end test for decode scenario:
    - First prefill: write initial tokens to KV cache
    - Then decode: append one token at a time and run attention
    """
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(42)
    
    num_query_heads, num_kv_heads = num_heads
    scale = head_size ** -0.5
    
    # Test configuration
    num_seqs = 2
    prefill_lens = [32, 48]  # Initial prefill lengths
    num_decode_steps = 5
    
    max_kv_len = max(prefill_lens) + num_decode_steps
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks_per_seq + 10
    
    # Create block tables
    block_tables = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32)
    block_idx = 0
    for seq_idx in range(num_seqs):
        for b in range(max_blocks_per_seq):
            block_tables[seq_idx, b] = block_idx
            block_idx += 1
            if block_idx >= num_blocks:
                block_idx = 0  # Wrap around (shouldn't happen with enough blocks)
    
    # Create KV caches
    k_pages, v_pages = create_flexi_kv_cache(num_blocks, block_size, num_kv_heads, head_size, dtype)
    key_cache_standard = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device='cuda')
    value_cache_standard = torch.zeros_like(key_cache_standard)
    
    k_ptrs, v_ptrs = prepare_flexi_kv_ptrs(k_pages, v_pages)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    v_scale = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    
    current_lens = prefill_lens.copy()
    
    # Step 1: Prefill
    total_prefill_tokens = sum(prefill_lens)
    query_prefill = torch.randn(total_prefill_tokens, num_query_heads, head_size, dtype=dtype)
    key_prefill = torch.randn(total_prefill_tokens, num_kv_heads, head_size, dtype=dtype)
    value_prefill = torch.randn(total_prefill_tokens, num_kv_heads, head_size, dtype=dtype)
    
    # Create prefill slot mapping
    slot_mapping_prefill = []
    for seq_idx in range(num_seqs):
        for pos in range(prefill_lens[seq_idx]):
            block_idx_for_pos = block_tables[seq_idx, pos // block_size].item()
            block_offset = pos % block_size
            slot = block_idx_for_pos * block_size + block_offset
            slot_mapping_prefill.append(slot)
    
    slot_mapping_prefill = torch.tensor(slot_mapping_prefill, dtype=torch.long, device='cuda')
    
    # Write prefill to both caches
    flexi_reshape_and_cache_flash(
        key_prefill, value_prefill, k_ptrs, v_ptrs,
        k_pages[0], v_pages[0],
        slot_mapping_prefill, "auto", k_scale, v_scale
    )
    
    for i, slot in enumerate(slot_mapping_prefill.tolist()):
        block_idx = slot // block_size
        block_offset = slot % block_size
        key_cache_standard[block_idx, block_offset] = key_prefill[i]
        value_cache_standard[block_idx, block_offset] = value_prefill[i]
    
    print(f"✓ Prefill completed: {prefill_lens}")
    
    # Step 2: Decode steps
    for decode_step in range(num_decode_steps):
        # Generate new tokens (one per sequence)
        new_query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
        new_key = torch.randn(num_seqs, num_kv_heads, head_size, dtype=dtype)
        new_value = torch.randn(num_seqs, num_kv_heads, head_size, dtype=dtype)
        
        # Create slot mapping for new tokens
        slot_mapping_decode = []
        for seq_idx in range(num_seqs):
            pos = current_lens[seq_idx]
            block_idx_for_pos = block_tables[seq_idx, pos // block_size].item()
            block_offset = pos % block_size
            slot = block_idx_for_pos * block_size + block_offset
            slot_mapping_decode.append(slot)
        
        slot_mapping_decode = torch.tensor(slot_mapping_decode, dtype=torch.long, device='cuda')
        
        # Write new tokens to both caches
        flexi_reshape_and_cache_flash(
            new_key, new_value, k_ptrs, v_ptrs,
            k_pages[0], v_pages[0],
            slot_mapping_decode, "auto", k_scale, v_scale
        )
        
        for i, slot in enumerate(slot_mapping_decode.tolist()):
            block_idx = slot // block_size
            block_offset = slot % block_size
            key_cache_standard[block_idx, block_offset] = new_key[i]
            value_cache_standard[block_idx, block_offset] = new_value[i]
        
        # Update sequence lengths
        for seq_idx in range(num_seqs):
            current_lens[seq_idx] += 1
        
        # Run attention
        cu_seqlens_q = torch.tensor([0] + list(range(1, num_seqs + 1)), dtype=torch.int32, device='cuda')
        seqused_k = torch.tensor(current_lens, dtype=torch.int32, device='cuda')
        
        output_flexi = flexi_flash_attn_varlen_func(
            q=new_query,
            k=k_pages,
            v=v_pages,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=max(current_lens),
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            fa_version=2,
            cached_k_ptrs=k_ptrs,
            cached_v_ptrs=v_ptrs
        )
        
        output_standard = flash_attn_varlen_func(
            q=new_query,
            k=key_cache_standard,
            v=value_cache_standard,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=max(current_lens),
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            fa_version=2
        )
        
        torch.testing.assert_close(
            output_flexi, output_standard,
            atol=2e-2, rtol=1e-2,
            msg=f"Decode step {decode_step}: Output mismatch! Max diff: {torch.max(torch.abs(output_flexi - output_standard))}"
        )
        
        print(f"✓ Decode step {decode_step + 1}: lens={current_lens}, max_diff={torch.max(torch.abs(output_flexi - output_standard)):.6f}")
    
    free_flexi_kv_ptrs(k_ptrs, v_ptrs)
    print("✓ All decode steps passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running End-to-End Flexi Attention Tests")
    print("=" * 60)
    
    # Run prefill test
    print("\n--- Prefill Test ---")
    test_flexi_e2e_prefill(
        num_heads=(8, 2),
        head_size=128,
        block_size=16,
        dtype=torch.bfloat16
    )
    
    # Run decode test
    print("\n--- Decode Test ---")
    test_flexi_e2e_decode(
        num_heads=(8, 2),
        head_size=128,
        block_size=16,
        dtype=torch.bfloat16
    )
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
