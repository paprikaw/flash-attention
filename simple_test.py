#!/usr/bin/env python3
"""
简化的 flexi flash attention 测试脚本
用于快速验证基本功能
"""

import torch
import sys

def simple_test():
    """简单的功能测试"""
    print("=" * 60)
    print("Flexi Flash Attention 简单测试")
    print("=" * 60)
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        sys.exit(1)
    print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    
    # 导入模块
    try:
        from vllm_flash_attn.flash_attn_interface import (
            flexi_flash_attn_varlen_func,
            is_fa_version_supported,
        )
        print("✓ 成功导入 vllm_flash_attn 模块")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("\n提示:")
        print("1. 确保已编译: cmake --build build --target _vllm_fa2_C -j")
        print("2. 设置 PYTHONPATH: export PYTHONPATH=$PWD:$PYTHONPATH")
        sys.exit(1)
    
    # 检查 FA2 支持
    if not is_fa_version_supported(2):
        print("❌ FA2 不支持")
        sys.exit(1)
    print("✓ FA2 支持")
    
    # 简单测试参数
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(42)
    
    # 参数
    num_seqs = 2
    query_lens = [4, 8]
    kv_lens = [32, 64]
    num_query_heads = 8
    num_kv_heads = 2
    head_size = 128
    block_size = 16
    dtype = torch.float16
    
    print(f"\n测试配置:")
    print(f"  Sequences: {num_seqs}")
    print(f"  Query lengths: {query_lens}")
    print(f"  KV lengths: {kv_lens}")
    print(f"  Heads: {num_query_heads}/{num_kv_heads}")
    print(f"  Head size: {head_size}")
    print(f"  Block size: {block_size}")
    print(f"  Dtype: {dtype}")
    
    # 创建数据
    max_kv_len = max(kv_lens)
    num_blocks_needed = sum((kv_len + block_size - 1) // block_size 
                           for kv_len in kv_lens)
    
    print(f"\n创建测试数据...")
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    
    # 创建 KV cache
    key_cache = torch.randn(num_blocks_needed, block_size, num_kv_heads, head_size, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    
    # 创建页面列表
    k_pages = [key_cache[i] for i in range(num_blocks_needed)]
    v_pages = [value_cache[i] for i in range(num_blocks_needed)]
    
    # cu_seqlens 和 seqused_k
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32)
    
    # Block table
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = []
    block_idx = 0
    for kv_len in kv_lens:
        num_blocks = (kv_len + block_size - 1) // block_size
        seq_blocks = list(range(block_idx, block_idx + num_blocks))
        # Pad to max_blocks_per_seq
        seq_blocks += [0] * (max_blocks_per_seq - len(seq_blocks))
        block_tables.append(seq_blocks)
        block_idx += num_blocks
    block_tables = torch.tensor(block_tables, dtype=torch.int32)
    
    print(f"  Query shape: {query.shape}")
    print(f"  Num pages: {len(k_pages)}")
    print(f"  Block table shape: {block_tables.shape}")
    
    # 运行测试
    try:
        print(f"\n运行 flexi_flash_attn_varlen_func...")
        output = flexi_flash_attn_varlen_func(
            q=query,
            k=k_pages,
            v=v_pages,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=max_kv_len,
            softmax_scale=head_size**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0.0,
            scheduler_metadata=None,
            fa_version=2
        )
        
        print(f"✓ 成功运行")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # 基本检查
        assert output.shape == query.shape, f"Shape mismatch: {output.shape} vs {query.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        
        print(f"\n✓ 所有检查通过")
        print("=" * 60)
        print("测试成功! ✓")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    sys.exit(0 if success else 1)
