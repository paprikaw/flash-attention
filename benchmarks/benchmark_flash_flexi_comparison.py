#!/usr/bin/env python3
"""
Benchmark script comparing three Flash Attention variants:
1. flash_attn_varlen_func - Standard Flash Attention with paged KV cache
2. flexi_flash_attn_varlen_func - Flexi version with two-level pointer lookup
3. flexi_direct_flash_attn_varlen_func - Flexi Direct version with single-level pointer lookup

Usage:
    python benchmarks/benchmark_flash_flexi_comparison.py
    python benchmarks/benchmark_flash_flexi_comparison.py --seqlen-q 512 --seqlen-k 2048
    python benchmarks/benchmark_flash_flexi_comparison.py --batch-sizes 1 4 8 16 --output results.csv
"""

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
    flexi_flash_attn_varlen_func,
    flexi_direct_flash_attn_varlen_func,
    block_table_to_ptr_tables,
    prepare_flexi_kv_ptrs,
    free_flexi_kv_ptrs,
    is_fa_version_supported,
)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    batch_size: int
    seqlen_q: int
    seqlen_k: int
    num_heads: int
    num_heads_k: int
    head_dim: int
    block_size: int
    dtype: torch.dtype
    causal: bool
    num_warmup: int = 10
    num_benchmark: int = 10


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    flash_time_ms: float
    flexi_time_ms: float
    flexi_direct_time_ms: float
    flash_tflops: float
    flexi_tflops: float
    flexi_direct_tflops: float
    
    @property
    def speedup_flexi_vs_flash(self) -> float:
        return self.flash_time_ms / self.flexi_time_ms if self.flexi_time_ms > 0 else 0
    
    @property
    def speedup_direct_vs_flash(self) -> float:
        return self.flash_time_ms / self.flexi_direct_time_ms if self.flexi_direct_time_ms > 0 else 0
    
    @property
    def speedup_direct_vs_flexi(self) -> float:
        return self.flexi_time_ms / self.flexi_direct_time_ms if self.flexi_direct_time_ms > 0 else 0


def compute_flops(batch_size: int, seqlen_q: int, seqlen_k: int, 
                  num_heads: int, head_dim: int, causal: bool) -> float:
    """Calculate theoretical FLOPs for attention computation."""
    # For attention: 2 * batch * heads * seqlen_q * seqlen_k * head_dim (for QK^T)
    #              + 2 * batch * heads * seqlen_q * seqlen_k * head_dim (for softmax @ V)
    # Total = 4 * batch * heads * seqlen_q * seqlen_k * head_dim
    # For causal, effective seqlen_k is halved on average
    effective_seqlen_k = seqlen_k / 2 if causal else seqlen_k
    return 4 * batch_size * num_heads * seqlen_q * effective_seqlen_k * head_dim


def time_to_tflops(flops: float, time_ms: float) -> float:
    """Convert time to TFLOPS."""
    if time_ms <= 0:
        return 0.0
    return (flops / (time_ms * 1e-3)) / 1e12


def create_paged_kv_cache(num_blocks: int, block_size: int, num_heads_k: int, 
                          head_dim: int, dtype: torch.dtype, device: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Create paged KV cache with random data."""
    k_cache = []
    v_cache = []
    for _ in range(num_blocks):
        k_cache.append(torch.randn(block_size, num_heads_k, head_dim, dtype=dtype, device=device))
        v_cache.append(torch.randn(block_size, num_heads_k, head_dim, dtype=dtype, device=device))
    return k_cache, v_cache


def create_block_table(batch_size: int, max_num_blocks_per_seq: int, 
                       num_blocks: int, device: str) -> torch.Tensor:
    """Create block_table for paged attention with sequential assignment."""
    block_table = torch.zeros(batch_size, max_num_blocks_per_seq, dtype=torch.int32, device=device)
    for i in range(batch_size):
        start_block = i * max_num_blocks_per_seq
        end_block = (i + 1) * max_num_blocks_per_seq
        block_table[i, :] = torch.arange(start_block, end_block, dtype=torch.int32, device=device)
    return block_table


def benchmark_kernel(func, num_warmup: int, num_benchmark: int, **kwargs) -> Tuple[float, float, float]:
    """
    Benchmark a kernel function using CUDA events for accurate timing.
    
    Returns:
        Tuple of (mean_time_ms, min_time_ms, max_time_ms)
    """
    # Warmup - use CUDA events and synchronize after each call to ensure 
    # JIT compilation completes. This prevents compilation overhead from 
    # bleeding into benchmark measurements.
    for _ in range(num_warmup):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = func(**kwargs)
        end.record()
        end.synchronize()  # This ensures the kernel fully completes including JIT
    
    # Benchmark using CUDA events
    times = []
    for _ in range(num_benchmark):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        _ = func(**kwargs)
        end_event.record()
        
        end_event.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    mean_time = sum(times) / len(times)
    return mean_time, min(times), max(times)


def run_benchmark(config: BenchmarkConfig, device: str = "cuda", verbose: bool = True,
                  global_warmup_done: bool = False) -> BenchmarkResult:
    """Run benchmark for all three kernel variants."""
    
    kBlockN = 64  # Flash attention block size
    
    # Calculate number of blocks needed
    seqlen_k_rounded = ((config.seqlen_k + kBlockN - 1) // kBlockN) * kBlockN
    max_num_blocks_per_seq = (seqlen_k_rounded // config.block_size) + 1
    num_blocks = config.batch_size * max_num_blocks_per_seq
    
    # Create query tensor
    total_q = config.batch_size * config.seqlen_q
    q = torch.randn(total_q, config.num_heads, config.head_dim, 
                    dtype=config.dtype, device=device)
    
    # Create paged KV cache
    k_cache, v_cache = create_paged_kv_cache(
        num_blocks, config.block_size, config.num_heads_k, 
        config.head_dim, config.dtype, device
    )
    k_meta = k_cache[0]
    v_meta = v_cache[0]
    
    # Create packed KV cache for regular flash attention
    k_cache_packed = torch.stack(k_cache, dim=0)
    v_cache_packed = torch.stack(v_cache, dim=0)
    
    # Create sequence length tensors
    cu_seqlens_q = torch.arange(
        0, (config.batch_size + 1) * config.seqlen_q, config.seqlen_q,
        dtype=torch.int32, device=device
    )
    seqused_k = torch.full((config.batch_size,), config.seqlen_k, 
                           dtype=torch.int32, device=device)
    
    # Create block table
    block_table = create_block_table(
        config.batch_size, max_num_blocks_per_seq, num_blocks, device
    )
    
    # Prepare cached pointers for flexi
    cached_k_ptrs, cached_v_ptrs = prepare_flexi_kv_ptrs(k_cache, v_cache)
    
    # Create direct pointer tables for flexi_direct
    k_page_ptrs = [k.data_ptr() for k in k_cache]
    v_page_ptrs = [v.data_ptr() for v in v_cache]
    k_ptr_table, v_ptr_table = block_table_to_ptr_tables(block_table, k_page_ptrs, v_page_ptrs)
    
    torch.cuda.synchronize()
    
    # ============ Global Warmup Phase ============
    # If this is the first run, do extra warmup for all kernels to eliminate
    # Python/PyTorch/CUDA initialization overhead
    if not global_warmup_done:
        global_warmup_iterations = 5
        if verbose:
            print("  [Global warmup: warming up all kernels...]")
        
        # Warmup Flash
        for _ in range(global_warmup_iterations):
            _ = flash_attn_varlen_func(
                q=q, k=k_cache_packed, v=v_cache_packed,
                cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
                max_seqlen_q=config.seqlen_q, max_seqlen_k=config.seqlen_k,
                causal=config.causal, block_table=block_table,
            )
        torch.cuda.synchronize()
        
        # Warmup Flexi
        for _ in range(global_warmup_iterations):
            _ = flexi_flash_attn_varlen_func(
                q=q, k_meta=k_meta, v_meta=v_meta, num_blocks=num_blocks,
                max_seqlen_q=config.seqlen_q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=config.seqlen_k, seqused_k=seqused_k,
                causal=config.causal, block_table=block_table,
                cached_k_ptrs=cached_k_ptrs, cached_v_ptrs=cached_v_ptrs,
            )
        torch.cuda.synchronize()
        
        # Warmup Flexi Direct
        for _ in range(global_warmup_iterations):
            _ = flexi_direct_flash_attn_varlen_func(
                q=q, k_meta=k_meta, v_meta=v_meta, num_blocks=num_blocks,
                k_ptr_table=k_ptr_table, v_ptr_table=v_ptr_table,
                max_seqlen_q=config.seqlen_q, cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=config.seqlen_k, seqused_k=seqused_k,
                causal=config.causal,
            )
        torch.cuda.synchronize()
        
        if verbose:
            print("  [Global warmup complete]")
    
    # Calculate theoretical FLOPs
    flops = compute_flops(
        config.batch_size, config.seqlen_q, config.seqlen_k,
        config.num_heads, config.head_dim, config.causal
    )
    
    # ============ Benchmark Regular Flash Attention ============
    flash_mean, flash_min, flash_max = benchmark_kernel(
        flash_attn_varlen_func,
        config.num_warmup, config.num_benchmark,
        q=q, k=k_cache_packed, v=v_cache_packed,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
        max_seqlen_q=config.seqlen_q, max_seqlen_k=config.seqlen_k,
        causal=config.causal, block_table=block_table,
    )
    
    # ============ Benchmark Flexi Flash Attention ============
    flexi_mean, flexi_min, flexi_max = benchmark_kernel(
        flexi_flash_attn_varlen_func,
        config.num_warmup, config.num_benchmark,
        q=q, k_meta=k_meta, v_meta=v_meta, num_blocks=num_blocks,
        max_seqlen_q=config.seqlen_q, cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=config.seqlen_k, seqused_k=seqused_k,
        causal=config.causal, block_table=block_table,
        cached_k_ptrs=cached_k_ptrs, cached_v_ptrs=cached_v_ptrs,
    )
    
    # ============ Benchmark Flexi Direct Flash Attention ============
    direct_mean, direct_min, direct_max = benchmark_kernel(
        flexi_direct_flash_attn_varlen_func,
        config.num_warmup, config.num_benchmark,
        q=q, k_meta=k_meta, v_meta=v_meta, num_blocks=num_blocks,
        k_ptr_table=k_ptr_table, v_ptr_table=v_ptr_table,
        max_seqlen_q=config.seqlen_q, cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=config.seqlen_k, seqused_k=seqused_k,
        causal=config.causal,
    )
    
    # Cleanup
    free_flexi_kv_ptrs(cached_k_ptrs, cached_v_ptrs)
    
    # Calculate TFLOPS
    flash_tflops = time_to_tflops(flops, flash_mean)
    flexi_tflops = time_to_tflops(flops, flexi_mean)
    direct_tflops = time_to_tflops(flops, direct_mean)
    
    result = BenchmarkResult(
        config=config,
        flash_time_ms=flash_mean,
        flexi_time_ms=flexi_mean,
        flexi_direct_time_ms=direct_mean,
        flash_tflops=flash_tflops,
        flexi_tflops=flexi_tflops,
        flexi_direct_tflops=direct_tflops,
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Config: batch={config.batch_size}, seqlen_q={config.seqlen_q}, "
              f"seqlen_k={config.seqlen_k}, heads={config.num_heads}/{config.num_heads_k}, "
              f"head_dim={config.head_dim}, causal={config.causal}")
        print(f"{'='*70}")
        print(f"{'Kernel':<20} {'Time (ms)':<15} {'TFLOPS':<12} {'vs Flash':<12} {'vs Flexi':<12}")
        print(f"{'-'*70}")
        print(f"{'Flash Attention':<20} {flash_mean:>10.3f}     {flash_tflops:>8.2f}     {'1.00x':>10}     {'-':>10}")
        print(f"{'Flexi':<20} {flexi_mean:>10.3f}     {flexi_tflops:>8.2f}     {result.speedup_flexi_vs_flash:>9.2f}x     {'1.00x':>10}")
        print(f"{'Flexi Direct':<20} {direct_mean:>10.3f}     {direct_tflops:>8.2f}     {result.speedup_direct_vs_flash:>9.2f}x     {result.speedup_direct_vs_flexi:>9.2f}x")
        print(f"{'='*70}")
        print(f"Timing details:")
        print(f"  Flash:        mean={flash_mean:.3f}ms, min={flash_min:.3f}ms, max={flash_max:.3f}ms")
        print(f"  Flexi:        mean={flexi_mean:.3f}ms, min={flexi_min:.3f}ms, max={flexi_max:.3f}ms")
        print(f"  Flexi Direct: mean={direct_mean:.3f}ms, min={direct_min:.3f}ms, max={direct_max:.3f}ms")
    
    return result


def run_benchmark_suite(configs: List[BenchmarkConfig], 
                        output_csv: Optional[str] = None,
                        verbose: bool = True) -> List[BenchmarkResult]:
    """Run benchmark suite and optionally save results to CSV."""
    
    results = []
    
    print("\n" + "=" * 80)
    print(" Flash Attention Benchmark Suite ".center(80, "="))
    print("=" * 80)
    print(f"Running {len(configs)} configurations...")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print("=" * 80)
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running benchmark...")
        try:
            # Only do global warmup on the first config
            result = run_benchmark(config, verbose=verbose, global_warmup_done=(i > 0))
            results.append(result)
        except Exception as e:
            print(f"Error running benchmark: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 105)
    print(" Summary ".center(105, "="))
    print("=" * 105)
    print(f"{'Config':<45} {'Heads':>10} {'Flash':>10} {'Flexi':>10} {'Direct':>10} {'Dir/Flash':>10} {'Dir/Flexi':>10}")
    print(f"{'':45} {'':>10} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'':>10} {'':>10}")
    print("-" * 105)
    
    for r in results:
        config_str = f"B{r.config.batch_size}_Q{r.config.seqlen_q}_K{r.config.seqlen_k}"
        heads_str = f"{r.config.num_heads}/{r.config.num_heads_k}"
        print(f"{config_str:<45} {heads_str:>10} {r.flash_time_ms:>10.3f} {r.flexi_time_ms:>10.3f} "
              f"{r.flexi_direct_time_ms:>10.3f} {r.speedup_direct_vs_flash:>9.2f}x {r.speedup_direct_vs_flexi:>9.2f}x")
    
    # Save to CSV if requested
    if output_csv:
        save_results_to_csv(results, output_csv)
        print(f"\nResults saved to: {output_csv}")
    
    return results


def save_results_to_csv(results: List[BenchmarkResult], filename: str):
    """Save benchmark results to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'batch_size', 'seqlen_q', 'seqlen_k', 'num_heads', 'num_heads_k',
            'head_dim', 'block_size', 'dtype', 'causal',
            'flash_time_ms', 'flexi_time_ms', 'flexi_direct_time_ms',
            'flash_tflops', 'flexi_tflops', 'flexi_direct_tflops',
            'speedup_flexi_vs_flash', 'speedup_direct_vs_flash', 'speedup_direct_vs_flexi'
        ])
        for r in results:
            writer.writerow([
                r.config.batch_size, r.config.seqlen_q, r.config.seqlen_k,
                r.config.num_heads, r.config.num_heads_k, r.config.head_dim,
                r.config.block_size, str(r.config.dtype), r.config.causal,
                r.flash_time_ms, r.flexi_time_ms, r.flexi_direct_time_ms,
                r.flash_tflops, r.flexi_tflops, r.flexi_direct_tflops,
                r.speedup_flexi_vs_flash, r.speedup_direct_vs_flash, r.speedup_direct_vs_flexi
            ])


def get_default_configs() -> List[BenchmarkConfig]:
    """Generate default benchmark configurations."""
    configs = []
    
    # Typical LLM configurations
    dtype = torch.bfloat16
    block_size = 16
    head_dim = 128
    
    # Various batch sizes, sequence lengths
    batch_sizes = [1, 4, 8, 16, 32]
    seqlen_q_vals = [4, 32, 128, 512]  # 1 for decode, others for prefill
    seqlen_k_vals = [256, 512, 1024, 2048, 4096]
    heads_configs = [(32, 8), (16, 4), (8, 8)]  # (num_q_heads, num_kv_heads)
    
    # Add key configurations
    for batch_size in batch_sizes:
        for seqlen_q in seqlen_q_vals:
            for seqlen_k in seqlen_k_vals:
                # Skip if seqlen_q > seqlen_k (not typical)
                if seqlen_q > seqlen_k:
                    continue
                for num_heads, num_heads_k in heads_configs:
                    # Limit total memory usage
                    if batch_size * seqlen_k * num_heads * head_dim > 2**28:
                        continue
                    
                    configs.append(BenchmarkConfig(
                        batch_size=batch_size,
                        seqlen_q=seqlen_q,
                        seqlen_k=seqlen_k,
                        num_heads=num_heads,
                        num_heads_k=num_heads_k,
                        head_dim=head_dim,
                        block_size=block_size,
                        dtype=dtype,
                        causal=True,
                    ))
    
    return configs


def get_quick_configs() -> List[BenchmarkConfig]:
    """Generate quick benchmark configurations for fast testing."""
    configs = []
    dtype = torch.bfloat16
    block_size = 16
    head_dim = 128
    
    quick_tests = [
        # (batch, seqlen_q, seqlen_k, num_heads, num_heads_k)
        (1, 1, 2048, 32, 8),      # Decode, long context
        (8, 1, 1024, 32, 8),      # Batch decode
        (1, 512, 512, 32, 8),     # Prefill
        (4, 256, 2048, 32, 8),    # Mixed
        (16, 32, 1024, 16, 4),    # High batch
        (32, 1, 512, 8, 8),       # Very high batch decode
        (1, 1024, 4096, 32, 8),   # Long prefill
        (8, 128, 2048, 32, 8),    # Medium batch prefill
    ]
    
    for batch, sq, sk, nh, nhk in quick_tests:
        configs.append(BenchmarkConfig(
            batch_size=batch,
            seqlen_q=sq,
            seqlen_k=sk,
            num_heads=nh,
            num_heads_k=nhk,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            causal=True,
        ))
    
    return configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Flash Attention, Flexi, and Flexi Direct kernels"
    )
    
    # Mode selection
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmark with fewer configurations")
    parser.add_argument("--full", action="store_true",
                        help="Run full benchmark suite")
    
    # Single config options
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (single config mode)")
    parser.add_argument("--seqlen-q", type=int, default=None,
                        help="Query sequence length (single config mode)")
    parser.add_argument("--seqlen-k", type=int, default=None,
                        help="Key sequence length (single config mode)")
    parser.add_argument("--num-heads", type=int, default=32,
                        help="Number of query heads")
    parser.add_argument("--num-heads-k", type=int, default=8,
                        help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Head dimension")
    parser.add_argument("--block-size", type=int, default=16,
                        help="KV cache block size")
    parser.add_argument("--causal", action="store_true", default=True,
                        help="Use causal attention")
    parser.add_argument("--no-causal", dest="causal", action="store_false",
                        help="Use non-causal attention")
    
    # Multiple values for sweep
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        help="List of batch sizes to test")
    parser.add_argument("--seqlen-qs", type=int, nargs="+",
                        help="List of query sequence lengths to test")
    parser.add_argument("--seqlen-ks", type=int, nargs="+",
                        help="List of key sequence lengths to test")
    
    # Benchmark parameters
    parser.add_argument("--num-warmup", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--num-benchmark", type=int, default=50,
                        help="Number of benchmark iterations")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file for results")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    
    # Check FA2 availability
    if not is_fa_version_supported(2):
        from vllm_flash_attn.flash_attn_interface import fa_version_unsupported_reason, FA2_AVAILABLE, FA2_UNAVAILABLE_REASON
        print(f"\nFlashAttention 2 is not supported!")
        print(f"  FA2_AVAILABLE: {FA2_AVAILABLE}")
        print(f"  Reason: {fa_version_unsupported_reason(2)}")
        print("\nPlease rebuild the library with: python setup.py develop")
        return
    
    torch.cuda.manual_seed_all(42)
    # torch.set_default_device("cuda")
    
    configs = []
    
    if args.quick:
        configs = get_quick_configs()
    elif args.full:
        configs = get_default_configs()
    elif args.batch_sizes or args.seqlen_qs or args.seqlen_ks:
        # Custom sweep
        batch_sizes = args.batch_sizes or [args.batch_size or 8]
        seqlen_qs = args.seqlen_qs or [args.seqlen_q or 1]
        seqlen_ks = args.seqlen_ks or [args.seqlen_k or 1024]
        
        for bs in batch_sizes:
            for sq in seqlen_qs:
                for sk in seqlen_ks:
                    configs.append(BenchmarkConfig(
                        batch_size=bs,
                        seqlen_q=sq,
                        seqlen_k=sk,
                        num_heads=args.num_heads,
                        num_heads_k=args.num_heads_k,
                        head_dim=args.head_dim,
                        block_size=args.block_size,
                        dtype=torch.bfloat16,
                        causal=args.causal,
                        num_warmup=args.num_warmup,
                        num_benchmark=args.num_benchmark,
                    ))
    elif args.batch_size and args.seqlen_q and args.seqlen_k:
        # Single config
        configs.append(BenchmarkConfig(
            batch_size=args.batch_size,
            seqlen_q=args.seqlen_q,
            seqlen_k=args.seqlen_k,
            num_heads=args.num_heads,
            num_heads_k=args.num_heads_k,
            head_dim=args.head_dim,
            block_size=args.block_size,
            dtype=torch.bfloat16,
            causal=args.causal,
            num_warmup=args.num_warmup,
            num_benchmark=args.num_benchmark,
        ))
    else:
        # Default: quick benchmark
        print("No configuration specified, running quick benchmark...")
        configs = get_quick_configs()
    
    # Update benchmark params for all configs
    for config in configs:
        config.num_warmup = args.num_warmup
        config.num_benchmark = args.num_benchmark
    
    results = run_benchmark_suite(
        configs, 
        output_csv=args.output,
        verbose=not args.quiet
    )
    
    print(f"\nBenchmark completed! Tested {len(results)} configurations.")


if __name__ == "__main__":
    main()
