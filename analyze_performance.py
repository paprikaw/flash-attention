#!/usr/bin/env python3
"""
Performance Analysis Tool for Flash Attention Variants

Automatically runs pytest and collects timing data from different variants:
- Regular Flash Attention
- FLEXI
- FLEXI DIRECT

Outputs a detailed summary with statistics for each stage.
"""

import subprocess
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict
import statistics


@dataclass
class TimingData:
    """Store timing measurements for a specific variant and stage."""
    before_params: List[float] = field(default_factory=list)
    after_params: List[float] = field(default_factory=list)
    pre_kernel_setup: List[float] = field(default_factory=list)
    kernel_execution: List[float] = field(default_factory=list)
    post_kernel_pre_reshape: List[float] = field(default_factory=list)
    total: List[float] = field(default_factory=list)
    
    # Debug timing (cycles)
    resolve_cycles: List[int] = field(default_factory=list)
    main_cycles: List[int] = field(default_factory=list)
    blocks: List[int] = field(default_factory=list)


def parse_timing_line(line: str, pattern: str) -> float:
    """Extract timing value from a line matching the pattern."""
    match = re.search(pattern + r':\s+([\d.]+)\s+ms', line)
    if match:
        return float(match.group(1))
    return None


def parse_debug_timing(line: str) -> tuple:
    """Extract cycle counts from DEBUG_TIMING line."""
    match = re.search(r'resolve:\s+(\d+),\s+main:\s+(\d+),\s+blocks:\s+(\d+)', line)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def run_pytest(test_path: str = "tests/test_flexi_direct.py::test_flexi_direct_performance") -> str:
    """Run pytest and capture output."""
    print(f"Running: pytest {test_path} -vs")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            ["pytest", test_path, "-vs"],
            capture_output=True,
            text=True,
            check=False
        )
        
        output = result.stdout + result.stderr
        
        # Print output in real-time style
        print(output)
        
        if result.returncode != 0:
            print(f"\n⚠️  Warning: pytest exited with code {result.returncode}")
        
        return output
        
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        sys.exit(1)


def parse_output(output: str) -> Dict[str, TimingData]:
    """Parse pytest output and extract timing data."""
    data = {
        'FLEXI DIRECT': TimingData(),
        'FLEXI': TimingData(),
        'Normal': TimingData()
    }
    
    current_variant = None
    lines = output.split('\n')
    
    for line in lines:
        # Detect which variant we're parsing
        if '[FLEXI DIRECT BENCHMARK]' in line:
            current_variant = 'FLEXI DIRECT'
        elif '[FLEXI BENCHMARK]' in line:
            current_variant = 'FLEXI'
        elif '[Normal BENCHMARK]' in line:
            current_variant = 'Normal'
        
        if current_variant is None:
            continue
        
        variant_data = data[current_variant]
        
        # Parse timing lines
        if 'Before params time:' in line:
            val = parse_timing_line(line, 'Before params time')
            if val is not None:
                variant_data.before_params.append(val)
        
        elif 'After params time:' in line:
            val = parse_timing_line(line, 'After params time')
            if val is not None:
                variant_data.after_params.append(val)
        
        elif 'Pre-kernel setup time:' in line:
            val = parse_timing_line(line, 'Pre-kernel setup time')
            if val is not None:
                variant_data.pre_kernel_setup.append(val)
        
        elif 'Kernel execution time:' in line:
            val = parse_timing_line(line, 'Kernel execution time')
            if val is not None:
                variant_data.kernel_execution.append(val)
        
        elif 'Post-kernel pre-reshape time:' in line:
            val = parse_timing_line(line, 'Post-kernel pre-reshape time')
            if val is not None:
                variant_data.post_kernel_pre_reshape.append(val)
        
        elif 'Total' in line and 'time:' in line:
            # Match both "Total time:" and "Total flexi_*_mha_varlen_fwd time:"
            val = parse_timing_line(line, 'Total.*time')
            if val is not None:
                variant_data.total.append(val)
        
        # Parse debug timing
        elif '[DEBUG_TIMING' in line:
            resolve, main, blocks = parse_debug_timing(line)
            if resolve is not None:
                variant_data.resolve_cycles.append(resolve)
                variant_data.main_cycles.append(main)
                variant_data.blocks.append(blocks)
    
    return data


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """Calculate mean, std, min, max from a list of values."""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
    
    # Skip first measurement (warmup)
    if len(values) > 1:
        values = values[1:]
    
    return {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }


def format_stat_line(label: str, stats: Dict[str, float], unit: str = "ms", width: int = 30) -> str:
    """Format a statistics line."""
    mean = stats['mean']
    std = stats['std']
    if unit == "M":
        return f"  {label:<{width}} {mean/1e6:7.1f} ± {std/1e6:4.1f} {unit}"
    else:
        return f"  {label:<{width}} {mean:7.3f} ± {std:5.3f} {unit}"


def print_summary(data: Dict[str, TimingData]):
    """Print detailed summary of timing data."""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Print for each variant
    for variant_name in ['Normal', 'FLEXI', 'FLEXI DIRECT']:
        variant_data = data[variant_name]
        
        print(f"\n{'─' * 80}")
        print(f"📊 {variant_name}")
        print(f"{'─' * 80}")
        
        if not variant_data.total:
            print("  ⚠️  No data collected")
            continue
        
        # Host-side timing
        print("\n  Host-side Timing:")
        if variant_data.before_params:
            stats = calculate_stats(variant_data.before_params)
            print(format_stat_line("Before params", stats))
        
        if variant_data.after_params:
            stats = calculate_stats(variant_data.after_params)
            print(format_stat_line("After params", stats))
        
        if variant_data.pre_kernel_setup:
            stats = calculate_stats(variant_data.pre_kernel_setup)
            print(format_stat_line("Pre-kernel setup", stats))
        
        # Kernel timing
        print("\n  Kernel Timing:")
        if variant_data.kernel_execution:
            stats = calculate_stats(variant_data.kernel_execution)
            print(format_stat_line("Kernel execution", stats))
        
        if variant_data.post_kernel_pre_reshape:
            stats = calculate_stats(variant_data.post_kernel_pre_reshape)
            print(format_stat_line("Post-kernel pre-reshape", stats))
        
        # GPU cycles
        if variant_data.resolve_cycles:
            print("\n  GPU Cycles (from kernel):")
            resolve_stats = calculate_stats(variant_data.resolve_cycles)
            main_stats = calculate_stats(variant_data.main_cycles)
            
            print(format_stat_line("Resolve cycles", resolve_stats, "M"))
            print(format_stat_line("Main kernel cycles", main_stats, "M"))
            
            total_cycles = [r + m for r, m in zip(variant_data.resolve_cycles, variant_data.main_cycles)]
            total_stats = calculate_stats(total_cycles)
            print(format_stat_line("Total cycles", total_stats, "M"))
        
        # Total timing
        print("\n  Total:")
        if variant_data.total:
            stats = calculate_stats(variant_data.total)
            print(format_stat_line("End-to-end time", stats))
    
    # Comparative analysis
    print(f"\n{'=' * 80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'=' * 80}")
    
    normal_total = calculate_stats(data['Normal'].total)['mean']
    flexi_total = calculate_stats(data['FLEXI'].total)['mean']
    direct_total = calculate_stats(data['FLEXI DIRECT'].total)['mean']
    
    normal_kernel = calculate_stats(data['Normal'].kernel_execution)['mean']
    flexi_kernel = calculate_stats(data['FLEXI'].kernel_execution)['mean']
    direct_kernel = calculate_stats(data['FLEXI DIRECT'].kernel_execution)['mean']
    
    if normal_total > 0 and flexi_total > 0 and direct_total > 0:
        print(f"\n  End-to-end Performance:")
        print(f"    Normal:        {normal_total:7.3f} ms  (baseline)")
        print(f"    FLEXI:         {flexi_total:7.3f} ms  ({flexi_total/normal_total:+6.1%})")
        print(f"    FLEXI DIRECT:  {direct_total:7.3f} ms  ({direct_total/normal_total:+6.1%})")
        
        print(f"\n  Kernel Performance:")
        print(f"    Normal:        {normal_kernel:7.3f} ms  (baseline)")
        print(f"    FLEXI:         {flexi_kernel:7.3f} ms  ({flexi_kernel/normal_kernel:+6.1%})")
        print(f"    FLEXI DIRECT:  {direct_kernel:7.3f} ms  ({direct_kernel/normal_kernel:+6.1%})")
        
        print(f"\n  FLEXI DIRECT vs FLEXI:")
        print(f"    Speedup:       {flexi_total/direct_total:.2f}x")
        print(f"    Time saved:    {(flexi_total - direct_total)*1000:.0f} μs")
    
    # Cycle analysis
    if data['Normal'].resolve_cycles and data['FLEXI'].resolve_cycles and data['FLEXI DIRECT'].resolve_cycles:
        normal_resolve = calculate_stats(data['Normal'].resolve_cycles)['mean']
        flexi_resolve = calculate_stats(data['FLEXI'].resolve_cycles)['mean']
        direct_resolve = calculate_stats(data['FLEXI DIRECT'].resolve_cycles)['mean']
        
        normal_main = calculate_stats(data['Normal'].main_cycles)['mean']
        flexi_main = calculate_stats(data['FLEXI'].main_cycles)['mean']
        direct_main = calculate_stats(data['FLEXI DIRECT'].main_cycles)['mean']
        
        print(f"\n  Cycle Breakdown:")
        print(f"    {'':20} {'Resolve':>12} {'Main':>12} {'Total':>12}")
        print(f"    {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        print(f"    {'Normal':20} {normal_resolve/1e6:10.1f} M {normal_main/1e6:10.1f} M {(normal_resolve+normal_main)/1e6:10.1f} M")
        print(f"    {'FLEXI':20} {flexi_resolve/1e6:10.1f} M {flexi_main/1e6:10.1f} M {(flexi_resolve+flexi_main)/1e6:10.1f} M")
        print(f"    {'FLEXI DIRECT':20} {direct_resolve/1e6:10.1f} M {direct_main/1e6:10.1f} M {(direct_resolve+direct_main)/1e6:10.1f} M")
        
        print(f"\n  Resolve Efficiency:")
        print(f"    FLEXI DIRECT vs Normal:  {direct_resolve/normal_resolve:+6.1%} ({(direct_resolve-normal_resolve)/1e6:+6.1f}M cycles)")
        print(f"    FLEXI DIRECT vs FLEXI:   {direct_resolve/flexi_resolve:+6.1%} ({(direct_resolve-flexi_resolve)/1e6:+6.1f}M cycles)")
    
    print(f"\n{'=' * 80}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze Flash Attention performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run default test
  %(prog)s --test tests/test_flexi_direct.py  # Run specific test
  %(prog)s --no-run --file output.txt         # Parse existing output
        """
    )
    parser.add_argument(
        '--test', '-t',
        default='tests/test_flexi_direct.py::test_flexi_direct_performance',
        help='Pytest test path to run (default: test_flexi_direct_performance)'
    )
    parser.add_argument(
        '--no-run',
        action='store_true',
        help='Skip running pytest, parse output from file instead'
    )
    parser.add_argument(
        '--file', '-f',
        help='Input file to parse (used with --no-run)'
    )
    
    args = parser.parse_args()
    
    if args.no_run:
        if not args.file:
            print("❌ Error: --file required when using --no-run")
            sys.exit(1)
        try:
            with open(args.file, 'r') as f:
                output = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    else:
        output = run_pytest(args.test)
    
    # Parse and analyze
    data = parse_output(output)
    print_summary(data)


if __name__ == '__main__':
    main()
