#!/usr/bin/env python3
"""
Flash Attention Benchmark Analyzer

自动运行 pytest 性能测试并分析各阶段的性能数据。
只捕获每组测试中的最后一次运行（排除预热）。

用法:
    python analyze_benchmark.py [options]

选项:
    --test-filter FILTER    pytest 测试过滤器 (默认: test_flexi_direct_performance)
    --runs N                每个测试重复次数 (默认: 1)
    --no-warmup             不跳过第一次运行
    --output FILE           输出文件名 (默认: benchmark_summary.txt)
"""

import subprocess
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict
import argparse


@dataclass
class BenchmarkRun:
    """单次基准测试运行的数据"""
    variant: str  # "FLEXI DIRECT", "FLEXI", "Normal"
    before_params: float = 0.0
    after_params: float = 0.0
    pre_kernel: float = 0.0
    kernel_exec: float = 0.0
    post_kernel: float = 0.0
    reshape_transpose: float = 0.0
    copy_time: float = 0.0
    lse_reshape: float = 0.0
    total: float = 0.0
    resolve_cycles: int = 0
    main_cycles: int = 0
    blocks: int = 0


@dataclass
class BenchmarkGroup:
    """一组测试的数据（多次运行）"""
    runs: List[BenchmarkRun] = field(default_factory=list)
    
    def get_last_run(self) -> BenchmarkRun:
        """获取最后一次运行（排除预热）"""
        return self.runs[-1] if self.runs else None


class BenchmarkAnalyzer:
    def __init__(self):
        self.groups = {
            "FLEXI DIRECT": BenchmarkGroup(),
            "FLEXI": BenchmarkGroup(),
            "Normal": BenchmarkGroup(),
        }
        
        # 正则表达式模式
        self.patterns = {
            'variant': re.compile(r'\[(FLEXI DIRECT|FLEXI|Normal) BENCHMARK\]'),
            'before_params': re.compile(r'Before params time: ([\d.]+) ms'),
            'after_params': re.compile(r'After params time: ([\d.]+) ms'),
            'pre_kernel': re.compile(r'Pre-kernel setup time: ([\d.]+) ms'),
            'kernel_exec': re.compile(r'Kernel execution time: ([\d.]+) ms'),
            'post_kernel': re.compile(r'Post-kernel pre-reshape time: ([\d.]+) ms'),
            'reshape_transpose': re.compile(r'Out reshape\+transpose time: ([\d.]+) ms'),
            'copy_time': re.compile(r'Out copy_ time: ([\d.]+) ms'),
            'lse_reshape': re.compile(r'LSE reshape time: ([\d.]+) ms'),
            'total': re.compile(r'Total.*?time: ([\d.]+) ms'),
            'timing': re.compile(r'\[DEBUG_TIMING.*?\] resolve: (\d+), main: (\d+), blocks: (\d+)'),
        }
    
    def parse_line(self, line: str, current_run: BenchmarkRun) -> bool:
        """解析一行输出，返回是否找到匹配"""
        for key, pattern in self.patterns.items():
            match = pattern.search(line)
            if match:
                if key == 'variant':
                    current_run.variant = match.group(1)
                elif key == 'timing':
                    current_run.resolve_cycles = int(match.group(1))
                    current_run.main_cycles = int(match.group(2))
                    current_run.blocks = int(match.group(3))
                else:
                    setattr(current_run, key, float(match.group(1)))
                return True
        return False
    
    def parse_output(self, output: str):
        """解析 pytest 输出"""
        current_run = None
        current_variant = None
        
        for line in output.split('\n'):
            # 检测新的测试运行开始
            variant_match = self.patterns['variant'].search(line)
            if variant_match:
                variant = variant_match.group(1)
                
                # 如果是新的变体或新的运行
                if current_variant != variant or current_run is None:
                    if current_run and current_run.total > 0:
                        # 保存之前的运行
                        self.groups[current_run.variant].runs.append(current_run)
                    
                    # 开始新的运行
                    current_run = BenchmarkRun(variant=variant)
                    current_variant = variant
            
            # 解析当前行
            if current_run:
                self.parse_line(line, current_run)
                
                # 如果遇到 total time，表示这次运行结束
                if 'Total' in line and 'time:' in line:
                    self.groups[current_run.variant].runs.append(current_run)
                    current_run = BenchmarkRun(variant=current_variant)
    
    def print_summary(self, output_file=None):
        """打印汇总报告（只使用最后一次运行）"""
        lines = []
        
        lines.append("=" * 80)
        lines.append("Flash Attention Benchmark Summary (Last Run Only)")
        lines.append("=" * 80)
        lines.append("")
        
        # 为每个变体打印最后一次运行的详细信息
        for variant_name in ["FLEXI DIRECT", "FLEXI", "Normal"]:
            group = self.groups[variant_name]
            run = group.get_last_run()
            
            if not run:
                lines.append(f"\n{variant_name}: No data")
                continue
            
            lines.append(f"\n{variant_name} (Last Run):")
            lines.append("-" * 80)
            lines.append(f"  Before params:        {run.before_params:8.3f} ms")
            lines.append(f"  After params:         {run.after_params:8.3f} ms")
            lines.append(f"  Pre-kernel setup:     {run.pre_kernel:8.3f} ms")
            lines.append(f"  Kernel execution:     {run.kernel_exec:8.3f} ms")
            lines.append(f"  Post-kernel:          {run.post_kernel:8.3f} ms")
            
            if run.reshape_transpose > 0:
                lines.append(f"  Reshape+transpose:    {run.reshape_transpose:8.3f} ms")
            if run.copy_time > 0:
                lines.append(f"  Copy time:            {run.copy_time:8.3f} ms")
            if run.lse_reshape > 0:
                lines.append(f"  LSE reshape:          {run.lse_reshape:8.3f} ms")
            
            lines.append(f"  Total:                {run.total:8.3f} ms")
            lines.append("")
            lines.append(f"  Resolve cycles:       {run.resolve_cycles:12,} ({run.resolve_cycles/1e6:.1f}M)")
            lines.append(f"  Main cycles:          {run.main_cycles:12,} ({run.main_cycles/1e6:.1f}M)")
            lines.append(f"  Total cycles:         {run.resolve_cycles + run.main_cycles:12,} ({(run.resolve_cycles + run.main_cycles)/1e6:.1f}M)")
            lines.append(f"  Blocks:               {run.blocks:12,}")
        
        # 比较分析
        direct_run = self.groups["FLEXI DIRECT"].get_last_run()
        flexi_run = self.groups["FLEXI"].get_last_run()
        normal_run = self.groups["Normal"].get_last_run()
        
        if direct_run and flexi_run and normal_run:
            lines.append("\n" + "=" * 80)
            lines.append("Performance Comparison (Last Run)")
            lines.append("=" * 80)
            lines.append("")
            
            lines.append("Total Time Comparison:")
            lines.append(f"  Normal:               {normal_run.total:8.3f} ms  (baseline)")
            lines.append(f"  FLEXI:                {flexi_run.total:8.3f} ms  ({flexi_run.total/normal_run.total:.2%})")
            lines.append(f"  FLEXI DIRECT:         {direct_run.total:8.3f} ms  ({direct_run.total/normal_run.total:.2%})")
            lines.append("")
            
            lines.append("Kernel Execution Comparison:")
            lines.append(f"  Normal:               {normal_run.kernel_exec:8.3f} ms")
            lines.append(f"  FLEXI:                {flexi_run.kernel_exec:8.3f} ms  (Δ {flexi_run.kernel_exec - normal_run.kernel_exec:+.3f} ms)")
            lines.append(f"  FLEXI DIRECT:         {direct_run.kernel_exec:8.3f} ms  (Δ {direct_run.kernel_exec - normal_run.kernel_exec:+.3f} ms)")
            lines.append("")
            
            lines.append("Resolve Cycles Comparison:")
            lines.append(f"  Normal:               {normal_run.resolve_cycles/1e6:6.1f}M cycles")
            lines.append(f"  FLEXI:                {flexi_run.resolve_cycles/1e6:6.1f}M cycles  ({flexi_run.resolve_cycles/normal_run.resolve_cycles:.2%})")
            lines.append(f"  FLEXI DIRECT:         {direct_run.resolve_cycles/1e6:6.1f}M cycles  ({direct_run.resolve_cycles/normal_run.resolve_cycles:.2%})")
            lines.append("")
            
            lines.append("Main Kernel Cycles Comparison:")
            lines.append(f"  Normal:               {normal_run.main_cycles/1e6:6.1f}M cycles")
            lines.append(f"  FLEXI:                {flexi_run.main_cycles/1e6:6.1f}M cycles  ({flexi_run.main_cycles/normal_run.main_cycles:.2%})")
            lines.append(f"  FLEXI DIRECT:         {direct_run.main_cycles/1e6:6.1f}M cycles  ({direct_run.main_cycles/normal_run.main_cycles:.2%})")
            lines.append("")
            
            # FLEXI DIRECT vs FLEXI 对比
            lines.append("FLEXI DIRECT vs FLEXI:")
            speedup = flexi_run.total / direct_run.total
            lines.append(f"  Speedup:              {speedup:.3f}x")
            lines.append(f"  Time saved:           {flexi_run.total - direct_run.total:+.3f} ms")
            lines.append(f"  Resolve improvement:  {(flexi_run.resolve_cycles - direct_run.resolve_cycles)/1e6:+.1f}M cycles")
        
        # 输出到文件和控制台
        output = "\n".join(lines)
        print(output)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(output)
            print(f"\n报告已保存到: {output_file}")
    
    def run_pytest(self, test_filter: str, runs: int = 1) -> str:
        """运行 pytest 并返回输出"""
        cmd = [
            "pytest",
            f"tests/test_flexi_direct.py::{test_filter}",
            "-vs",
        ]
        
        print(f"运行命令: {' '.join(cmd)}")
        print(f"重复次数: {runs}")
        print("=" * 80)
        
        all_output = []
        for i in range(runs):
            if runs > 1:
                print(f"\n运行 {i+1}/{runs}...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            
            all_output.append(result.stdout)
            
            if result.returncode != 0:
                print(f"警告: pytest 返回非零退出码 {result.returncode}")
                if result.stderr:
                    print("错误输出:")
                    print(result.stderr)
        
        return "\n".join(all_output)


def main():
    parser = argparse.ArgumentParser(
        description="分析 Flash Attention 性能测试结果（只使用最后一次运行）"
    )
    parser.add_argument(
        "--test-filter",
        default="test_flexi_direct_performance",
        help="pytest 测试过滤器"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="每个测试重复次数"
    )
    parser.add_argument(
        "--output",
        default="benchmark_summary.txt",
        help="输出文件名"
    )
    parser.add_argument(
        "--input",
        help="从文件读取 pytest 输出而不是运行测试"
    )
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer()
    
    if args.input:
        print(f"从文件读取: {args.input}")
        with open(args.input, 'r') as f:
            output = f.read()
    else:
        output = analyzer.run_pytest(args.test_filter, args.runs)
    
    analyzer.parse_output(output)
    analyzer.print_summary(args.output)


if __name__ == "__main__":
    main()
