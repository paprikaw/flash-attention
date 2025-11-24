#!/bin/bash
# 运行 flexi flash attention 测试脚本

set -e  # 遇到错误立即退出

echo "========================================"
echo "运行 flexi_flash_attn 测试"
echo "========================================"

# 设置 PYTHONPATH，确保能找到编译的模块
export PYTHONPATH=/home/student.unimelb.edu.au/bxb1/flash-attention:$PYTHONPATH

cd /home/student.unimelb.edu.au/bxb1/flash-attention

# 方法1: 运行整个测试文件
echo ""
echo "方法1: 运行所有测试（可能需要很长时间）"
echo "--------------------------------------"
echo "命令: pytest tests/test_vllm_flexi_flash_attn.py -v"
echo ""

# 方法2: 运行单个测试（推荐用于调试）
echo "方法2: 运行单个测试配置（推荐）"
echo "--------------------------------------"
echo "命令: pytest tests/test_vllm_flexi_flash_attn.py::test_flash_attn_kv_split -v -k 'head_size-128 and dtype0 and block_size-16'"
echo ""

# 方法3: 运行特定参数组合
echo "方法3: 只测试 head_size=128"
echo "--------------------------------------"
echo "命令: pytest tests/test_vllm_flexi_flash_attn.py -v -k 'head_size-128'"
echo ""

# 方法4: 显示详细输出
echo "方法4: 显示详细输出（包括 print 语句）"
echo "--------------------------------------"
echo "命令: pytest tests/test_vllm_flexi_flash_attn.py -v -s"
echo ""

echo "========================================"
echo "选择运行方式："
echo "========================================"
echo "1. 快速测试（只测试一个配置）"
echo "2. 中等测试（head_size=128）"
echo "3. 完整测试（所有配置）"
echo "4. 自定义"
echo ""
read -p "请选择 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "运行快速测试..."
        pytest tests/test_vllm_flexi_flash_attn.py::test_flash_attn_kv_split \
            -v -s \
            -k 'head_size-128 and dtype0 and block_size-16 and soft_cap0 and num_blocks-2048 and aot_schedule-True'
        ;;
    2)
        echo ""
        echo "运行中等测试（head_size=128）..."
        pytest tests/test_vllm_flexi_flash_attn.py -v -s -k 'head_size-128'
        ;;
    3)
        echo ""
        echo "运行完整测试（所有配置）..."
        pytest tests/test_vllm_flexi_flash_attn.py -v -s
        ;;
    4)
        echo ""
        echo "运行自定义测试..."
        read -p "请输入 pytest 参数（例如: -k 'head_size-256'）: " custom_args
        pytest tests/test_vllm_flexi_flash_attn.py -v -s $custom_args
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "测试完成！"
