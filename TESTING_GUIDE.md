# Flexi Flash Attention 测试指南

## 测试文件位置
`/home/student.unimelb.edu.au/bxb1/flash-attention/tests/test_vllm_flexi_flash_attn.py`

## 发现的问题及修复

### ✅ 已修复
1. **assert 语法错误**：原来的 `torch.testing.assert_close(...), f"message"` 语法错误
   - 修复：分开写 assert 和 print 语句

### ⚠️  需要注意的问题

1. **k_pages/v_pages 创建**
   ```python
   k_pages = [key_cache[i] for i in range(num_blocks)]
   v_pages = [value_cache[i] for i in range(num_blocks)]
   ```
   
   **可能的问题**：
   - 这会创建 `num_blocks` 个页面（例如 32768 个）
   - 但实际使用的只有 `block_tables` 中引用的那些
   - 可能造成内存浪费和不必要的数据传输
   
   **建议修改**：
   ```python
   # 只创建实际需要的页面
   unique_blocks = torch.unique(block_tables).tolist()
   k_pages = [key_cache[i] for i in unique_blocks]
   v_pages = [value_cache[i] for i in unique_blocks]
   # 可能需要重新映射 block_tables
   ```

2. **fa_version 硬编码**
   ```python
   assert fa_version == 2
   ```
   如果参数化测试中包含 FA3，这个 assert 会失败。建议改为：
   ```python
   if fa_version != 2:
       pytest.skip(f"FA version {fa_version} not supported in flexi interface")
   ```

3. **scheduler_metadata**
   ```python
   scheduler_metadata = None
   ```
   当 `aot_schedule=True` 时，可能需要提供有效的 scheduler_metadata。

## 运行测试的方法

### 前置条件
1. 编译项目：
   ```bash
   cd /home/student.unimelb.edu.au/bxb1/flash-attention/build
   cmake ..
   cmake --build . --target _vllm_fa2_C -j
   ```

2. 设置 Python 路径：
   ```bash
   export PYTHONPATH=/home/student.unimelb.edu.au/bxb1/flash-attention:$PYTHONPATH
   ```

### 方法1：使用测试脚本（推荐）
```bash
cd /home/student.unimelb.edu.au/bxb1/flash-attention
chmod +x run_flexi_test.sh
./run_flexi_test.sh
```

### 方法2：直接使用 pytest

#### 快速测试（单个配置）
```bash
cd /home/student.unimelb.edu.au/bxb1/flash-attention
pytest tests/test_vllm_flexi_flash_attn.py::test_flash_attn_kv_split \
    -v -s \
    -k 'head_size-128 and dtype0 and block_size-16'
```

#### 测试特定 head_size
```bash
# 只测试 head_size=128
pytest tests/test_vllm_flexi_flash_attn.py -v -s -k 'head_size-128'

# 只测试 head_size=256
pytest tests/test_vllm_flexi_flash_attn.py -v -s -k 'head_size-256'
```

#### 测试特定数据类型
```bash
# 只测试 float16
pytest tests/test_vllm_flexi_flash_attn.py -v -s -k 'dtype0'

# 只测试 bfloat16
pytest tests/test_vllm_flexi_flash_attn.py -v -s -k 'dtype1'
```

#### 运行所有测试
```bash
pytest tests/test_vllm_flexi_flash_attn.py -v -s
```

### 方法3：使用 Python 直接运行
```bash
cd /home/student.unimelb.edu.au/bxb1/flash-attention
python -m pytest tests/test_vllm_flexi_flash_attn.py -v -s
```

## pytest 参数说明
- `-v` 或 `--verbose`: 显示详细输出
- `-s`: 显示 print 语句输出
- `-k EXPRESSION`: 按表达式过滤测试
- `-x`: 遇到第一个失败就停止
- `--tb=short`: 显示简短的错误信息
- `--tb=long`: 显示完整的错误信息

## 常用过滤表达式

### 组合条件
```bash
# head_size=128 且 block_size=16
-k 'head_size-128 and block_size-16'

# head_size=128 或 head_size=256
-k 'head_size-128 or head_size-256'

# 排除 soft_cap
-k 'not soft_cap'

# head_size=128 且不使用 soft_cap
-k 'head_size-128 and not soft_cap'
```

### 参数含义对照
- `dtype0` = torch.float16
- `dtype1` = torch.bfloat16
- `block_size-16` = block_size=16
- `block_size-32` = block_size=32
- `soft_cap0` = soft_cap=None
- `soft_cap-10.0` = soft_cap=10.0
- `num_blocks-2048` = num_blocks=2048
- `num_blocks-32768` = num_blocks=32768
- `aot_schedule-True` = aot_schedule=True
- `aot_schedule-False` = aot_schedule=False

## 调试技巧

### 1. 添加更多输出
在测试函数中添加：
```python
print(f"Testing: heads={num_heads}, head_size={head_size}, dtype={dtype}")
print(f"Query shape: {query.shape}")
print(f"Output shape: {output.shape}")
```

### 2. 检查中间结果
```python
print(f"max_query_len: {max_query_len}, max_kv_len: {max_kv_len}")
print(f"Number of pages: {len(k_pages)}")
print(f"Block tables shape: {block_tables.shape}")
```

### 3. 使用 pdb 调试器
```bash
pytest tests/test_vllm_flexi_flash_attn.py -v -s --pdb
```
遇到错误时会自动进入调试器。

### 4. 只运行失败的测试
```bash
pytest tests/test_vllm_flexi_flash_attn.py -v -s --lf
```

## 预期测试时间
- 单个配置：~5-30秒
- head_size=128 全部配置：~5-10分钟
- 所有配置：~30-60分钟（取决于参数组合数量）

## 常见错误及解决方案

### 错误1: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'vllm_flash_attn'
```
**解决**：确保设置了 PYTHONPATH 并且已编译项目

### 错误2: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**解决**：
- 减少 num_blocks 参数（使用 2048 而不是 32768）
- 减少测试的参数组合
- 使用 `-k` 过滤只运行部分测试

### 错误3: Shape mismatch
```
AssertionError: Shape mismatch: torch.Size([...]) vs torch.Size([...])
```
**解决**：检查 flexi_flash_attn_varlen_func 的输出格式是否正确

### 错误4: Assertion failed
```
AssertionError: Tensor-likes are not close!
```
**解决**：这可能是数值精度问题，检查：
- tolerance 设置是否合理（atol, rtol）
- 是否需要调整实现
- 打印 max_diff 查看具体差异大小

## 下一步
1. 运行快速测试验证基本功能
2. 逐步增加测试覆盖范围
3. 根据失败情况调整实现或测试参数
