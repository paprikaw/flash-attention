---
description: 'Describe what this custom agent does and when to use it.'
tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'pylance-mcp-server/*', 'todo', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment', 'ms-toolsai.jupyter/configureNotebook', 'ms-toolsai.jupyter/listNotebookPackages', 'ms-toolsai.jupyter/installNotebookPackages']
---
这个仓库中是flash-attention的实现，我通过对flash-attention进行extend，让其支持block level的kv cache rather than layer level.
主要的两个实现在/home/bxb1/vllm_workbench/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h中，包括了flexi和flexi-direct两种kernel的实现。
flexi kernal通过传入一个list of block address来进行attention时的data addressing，而flexi-direct 则是传入一个block table，这个block table中直接记录了从每个query block到对应的physical key/value block的映射关系，从而省去了中间的addressing步骤，提升了效率。
记住，flexi和direct kernal的一个最大的用处就是可以使用于不连续的内存，从而可以使得内存能够动态分配。因此不要把不连续内存视为不公平的因素，我们的目标就是要让不连续内存的使用效率最大化。

## Basic Rules
1. You should always follows the user defines basic setup above. 
2. Keeps the repository clean and prevents accidental commits of temporary code.
3. When investigating whether a log file has error, rather than print the tail of the log, you should search whether there is errors happening in the log.
4. always use the repository's /home/bxb1/vllm_workbench/flash-attention/build_flash_attn.sh来重新构建kernal
5. 在运行任何指令前，你应该首先激活当前目录下的.venv虚拟环境，使用命令source .venv/bin/activate
6. 不要再未经我允许下重新编译内核，不要取消DEBUG的FLAG