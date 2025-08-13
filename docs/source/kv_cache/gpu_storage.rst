GPU存储后端
==================

.. _gpu-storage-overview:

概述
--------

GPU存储后端将LMCache的prefix cache chunk存储在GPU显存中，而不是传统的CPU内存。这提供了以下优势：

- **更快的访问速度**: GPU显存访问比CPU内存更快
- **减少CPU-GPU数据传输**: 数据直接在GPU上存储和访问
- **NVLink传输**: 支持使用NVLink在多个LMCache实例之间高速传输数据
- **更好的内存管理**: 专门的GPU内存分配器，避免内存碎片

架构
--------

GPU存储后端包含以下主要组件：

1. **GPUMemoryAllocator**: GPU内存分配器，管理GPU显存池
2. **GPUStorageBackend**: 存储后端实现，提供标准的存储接口
3. **NVLinkTransferManager**: NVLink传输管理器，处理实例间数据传输
4. **NVLinkConnection**: NVLink连接，支持NCCL和ZMQ传输

配置方式
-----------------------------------------

**1. 环境变量配置:**

.. code-block:: bash

    # 启用GPU存储
    export LMCACHE_ENABLE_GPU_STORAGE=True
    # GPU内存大小（GB）
    export LMCACHE_GPU_MEMORY_GB=8.0
    # 启用NVLink传输
    export LMCACHE_ENABLE_NVLINK_TRANSFER=True
    # 本地地址
    export LMCACHE_NVLINK_LOCAL_ADDRESS="localhost:5555"
    # 目标地址（逗号分隔）
    export LMCACHE_NVLINK_TARGET_ADDRESSES="192.168.1.100:5555,192.168.1.101:5555"

**2. 配置文件方式:**

通过 ``LMCACHE_CONFIG_FILE=your-lmcache-config.yaml`` 指定配置文件

示例 ``gpu_storage_config.yaml``:

.. code-block:: yaml

    # 基本配置
    chunk_size: 256
    local_cpu: false
    max_local_cpu_size: 0.0
    
    # GPU存储配置
    enable_gpu_storage: true
    gpu_memory_gb: 8.0
    enable_nvlink_transfer: true
    
    # NVLink传输配置
    nvlink_local_address: "localhost:5555"
    nvlink_target_addresses:
      - "192.168.1.100:5555"
      - "192.168.1.101:5555"

**3. 启动脚本示例:**

.. code-block:: bash

    #!/bin/bash
    
    # 设置环境变量
    export LMCACHE_USE_EXPERIMENTAL=True
    export LMCACHE_CONFIG_FILE=./examples/gpu_storage_config.yaml
    export LMCACHE_ENABLE_GPU_STORAGE=True
    export LMCACHE_GPU_MEMORY_GB=8.0
    export LMCACHE_ENABLE_NVLINK_TRANSFER=True
    
    # 启动vLLM服务器
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --port 8000 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer"}'

GPU内存管理
------------------------------

GPU存储后端使用专门的内存分配器来管理GPU显存：

- **预分配内存池**: 启动时预分配一大块GPU内存
- **动态分配**: 根据需求动态分配和释放内存块
- **内存合并**: 自动合并相邻的空闲内存块
- **内存统计**: 提供内存使用情况的统计信息

NVLink传输
------------------------------

NVLink传输支持两种传输方式：

1. **NCCL传输**: 使用NVIDIA Collective Communications Library进行高速GPU间传输
2. **ZMQ传输**: 使用ZeroMQ进行网络传输，支持跨节点传输

传输特性：

- **异步传输**: 支持异步数据传输，不阻塞主线程
- **批量传输**: 支持批量数据传输，提高传输效率
- **错误处理**: 完善的错误处理和重试机制
- **传输统计**: 提供详细的传输统计信息

性能优化建议
------------------------------

1. **GPU内存大小**: 根据模型大小和并发请求数调整GPU内存大小
2. **NVLink带宽**: 确保NVLink带宽足够支持数据传输需求
3. **网络配置**: 对于跨节点传输，确保网络带宽和延迟满足要求
4. **内存碎片**: 定期监控内存碎片情况，必要时重启服务

使用示例
------------------------------

**单节点GPU存储:**

.. code-block:: python

    from lmcache.v1.config import LMCacheEngineConfig
    
    # 创建GPU存储配置
    config = LMCacheEngineConfig.from_defaults(
        enable_gpu_storage=True,
        gpu_memory_gb=8.0,
        enable_nvlink_transfer=False  # 单节点不需要NVLink传输
    )

**多节点GPU存储:**

.. code-block:: python

    # 节点1配置
    config1 = LMCacheEngineConfig.from_defaults(
        enable_gpu_storage=True,
        gpu_memory_gb=8.0,
        enable_nvlink_transfer=True,
        nvlink_local_address="192.168.1.100:5555",
        nvlink_target_addresses=["192.168.1.101:5555"]
    )
    
    # 节点2配置
    config2 = LMCacheEngineConfig.from_defaults(
        enable_gpu_storage=True,
        gpu_memory_gb=8.0,
        enable_nvlink_transfer=True,
        nvlink_local_address="192.168.1.101:5555",
        nvlink_target_addresses=["192.168.1.100:5555"]
    )

故障排除
------------------------------

1. **GPU内存不足**: 减少 `gpu_memory_gb` 或增加GPU内存
2. **NVLink连接失败**: 检查NVLink硬件连接和网络配置
3. **传输超时**: 增加超时时间或检查网络带宽
4. **内存碎片**: 重启服务或调整内存分配策略

限制和注意事项
------------------------------

1. **GPU内存限制**: 受限于GPU显存大小
2. **NVLink硬件要求**: 需要支持NVLink的GPU
3. **网络依赖**: 跨节点传输依赖网络性能
4. **兼容性**: 需要CUDA 11.8+和PyTorch 2.0+ 