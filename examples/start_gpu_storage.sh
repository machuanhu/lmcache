#!/bin/bash

# LMCache GPU存储启动脚本示例
# 此脚本展示了如何启动使用GPU存储的LMCache实例

# 设置环境变量
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CONFIG_FILE=./examples/gpu_storage_config.yaml

# GPU存储相关环境变量
export LMCACHE_ENABLE_GPU_STORAGE=True
export LMCACHE_GPU_MEMORY_GB=8.0
export LMCACHE_ENABLE_NVLINK_TRANSFER=True
export LMCACHE_NVLINK_LOCAL_ADDRESS="localhost:5555"
export LMCACHE_NVLINK_TARGET_ADDRESSES="192.168.1.100:5555,192.168.1.101:5555"

# vLLM相关环境变量
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0

# 启动vLLM服务器
echo "启动LMCache GPU存储实例..."
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --disable-log-requests \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "gpu_storage_instance"}}' 