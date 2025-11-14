PYTHONHASHSEED=0 \
CUDA_LAUNCH_BLOCKING=1 \
TORCH_USE_CUDA_DSA=1 \
NCCL_SOCKET_IFNAME=eth0 \
GLOO_SOCKET_IFNAME=eth0 \
LMCACHE_RANK=1 \
LMCACHE_WORLD_SIZE=8 \
LMCACHE_LOG_LEVEL=DEBUG \
CUDA_VISIBLE_DEVICES=1 \
NCCL_NVLINK_DISABLE=0 \
NCCL_P2P_DISABLE=0 \
LMCACHE_CONFIG_FILE=lmcache_config2.yaml \
vllm serve ~/meta-llama/Llama-2-13b-chat-hf \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.5 \
    --host 127.0.0.1 \
    --port 8001 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' > vllm_cachehub2.log 2>&1