PYTHONHASHSEED=0 \
NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1 \
NCCL_IB_DISABLE=0 \
NCCL_SOCKET_IFNAME=bond0 \
LMCACHE_RANK=0 \
LMCACHE_WORLD_SIZE=2 \
LMCACHE_LOG_LEVEL=DEBUG \
CUDA_VISIBLE_DEVICES=0 \
NCCL_NVLINK_DISABLE=0 \
NCCL_P2P_DISABLE=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
LMCACHE_CONFIG_FILE=lmcache_config1.yaml \
vllm serve /root/.cache/modelscope/hub/models/LLM-Research/llama-2-7b \
    --max-model-len 4096\
    --gpu-memory-utilization 0.5 \
    --port 8000 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' > vllm_cachehub1.log 2>&1