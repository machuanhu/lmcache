PYTHONHASHSEED=0 \
LMCACHE_RANK=1 \
LMCACHE_WORLD_SIZE=2 \
LMCACHE_LOG_LEVEL=DEBUG \
CUDA_VISIBLE_DEVICES=1 \
NCCL_NVLINK_DISABLE=0 \
NCCL_P2P_DISABLE=0 \
LMCACHE_CONFIG_FILE=lmcache_config2.yaml \
vllm serve ~/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e/ \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --port 8001 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' > vllm_lmcache2.log 2>&1