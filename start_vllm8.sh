PYTHONHASHSEED=0 \
NCCL_SOCKET_IFNAME=eth0 \
GLOO_SOCKET_IFNAME=eth0 \
LMCACHE_RANK=7 \
LMCACHE_WORLD_SIZE=8 \
LMCACHE_LOG_LEVEL=DEBUG \
CUDA_VISIBLE_DEVICES=7 \
NCCL_NVLINK_DISABLE=0 \
NCCL_P2P_DISABLE=0 \
LMCACHE_CONFIG_FILE=lmcache_config8.yaml \
vllm serve ~/meta-llama/Llama-2-13b-chat-hf \
    --max-model-len 4096\
    --port 8007 \
    > vllm8.log 2>&1
    #--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' > vllm_cachehub8.log 2>&1


