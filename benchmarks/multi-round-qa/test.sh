python3 multi-round-qa.py \
    --num-users 3 \
    --num-rounds 10 \
    --qps 3 \
    --shared-system-prompt 1 \
    --user-history-prompt 4096 \
    --answer-len 100 \
    --model /root/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e/ \
    --base-url http://localhost:9000/v1 \
    --sharegpt