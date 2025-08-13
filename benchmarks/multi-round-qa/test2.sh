python3 multi-round-qa.py \
    --num-users 5 \
    --num-rounds 2 \
    --qps 5 \
    --shared-system-prompt 1 \
    --user-history-prompt 1000 \
    --answer-len 100 \
    --model Qwen/Qwen2-7B \
    --base-url http://localhost:8001/v1 \
    --sharegpt