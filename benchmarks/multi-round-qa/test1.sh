python3 multi-round-qa.py \
    --num-users 2 \
    --num-rounds 2 \
    --qps 2 \
    --shared-system-prompt 1 \
    --user-history-prompt 3000 \
    --answer-len 100 \
    --model Qwen/Qwen2-7B \
    --base-url http://localhost:8000/v1 \
    --sharegpt