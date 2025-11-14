#!/bin/bash

QPS_VALUE="${1:-16}"
BATCH_CONVERSATIONS="${2:-1}"

python3 multi-round-qa.py \
    --num-users 5 \
    --num-rounds 5 \
    --qps 10\
    --shared-system-prompt 1 \
    --user-history-prompt 64 \
    --answer-len 100 \
    --model /root/meta-llama/Llama-2-13b-chat-hf \
    --base-url http://localhost:8100/v1 \
    --sharegpt \
    --time 20 \
    --batch-conversations 5