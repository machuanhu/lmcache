#!/bin/bash

# 测试批量conversation功能
# 将5个conversation合并成一个prompt发送给LLM

python3 multi-round-qa.py \
    --num-users 3 \
    --num-rounds 10 \
    --qps 3 \
    --shared-system-prompt 1 \
    --user-history-prompt 4096 \
    --answer-len 100 \
    --model /root/.cache/modelscope/hub/models/LLM-Research/llama-2-7b \
    --base-url http://localhost:9000/v1 \
    --sharegpt \
    --batch-conversations 5 