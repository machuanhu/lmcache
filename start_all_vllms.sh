#!/bin/bash

# 启动所有 8 个 vllm 实例
# 每个实例在后台运行

echo "Starting all 8 vllm instances..."

# 启动实例 1-8
nohup bash start_vllm1.sh &
echo "Started vllm1 (GPU 0, Port 8000)"

nohup bash start_vllm2.sh &
echo "Started vllm2 (GPU 1, Port 8001)"

nohup bash start_vllm3.sh &
echo "Started vllm3 (GPU 2, Port 8002)"

nohup bash start_vllm4.sh &
echo "Started vllm4 (GPU 3, Port 8003)"



echo ""
echo "All 8 instances started in background."
echo "Check logs: vllm1.log to vllm8.log"
echo ""
echo "To stop all instances, run: pkill -f 'vllm serve'"


