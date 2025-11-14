#!/bin/bash

# 启动所有 8 个 lmcache 实例
# 每个实例在后台运行

echo "Starting all 8 lmcache instances..."

# 启动实例 1-8
nohup bash start_cachehub1.sh &
echo "Started cachehub1 (GPU 0, Port 8000)"

nohup bash start_cachehub2.sh &
echo "Started cachehub2 (GPU 1, Port 8001)"

nohup bash start_cachehub3.sh &
echo "Started cachehub3 (GPU 2, Port 8002)"

nohup bash start_cachehub4.sh &
echo "Started cachehub4 (GPU 3, Port 8003)"

nohup bash start_cachehub5.sh &
echo "Started cachehub5 (GPU 4, Port 8004)"

nohup bash start_cachehub6.sh &
echo "Started cachehub6 (GPU 5, Port 8005)"

nohup bash start_cachehub7.sh &
echo "Started cachehub7 (GPU 6, Port 8006)"

nohup bash start_cachehub8.sh &
echo "Started cachehub8 (GPU 7, Port 8007)"

echo ""
echo "All 8 instances started in background."
echo "Check logs: vllm_cachehub1.log to vllm_cachehub8.log"
echo ""
echo "To stop all instances, run: pkill -f 'vllm serve'"


