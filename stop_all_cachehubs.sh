#!/bin/bash

# 停止所有 lmcache 实例

echo "Stopping all lmcache instances..."

# 查找并停止所有 vllm serve 进程
pkill -f 'vllm serve'

echo "All instances stopped."
echo ""
echo "To verify, check if processes are still running:"
echo "  ps aux | grep 'vllm serve'"


