#!/usr/bin/env python3
"""
NCCL GPU传输示例脚本

这个脚本演示了如何使用LMCache的NCCL GPU传输功能。
需要在不同的终端中运行不同的rank。

使用方法:
1. 设置环境变量并启动服务器:
   export LMCACHE_RANK=0
   export LMCACHE_WORLD_SIZE=2
   export CUDA_VISIBLE_DEVICES=0
   python nccl_gpu_transfer_example.py server

2. 在另一个终端启动客户端:
   export LMCACHE_RANK=1
   export LMCACHE_WORLD_SIZE=2
   export CUDA_VISIBLE_DEVICES=1
   python nccl_gpu_transfer_example.py client
"""

import os
import sys
import asyncio
import time
import torch

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from lmcache.v1.distributed_server.nccl_manager import get_nccl_manager
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj


async def run_server():
    """运行服务器端"""
    print(f"启动服务器 - Rank: {os.getenv('LMCACHE_RANK')}")
    
    # 初始化NCCL管理器
    nccl_manager = get_nccl_manager()
    
    # 创建一些测试数据在GPU上
    test_tensor = torch.randn(1000, 1000, device='cuda')
    test_memory_obj = MemoryObj(
        tensor=test_tensor,
        byte_array=test_tensor.cpu().numpy().tobytes(),
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1
    )
    
    # 模拟存储管理器
    storage = {CacheEngineKey(fmt="test", model_name="test", world_size=2, worker_id=0, chunk_hash="test_key"): test_memory_obj}
    
    print("服务器准备就绪，等待客户端请求...")
    
    # 模拟处理客户端请求
    while True:
        await asyncio.sleep(1)
        # 这里可以添加实际的socket监听逻辑


async def run_client():
    """运行客户端"""
    print(f"启动客户端 - Rank: {os.getenv('LMCACHE_RANK')}")
    
    # 初始化NCCL管理器
    nccl_manager = get_nccl_manager()
    
    # 等待NCCL初始化
    await asyncio.sleep(2)
    
    if not nccl_manager.is_available():
        print("NCCL不可用，退出")
        return
    
    # 模拟从服务器获取数据
    test_key = CacheEngineKey(fmt="test", model_name="test", world_size=2, worker_id=0, chunk_hash="test_key")
    
    print("尝试从rank 0获取数据...")
    
    # 模拟接收数据
    memory_obj = nccl_manager.recv_gpu_data(
        source_rank=0,
        shape=torch.Size([1000, 1000, 0, 0]),
        dtype=torch.float32
    )
    
    if memory_obj is not None:
        print(f"成功接收数据，形状: {memory_obj.tensor.shape}")
        print(f"数据设备: {memory_obj.tensor.device}")
        print(f"数据前几个值: {memory_obj.tensor.flatten()[:5]}")
    else:
        print("接收数据失败")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['server', 'client']:
        print("使用方法: python nccl_gpu_transfer_example.py [server|client]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    # 检查环境变量
    rank = os.getenv('LMCACHE_RANK')
    world_size = os.getenv('LMCACHE_WORLD_SIZE')
    cuda_device = os.getenv('CUDA_VISIBLE_DEVICES')
    
    if not all([rank, world_size, cuda_device]):
        print("请设置环境变量: LMCACHE_RANK, LMCACHE_WORLD_SIZE, CUDA_VISIBLE_DEVICES")
        sys.exit(1)
    
    print(f"配置 - Rank: {rank}, World Size: {world_size}, CUDA Device: {cuda_device}")
    
    if mode == 'server':
        asyncio.run(run_server())
    else:
        asyncio.run(run_client())


if __name__ == "__main__":
    main() 