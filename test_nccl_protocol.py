#!/usr/bin/env python3
"""
测试NCCL协议格式的脚本
"""

import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from lmcache.v1.protocol import ClientMetaMessage, ServerMetaMessage, Constants
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
import torch


def test_protocol_serialization():
    """测试协议序列化和反序列化"""
    print("测试协议序列化和反序列化...")
    
    # 创建测试key
    key = CacheEngineKey(
        fmt="test",
        model_name="test_model",
        world_size=2,
        worker_id=0,
        chunk_hash="test_hash"
    )
    
    # 测试ClientMetaMessage
    client_msg = ClientMetaMessage(
        command=Constants.CLIENT_GET,
        key=key,
        length=0,
        fmt=MemoryFormat.KV_2LTD,
        dtype=torch.float16,
        shape=torch.Size([1000, 1000, 0, 0]),
        source_rank=1,
        target_rank=0
    )
    
    # 序列化
    serialized = client_msg.serialize()
    print(f"ClientMetaMessage序列化长度: {len(serialized)}")
    
    # 反序列化
    deserialized = ClientMetaMessage.deserialize(serialized)
    print(f"反序列化结果: command={deserialized.command}, source_rank={deserialized.source_rank}, target_rank={deserialized.target_rank}")
    
    # 测试ServerMetaMessage
    server_msg = ServerMetaMessage(
        code=Constants.GPU_SUCCESS,
        length=4000000,  # 1000*1000*4 bytes
        fmt=MemoryFormat.KV_2LTD,
        dtype=torch.float16,
        shape=torch.Size([1000, 1000, 0, 0]),
        source_rank=0,
        target_rank=1
    )
    
    # 序列化
    serialized = server_msg.serialize()
    print(f"ServerMetaMessage序列化长度: {len(serialized)}")
    
    # 反序列化
    deserialized = ServerMetaMessage.deserialize(serialized)
    print(f"反序列化结果: code={deserialized.code}, source_rank={deserialized.source_rank}, target_rank={deserialized.target_rank}")
    
    print("协议测试通过！")


def test_constants():
    """测试常量定义"""
    print("\n测试常量定义...")
    print(f"CLIENT_GET: {Constants.CLIENT_GET}")
    print(f"SERVER_SUCCESS: {Constants.SERVER_SUCCESS}")
    print(f"GPU_SUCCESS: {Constants.GPU_SUCCESS}")
    print(f"SERVER_FAIL: {Constants.SERVER_FAIL}")


if __name__ == "__main__":
    test_constants()
    test_protocol_serialization() 