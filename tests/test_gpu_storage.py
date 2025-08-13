#!/usr/bin/env python3
"""
GPU存储功能测试脚本
"""

import os
import sys
import torch
import unittest
from typing import Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lmcache.v1.storage_backend.gpu_storage_backend import (
    GPUMemoryAllocator, 
    GPUStorageBackend
)
from lmcache.v1.storage_backend.nvlink_transfer import (
    NVLinkTransferManager,
    NVLinkConnection
)
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, MemoryFormat
from lmcache.v1.cache_engine_key import CacheEngineKey


class TestGPUMemoryAllocator(unittest.TestCase):
    """测试GPU内存分配器"""
    
    def setUp(self):
        """设置测试环境"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA不可用")
        
        self.device = "cuda:0"
        self.max_memory_gb = 1.0  # 使用1GB进行测试
        self.allocator = GPUMemoryAllocator(self.device, self.max_memory_gb)
    
    def test_allocate_and_free(self):
        """测试内存分配和释放"""
        # 分配内存
        shape = torch.Size([2, 32, 256, 128])
        dtype = torch.float16
        memory_obj = self.allocator.allocate(shape, dtype)
        
        self.assertIsNotNone(memory_obj)
        self.assertEqual(memory_obj.tensor.shape, shape)
        self.assertEqual(memory_obj.tensor.dtype, dtype)
        self.assertEqual(memory_obj.tensor.device.type, "cuda")
        
        # 释放内存
        self.allocator.free(memory_obj)
        
        # 验证内存已释放
        allocated, total = self.allocator.get_memory_usage()
        self.assertEqual(allocated, 0)
    
    def test_multiple_allocations(self):
        """测试多次分配"""
        shape = torch.Size([2, 16, 128, 64])
        dtype = torch.float16
        
        memory_objs = []
        for i in range(5):
            memory_obj = self.allocator.allocate(shape, dtype)
            self.assertIsNotNone(memory_obj)
            memory_objs.append(memory_obj)
        
        # 释放部分内存
        for i in range(0, 5, 2):
            self.allocator.free(memory_objs[i])
        
        # 验证内存使用情况
        allocated, total = self.allocator.get_memory_usage()
        self.assertGreater(allocated, 0)
        self.assertLess(allocated, total)
    
    def test_memory_exhaustion(self):
        """测试内存耗尽"""
        # 分配大量内存直到耗尽
        shape = torch.Size([2, 32, 256, 128])
        dtype = torch.float16
        
        memory_objs = []
        while True:
            memory_obj = self.allocator.allocate(shape, dtype)
            if memory_obj is None:
                break
            memory_objs.append(memory_obj)
        
        # 验证至少分配了一些内存
        self.assertGreater(len(memory_objs), 0)
        
        # 释放所有内存
        for memory_obj in memory_objs:
            self.allocator.free(memory_obj)


class TestGPUStorageBackend(unittest.TestCase):
    """测试GPU存储后端"""
    
    def setUp(self):
        """设置测试环境"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA不可用")
        
        self.device = "cuda:0"
        self.max_memory_gb = 1.0
        self.backend = GPUStorageBackend(
            device=self.device,
            max_memory_gb=self.max_memory_gb,
            enable_nvlink_transfer=False  # 测试时不启用NVLink传输
        )
    
    def test_put_and_get(self):
        """测试存储和获取"""
        # 创建测试数据
        key = CacheEngineKey("test_format", "test_model", 1, 0, 12345)
        tensor = torch.randn(2, 16, 128, 64, dtype=torch.float16, device=self.device)
        memory_obj = MemoryObj(
            tensor=tensor,
            metadata=MemoryObjMetadata(fmt=MemoryFormat.KV_2LTD, ref_count=1)
        )
        
        # 存储数据
        self.backend.put(key, memory_obj)
        
        # 验证数据存在
        self.assertTrue(self.backend.contains(key))
        
        # 获取数据
        retrieved_obj = self.backend.get(key)
        self.assertIsNotNone(retrieved_obj)
        self.assertTrue(torch.allclose(retrieved_obj.tensor, tensor))
    
    def test_remove(self):
        """测试数据移除"""
        key = CacheEngineKey("test_format", "test_model", 1, 0, 12345)
        tensor = torch.randn(2, 16, 128, 64, dtype=torch.float16, device=self.device)
        memory_obj = MemoryObj(
            tensor=tensor,
            metadata=MemoryObjMetadata(fmt=MemoryFormat.KV_2LTD, ref_count=1)
        )
        
        # 存储数据
        self.backend.put(key, memory_obj)
        self.assertTrue(self.backend.contains(key))
        
        # 移除数据
        self.backend.remove(key)
        self.assertFalse(self.backend.contains(key))
    
    def test_overwrite(self):
        """测试数据覆盖"""
        key = CacheEngineKey("test_format", "test_model", 1, 0, 12345)
        
        # 存储第一个数据
        tensor1 = torch.randn(2, 16, 128, 64, dtype=torch.float16, device=self.device)
        memory_obj1 = MemoryObj(
            tensor=tensor1,
            metadata=MemoryObjMetadata(fmt=MemoryFormat.KV_2LTD, ref_count=1)
        )
        self.backend.put(key, memory_obj1)
        
        # 存储第二个数据（覆盖）
        tensor2 = torch.randn(2, 16, 128, 64, dtype=torch.float16, device=self.device)
        memory_obj2 = MemoryObj(
            tensor=tensor2,
            metadata=MemoryObjMetadata(fmt=MemoryFormat.KV_2LTD, ref_count=1)
        )
        self.backend.put(key, memory_obj2)
        
        # 验证获取的是第二个数据
        retrieved_obj = self.backend.get(key)
        self.assertTrue(torch.allclose(retrieved_obj.tensor, tensor2))


class TestNVLinkTransfer(unittest.TestCase):
    """测试NVLink传输"""
    
    def setUp(self):
        """设置测试环境"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA不可用")
    
    def test_transfer_manager_creation(self):
        """测试传输管理器创建"""
        manager = NVLinkTransferManager(
            local_address="localhost:5555",
            local_device="cuda:0",
            enable_nccl=False,  # 测试时禁用NCCL
            enable_zmq=True
        )
        
        self.assertIsNotNone(manager)
        self.assertEqual(manager.local_address, "localhost:5555")
        self.assertEqual(manager.local_device, "cuda:0")
    
    def test_connection_creation(self):
        """测试连接创建"""
        connection = NVLinkConnection(
            local_address="localhost:5555",
            target_address="localhost:5556",
            local_device="cuda:0",
            enable_nccl=False,
            enable_zmq=True
        )
        
        self.assertIsNotNone(connection)
        self.assertEqual(connection.local_address, "localhost:5555")
        self.assertEqual(connection.target_address, "localhost:5556")


def run_tests():
    """运行所有测试"""
    # 设置测试环境
    os.environ['LMCACHE_USE_EXPERIMENTAL'] = 'True'
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_suite.addTest(unittest.makeSuite(TestGPUMemoryAllocator))
    test_suite.addTest(unittest.makeSuite(TestGPUStorageBackend))
    test_suite.addTest(unittest.makeSuite(TestNVLinkTransfer))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 