#!/usr/bin/env python3
"""
测试GPU存储数据流
验证vLLM是否直接与GPU显存交互，而不是通过CPU内存中转
"""

import os
import sys
import torch
import unittest
from typing import Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lmcache.v1.storage_backend.gpu_storage_backend import GPUStorageBackend
from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, MemoryFormat
from lmcache.v1.cache_engine_key import CacheEngineKey


class TestGPUStorageDataFlow(unittest.TestCase):
    """测试GPU存储数据流"""
    
    def setUp(self):
        """设置测试环境"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA不可用")
        
        self.device = "cuda:0"
        self.backend = GPUStorageBackend(
            device=self.device,
            max_memory_gb=1.0,
            enable_nvlink_transfer=False
        )
        
        # 模拟vLLM的KV cache
        self.kvcaches = [
            (torch.randn(2, 256, 32, 128, dtype=torch.float16, device=self.device),
             torch.randn(2, 256, 32, 128, dtype=torch.float16, device=self.device))
        ]
        
        # 创建GPU连接器
        self.gpu_connector = VLLMPagedMemGPUConnectorV2(
            hidden_dim_size=128,
            num_layers=1,
            use_gpu=True,
            chunk_size=256,
            dtype=torch.float16,
            device=self.device
        )
    
    def test_direct_gpu_storage(self):
        """测试直接GPU存储"""
        # 创建测试数据（直接在GPU上）
        key = CacheEngineKey("test_format", "test_model", 1, 0, 12345)
        gpu_tensor = torch.randn(2, 1, 256, 128, dtype=torch.float16, device=self.device)
        memory_obj = MemoryObj(
            tensor=gpu_tensor,
            metadata=MemoryObjMetadata(fmt=MemoryFormat.KV_2LTD, ref_count=1)
        )
        
        # 验证数据在GPU上
        self.assertEqual(memory_obj.tensor.device.type, "cuda")
        
        # 存储到GPU存储后端
        self.backend.put(key, memory_obj)
        
        # 从GPU存储后端获取数据
        retrieved_obj = self.backend.get(key)
        
        # 验证获取的数据也在GPU上
        self.assertIsNotNone(retrieved_obj)
        self.assertEqual(retrieved_obj.tensor.device.type, "cuda")
        self.assertTrue(torch.allclose(retrieved_obj.tensor, gpu_tensor))
    
    def test_gpu_connector_to_gpu_storage(self):
        """测试GPU连接器与GPU存储的交互"""
        # 创建测试数据
        key = CacheEngineKey("test_format", "test_model", 1, 0, 12345)
        
        # 模拟从vLLM GPU cache提取数据
        start, end = 0, 256
        slot_mapping = torch.arange(256, dtype=torch.long, device=self.device)
        
        # 分配GPU内存对象
        shape = self.gpu_connector.get_shape(end - start)
        memory_obj = self.backend.allocate(shape, torch.float16)
        
        self.assertIsNotNone(memory_obj)
        self.assertEqual(memory_obj.tensor.device.type, "cuda")
        
        # 使用GPU连接器从vLLM提取数据到GPU存储
        self.gpu_connector.from_gpu(
            memory_obj, start, end,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping
        )
        
        # 存储到GPU存储后端
        self.backend.put(key, memory_obj)
        
        # 从GPU存储后端获取数据
        retrieved_obj = self.backend.get(key)
        
        # 验证数据仍在GPU上
        self.assertIsNotNone(retrieved_obj)
        self.assertEqual(retrieved_obj.tensor.device.type, "cuda")
        
        # 使用GPU连接器将数据写回vLLM
        self.gpu_connector.to_gpu(
            retrieved_obj, start, end,
            kvcaches=self.kvcaches,
            slot_mapping=slot_mapping
        )
    
    def test_no_cpu_transfer(self):
        """测试没有CPU传输"""
        # 创建测试数据
        key = CacheEngineKey("test_format", "test_model", 1, 0, 12345)
        gpu_tensor = torch.randn(2, 1, 256, 128, dtype=torch.float16, device=self.device)
        memory_obj = MemoryObj(
            tensor=gpu_tensor,
            metadata=MemoryObjMetadata(fmt=MemoryFormat.KV_2LTD, ref_count=1)
        )
        
        # 存储数据
        self.backend.put(key, memory_obj)
        
        # 获取数据
        retrieved_obj = self.backend.get(key)
        
        # 验证数据从未离开GPU
        self.assertEqual(retrieved_obj.tensor.device.type, "cuda")
        
        # 检查数据是否与原始数据相同
        self.assertTrue(torch.allclose(retrieved_obj.tensor, gpu_tensor))
    
    def test_batched_operations(self):
        """测试批量操作"""
        keys = []
        memory_objs = []
        
        # 创建多个测试数据
        for i in range(3):
            key = CacheEngineKey("test_format", "test_model", 1, 0, 12345 + i)
            gpu_tensor = torch.randn(2, 1, 256, 128, dtype=torch.float16, device=self.device)
            memory_obj = MemoryObj(
                tensor=gpu_tensor,
                metadata=MemoryObjMetadata(fmt=MemoryFormat.KV_2LTD, ref_count=1)
            )
            keys.append(key)
            memory_objs.append(memory_obj)
        
        # 批量存储
        for key, memory_obj in zip(keys, memory_objs):
            self.backend.put(key, memory_obj)
        
        # 批量获取
        retrieved_objs = self.backend.batched_get_blocking(keys)
        
        # 验证所有数据都在GPU上
        for retrieved_obj in retrieved_objs:
            self.assertIsNotNone(retrieved_obj)
            self.assertEqual(retrieved_obj.tensor.device.type, "cuda")
    
    def test_memory_allocation(self):
        """测试内存分配"""
        # 测试不同大小的内存分配
        shapes = [
            torch.Size([2, 1, 256, 128]),
            torch.Size([2, 1, 512, 128]),
            torch.Size([2, 1, 1024, 128])
        ]
        
        for shape in shapes:
            memory_obj = self.backend.allocate(shape, torch.float16)
            self.assertIsNotNone(memory_obj)
            self.assertEqual(memory_obj.tensor.device.type, "cuda")
            self.assertEqual(memory_obj.tensor.shape, shape)
            
            # 释放内存
            self.backend.memory_allocator.free(memory_obj)
    
    def test_memory_usage_tracking(self):
        """测试内存使用跟踪"""
        # 获取初始内存使用情况
        initial_allocated, total = self.backend.get_memory_usage()
        
        # 分配内存
        memory_obj = self.backend.allocate(torch.Size([2, 1, 256, 128]), torch.float16)
        
        # 检查内存使用增加
        allocated, total = self.backend.get_memory_usage()
        self.assertGreater(allocated, initial_allocated)
        
        # 释放内存
        self.backend.memory_allocator.free(memory_obj)
        
        # 检查内存使用减少
        final_allocated, total = self.backend.get_memory_usage()
        self.assertEqual(final_allocated, initial_allocated)


def run_dataflow_tests():
    """运行数据流测试"""
    # 设置测试环境
    os.environ['LMCACHE_USE_EXPERIMENTAL'] = 'True'
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestGPUStorageDataFlow))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_dataflow_tests()
    sys.exit(0 if success else 1) 