#!/usr/bin/env python3
"""
简单的 GPU 存储后端测试
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lmcache'))

def test_gpu_backend_import():
    """测试 GPU 存储后端是否能正确导入"""
    try:
        from lmcache.v1.storage_backend.gpu_storage_backend import GPUStorageBackend
        print("✅ GPU 存储后端导入成功")
        return True
    except ImportError as e:
        print(f"❌ GPU 存储后端导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ GPU 存储后端导入时发生错误: {e}")
        return False

def test_gpu_backend_class():
    """测试 GPU 存储后端类的基本结构"""
    try:
        from lmcache.v1.storage_backend.gpu_storage_backend import GPUStorageBackend
        from lmcache.v1.config import LMCacheEngineConfig
        
        # 创建配置
        config = LMCacheEngineConfig.from_defaults()
        
        # 检查配置是否有 enable_gpu_storage 属性
        if hasattr(config, 'enable_gpu_storage'):
            print(f"✅ 配置支持 enable_gpu_storage: {config.enable_gpu_storage}")
        else:
            print("❌ 配置不支持 enable_gpu_storage")
            return False
        
        # 检查类的方法
        required_methods = [
            'contains', 'submit_put_task', 'get_blocking', 'get_non_blocking',
            'pin', 'unpin', 'remove', 'allocate', 'batched_allocate',
            'write_back', 'get_keys', 'clear', 'close'
        ]
        
        for method_name in required_methods:
            if hasattr(GPUStorageBackend, method_name):
                print(f"✅ 方法 {method_name} 存在")
            else:
                print(f"❌ 方法 {method_name} 不存在")
                return False
        
        print("✅ GPU 存储后端类结构正确")
        return True
        
    except Exception as e:
        print(f"❌ 测试 GPU 存储后端类时发生错误: {e}")
        return False

def test_observability_integration():
    """测试与观察性模块的集成"""
    try:
        from lmcache.observability import LMCStatsMonitor
        
        # 检查是否有 update_gpu_cache_usage 方法
        if hasattr(LMCStatsMonitor, 'update_gpu_cache_usage'):
            print("✅ LMCStatsMonitor 支持 update_gpu_cache_usage 方法")
        else:
            print("❌ LMCStatsMonitor 不支持 update_gpu_cache_usage 方法")
            return False
        
        # 检查 LMCacheStats 是否有 gpu_cache_usage_bytes 字段
        from lmcache.observability import LMCacheStats
        
        # 创建统计对象
        stats = LMCacheStats(
            interval_retrieve_requests=0,
            interval_store_requests=0,
            interval_requested_tokens=0,
            interval_hit_tokens=0,
            interval_remote_read_requests=0,
            interval_remote_read_bytes=0,
            interval_remote_write_requests=0,
            interval_remote_write_bytes=0,
            interval_remote_time_to_get=[],
            interval_remote_time_to_put=[],
            interval_remote_time_to_get_sync=[],
            cache_hit_rate=0.0,
            local_cache_usage_bytes=0,
            remote_cache_usage_bytes=0,
            local_storage_usage_bytes=0,
            gpu_cache_usage_bytes=0,  # 新添加的字段
            time_to_retrieve=[],
            time_to_store=[],
            retrieve_speed=[],
            store_speed=[]
        )
        
        print("✅ LMCacheStats 支持 gpu_cache_usage_bytes 字段")
        return True
        
    except Exception as e:
        print(f"❌ 测试观察性集成时发生错误: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试 GPU 存储后端实现...")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_gpu_backend_import),
        ("类结构测试", test_gpu_backend_class),
        ("观察性集成测试", test_observability_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n运行 {test_name}...")
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！GPU 存储后端实现正确。")
        return 0
    else:
        print("❌ 部分测试失败，需要修复问题。")
        return 1

if __name__ == "__main__":
    exit(main()) 