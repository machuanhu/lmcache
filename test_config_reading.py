#!/usr/bin/env python3
"""
测试配置文件读取的脚本
"""

import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

from lmcache.v1.config import LMCacheEngineConfig


def test_config_file_reading():
    """测试配置文件读取"""
    print("=== 测试配置文件读取 ===")
    
    # 测试读取配置文件
    config_file = "lmcache_config2.yaml"
    if os.path.exists(config_file):
        print(f"读取配置文件: {config_file}")
        config = LMCacheEngineConfig.from_file(config_file)
        
        print(f"enable_gpu_storage: {config.enable_gpu_storage}")
        print(f"gpu_memory_gb: {config.gpu_memory_gb}")
        print(f"chunk_size: {config.chunk_size}")
        print(f"local_cpu: {config.local_cpu}")
        print(f"max_local_cpu_size: {config.max_local_cpu_size}")
        print(f"enable_p2p: {config.enable_p2p}")
        print(f"lookup_url: {config.lookup_url}")
        print(f"distributed_url: {config.distributed_url}")
    else:
        print(f"配置文件 {config_file} 不存在")


def test_env_config_reading():
    """测试环境变量配置读取"""
    print("\n=== 测试环境变量配置读取 ===")
    
    # 设置环境变量
    os.environ["LMCACHE_ENABLE_GPU_STORAGE"] = "true"
    os.environ["LMCACHE_GPU_MEMORY_GB"] = "10.0"
    
    config = LMCacheEngineConfig.from_env()
    
    print(f"enable_gpu_storage: {config.enable_gpu_storage}")
    print(f"gpu_memory_gb: {config.gpu_memory_gb}")


def test_default_config():
    """测试默认配置"""
    print("\n=== 测试默认配置 ===")
    
    config = LMCacheEngineConfig.from_defaults()
    
    print(f"enable_gpu_storage: {config.enable_gpu_storage}")
    print(f"gpu_memory_gb: {config.gpu_memory_gb}")


def main():
    print("LMCache配置读取测试")
    print("=" * 50)
    
    test_config_file_reading()
    test_env_config_reading()
    test_default_config()
    
    print("\n=== 总结 ===")
    print("1. 配置文件中的enable_gpu_storage和gpu_memory_gb现在应该能正确读取")
    print("2. 环境变量LMCACHE_ENABLE_GPU_STORAGE和LMCACHE_GPU_MEMORY_GB也能正确读取")
    print("3. 如果仍然显示False和None，请检查配置文件格式和环境变量设置")


if __name__ == "__main__":
    main() 