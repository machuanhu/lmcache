#!/usr/bin/env python3
"""
测试配置类版本的脚本
"""

import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

def test_config_class():
    """测试使用的是哪个配置类"""
    print("=== 测试配置类版本 ===")
    
    # 检查环境变量
    use_experimental = os.getenv("LMCACHE_USE_EXPERIMENTAL", "True")
    print(f"LMCACHE_USE_EXPERIMENTAL: {use_experimental}")
    
    # 根据环境变量导入不同的配置类
    if use_experimental.lower() in ("true", "1", "yes"):
        print("使用新版配置类 (lmcache.v1.config.LMCacheEngineConfig)")
        try:
            from lmcache.v1.config import LMCacheEngineConfig
            config_class = "V1"
        except ImportError as e:
            print(f"导入新版配置类失败: {e}")
            config_class = "ERROR"
    else:
        print("使用旧版配置类 (lmcache.config.LMCacheEngineConfig)")
        try:
            from lmcache.config import LMCacheEngineConfig
            config_class = "LEGACY"
        except ImportError as e:
            print(f"导入旧版配置类失败: {e}")
            config_class = "ERROR"
    
    if config_class != "ERROR":
        # 检查配置类是否有GPU存储相关字段
        if hasattr(LMCacheEngineConfig, 'enable_gpu_storage'):
            print("✅ 配置类支持 enable_gpu_storage 字段")
        else:
            print("❌ 配置类不支持 enable_gpu_storage 字段")
            
        if hasattr(LMCacheEngineConfig, 'gpu_memory_gb'):
            print("✅ 配置类支持 gpu_memory_gb 字段")
        else:
            print("❌ 配置类不支持 gpu_memory_gb 字段")
            
        if hasattr(LMCacheEngineConfig, 'max_local_cpu_size'):
            print("✅ 配置类支持 max_local_cpu_size 字段")
        else:
            print("❌ 配置类不支持 max_local_cpu_size 字段")
    
    print(f"\n配置类版本: {config_class}")
    
    # 测试配置文件读取
    config_file = os.getenv("LMCACHE_CONFIG_FILE")
    if config_file and os.path.exists(config_file):
        print(f"\n测试读取配置文件: {config_file}")
        try:
            config = LMCacheEngineConfig.from_file(config_file)
            print(f"配置读取成功")
            
            # 尝试访问字段
            if hasattr(config, 'enable_gpu_storage'):
                print(f"enable_gpu_storage: {config.enable_gpu_storage}")
            else:
                print("enable_gpu_storage: 字段不存在")
                
            if hasattr(config, 'gpu_memory_gb'):
                print(f"gpu_memory_gb: {config.gpu_memory_gb}")
            else:
                print("gpu_memory_gb: 字段不存在")
                
            if hasattr(config, 'max_local_cpu_size'):
                print(f"max_local_cpu_size: {config.max_local_cpu_size}")
            else:
                print("max_local_cpu_size: 字段不存在")
                
        except Exception as e:
            print(f"配置文件读取失败: {e}")
    else:
        print(f"\n配置文件不存在或未设置: {config_file}")


def main():
    print("LMCache配置类测试")
    print("=" * 50)
    
    test_config_class()
    
    print("\n=== 总结 ===")
    print("如果使用的是旧版配置类，它不支持 enable_gpu_storage 和 gpu_memory_gb 字段")
    print("只有新版配置类 (LMCACHE_USE_EXPERIMENTAL=True) 才支持这些字段")
    print("这就是为什么 max_local_cpu_size 能被识别，但 GPU 存储配置不能的原因")


if __name__ == "__main__":
    main() 