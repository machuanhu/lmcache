# LMCache 配置类对比：实验性 vs 传统

## 概述

LMCache 支持两种配置类：
- **实验性配置类** (`LMCACHE_USE_EXPERIMENTAL=True`)：`lmcache.v1.config.LMCacheEngineConfig`
- **传统配置类** (`LMCACHE_USE_EXPERIMENTAL=False`)：`lmcache.config.LMCacheEngineConfig`

## 主要差异对比

### 1. 配置字段支持

| 配置字段 | 实验性配置类 | 传统配置类 | 说明 |
|---------|-------------|-----------|------|
| `chunk_size` | ✅ | ✅ | KV cache chunk大小 |
| `local_cpu` | ✅ | ❌ | 是否启用CPU存储 |
| `max_local_cpu_size` | ✅ | ✅ | CPU存储大小(GB) |
| `local_disk` | ✅ | ❌ | 本地磁盘路径 |
| `max_local_disk_size` | ✅ | ❌ | 本地磁盘大小(GB) |
| `remote_url` | ✅ | ✅ | 远程存储URL |
| `remote_serde` | ✅ | ✅ | 远程序列化格式 |
| `enable_gpu_storage` | ✅ | ❌ | **GPU存储支持** |
| `gpu_memory_gb` | ✅ | ❌ | **GPU内存大小** |
| `enable_p2p` | ✅ | ❌ | P2P共享 |
| `lookup_url` | ✅ | ❌ | 查找服务器URL |
| `distributed_url` | ✅ | ❌ | 分布式服务器URL |
| `enable_blending` | ✅ | ✅ | KV混合 |
| `enable_nixl` | ✅ | ❌ | Nixl后端 |
| `external_lookup_client` | ✅ | ❌ | 外部查找客户端 |
| `weka_path` | ✅ | ❌ | WekaFS路径 |
| `gds_path` | ✅ | ❌ | GDS路径 |
| `cufile_buffer_size` | ✅ | ❌ | CuFile缓冲区大小 |

### 2. 功能支持差异

#### 实验性配置类支持的功能：
- ✅ **GPU存储后端** - 直接在GPU显存中存储KV cache
- ✅ **P2P共享** - 实例间对等共享
- ✅ **Nixl后端** - 高性能分布式存储
- ✅ **WekaFS后端** - 高性能分布式文件系统
- ✅ **GDS后端** - GPU Direct Storage
- ✅ **外部查找客户端** - 支持Mooncake等外部存储
- ✅ **更丰富的配置选项** - 支持更多高级功能

#### 传统配置类支持的功能：
- ❌ GPU存储后端
- ❌ P2P共享
- ❌ Nixl后端
- ❌ WekaFS后端
- ❌ GDS后端
- ❌ 外部查找客户端
- ✅ 基本的CPU/磁盘/远程存储

### 3. 配置读取方式

#### 实验性配置类：
```python
# 支持配置文件和环境变量
config = LMCacheEngineConfig.from_file("config.yaml")
config = LMCacheEngineConfig.from_env()
config = LMCacheEngineConfig.from_defaults()
```

#### 传统配置类：
```python
# 支持配置文件和环境变量
config = LMCacheEngineConfig.from_file("config.yaml")
config = LMCacheEngineConfig.from_env()
config = LMCacheEngineConfig.from_legacy()
```

### 4. 默认值差异

| 配置项 | 实验性配置类默认值 | 传统配置类默认值 |
|--------|------------------|-----------------|
| `chunk_size` | 256 | 256 |
| `local_cpu` | True | N/A |
| `max_local_cpu_size` | 5.0 GB | 5 GB |
| `enable_gpu_storage` | False | N/A |
| `gpu_memory_gb` | 8.0 | N/A |
| `enable_p2p` | False | N/A |
| `enable_blending` | False | False |

### 5. 环境变量映射

#### 实验性配置类环境变量：
```bash
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5.0
export LMCACHE_ENABLE_GPU_STORAGE=True
export LMCACHE_GPU_MEMORY_GB=8.0
export LMCACHE_ENABLE_P2P=True
export LMCACHE_LOOKUP_URL="localhost:6379"
export LMCACHE_DISTRIBUTED_URL="localhost:8200"
```

#### 传统配置类环境变量：
```bash
export LMCACHE_USE_EXPERIMENTAL=False
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_MAX_LOCAL_CACHE_SIZE=5
export LMCACHE_REMOTE_URL="redis://localhost:6379"
export LMCACHE_REMOTE_SERDE="torch"
```

## 使用建议

### 1. 何时使用实验性配置类：
- 需要使用GPU存储功能
- 需要P2P共享功能
- 需要使用Nixl、WekaFS、GDS等高级后端
- 需要外部查找客户端
- 需要最新的功能和性能优化

### 2. 何时使用传统配置类：
- 只需要基本的CPU/磁盘/远程存储
- 兼容性要求较高
- 不需要新功能

### 3. 迁移建议：
- **强烈建议**使用实验性配置类
- 传统配置类已被标记为"deprecated"
- 未来版本可能会移除传统配置类

## 配置示例

### 实验性配置类示例：
```yaml
# config.yaml
chunk_size: 256
local_cpu: false
max_local_cpu_size: 8.0
enable_gpu_storage: true
gpu_memory_gb: 5.0
enable_p2p: true
lookup_url: "localhost:6379"
distributed_url: "localhost:8201"
external_lookup_client: null
```

### 传统配置类示例：
```yaml
# config.yaml
chunk_size: 256
local_device: "cpu"
max_local_cache_size: 5
remote_url: "redis://localhost:6379"
remote_serde: "torch"
```

## 总结

实验性配置类提供了更丰富的功能和更好的性能，是LMCache的未来发展方向。建议所有新项目都使用实验性配置类，以获得最佳的功能支持和性能表现。 