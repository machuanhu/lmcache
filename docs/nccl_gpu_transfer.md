# NCCL GPU传输功能

## 概述

LMCache现在支持通过NCCL在GPU显存之间进行高速数据传输，这比传统的socket传输具有更低的延迟和更高的带宽。

## 功能特性

- **GPU显存间直接传输**: 数据直接在GPU显存之间传输，无需经过CPU内存
- **NVLink支持**: 利用NVLink高速互连进行数据传输
- **自动降级**: 如果NCCL不可用，自动降级到传统socket传输
- **Rank管理**: 通过环境变量配置每个实例的rank号

## 环境变量配置

每个LMCache实例需要设置以下环境变量：

```bash
# 当前实例的rank号（从0开始）
export LMCACHE_RANK=0

# 总实例数量
export LMCACHE_WORLD_SIZE=2

# 当前实例使用的GPU设备
export CUDA_VISIBLE_DEVICES=0
```

## 工作流程

### 1. 请求方（客户端）

1. 发送带rank信息的get请求
2. 接收ServerMetaMessage响应
3. 如果响应码是`GPU_SUCCESS`，使用NCCL接收数据
4. 如果响应码是`SERVER_SUCCESS`，使用传统socket接收数据

### 2. 接收方（服务器）

1. 接收带rank信息的get请求
2. 从GPU存储后端获取数据
3. 如果数据在GPU上且NCCL可用，发送`GPU_SUCCESS`响应并使用NCCL发送数据
4. 否则发送`SERVER_SUCCESS`响应并使用传统socket发送数据

## 代码示例

### 启动服务器（Rank 0）

```bash
export LMCACHE_RANK=0
export LMCACHE_WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0

python -m lmcache.v1.server --host 0.0.0.0 --port 8080
```

### 启动客户端（Rank 1）

```bash
export LMCACHE_RANK=1
export LMCACHE_WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=1

python -c "
from lmcache.v1.distributed_server.nccl_manager import get_nccl_manager
import torch

# 初始化NCCL管理器
nccl_manager = get_nccl_manager()

# 接收GPU数据
memory_obj = nccl_manager.recv_gpu_data(
    source_rank=0,
    shape=torch.Size([1000, 1000, 0, 0]),
    dtype=torch.float32
)

print(f'接收到的数据形状: {memory_obj.tensor.shape}')
"
```

## 协议变更

### ClientMetaMessage

新增字段：
- `source_rank`: 请求方的rank号
- `target_rank`: 接收方的rank号

### ServerMetaMessage

新增字段：
- `source_rank`: 发送方的rank号
- `target_rank`: 接收方的rank号

### 状态码

新增状态码：
- `GPU_SUCCESS = 201`: GPU传输成功

## 性能优势

1. **低延迟**: 数据直接在GPU显存间传输，无需经过CPU内存
2. **高带宽**: 利用NVLink的高带宽特性
3. **零拷贝**: 避免不必要的数据拷贝操作
4. **并行传输**: 支持多个GPU之间的并行数据传输

## 注意事项

1. **NCCL初始化**: 确保所有实例的NCCL进程组正确初始化
2. **GPU设备**: 确保每个实例使用不同的GPU设备
3. **网络配置**: 对于多机部署，需要配置正确的网络环境变量
4. **错误处理**: NCCL传输失败时会自动降级到socket传输

## 故障排除

### NCCL初始化失败

```bash
# 检查环境变量
echo $LMCACHE_RANK
echo $LMCACHE_WORLD_SIZE
echo $CUDA_VISIBLE_DEVICES

# 检查NCCL环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果InfiniBand有问题
```

### GPU传输失败

1. 检查GPU设备是否正确配置
2. 确认NVLink连接正常
3. 查看日志中的错误信息
4. 系统会自动降级到socket传输

## 测试

运行测试脚本：

```bash
# 终端1
export LMCACHE_RANK=0
export LMCACHE_WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0
python examples/nccl_gpu_transfer_example.py server

# 终端2
export LMCACHE_RANK=1
export LMCACHE_WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=1
python examples/nccl_gpu_transfer_example.py client
``` 