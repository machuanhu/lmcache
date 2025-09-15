# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
import os
import threading
from typing import Optional
import time

# Third Party
import torch
import torch.distributed as dist

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryObj, GPUMemoryAllocator, MemoryFormat

logger = init_logger(__name__)


class NCCLManager:
    """
    NCCL管理器，用于处理GPU显存间的数据传输
    """
    
    def __init__(self, 
                 gpu_buffer_size_gb: int = 8,  # 默认8GB GPU缓冲区
                 dtype: torch.dtype = torch.float16,
                 device: Optional[str] = "cuda"):
        self.initialized = False
        self.rank = int(os.getenv("LMCACHE_RANK", "0"))
        self.world_size = int(os.getenv("LMCACHE_WORLD_SIZE", "1"))
        self.device = device

        # 自动设置MASTER_ADDR和MASTER_PORT的默认值
        if os.getenv("MASTER_ADDR") is None:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if os.getenv("MASTER_PORT") is None:
            os.environ["MASTER_PORT"] = "6190"
        
        self.init_lock = threading.Lock()
        
        # 初始化GPU内存分配器
        self._init_gpu_buffer_allocator(gpu_buffer_size_gb, dtype)
        
        self._init_nccl()
        
        logger.info(f"NCCL管理器初始化完成 - Rank: {self.rank}, World Size: {self.world_size}, Device: {self.device}")
    
    def _init_gpu_buffer_allocator(self, gpu_buffer_size_gb: int, dtype: torch.dtype):
        """
        初始化GPU内存分配器
        
        Args:
            gpu_buffer_size_mb: GPU缓冲区大小（MB）
            dtype: 数据类型
        """
        try:
            # 计算GPU缓冲区大小（字节）
            gpu_buffer_size = gpu_buffer_size_gb * 1024 * 1024 * 1024  # MB to bytes
            
            # 创建GPU内存分配器
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                size=gpu_buffer_size,
                device=self.device
            )
            
            self.dtype = dtype
            self.gpu_buffer_size = gpu_buffer_size
            
            logger.info(f"GPU内存分配器初始化成功 - 大小: {gpu_buffer_size_gb}GB, 设备: {self.device}, 数据类型: {dtype}")
            
        except Exception as e:
            logger.error(f"GPU内存分配器初始化失败: {e}")
            self.gpu_buffer_allocator = None
            self.dtype = dtype
            self.gpu_buffer_size = 0
    
    def _init_nccl(self):
        """初始化NCCL后端"""
        with self.init_lock:
            if not self.initialized:
                try:
                    # 初始化进程组
                    dist.init_process_group(
                        backend='nccl',
                        rank=self.rank,
                        world_size=self.world_size,
                        init_method='env://'
                    )
                    self.initialized = True
                    logger.info(f"NCCL进程组初始化成功 - Rank {self.rank}")
                except Exception as e:
                    logger.error(f"NCCL初始化失败: {e}")
                    # 检查是否已经有默认通信组，如果有则销毁
                    if dist.is_initialized():
                        try:
                            dist.destroy_process_group()
                            logger.info("已销毁之前创建的通信组，准备重新初始化。")
                        except Exception as destroy_e:
                            logger.error(f"销毁通信组失败: {destroy_e}")
                    # 再次尝试初始化
                    try:
                        dist.init_process_group(
                            backend='nccl',
                            rank=self.rank,
                            world_size=self.world_size,
                            init_method='env://'
                        )
                        self.initialized = True

                        # NCCL预热传输：每个rank依次与其他所有rank进行一次点对点通信，确保所有通信通道建立
                        try:
                            dummy_tensor = torch.zeros(1, device=f"cuda:0")
                            for peer_rank in range(self.world_size):
                                if peer_rank == self.rank:
                                    continue
                                if self.rank < peer_rank:
                                    dist.send(dummy_tensor, dst=peer_rank)
                                    logger.info(f"NCCL预热：rank {self.rank} 已向 rank {peer_rank} 发送预热张量")
                                    dist.recv(dummy_tensor, src=peer_rank)
                                    logger.info(f"NCCL预热：rank {self.rank} 已从 rank {peer_rank} 接收预热张量")
                                else:
                                    dist.recv(dummy_tensor, src=peer_rank)
                                    logger.info(f"NCCL预热：rank {self.rank} 已从 rank {peer_rank} 接收预热张量")
                                    dist.send(dummy_tensor, dst=peer_rank)
                                    logger.info(f"NCCL预热：rank {self.rank} 已向 rank {peer_rank} 发送预热张量")
                            logger.info(f"NCCL进程组重新初始化成功 - Rank {self.rank}")
                        except Exception as warmup_e:
                            logger.warning(f"NCCL预热传输失败: {warmup_e}")
                    except Exception as e2:
                        logger.error(f"NCCL重新初始化仍然失败: {e2}")
                        # 如果NCCL初始化失败，仍然可以运行，但GPU传输功能不可用
                        self.initialized = False
    
    def is_available(self) -> bool:
        """检查NCCL是否可用"""
        return self.initialized and dist.is_initialized()
    
    def allocate_gpu_buffer(self, shape: torch.Size, fmt: MemoryFormat = MemoryFormat.KV_2LTD) -> Optional[MemoryObj]:
        """
        使用GPU内存分配器分配GPU缓冲区
        
        Args:
            shape: 张量形状
            fmt: 内存格式
            
        Returns:
            Optional[MemoryObj]: 分配的内存对象，失败时返回None
        """
        if self.gpu_buffer_allocator is None:
            logger.warning("GPU内存分配器未初始化，无法分配GPU缓冲区")
            return None
        
        try:
            memory_obj = self.gpu_buffer_allocator.allocate(
                shape=shape,
                dtype=self.dtype,
                fmt=fmt
            )
            
            if memory_obj is not None:
                memory_obj.ref_count_up()
                logger.debug(f"成功分配GPU缓冲区 - 形状: {shape}, 格式: {fmt}")
            else:
                logger.warning(f"GPU缓冲区分配失败 - 形状: {shape}, 格式: {fmt}")
            
            return memory_obj
            
        except Exception as e:
            logger.error(f"GPU缓冲区分配异常: {e}")
            return None
    
    def free_gpu_buffer(self, memory_obj: MemoryObj):
        """
        释放GPU缓冲区
        
        Args:
            memory_obj: 要释放的内存对象
        """
        if self.gpu_buffer_allocator is None:
            logger.warning("GPU内存分配器未初始化，无法释放GPU缓冲区")
            return
        
        try:
            # 使用引用计数机制释放内存
            memory_obj.ref_count_down()
            logger.debug("GPU缓冲区已释放")
            
        except Exception as e:
            logger.error(f"GPU缓冲区释放异常: {e}")
    
    def send_gpu_data(self, memory_obj: MemoryObj, target_rank: int) -> bool:
        """
        通过NCCL发送GPU数据到目标rank
        
        Args:
            memory_obj: 要发送的内存对象
            target_rank: 目标rank号
            
        Returns:
            bool: 发送是否成功
        """
        if not self.is_available():
            logger.warning("NCCL不可用，无法发送GPU数据")
            return False
        
        if target_rank >= self.world_size:
            logger.error(f"目标rank {target_rank} 超出world_size {self.world_size}")
            return False
        
        try:
            # 确保数据在GPU上
            if memory_obj.tensor.device.type != "cuda":
                logger.warning("数据不在GPU上，移动到GPU")
                memory_obj.tensor = memory_obj.tensor.cuda(self.device_id)
            
            # 使用NCCL发送数据
            dist.send(memory_obj.tensor, dst=target_rank)
            logger.debug(f"成功发送数据到rank {target_rank}")
            return True
            
        except Exception as e:
            logger.error(f"发送数据到rank {target_rank} 失败: {e}")
            return False
    
    def recv_gpu_data(self, source_rank: int, shape: torch.Size, dtype: torch.dtype, storage_manager=None) -> Optional[MemoryObj]:
        """
        通过NCCL从源rank接收GPU数据
        
        Args:
            source_rank: 源rank号
            shape: 数据形状
            dtype: 数据类型
            storage_manager: 存储管理器，用于分配内存
            
        Returns:
            Optional[MemoryObj]: 接收到的内存对象，失败时返回None
        """
        if not self.is_available():
            logger.warning("NCCL不可用，无法接收GPU数据")
            return None
        
        if source_rank >= self.world_size:
            logger.error(f"源rank {source_rank} 超出world_size {self.world_size}")
            return None
        
        try:
            # 优先使用GPU内存分配器分配内存（如果dtype匹配）
            memory_obj = None
            memory_obj = self.allocate_gpu_buffer(shape, MemoryFormat.KV_2LTD)
            if memory_obj is not None:
                logger.debug("使用GPU内存分配器分配内存")
            
            if memory_obj is None:
                # 如果GPU内存分配器分配失败或不匹配dtype，回退到存储管理器
                if storage_manager is not None:
                    memory_obj = storage_manager.allocate(
                        shape,
                        dtype,
                        MemoryFormat.KV_2LTD,  # 默认格式
                    )
                    t_allocate = time.perf_counter()
                    logger.info(f"存储管理器分配内存耗时: {t_allocate - time.perf_counter():.6f} 秒")
                    if memory_obj is None:
                        logger.error("存储管理器分配内存失败")
                        return None
                else:
                    # 最后回退到直接分配GPU内存
                    from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
                    tensor = torch.empty(shape, dtype=dtype, device=self.device)
                    metadata = MemoryObjMetadata(
                        shape=shape,
                        dtype=dtype,
                        address=0,
                        phy_size=tensor.numel() * dtype.itemsize,
                        ref_count=1,
                        is_pin=False,
                        fmt=MemoryFormat.KV_2LTD
                    )
                    memory_obj = TensorMemoryObj(
                        raw_data=tensor.view(torch.uint8).flatten(),
                        metadata=metadata,
                        parent_allocator=None
                    )
                    logger.info("使用直接GPU内存分配")
            
            # 确保内存对象在GPU上
            if memory_obj.tensor.device.type != "cuda":
                logger.warning("分配的内存对象不在GPU上，移动到GPU")
                memory_obj.tensor = memory_obj.tensor.cuda(self.device_id)
            
            # 使用NCCL接收数据到内存对象
            t_start = time.perf_counter()
            dist.recv(memory_obj.tensor, src=source_rank)
            t_recv = time.perf_counter()
            logger.info(f"NCCL接收数据耗时: {t_recv - t_start:.6f} 秒")
            logger.debug(f"成功从rank {source_rank} 接收数据")
            return memory_obj
            
        except Exception as e:
            logger.error(f"从rank {source_rank} 接收数据失败: {e}")
            return None
    
    def get_gpu_buffer_info(self) -> dict:
        """
        获取GPU缓冲区信息
        
        Returns:
            dict: GPU缓冲区信息
        """
        if self.gpu_buffer_allocator is None:
            return {
                "initialized": False,
                "buffer_size": 0,
                "dtype": str(self.dtype) if hasattr(self, 'dtype') else "unknown"
            }
        
        try:
            # 检查内存分配器状态
            memcheck_result = self.gpu_buffer_allocator.memcheck()
            return {
                "initialized": True,
                "buffer_size": self.gpu_buffer_size,
                "buffer_size_mb": self.gpu_buffer_size / (1024 * 1024),
                "dtype": str(self.dtype),
                "device": self.device,
                "memcheck_passed": memcheck_result
            }
        except Exception as e:
            logger.error(f"获取GPU缓冲区信息失败: {e}")
            return {
                "initialized": True,
                "buffer_size": self.gpu_buffer_size,
                "buffer_size_mb": self.gpu_buffer_size / (1024 * 1024),
                "dtype": str(self.dtype),
                "device": self.device,
                "memcheck_passed": False,
                "error": str(e)
            }
    
    def close(self):
        """关闭NCCL管理器"""
        if self.initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.initialized = False
            logger.info("NCCL管理器已关闭")


# 全局NCCL管理器实例
_nccl_manager: Optional[NCCLManager] = None


def get_nccl_manager(gpu_buffer_size_gb: int = 8, 
                    dtype: torch.dtype = torch.float16,
                    device: Optional[str] = "cuda") -> NCCLManager:
    """
    获取全局NCCL管理器实例
    
    Args:
        gpu_buffer_size_gb: GPU缓冲区大小（GB），默认8GB
        dtype: 数据类型，默认float16
        device: 设备，默认自动检测
        
    Returns:
        NCCLManager: NCCL管理器实例
    """
    global _nccl_manager
    if _nccl_manager is None:
        _nccl_manager = NCCLManager(
            gpu_buffer_size_gb=gpu_buffer_size_gb,
            dtype=dtype,
            device=device
        )
    return _nccl_manager 