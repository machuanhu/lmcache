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
import socket
import threading
from typing import List, Optional, no_type_check
import asyncio

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.protocol import ClientMetaMessage, Constants, ServerMetaMessage
from lmcache.v1.storage_backend.connector.abstract_connector import (
    RemoteConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)


# TODO: performance optimization for this class, consider using C/C++/Rust
# for communication + deserialization
class LMCServerConnector(RemoteConnector):
    def __init__(
        self,
        host: str,
        port: int,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        self.async_socket_lock = asyncio.Lock()
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        
        # 初始化rank信息
        self.rank = int(os.getenv("LMCACHE_RANK", "0"))
        self.world_size = int(os.getenv("LMCACHE_WORLD_SIZE", "1"))

    # TODO(Jiayi): This should be an async function
    def receive_all(self, meta: ServerMetaMessage) -> Optional[MemoryObj]:
        received = 0
        n = meta.length

        # TODO(Jiayi): Format will be used once we support
        # compressed memory format
        memory_obj = self.local_cpu_backend.allocate(
            meta.shape,
            meta.dtype,
            meta.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        buffer = memory_obj.byte_array
        view = memoryview(buffer)

        while received < n:
            num_bytes = self.client_socket.recv_into(view[received:], n - received)
            if num_bytes == 0:
                return None
            received += num_bytes

        return memory_obj

    async def exists(self, key: CacheEngineKey) -> bool:
        # logger.debug("Call to exists()!")

        async with self.async_socket_lock:
            self.client_socket.sendall(
                ClientMetaMessage(
                    Constants.CLIENT_EXIST,
                    key,
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                    self.rank,  # source_rank
                    0,  # target_rank
                ).serialize()
            )

            response = self.client_socket.recv(ServerMetaMessage.packlength())

        return ServerMetaMessage.deserialize(response).code == Constants.SERVER_SUCCESS

    async def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        # logger.debug("Async call to put()!")

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        async with self.async_socket_lock:
            await self.loop.sock_sendall(
                self.client_socket,
                ClientMetaMessage(
                    Constants.CLIENT_PUT,
                    key,
                    len(kv_bytes),
                    memory_format,
                    kv_dtype,
                    kv_shape,
                    self.rank,  # source_rank
                    0,  # target_rank
                ).serialize(),
            )

            await self.loop.sock_sendall(self.client_socket, kv_bytes)

    # TODO(Jiayi): This should be an async function
    @_lmcache_nvtx_annotate
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # NOTE(Jiayi): Not using any await in the following as
        # we don't want to yield control to other tasks which could
        # sacrifice the performance loading to trade the performance of
        # saving
        async with self.async_socket_lock:
            self.client_socket.sendall(
                ClientMetaMessage(
                    Constants.CLIENT_GET,
                    key,
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                    self.rank,  # source_rank
                    0,  # target_rank
                ).serialize()
            )

            data = self.client_socket.recv(ServerMetaMessage.packlength())

        meta = ServerMetaMessage.deserialize(data)
        
        # 检查是否是GPU传输响应
        if meta.code == Constants.GPU_SUCCESS:
            logger.debug("接收到GPU传输响应，使用NCCL接收数据")
            # 这里需要集成NCCL管理器来接收GPU数据
            # 暂时返回None，实际实现需要NCCL管理器
            return None
        elif meta.code != Constants.SERVER_SUCCESS:
            return None

        async with self.async_socket_lock:
            memory_obj = self.receive_all(meta)

        return memory_obj

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        async with self.async_socket_lock:
            self.client_socket.close()
        logger.info("Closed the lmserver connection")
