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
from typing import Optional
import asyncio
import socket
import threading
import time
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.distributed_server.abstract_server import (  # noqa: E501
    DistributedServerInterface,
)
from lmcache.v1.distributed_server.nccl_manager import get_nccl_manager
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.protocol import ClientMetaMessage, Constants, ServerMetaMessage
from lmcache.v1.storage_backend.storage_manager import StorageManager

logger = init_logger(__name__)

# TODO(Jiayi): Logic related to "put" and "exists" is not implemented yet.
# Need to think when it's needed.

# TODO(Jiayi): Need to make `handle_get` async as blocking get from disk
# will affect the performance. Another simpler and cleaner option is to make
# `handle_get` always blocking but make disk loading always async.

# TODO(Jiayi): Need to find a way to make the code more concise.
# For example, consider reusing code from remote cache server?


class NaiveDistributedServer(DistributedServerInterface):
    def __init__(
        self,
        storage_manager: StorageManager,
        lookup_server: LookupServerInterface,
        loop: asyncio.AbstractEventLoop,
        config: LMCacheEngineConfig,
    ):
        self.storage_manager = storage_manager
        self.lookup_server = lookup_server

        self.url = config.distributed_url
        assert self.url is not None
        host, port = self.url.split(":")
        self.host = host
        self.port = int(port)

        # 初始化rank信息
        self.rank = int(os.getenv("LMCACHE_RANK", "0"))
        self.world_size = int(os.getenv("LMCACHE_WORLD_SIZE", "1"))
        
        # 初始化NCCL管理器
        self.nccl_manager = get_nccl_manager()

        self.loop = loop
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.start(), self.loop)

        self.async_socket_lock = asyncio.Lock()

    async def handle_get(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Handle get from the peer.
        This function is blocking for now but should be non-blocking.
        """
        memory_obj = self.storage_manager.get(key)
        if memory_obj is not None:
            logger.debug(f"get kv cache success")
        return memory_obj

    def receive_all_client(
        self,
        meta: ServerMetaMessage,
        client_socket: socket.socket,
    ) -> Optional[MemoryObj]:
        received = 0
        n = meta.length
        logger.debug(f"received meta.length {meta.length}")

        # TODO(Jiayi): Format will be used once we support
        # compressed memory format
        if meta.dtype is None:
            logger.warning("meta.dtype is None, cannot allocate memory")
            return None
        memory_obj = self.storage_manager.allocate(
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
            num_bytes = client_socket.recv_into(view[received:], n - received)
            if num_bytes == 0:
                return None
            received += num_bytes

        return memory_obj

    async def issue_get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Perform get from the peer.
        This function can be blocking for now.
        """
        import time
        t_start = time.perf_counter()
        # `url` has the format host:port
        host_and_port = self.lookup_server.lookup(key)
        if host_and_port is None:
            return None
        host, port = host_and_port
        t_lookup = time.perf_counter()
        logger.info(f"lookup耗时: {t_lookup - t_start:.6f} 秒")

        # TODO(Jiayi): Cache the hot client sockets if possible.
        # For example, retrieving 100 chunks could create 100 the same
        # connection for 100 times.
        # However, too many live sockets could cause file descriptor exhaustion
        # (i.e., Too many open files).
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((host, port))
            logger.debug(f"Peer connection created at {host}:{port}")
            t_connect = time.perf_counter()
            logger.info(f"connect耗时: {t_connect - t_lookup:.6f} 秒")
            async with self.async_socket_lock:
                client_socket.sendall(
                    ClientMetaMessage(
                        Constants.CLIENT_GET,
                        key,
                        0,
                        MemoryFormat(1),
                        torch.float16,
                        torch.Size([0, 0, 0, 0]),
                        self.rank,  # 添加source_rank
                        0,  # target_rank将在后续确定
                    ).serialize()
                )
                data = client_socket.recv(ServerMetaMessage.packlength())
            meta = ServerMetaMessage.deserialize(data)
            t_deserialize = time.perf_counter()
            logger.info(f"deserialize耗时: {t_deserialize - t_connect:.6f} 秒")

            # 检查是否是GPU传输响应
            if meta.code == Constants.GPU_SUCCESS:
                logger.debug("接收到GPU传输响应，使用NCCL接收数据")
                # 使用NCCL接收GPU数据
                memory_obj = self.nccl_manager.recv_gpu_data(
                    meta.source_rank,  # 从源rank接收
                    meta.shape,
                    meta.dtype,
                    self.storage_manager  # 传入存储管理器
                )
                if memory_obj is not None:
                    logger.debug(f"NCCL接收数据成功，数据长度: {len(memory_obj.byte_array)}")
                t_end = time.perf_counter()
                logger.info(f"NCCL接收数据耗时: {t_end - t_deserialize:.6f} 秒")
                logger.info(f"issue_get整体耗时: {t_end - t_start:.6f} 秒")
                return memory_obj
            elif meta.code != Constants.SERVER_SUCCESS:
                logger.debug(f"meta.code {meta.code}")
                return None

            async with self.async_socket_lock:
                memory_obj = self.receive_all_client(meta, client_socket)
            logger.debug(f"get kv cache success")
            if memory_obj is not None:
                logger.debug(f"received data length is {len(memory_obj.byte_array)}")
            t_end = time.perf_counter()
            logger.info(f"issue_get整体耗时: {t_end - t_start:.6f} 秒")
            return memory_obj
        finally:
            try:
                client_socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            client_socket.close()

    async def receive_all_server(self, reader, n):
        data = bytearray()
        while len(data) < n:
            packet = await reader.read(n - len(data))
            if not packet:
                return None  # Client disconnected
            data.extend(packet)
        return data

    async def handle_client(self, reader, writer):
        """
        Handle the client.
        """
        addr = writer.get_extra_info("peername")
        logger.info(f"Connected by {addr}")
        try:
            while True:
                header = await self.receive_all_server(
                    reader, ClientMetaMessage.packlength()
                )
                if not header:
                    break
                meta = ClientMetaMessage.deserialize(header)

                match meta.command:
                    case Constants.CLIENT_GET:
                        t0 = time.perf_counter()
                        logger.debug(f"handle_get start:{meta.key}")
                        memory_obj = await self.handle_get(meta.key)

                        t1 = time.perf_counter()

                        if memory_obj is not None:
                            # 检查数据是否在GPU上且NCCL可用
                            if (memory_obj.tensor.device.type == "cuda" and 
                                self.nccl_manager.is_available() and
                                meta.source_rank != self.rank):  # 避免自己给自己发送
                                
                                logger.debug(f"数据在GPU上，使用NCCL发送到rank {meta.source_rank}")
                                
                                # 发送GPU_SUCCESS响应
                                writer.write(
                                    ServerMetaMessage(
                                        Constants.GPU_SUCCESS,
                                        len(memory_obj.byte_array),
                                        memory_obj.get_memory_format(),
                                        memory_obj.get_dtype(),
                                        memory_obj.get_shape(),
                                        self.rank,  # source_rank (当前服务器)
                                        meta.source_rank,  # target_rank (请求方)
                                    ).serialize()
                                )
                                await writer.drain()

                                t2 = time.perf_counter()
                                
                                # 使用NCCL发送GPU数据
                                success = self.nccl_manager.send_gpu_data(memory_obj, meta.source_rank)
                                if success:
                                    logger.debug(f"NCCL发送数据成功到rank {meta.source_rank}")
                                else:
                                    logger.error(f"NCCL发送数据失败到rank {meta.source_rank}")
                                
                                memory_obj.ref_count_down()

                                t3 = time.perf_counter()
                                
                            else:
                                # 使用传统socket传输
                                logger.debug(f"sended meta.length {len(memory_obj.byte_array)}")
                                writer.write(
                                    ServerMetaMessage(
                                        Constants.SERVER_SUCCESS,
                                        len(memory_obj.byte_array),
                                        memory_obj.get_memory_format(),
                                        memory_obj.get_dtype(),
                                        memory_obj.get_shape(),
                                        self.rank,  # source_rank
                                        meta.source_rank,  # target_rank
                                    ).serialize()
                                )
                                await writer.drain()

                                t2 = time.perf_counter()

                                writer.write(memory_obj.byte_array)
                                await writer.drain()
                                memory_obj.ref_count_down()

                                t3 = time.perf_counter()
                            logger.info(
                                    f"Time to get data: {t1 - t0}, "
                                    f"time to send meta: {t2 - t1}, "
                                    f"time to send data: {t3 - t2}"
                                )   
                        else:
                            writer.write(
                                ServerMetaMessage(
                                    Constants.SERVER_FAIL,
                                    0,
                                    MemoryFormat(1),
                                    torch.float16,
                                    torch.Size((0, 0, 0, 0)),
                                    self.rank,  # source_rank
                                    meta.source_rank,  # target_rank
                                ).serialize()
                            )
                            await writer.drain()
        except ConnectionResetError as e:
            logger.warning(f"Connection reset by peer for {addr}: {e}")
        except ConnectionAbortedError as e:
            logger.warning(f"Connection aborted for {addr}: {e}")
        except BrokenPipeError as e:
            logger.warning(f"Broken pipe for {addr}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error handling client {addr}: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.debug(f"Error closing writer for {addr}: {e}")

    async def start(self):
        """
        Start the server.
        """
        logger.debug(f"start server")
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addr = server.sockets[0].getsockname()
        logger.info(f"Server started at {addr}")

        async with server:
            await server.serve_forever()

    def close(self):
        """
        Close the server.
        """
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join()
        
        # 关闭NCCL管理器
        if hasattr(self, 'nccl_manager'):
            self.nccl_manager.close()
