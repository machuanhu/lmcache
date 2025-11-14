# Standard
from dataclasses import dataclass
from typing import Optional
import argparse
import asyncio
import json
import logging
import time

# Third Party
from utils import AsyncLoopWrapper, init_logger
import openai
import pandas as pd

logger = init_logger(__name__, logging.INFO)


@dataclass
class WorkloadConfig:
    # Max number of users in the system concurrently
    num_users: int

    # Length of shared system prompt
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Length of the answer in one round
    answer_len: int

    # Number of rounds in the conversation
    num_rounds: int

    # Overall QPS
    qps: int

    # Model name
    model: str

    # Whether to include user id in request header
    enable_user_id: bool

    # Number of conversations to batch together (new parameter)
    batch_conversations: int = 1


@dataclass
class UserConfig:
    # User id
    user_id: int

    # System prompt length
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Answer length
    answer_len: int

    # Gap between two requests
    gap_between_requests: int

    # Num rounds
    num_rounds: int

    # Whether to include user id in request header
    enable_user_id: bool

    # Number of conversations to batch together
    batch_conversations: int

    @staticmethod
    def new_user_config(user_id: int, workload_config: WorkloadConfig) -> "UserConfig":
        return UserConfig(
            user_id=user_id,
            system_prompt_len=workload_config.system_prompt_len,
            user_info_len=workload_config.user_info_len,
            answer_len=workload_config.answer_len,
            gap_between_requests=workload_config.num_users / workload_config.qps,
            num_rounds=workload_config.num_rounds,
            enable_user_id=workload_config.enable_user_id,
            batch_conversations=workload_config.batch_conversations,
        )


class ChatHistory:
    def __init__(
        self,
    ):
        self.history = []

    def on_user_query(self, query: str):
        if len(self.history) == 0:
            self.history.append({"role": "user", "content": query})
        else:
            assert self.history[-1]["role"] == "assistant", "Expect system response"
            self.history.append({"role": "user", "content": query})

    def on_system_response(self, response: str):
        assert len(self.history) > 0, "Expect user query"
        assert self.history[-1]["role"] == "user", "Expect user query"
        self.history.append({"role": "assistant", "content": response})

    def get_messages_for_openai(self):
        return self.history

    def __len__(self):
        return len(self.history)


@dataclass
class Response:
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float


class RequestExecutor:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()
        self.request_history = []

    async def _async_launch_request(self, messages, max_tokens, extra_headers=None):
        start_time = time.time()
        first_token_time = None
        words = ""

        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0,
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
            extra_headers=extra_headers,
        )

        async for tok in response:
            if not tok.choices:
                continue
            chunk_message = tok.choices[0].delta.content
            if chunk_message is not None:
                if first_token_time is None and chunk_message != "":
                    first_token_time = time.time()
                words += chunk_message
        tokens_out = tok.usage.completion_tokens
        tokens_prefill = tok.usage.prompt_tokens

        return Response(
            body=words,
            ttft=first_token_time - start_time,
            generation_time=time.time() - first_token_time,
            prompt_tokens=tokens_prefill,
            generation_tokens=tokens_out,
            launch_time=start_time,
            finish_time=time.time(),
        )

    def launch_request(
        self,
        chat_history: ChatHistory,
        max_tokens: int,
        finish_callback,
        extra_headers=None,
    ):
        """
        finish_callback: Callable[[Response], None]
        """
        messages = chat_history.get_messages_for_openai()
        real_callback = lambda x: finish_callback(x.result())
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(messages, max_tokens, extra_headers),
            self.loop,
        )
        future.add_done_callback(real_callback)


class UserSession:
    def __init__(self, user_config: UserConfig, use_sharegpt=False, sharegpt_data=None):
        self.user_config = user_config
        self.last_request_time = None
        self.chat_history = ChatHistory()
        self.question_id = 0
        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self.sharegpt_data = sharegpt_data
            if self.sharegpt_data["num_round"] % 2 == 0:
                self.start_with_gpt = False
            else:
                self.start_with_gpt = True

        self.has_unfinished_request = False
        self.last_unfinished_log = 0

        self.prompt_lengths = []
        self.generation_lengths = []
        self.ttfts = []
        self.generation_times = []
        self.launch_times = []
        self.finish_times = []

        self.finished = False
        self.skipped_requests = 0  # 被跳过的请求数量（因为 token 限制）
        self.data_insufficient = 0  # 因为数据不足而提前结束的次数

    @staticmethod
    def _estimate_tokens(messages):
        """估算 messages 的 token 数量
        使用更准确的估算方法：
        - 对于英文：1 token ≈ 4 字符
        - 对于中文：1 token ≈ 1.5 字符
        - 使用更保守的估算（字符数 / 2.5）以确保不会低估
        """
        total_chars = 0
        for msg in messages:
            # 每个消息有 role 和 content，加上一些格式开销
            content = msg.get("content", "")
            total_chars += len(content) + 15  # 15 字符用于 role、格式开销和特殊token
        
        # 保守估算：使用字符数 / 2.5，这样对于混合文本（中英文）都能有较好的估算
        # 这个比例比实际略高，但更安全，避免低估
        estimated_tokens = int(total_chars / 2.5)
        return estimated_tokens

    def _update_result(self, response: Response):
        self.prompt_lengths.append(response.prompt_tokens)
        self.generation_lengths.append(response.generation_tokens)
        self.ttfts.append(response.ttft)
        self.generation_times.append(response.generation_time)
        self.launch_times.append(response.launch_time)
        self.finish_times.append(response.finish_time)

    def _build_system_prompt(self):
        def gen_dummy_text(length):
            return " ".join(["hi"] * length)

        dummy_text_sys = gen_dummy_text(self.user_config.system_prompt_len)
        dummy_text_user = gen_dummy_text(self.user_config.user_info_len)
        system_prompt = (
            f"Hi, here's some system prompt: {dummy_text_sys}."
            + f"For user {self.user_config.user_id}, "
            + f"here are some other context: {dummy_text_user}."
        )
        return system_prompt

    def _build_new_question(self):
        return (
            f"Here's question #{self.question_id + 1}: can you tell me "
            + "a new long story with a happy ending?"
        )

    def _build_batched_sharegpt_prompt(self):
        """构建包含多个conversation的合并prompt
        每个conversation包含一个完整的对话对（问题+回答），即2个turn
        """
        if not self.use_sharegpt:
            return self._build_new_question()
        
        batch_size = self.user_config.batch_conversations
        # 每个对话对包含2个turn（问题和回答），所以起始索引需要乘以2
        start_idx = self.question_id * batch_size * 2
        
        # 检查是否有足够的数据（至少需要 batch_size * 2 个 turn）
        if start_idx + batch_size * 2 > len(self.sharegpt_data["conversations"]):
            return None
        
        # 构建合并的prompt
        batched_prompt = ""
        actual_count = 0
        
        for i in range(batch_size):
            # 每个对话对包含2个连续的turn
            # 根据 start_with_gpt 判断哪个是问题，哪个是回答
            turn1_idx = start_idx + i * 2
            turn2_idx = start_idx + i * 2 + 1
            
            if turn2_idx >= len(self.sharegpt_data["conversations"]):
                break
            
            turn1 = self.sharegpt_data["conversations"][turn1_idx]
            turn2 = self.sharegpt_data["conversations"][turn2_idx]
            
            # 确定问题和回答
            if self.start_with_gpt:
                # 偶数索引是GPT回复，奇数索引是人类问题
                question = turn2  # 奇数索引
                answer = turn1    # 偶数索引
            else:
                # 偶数索引是人类问题，奇数索引是GPT回复
                question = turn1  # 偶数索引
                answer = turn2    # 奇数索引
            
            # 添加完整的对话对（问题+回答）
            batched_prompt += f"\n\n--- Conversation {actual_count + 1} ---\n"
            batched_prompt += f"Question: {question['value']}\n"
            batched_prompt += f"Answer: {answer['value']}\n"
            actual_count += 1
        
        if not batched_prompt.strip() or actual_count == 0:
            return None
            
        # 添加指令
        instruction = f"\n\nPlease respond to all {actual_count} conversations above. Provide comprehensive answers for each conversation."
        batched_prompt = instruction + batched_prompt
        
        # max_tokens 应该表示模型要生成的回答的 token 数量
        # 在批量模式下，虽然 prompt 包含多个对话对，但 max_tokens 应该使用 answer_len
        # 因为 answer_len 是用户设置的每个回答的长度
        max_tokens = self.user_config.answer_len
        
        return batched_prompt, max_tokens

    def _launch_new_request(self, timestamp: float, request_executor: RequestExecutor):
        if self.use_sharegpt and self.user_config.batch_conversations > 1:
            # 使用批量模式
            result = self._build_batched_sharegpt_prompt()
            if result is None:
                self.data_insufficient += 1
                logger.warning(
                    f"User {self.user_config.user_id} request {self.question_id} "
                    f"insufficient ShareGPT data, finishing session "
                    f"(completed {len(self.prompt_lengths)} requests, "
                    f"skipped {self.skipped_requests} requests)"
                )
                self.finished = True
                return
            prompt, max_tokens = result
        else:
            # 使用原始模式
            if self.use_sharegpt:
                # 检查数据是否足够
                if self.start_with_gpt:
                    # 偶数索引是回答，奇数索引是问题
                    prompt_index = 2 * self.question_id + 1
                    answer_index = 2 * self.question_id
                else:
                    # 偶数索引是问题，奇数索引是回答
                    prompt_index = 2 * self.question_id
                    answer_index = 2 * self.question_id + 1
                
                # 检查索引是否超出范围
                if prompt_index >= len(self.sharegpt_data["conversations"]) or answer_index >= len(self.sharegpt_data["conversations"]):
                    self.data_insufficient += 1
                    logger.warning(
                        f"User {self.user_config.user_id} request {self.question_id} "
                        f"insufficient ShareGPT data, finishing session "
                        f"(completed {len(self.prompt_lengths)} requests, "
                        f"skipped {self.skipped_requests} requests)"
                    )
                    self.finished = True
                    return
                
                prompt = self.sharegpt_data["conversations"][prompt_index]["value"]
                
                # 计算max_tokens（从对应的回答中获取）
                if answer_index < len(self.sharegpt_data["conversations"]):
                    conversation = self.sharegpt_data["conversations"][answer_index]
                    max_tokens = conversation.get("num_tokens", self.user_config.answer_len)
                else:
                    max_tokens = self.user_config.answer_len
                max_tokens = min(max_tokens, self.user_config.answer_len)
            else:
                prompt = self._build_new_question()
                max_tokens = self.user_config.answer_len
        
        # 添加系统提示词（如果是第一轮）
        if len(self.chat_history) == 0:
            prompt = self._build_system_prompt() + prompt
        
        self.chat_history.on_user_query(prompt)
        
        # 检查 token 长度，确保 messages + max_tokens <= 4096
        # 使用阈值 4000，留出安全余量（4096 - 4000 = 96 tokens 的缓冲）
        # 这样可以允许 prompt tokens 接近 4000
        messages = self.chat_history.get_messages_for_openai()
        estimated_messages_tokens = self._estimate_tokens(messages)
        # 总需求 = messages tokens + completion tokens (max_tokens)
        total_estimated_tokens = estimated_messages_tokens + max_tokens
        
        # 模型的最大上下文长度是 4096，我们需要确保不超过这个限制
        # 使用 4000 作为阈值，允许 prompt tokens 接近 4000
        # 留出 96 tokens 的安全余量（用于格式开销和估算误差）
        max_allowed_total_tokens = 4000
        max_allowed_messages_tokens = max_allowed_total_tokens - max_tokens
        
        if estimated_messages_tokens > max_allowed_messages_tokens or total_estimated_tokens > max_allowed_total_tokens:
            self.skipped_requests += 1
            logger.warning(
                f"User {self.user_config.user_id} request {self.question_id} "
                f"estimated tokens (messages: {estimated_messages_tokens}, "
                f"completion: {max_tokens}, total: {total_estimated_tokens}) "
                f"exceeds limit, skipping this request (total skipped: {self.skipped_requests})"
            )
            # 从 chat_history 中移除刚添加的 prompt，因为请求被跳过了
            self.chat_history.history.pop()
            # 更新 question_id 并继续
            self.question_id += 1
            # 如果已经达到最大轮数，标记为完成
            if self.question_id >= self.user_config.num_rounds:
                self.finished = True
            return
        
        logger.debug(
            f"User {self.user_config.user_id} issues request {self.question_id} "
            f"(estimated tokens: messages={estimated_messages_tokens}, "
            f"completion={max_tokens}, total={total_estimated_tokens})"
        )
        
        # 更新question_id
        self.question_id += 1
            
        request_executor.launch_request(
            self.chat_history,
            max_tokens,
            self._on_request_finished,
            extra_headers={"x-user-id": str(self.user_config.user_id)},
        )
        self.has_unfinished_request = True
        self.last_request_time = timestamp

    def _on_request_finished(self, response: Response):
        self.chat_history.on_system_response(response.body)
        self.has_unfinished_request = False
        logger.debug(
            f"User {self.user_config.user_id} finished one request. "
            f"Prompt tokens: {response.prompt_tokens}, "
            f"generation tokens: {response.generation_tokens}"
        )
        self._update_result(response)

    def set_internal_state(self, offset: float, timestamp: float):
        """Tell the session is the 'offset' seconds after the start"""
        assert len(self.chat_history) == 0, (
            "Internal state should be set before the first request"
        )

        num_passed_questions = int(offset / self.user_config.gap_between_requests) + 1

        passed_time = (num_passed_questions - 1) * self.user_config.gap_between_requests

        self.last_request_time = timestamp - offset + passed_time
        self.question_id = num_passed_questions
        logger.debug(
            f"Set internal state for user {self.user_config.user_id}, "
            f"question_id: {self.question_id}, "
            f"last_request_time: {self.last_request_time}"
        )

    def step(self, timestamp: float, request_executor: RequestExecutor):
        if (
            self.question_id >= self.user_config.num_rounds
            and not self.has_unfinished_request
        ):
            self.finished = True
            return

        if self.last_request_time is None:
            self._launch_new_request(timestamp, request_executor)
            return

        if timestamp - self.last_request_time > self.user_config.gap_between_requests:
            if self.has_unfinished_request:
                if timestamp - self.last_unfinished_log > 10:
                    logger.warning(
                        f"User {self.user_config.user_id} has an unfinished "
                        "request and unable to fit the QPS requirement."
                    )
                    self.last_unfinished_log = timestamp
                return

            self._launch_new_request(timestamp, request_executor)
            return

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["prompt_tokens"] = self.prompt_lengths
        df["generation_tokens"] = self.generation_lengths
        df["ttft"] = self.ttfts
        df["generation_time"] = self.generation_times
        df["user_id"] = self.user_config.user_id
        df["question_id"] = range(1, len(self.prompt_lengths) + 1)
        df["launch_time"] = self.launch_times
        df["finish_time"] = self.finish_times
        return df


class UserSessionManager:
    def __init__(
        self,
        workload_config: WorkloadConfig,
        init_user_id=0,
        use_sharegpt=False,
    ):
        self.workload_config = workload_config
        self.sessions = []

        gap_between_requests_per_user = workload_config.num_users / workload_config.qps
        session_alive_time = gap_between_requests_per_user * (
            workload_config.num_rounds - 1
        )
        self.gap_between_users = session_alive_time / (workload_config.num_users + 0)
        self.ramp_up_time = workload_config.num_users * self.gap_between_users

        logger.info(
            f"Gap between users: {self.gap_between_users} secs.\n"
            f"Gap between user reqs: {gap_between_requests_per_user} secs.\n"
            f"Expected length of user session: {session_alive_time} secs."
        )

        self.user_id = init_user_id
        self.last_user_join = 0
        self.session_summaries = []
        self.start_time = None

        self.need_ramp_up = True
        
        # 维护统计信息，因为 session_summaries 中存储的是 DataFrame，不是 UserSession 对象
        self.session_stats = []  # 存储每个会话的统计信息

        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self._load_sharegpt_data()

    def _load_sharegpt_data(self):
        with open("ShareGPT.json", "r", encoding="utf-8") as file:
            self.sharegpt_data = json.load(file)
        
        # 计算需要的最少轮数
        required_rounds = self.workload_config.num_rounds * self.workload_config.batch_conversations
        
        original_count = len(self.sharegpt_data)
        self.sharegpt_data = [
            d
            for d in self.sharegpt_data
            if d["num_round"] >= required_rounds
        ]
        filtered_count = len(self.sharegpt_data)
        
        logger.info(
            f"Loaded ShareGPT data: {original_count} users total, "
            f"{filtered_count} users have enough rounds (required >= {required_rounds} rounds)"
        )

    def _ramp_up(self, timestamp: float, ramp_up_time: float):
        for i in range(self.workload_config.num_users):
            new_session = self._create_user_session()
            offset = ramp_up_time - i * self.gap_between_users
            if offset < 0:
                break
            new_session.set_internal_state(offset, timestamp)
        self.need_ramp_up = False

    def _create_user_session(self):
        self.user_id += 1
        user_config = UserConfig.new_user_config(self.user_id, self.workload_config)
        if self.use_sharegpt:
            user_session = UserSession(
                user_config, self.use_sharegpt, self.sharegpt_data[self.user_id]
            )
        else:
            user_session = UserSession(user_config, self.use_sharegpt)
        self.sessions.append(user_session)
        return user_session

    def _remove_finished_sessions(self):
        sessions_to_remove = [s for s in self.sessions if s.finished]
        if len(sessions_to_remove) > 0:
            total_skipped = sum(s.skipped_requests for s in sessions_to_remove)
            total_data_insufficient = sum(s.data_insufficient for s in sessions_to_remove)
            logger.info(
                f"Removing {len(sessions_to_remove)} finished sessions, now "
                f"active users: {len(self.sessions) - len(sessions_to_remove)}. "
                f"Total skipped requests: {total_skipped}, "
                f"data insufficient: {total_data_insufficient}"
            )
            for session in sessions_to_remove:
                # 保存统计信息
                self.session_stats.append({
                    'skipped_requests': session.skipped_requests,
                    'data_insufficient': session.data_insufficient,
                })
                # 保存 DataFrame
                self.session_summaries.append(session.summary())
        self.sessions = [s for s in self.sessions if not s.finished]

    def all_users_finished(self):
        """检查是否所有用户都完成了所有轮次"""
        # 如果还在 ramp_up 阶段，还没完成
        if self.need_ramp_up:
            return False
        
        # 如果还有活跃的会话，还没完成
        if len(self.sessions) > 0:
            return False
        
        # 检查是否有未完成的请求（pending requests）
        # 虽然会话已经完成，但可能还有请求在处理中
        # 这个检查在 step 方法中已经处理了，当会话完成时会自动移除
        # 所以如果 sessions 为空，说明所有请求都已完成
        
        return True

    def step(self, timestamp: float, executor: RequestExecutor):
        if self.need_ramp_up:
            self._ramp_up(timestamp, self.ramp_up_time)

        if self.start_time is None:
            self.start_time = timestamp

        if timestamp - self.last_user_join > self.gap_between_users:
            self._create_user_session()
            self.last_user_join = timestamp
            logger.info(
                f"Joined a new user {self.user_id}, "
                f"now active users: {len(self.sessions)}"
            )

        for session in self.sessions:
            session.step(timestamp, executor)

        self._remove_finished_sessions()

    @staticmethod
    def ProcessSummary(
        df: pd.DataFrame,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        pending_queries: int = 0,
        qps: Optional[int] = None,
    ):
        # 统计所有数据，不进行时间范围过滤
        # 使用实际的第一个请求启动时间和最后一个请求完成时间
        if len(df) == 0:
            launched_queries = 0
            actual_start_time = start_time if start_time else 0
            actual_end_time = end_time if end_time else 0
        else:
            # 统计所有已完成的请求
            launched_queries = len(df)
            # 使用实际的第一个请求启动时间和最后一个请求完成时间
            actual_start_time = df["launch_time"].min()
            actual_end_time = df["finish_time"].max()
            # 如果提供了时间范围参数，使用它们作为显示范围，但不用于过滤数据
            if start_time is not None:
                actual_start_time = min(actual_start_time, start_time)
            if end_time is not None:
                actual_end_time = max(actual_end_time, end_time)

        logger.debug(
            f"Launched queries: {launched_queries}, "
            f"pending queries: {pending_queries}, "
            f"finished queries: {len(df)}"
        )

        if qps is None:
            qps = 0.0

        # 使用实际的时间范围（从第一个请求启动到最后一个请求完成）
        total_time = actual_end_time - actual_start_time if actual_end_time > actual_start_time else 0.0

        total_requests = launched_queries + pending_queries
        actual_qps = total_requests / total_time if total_time > 0 else 0.0

        total_finished_requests = len(df)
        finished_qps = total_finished_requests / total_time if total_time > 0 else 0.0

        total_prompt_tokens = df["prompt_tokens"].sum()
        total_generation_tokens = df["generation_tokens"].sum()
        average_prefill_speed = total_prompt_tokens / total_time if total_time > 0 else 0.0
        average_generation_speed = total_generation_tokens / total_time if total_time > 0 else 0.0
        average_generation_speed_per_request = (
            df["generation_tokens"] / df["generation_time"]
        ).mean()
        average_ttft = df["ttft"].mean()
        
        # Average tokens per request
        average_prompt_tokens_per_request = df["prompt_tokens"].mean()
        average_generation_tokens_per_request = df["generation_tokens"].mean()
        average_total_tokens_per_request = (df["prompt_tokens"] + df["generation_tokens"]).mean()
        
        logger.info("Calculating performance summary")
        print("\n")
        print("==================== Performance summary ======================")
        print(f"  Total launched requests: {launched_queries}\n")
        
        print(f"  Total finished requests: {total_finished_requests}\n")
        
        # 显示配置的目标 QPS 和实际测量的 QPS
        print(f"  Target QPS (configured): {qps:.4f} reqs/s\n")
        print(f"  Actual QPS (measured): {actual_qps:.4f} reqs/s\n")

        print(f"  Processing speed: {finished_qps:.4f} reqs/s\n")

        print(f"  Requests on-the-fly: {pending_queries}\n")

        print(
            "  Input tokens per second: "
            f"{average_prefill_speed:.4f} tokens/s\n"
        )

        print(
            "  Output tokens per second: "
            f"{average_generation_speed:.4f} tokens/s\n"
        )

        print(
            "  Average generation throughput (per request): "
            f"{average_generation_speed_per_request:.4f} "
            "tokens/req/s\n"
        )

        print(f"  Average TTFT: {average_ttft:.4f}s\n")

        print(
            "  Average prompt tokens per request: "
            f"{average_prompt_tokens_per_request:.2f} tokens\n"
        )

        print(
            "  Average generation tokens per request: "
            f"{average_generation_tokens_per_request:.2f} tokens\n"
        )

        print(
            "  Average total tokens per request: "
            f"{average_total_tokens_per_request:.2f} tokens\n"
        )

        print(f"Time range: {actual_start_time:.2f} - {actual_end_time:.2f} ({total_time:.2f}s)")

        print("===============================================================")
        print("\n")
        return df

    def summary(self, start_time: float, end_time: float) -> pd.DataFrame:
        if len(self.session_summaries) == 0 and len(self.sessions) == 0:
            return pd.DataFrame()

        df = pd.concat(
            [s for s in self.session_summaries] + [s.summary() for s in self.sessions]
        )
        pending_queries = len([s for s in self.sessions if s.has_unfinished_request])
        
        # 使用实际的测试时间范围（从测试启动到当前时间或最后一个请求完成时间）
        # 不进行数据过滤，统计所有数据
        if len(df) > 0:
            actual_start_time = min(self.start_time if self.start_time else df["launch_time"].min(), 
                                   df["launch_time"].min())
            actual_end_time = max(end_time, df["finish_time"].max())
        else:
            actual_start_time = self.start_time if self.start_time else start_time
            actual_end_time = end_time
        
        qps = self.workload_config.qps
        
        # 收集统计信息
        # 从已完成的会话统计信息中获取
        total_skipped = sum(stat['skipped_requests'] for stat in self.session_stats)
        total_data_insufficient = sum(stat['data_insufficient'] for stat in self.session_stats)
        # 从当前活跃的会话中获取
        total_skipped += sum(s.skipped_requests for s in self.sessions)
        total_data_insufficient += sum(s.data_insufficient for s in self.sessions)
        
        expected_requests = self.workload_config.num_users * self.workload_config.num_rounds
        actual_requests = len(df)
        
        logger.info(
            f"Request statistics: Expected {expected_requests} requests, "
            f"actual {actual_requests} requests, "
            f"skipped {total_skipped} requests, "
            f"data insufficient {total_data_insufficient} times"
        )

        df = UserSessionManager.ProcessSummary(
            df, actual_start_time, actual_end_time, pending_queries, qps
        )
        return df


def warmup_engine(executor):
    logger.info("Warming up the engine")
    for i in range(10):
        chat_history = ChatHistory()
        chat_history.on_user_query(
            f"WARMUP: Hi, I'm user {i}. Here are some text: 'hi '."
        )
        executor.launch_request(chat_history, 100, lambda x: None)

    AsyncLoopWrapper.WaitLoop()


def parse_arguments() -> WorkloadConfig:
    parser = argparse.ArgumentParser(description="Parse benchmark configurations.")

    parser.add_argument(
        "--num-users",
        type=int,
        required=True,
        help="Max number of users in the system concurrently",
    )
    parser.add_argument(
        "--shared-system-prompt",
        type=int,
        required=True,
        help="Length of the shared system prompt (tokens)",
    )
    parser.add_argument(
        "--user-history-prompt",
        type=int,
        required=True,
        help="Length of the user-specific history prompt (tokens)",
    )
    parser.add_argument(
        "--answer-len",
        type=int,
        required=True,
        help="Length of the answer in one round",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        required=True,
        help="Number of rounds in the conversation",
    )
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Base URL of the serving engine endpoint",
    )
    parser.add_argument(
        "--time",
        type=int,
        required=False,
        help="The time to run the simulation in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name (ended with csv or txt) for the summary csv and txt",
    )
    parser.add_argument(
        "--init-user-id",
        type=int,
        default=0,
        help="The initial user id to start with",
    )
    parser.add_argument(
        "--request-with-user-id",
        action="store_true",
        help="Whether to enable user id in the request headers",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="The time between two summary loggings in seconds",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to enable verbose logging",
    )
    parser.add_argument(
        "--sharegpt",
        action="store_true",
        help="Whether to use ShareGPT dataset",
    )
    parser.add_argument(
        "--batch-conversations",
        type=int,
        default=1,
        help="Number of conversations to batch together in one request (default: 1)",
    )
    args = parser.parse_args()
    return args


def parse_process_summary():
    parser = argparse.ArgumentParser(
        description="Parse benchmark configurations.", add_help=False
    )

    parser.add_argument("--process-summary", type=str, default=None)

    args, _ = parser.parse_known_args()
    return args


def process_output(filename):
    logger.warning(
        f"Processing the existing summary file {filename}"
        ", ignoring all the other arguments"
    )
    UserSessionManager.ProcessSummary(pd.read_csv(filename), pending_queries=0)


def main():
    args = parse_process_summary()
    if args.process_summary:
        process_output(args.process_summary)
        return

    args = parse_arguments()
    if args.verbose:
        global logger
        logger = init_logger(__name__, level=logging.DEBUG)

    step_interval = 0.1

    executor = RequestExecutor(
        base_url=args.base_url, api_key="EMPTY", model=args.model
    )

    warmup_engine(executor)
    workload_config = WorkloadConfig(
        num_users=args.num_users,
        system_prompt_len=args.shared_system_prompt,
        user_info_len=args.user_history_prompt,
        answer_len=args.answer_len,
        num_rounds=args.num_rounds,
        qps=args.qps,
        model=args.model,
        enable_user_id=args.request_with_user_id,
        batch_conversations=args.batch_conversations,
    )

    manager = UserSessionManager(
        workload_config,
        init_user_id=args.init_user_id,
        use_sharegpt=args.sharegpt,
    )

    num_steps = 0
    start_time = time.time()
    last_summary_time = start_time
    max_wait_time = 3600  # 最大等待时间（1小时），避免无限等待
    try:
        while True:
            num_steps += 1
            manager.step(time.time(), executor)
            time.sleep(step_interval)

            if time.time() - last_summary_time > args.log_interval:
                manager.summary(last_summary_time, time.time())
                last_summary_time = time.time()

            # 如果指定了时间限制，优先使用时间限制
            if args.time is not None and time.time() - start_time > args.time:
                logger.info(f"Time limit ({args.time}s) reached, stopping")
                break
            
            # 如果所有用户都完成了所有轮次，退出循环
            if manager.all_users_finished():
                logger.info("All users have completed all rounds, stopping")
                break
            
            # 防止无限等待，设置最大等待时间
            if time.time() - start_time > max_wait_time:
                logger.warning(f"Maximum wait time ({max_wait_time}s) reached, stopping")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted, waiting for the final result")

    AsyncLoopWrapper.StopLoop()

    logger.info(f"Finished benchmarking, dumping summary to {args.output}")
    summary = manager.summary(0, time.time())
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
