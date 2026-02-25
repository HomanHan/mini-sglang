from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger
from transformers import AutoTokenizer, LlamaTokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    # 初始化三个 ZMQ 队列
    # send_backend: 发往 Scheduler (带 input_ids)
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    # send_frontend: 发往 API Server (带文本)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    # recv_listener: 接收所有消息 (来自 API Server 或 Scheduler)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(tokenizer)
    detokenize_manager = DetokenizeManager(tokenizer)

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())  # 1. 批量接收消息 (Batching)
            # 贪婪地多读一些，凑够 local_bs (Batch Size)，提高吞吐量
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            # 2. 分类处理
            # DetokenizeMsg: 来自 Scheduler，包含 next_token_id
            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            # TokenizeMsg: 来自 API Server，包含 prompt 文本
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            assert len(detokenize_msg) + len(tokenize_msg) == len(pending_msg)

            # 3. 处理 Detokenize (ID -> Text)
            if len(detokenize_msg) > 0:
                # 批量 detokenize，效率更高
                replies = detokenize_manager.detokenize(detokenize_msg)
                # 封装成 UserReply 发回前端
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            # 4. 处理 Tokenize (Text -> ID)
            if len(tokenize_msg) > 0:
                # 批量 Tokenize
                tensors = tokenize_manager.tokenize(tokenize_msg)
                # 封装成 UserMsg 发往后端 Scheduler
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass
