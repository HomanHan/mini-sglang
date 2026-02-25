from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)  # ZMQ 传输字节流，要求对象必须序列化

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):  # id -> text
    uid: int
    next_token: int  # 只有一个整数 ID
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg): # text -> id
    uid: int
    text: str | List[Dict[str, str]]  # 原始文本(str) 或 对话历史(dict)
    sampling_params: SamplingParams  # 采样参数


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
