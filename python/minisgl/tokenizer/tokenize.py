from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from minisgl.message import TokenizeMsg

if TYPE_CHECKING:
    from transformers import LlamaTokenizer


class TokenizeManager:
    def __init__(self, tokenizer: LlamaTokenizer) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            # 1. 处理 Chat Template
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(  # 把多轮 {role, content} 拼成一个完整字符串 prompt
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text

            # 2. 调用 HuggingFace Tokenizer
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            # 3. 展平并转为 int32 (为了后续序列化和 C++ 兼容)
            results.append(
                input_ids.view(-1).to(torch.int32)
            )  # 二维 Tensor [1, seq_len] 展平为一维，并强制转为 int32
        return results
