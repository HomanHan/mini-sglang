from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    GemmaRMSNorm,
    GemmaRMSNormFused,
    LinearOProj,
    LinearQKVMerged,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3_5MLP

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3_5Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.head_dim = config.head_dim
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.attn_output_gate = config.attn_output_gate

        # Qwen3.5 can fold attention output gate into q projection.
        q_heads_for_proj = self.num_qo_heads * (2 if self.attn_output_gate else 1)
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=self.head_dim,
            num_qo_heads=q_heads_for_proj,
            num_kv_heads=self.num_kv_heads,
            has_bias=False,
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=self.head_dim,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            input_size=self.head_dim * self.num_qo_heads,
            output_size=config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x

        if self.attn_output_gate:
            qo_attn_dim = self.head_dim * self.num_qo_heads
            kv_attn_dim = self.head_dim * self.num_kv_heads
            q_gate, k, v = qkv.split([qo_attn_dim * 2, kv_attn_dim, kv_attn_dim], dim=-1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            attn_output = self.attn.forward(torch.cat([q, k, v], dim=-1))
            attn_output = attn_output * torch.sigmoid(gate)
        else:
            attn_output = self.attn.forward(qkv)

        return self.o_proj.forward(attn_output)


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3_5Attention(config, layer_id)
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3_5ForCausalLM"]
