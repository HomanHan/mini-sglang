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
    LinearColParallelMerged,
    LinearColParallel,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from minisgl.utils import nvtx_annotate, get_rope_config

from .base import BaseLLMModel
from .utils import Qwen2MoeMLP as Qwen3_5MLP

if TYPE_CHECKING:
    from .config import ModelConfig
    from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig

from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.utils import (
    set_weight_attrs,
)
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader


class Qwen3_5GatedDeltaNet(BaseOP):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # Conv1d layer
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = LinearColParallel(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            has_bias=False,
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # Split projection layers (following vLLM's implementation)
        # Instead of fused in_proj_qkvz and in_proj_ba, use separate layers
        self.in_proj_qkv = LinearColParallelMerged(
            input_size=self.hidden_size,
            output_sizes=[self.key_dim, self.key_dim, self.value_dim],
            has_bias=False,
        )
        self.in_proj_z = LinearColParallel(
            input_size=self.hidden_size,
            output_size=self.value_dim,
            has_bias=False,
        )
        self.in_proj_b = LinearColParallel(
            input_size=self.hidden_size, output_size=self.num_v_heads, has_bias=False
        )
        self.in_proj_a = LinearColParallel(
            input_size=self.hidden_size,
            output_size=self.num_v_heads,
            has_bias=False,
        )

        # Conv1d weight loader setup
        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        query_key_settings,
                        query_key_settings,
                        value_settings,
                    ]
                )
            },
        )

        # State parameters
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.attn_tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads // self.attn_tp_size, dtype=torch.float32),
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        # RadixLinearAttention layer
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads // self.attn_tp_size,
            num_k_heads=self.num_k_heads // self.attn_tp_size,
            num_v_heads=self.num_v_heads // self.attn_tp_size,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

        # Normalization layer
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.get_device_module().current_device(),
            dtype=config.torch_dtype,
        )

        # Output projection
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("out_proj", prefix),
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkv,
        z,
        b,
        a,
    ):
        raise NotImplementedError("Qwen3.5 Series dont need to fix query key value ordering")

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        seq_len, _ = hidden_states.shape

        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b, _ = self.in_proj_b(hidden_states)
        a, _ = self.in_proj_a(hidden_states)

        b = b.contiguous()
        a = a.contiguous()

        core_attn_out = self.attn(
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        output, _ = self.out_proj(core_attn_out)
        return output


class Qwen3_5AttentionDecoderLayer(BaseOP):
    def __init__(self, config: Qwen3_5TextConfig, layer_id: int):
        self.config = config
        self.head_dim = config.head_dim
        self.layer_id = layer_id
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads

        # Rotary embedding parameters
        self.rope_theta, rope_scaling = get_rope_config(config)
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # If rope_scaling doesn't specify a scaling type, treat as no scaling
        if rope_scaling and not ("rope_type" in rope_scaling or "type" in rope_scaling):
            rope_scaling = None

        # get rope config using sglang utility
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        # Gated Attention
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        # qkv projection
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=self.head_dim,
            num_qo_heads=self.total_num_heads * (1 + self.attn_output_gate),
            num_kv_heads=self.total_num_kv_heads,
            has_bias=False,
        )

        # Output projection
        self.o_proj = LinearOProj(
            input_size=self.head_dim * self.total_num_heads,
            output_size=config.hidden_size,
            has_bias=False,
        )

        # Normalization layers
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Attention layer, with RadixAttention & PagedAttention & Attention Backend
        self.self_attn = AttentionLayer(  # need to pass rope
            num_qo_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            head_dim=self.head_dim,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )

        # Dense MLP for non-MoE variant
        if config.model_type == "qwen3_5_text":
            self.mlp = Qwen3_5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
            is_layer_sparse = False
            is_previous_layer_sparse = False
            is_next_layer_sparse = False
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")

        self.input_layernorm = GemmaRMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNormFused(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @nvtx_annotate("Layer_{}", layer_id_field="layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        qkv = self.qkv_proj.forward(x)
        del x

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [
                    self.head_dim * self.total_num_heads * 2,
                    self.head_dim * self.total_num_kv_heads,
                    self.head_dim * self.total_num_kv_heads,
                ],
                dim=-1,
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(
                *orig_shape, self.total_num_heads, -1
            )  # shape: [seq_len, batch_size, num_heads, head_dim * 2]
            q, gate = torch.chunk(
                q_gate, 2, dim=-1
            )  # shape of q and gate: [seq_len, batch_size, num_heads, head_dim]
            q = q.reshape(*orig_shape, -1)  # shape: [seq_len, batch_size, num_heads * head_dim]
            gate = gate.reshape(
                *orig_shape, -1
            )  # shape: [seq_len, batch_size, num_heads * head_dim]
            qkv = torch.cat([q, k, v], dim=-1)

        attn_output = self.self_attn.forward(qkv)
        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output = self.o_proj.forward(attn_output)
        x, residual = self.post_attention_layernorm.forward(output, residual)
        x = self.mlp.forward(x)

        return x, residual


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
