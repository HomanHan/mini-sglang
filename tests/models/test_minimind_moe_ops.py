import math

import torch
import torch.nn.functional as F

from minisgl.attention.fa import _fa_sgl_impl
from minisgl.distributed import set_tp_info
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.layers.rotary import RotaryEmbedding
from minisgl.moe.fused import FusedMoe


def _attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_len, k_len = q.shape[0], k.shape[0]
    repeats = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(q.shape[-1])
    q_pos = torch.arange(q_len, device=q.device) + k_len - q_len
    mask = torch.arange(k_len, device=q.device)[None, :] <= q_pos[:, None]
    probs = scores.masked_fill(~mask[None, :, :], -torch.inf).softmax(dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v.float())


def test_rope_head_dim_96():
    torch.manual_seed(0)
    with torch.device("cuda"):
        rope = RotaryEmbedding(96, 96, 128, 1_000_000.0)

    positions = torch.tensor([0, 1, 7, 31], device="cuda", dtype=torch.int32)
    query = torch.randn(4, 8 * 96, device="cuda", dtype=torch.float16)
    key = torch.randn(4, 4 * 96, device="cuda", dtype=torch.float16)
    query_input, key_input = query.clone(), key.clone()
    rope.forward(positions, query, key)

    inv_freq = 1.0 / 1_000_000.0 ** (torch.arange(0, 96, 2, device="cuda") / 96)
    angles = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[:, None, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[:, None, :]

    def reference(x: torch.Tensor, heads: int) -> torch.Tensor:
        x = x.view(4, heads, 96).float()
        rotated = torch.cat((-x[..., 48:], x[..., :48]), dim=-1)
        return (x * cos + rotated * sin).flatten(1)

    torch.testing.assert_close(query.float(), reference(query_input, 8), atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(key.float(), reference(key_input, 4), atol=2e-3, rtol=2e-3)


def test_fa3_gqa_head_dim_96():
    for batch_size, q_len in ((1, 17), (1, 5), (1, 1), (4, 17)):
        torch.manual_seed(batch_size * 100 + q_len)
        k_len = 17
        q = torch.randn(batch_size, q_len, 8, 96, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, k_len, 4, 96, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, k_len, 4, 96, device="cuda", dtype=torch.float16)
        cu_q = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * q_len
        cu_k = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * k_len

        output = _fa_sgl_impl(
            q=q.flatten(0, 1),
            k_cache=k.flatten(0, 1)[:, None],
            v_cache=v.flatten(0, 1)[:, None],
            page_table=torch.arange(
                batch_size * k_len, device="cuda", dtype=torch.int32
            ).view(batch_size, k_len),
            cache_seqlens=torch.full(
                (batch_size,), k_len, device="cuda", dtype=torch.int32
            ),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_len,
            softmax_scale=96**-0.5,
            version=3,
        )

        assert torch.isfinite(output).all()
        output = output.view(batch_size, q_len, 8, 96)
        for index in range(batch_size):
            reference = _attention_reference(q[index], k[index], v[index])
            torch.testing.assert_close(output[index].float(), reference, atol=4e-3, rtol=4e-3)


def test_fa3_varlen_batch_head_dim_96():
    torch.manual_seed(2)
    lengths = (11, 17, 11, 17)
    q_base = [
        torch.randn(length, 8, 96, device="cuda", dtype=torch.float16)
        for length in lengths[:2]
    ]
    k_base = [
        torch.randn(length, 4, 96, device="cuda", dtype=torch.float16)
        for length in lengths[:2]
    ]
    v_base = [
        torch.randn(length, 4, 96, device="cuda", dtype=torch.float16)
        for length in lengths[:2]
    ]
    queries = q_base * 2
    keys = k_base * 2
    values = v_base * 2
    cu_seqlens = torch.tensor((0, 11, 28, 39, 56), device="cuda", dtype=torch.int32)
    page_table = torch.zeros(4, 17, device="cuda", dtype=torch.int32)
    offset = 0
    for index, length in enumerate(lengths):
        page_table[index, :length] = torch.arange(offset, offset + length, device="cuda")
        offset += length

    output = _fa_sgl_impl(
        q=torch.cat(queries),
        k_cache=torch.cat(keys)[:, None],
        v_cache=torch.cat(values)[:, None],
        page_table=page_table,
        cache_seqlens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=17,
        softmax_scale=96**-0.5,
        version=3,
    )

    outputs = output.split(lengths)
    for index in range(4):
        reference = _attention_reference(queries[index], keys[index], values[index])
        torch.testing.assert_close(outputs[index].float(), reference, atol=4e-3, rtol=4e-3)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], outputs[3], rtol=0, atol=0)


def test_kv_cache_head_dim_96():
    set_tp_info(0, 1)
    cache = MHAKVCache(
        num_kv_heads=4,
        num_layers=2,
        head_dim=96,
        num_pages=16,
        page_size=1,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )
    indices = torch.tensor([1, 5, 9], device="cuda", dtype=torch.int32)
    key = torch.randn(3, 4 * 96, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    cache.store_kv(key, value, indices, layer_id=1)

    assert cache.k_cache(1).shape == (16, 1, 4, 96)
    torch.testing.assert_close(cache.k_cache(1).view(-1, 4 * 96)[indices], key, rtol=0, atol=0)
    torch.testing.assert_close(cache.v_cache(1).view(-1, 4 * 96)[indices], value, rtol=0, atol=0)


def test_fused_moe_minimind_shape():
    torch.manual_seed(1)
    tokens, experts, hidden, intermediate = 8, 4, 768, 2432
    inputs = torch.randn(tokens, hidden, device="cuda", dtype=torch.float16)
    w1 = torch.randn(
        experts, 2 * intermediate, hidden, device="cuda", dtype=torch.float16
    ) / math.sqrt(hidden)
    w2 = torch.randn(
        experts, hidden, intermediate, device="cuda", dtype=torch.float16
    ) / math.sqrt(intermediate)
    routing = torch.full((tokens, experts), -20.0, device="cuda", dtype=torch.float16)
    routing[torch.arange(tokens, device="cuda"), torch.arange(tokens, device="cuda") % experts] = 20

    reference = torch.empty_like(inputs)
    for expert in range(experts):
        rows = torch.arange(tokens, device="cuda") % experts == expert
        gate, up = F.linear(inputs[rows], w1[expert]).chunk(2, dim=-1)
        reference[rows] = F.linear(F.silu(gate) * up, w2[expert])

    output = FusedMoe().forward(inputs.clone(), w1, w2, routing, topk=1, renormalize=True)
    cosine = F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0)
    assert torch.isfinite(output).all()
    assert cosine > 0.999
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)

    repeated_inputs = torch.randn(8, hidden, device="cuda", dtype=torch.float16).repeat(4, 1)
    repeated_routing = torch.randn(8, experts, device="cuda", dtype=torch.float16).repeat(4, 1)
    repeated_output = FusedMoe().forward(
        repeated_inputs, w1, w2, repeated_routing, topk=1, renormalize=True
    ).view(4, 8, hidden)
    for index in range(1, 4):
        torch.testing.assert_close(repeated_output[index], repeated_output[0], rtol=0, atol=0)


if __name__ == "__main__":
    test_rope_head_dim_96()
    test_fa3_gqa_head_dim_96()
    test_fa3_varlen_batch_head_dim_96()
    test_kv_cache_head_dim_96()
    test_fused_moe_minimind_shape()
    print("MiniMind MoE operator tests passed")
