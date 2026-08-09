# MiniMind MoE 适配测试记录

## 1. 测试范围

- 日期：2026-08-08
- 模型：`~/workspace/minimind/out/hf/kd_teacher_full_sft_v2_768_moe`
- 模型形状：hidden 768，8 层，Q 头 8，KV 头 4，head dim 96，专家 4，top-k 1
- 路径：FP16、`--attn fa`、`--moe-backend fused`
- 硬件：RTX 3090，驱动 570.169
- 软件：PyTorch 2.9.1+cu128、Transformers 4.57.3、sgl-kernel 0.3.21、FlashInfer 0.6.16

当前实现直接使用 96 维 FA3。旧计划中的 96 到 128 padding、FlashInfer 验收不再适用。

测试依赖当前 `.venv` 中的本地补丁：`sgl_kernel/flash_attn.py` 捕获 `(ImportError, AttributeError)`。重装 sgl-kernel 后需重新确认该补丁。

## 2. 测试文件

- `tests/models/test_minimind_moe_ops.py`：RoPE、FA、KV Cache、fused MoE 算子测试。
- `tests/models/run_minimind_moe_e2e.py`：端到端、cache、CUDA Graph、batch 和轻量性能测试。

端到端测试读取：

- `docs/minimind-moe-baseline/inputs.json`
- `docs/minimind-moe-baseline/transformers_baseline.pt`

生成基线：

```bash
cd ~/workspace/mini-sglang
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2 python ~/workspace/minimind/scripts/freeze_moe_baseline.py
```

运行器默认读取该目录。当前旧产物在 `docs/output` 时，可传 `--baseline-dir docs/output`。

## 3. 测试方法

### 3.1 算子

- RoPE：96 维输出与非交错 PyTorch 参考实现比较。
- FA3：Q8/KV4/D96 与 FP32 Attention 参考比较。
- FA3 场景：fresh prefill、partial extend、decode、batch=4、变长 batch。
- KV Cache：检查 96 维实际形状，并精确比较 store 前后数据。
- MoE：使用 E4/top1/H768/I2432，对比逐专家 PyTorch SwiGLU；重复输入另做逐位一致性检查。

```bash
cd ~/workspace/mini-sglang
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2 python tests/models/test_minimind_moe_ops.py
```

### 3.2 端到端

先生成单请求基线，其他配置与该基线比较：

`LLM.generate()` 不返回 logits，因此端到端只比较 token IDs；数值误差由算子测试覆盖。

```bash
export MINIMIND_MOE_MODEL=~/workspace/minimind/out/hf/kd_teacher_full_sft_v2_768_moe

CUDA_VISIBLE_DEVICES=2 python tests/models/run_minimind_moe_e2e.py \
  --model "$MINIMIND_MOE_MODEL" --scenario base \
  --output /tmp/minimind-moe-base.json

CUDA_VISIBLE_DEVICES=2 python tests/models/run_minimind_moe_e2e.py \
  --model "$MINIMIND_MOE_MODEL" --scenario chunk \
  --reference /tmp/minimind-moe-base.json --output /tmp/minimind-moe-chunk.json

CUDA_VISIBLE_DEVICES=2 python tests/models/run_minimind_moe_e2e.py \
  --model "$MINIMIND_MOE_MODEL" --scenario radix \
  --reference /tmp/minimind-moe-base.json --output /tmp/minimind-moe-radix.json

CUDA_VISIBLE_DEVICES=2 python tests/models/run_minimind_moe_e2e.py \
  --model "$MINIMIND_MOE_MODEL" --scenario batch \
  --reference /tmp/minimind-moe-base.json --output /tmp/minimind-moe-batch.json

CUDA_VISIBLE_DEVICES=2 python tests/models/run_minimind_moe_e2e.py \
  --model "$MINIMIND_MOE_MODEL" --scenario graph \
  --reference /tmp/minimind-moe-batch.json --output /tmp/minimind-moe-graph.json
```

`base`、`batch`、`graph` 会先写 JSON，再因严格 token 不一致返回非零。该退出码是测试结果。

配置矩阵：

| 场景 | Cache | CUDA Graph | Max extend | 请求 |
| --- | --- | ---: | ---: | --- |
| base | naive | 0 | 8192 | 单请求，长度 32/128/768 和两组 chat |
| chunk | naive | 0 | 128 | 长度 768，连续运行两次 |
| radix | radix | 0 | 8192 | 128 前缀、768 部分命中、768 完整命中 |
| batch | naive | 0 | 8192 | batch=8 |
| graph | naive | 8 | 8192 | batch=8 |

## 4. 正确性结果

### 4.1 算子结果

| 项目 | 结果 |
| --- | --- |
| RoPE D96 | 通过，`atol=rtol=2e-3` |
| FA3 prefill/extend/decode | 通过，全部有限，`atol=rtol=4e-3` |
| FA3 固定长度和变长 batch | 通过，重复输入逐位一致 |
| KV Cache D96 | 通过，shape 和 store 数据精确一致 |
| fused MoE 实际形状 | 通过，cosine 大于 0.999，`atol=rtol=2e-2` |

### 4.2 端到端结果

| 场景 | 结果 |
| --- | --- |
| Transformers 对齐 | 严格未通过；5/5 首 token 一致，4/5 完整序列一致 |
| chat_multi | 前 24 个生成 token 一致，第 25 个开始分叉 |
| chunked prefill | 与 base 完整一致；连续两次一致 |
| radix 部分和完整命中 | ids_128、ids_768、重复 ids_768 均与 base 完整一致 |
| batch=8，graph 关闭 | 严格未通过；8/8 首 token 一致，最终运行 2/8 完整一致 |
| batch=8，graph 开启 | 严格未通过；与 graph0 batch 首 token 8/8 一致，完整序列 4/8 一致 |

所有请求均正常完成。batch 和 CUDA Graph 的完整 greedy token 不满足旧计划的严格一致标准。相同 prompt 在同一 batch 中也可能在后续 token 分叉。FA 变长 batch 和 fused MoE 重复输入单测均通过，当前未继续修改内核。

## 5. 服务和 TP

启动模板：

```bash
CUDA_VISIBLE_DEVICES=<GPU列表> python -m minisgl \
  --model "$MINIMIND_MOE_MODEL" --dtype float16 \
  --tp-size <TP> --attn fa --moe-backend fused \
  --cache-type <CACHE> --graph 0 --max-prefill-length 128 \
  --max-seq-len-override 2048 --host 127.0.0.1 --port <端口>
```

| 模式 | GPU 列表 | TP | Cache | 端口 | 附加参数 |
| --- | --- | ---: | --- | ---: | --- |
| TP1 | `2` | 1 | radix | 30123 | 无 |
| TP2 | `2,3` | 2 | naive | 30124 | `--num-pages 8192` |
| TP4 | `4,5,6,7` | 4 | naive | 30125 | `--num-pages 8192` |

请求：

```bash
curl http://127.0.0.1:30123/v1/models

curl http://127.0.0.1:30123/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimind","messages":[{"role":"user","content":"用一句话解释 MoE。"}],"temperature":0,"top_k":1,"max_tokens":8,"stream":false}'

curl --no-buffer http://127.0.0.1:30123/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimind","messages":[{"role":"user","content":"计算 2+2。"}],"temperature":0,"top_k":1,"max_tokens":8,"stream":true}'
```

| 项目 | 结果 |
| --- | --- |
| TP1 `/v1/models` | HTTP 200 |
| TP1 非流式 chat | HTTP 200，正常返回文本 |
| TP1 流式 chat | HTTP 200，正常结束于 `[DONE]` |
| TP2 | 启动和请求通过；8-token 文本与 TP1 一致 |
| TP4 | 失败；前端启动后，首个请求触发 KV store JIT 编译错误 |

TP4 失败链路：

```text
KV heads 4 / TP4
-> local KV heads 1
-> FP16 单 token KV 大小 = 1 * 96 * 2 = 192 bytes
-> store_cache element_size=192
-> warp::copy<192> 无可用的 4/8/16-byte warp package
-> Unsupported memory package size
```

对应位置：

- `python/minisgl/kvcache/mha_pool.py:27`
- `python/minisgl/kernel/store.py:40`
- `python/minisgl/kernel/csrc/include/minisgl/warp.cuh:25`

该问题属于 KV store 内核，不属于 FA attention。TP4 当前不能标记为支持。

## 6. 轻量性能结果

同一张 RTX 3090、FP16、相同 token IDs。模型加载不计时；预热一次，每项运行两次取平均。mini-SGLang 使用 naive cache、CUDA Graph 8 和测试专用 8192-token KV Cache。

```bash
for RUNTIME in native transformers minisgl; do
  CUDA_VISIBLE_DEVICES=2 python tests/models/run_minimind_moe_e2e.py \
    --model "$MINIMIND_MOE_MODEL" --scenario perf --runtime "$RUNTIME" \
    --output "/tmp/minimind-moe-perf-$RUNTIME.json"
done
```

单位为 output token/s：

| 输入/输出/batch | MiniMind 原生 | Transformers | mini-SGLang | 相对原生 | 相对 Transformers |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/128/1 | 66.5 | 72.9 | 1019.9 | 15.3x | 14.0x |
| 128/128/8 | 359.5 | 384.7 | 6166.9 | 17.2x | 16.0x |
| 768/32/1 | 65.7 | 70.4 | 939.7 | 14.3x | 13.4x |
| 32/256/8 | 364.0 | 390.0 | 6479.6 | 17.8x | 16.6x |

吞吐按实际返回 token 数计算。峰值 PyTorch allocated memory：原生 0.456 GiB，Transformers 0.409 GiB，mini-SGLang 0.505 GiB。mini-SGLang 使用了较小的测试 KV Cache，该显存值不能代表默认服务配置。

这组数据只用于确认加速方向。它不包含 TTFT、TPOT、在线并发和长时间稳定性，不能代替正式 benchmark。

## 7. 结论

1. TP1 和 TP2 的原生 D96 FA 路径可运行，核心算子、chunk、radix 和 API 均通过。
2. mini-SGLang 在本次轻量测试中明显快于 MiniMind 原生和 Transformers。
3. Transformers 基线、batch 和 graph 的完整 greedy 序列均未全部一致，严格正确性验收未完成。
4. TP4 被 192-byte KV store 内核限制阻塞，当前不能使用。
