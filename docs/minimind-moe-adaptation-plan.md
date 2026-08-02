# MiniMind MoE 接入 mini-SGLang 实施计划

## 1. 目标

将 `~/workspace/minimind/out/` 中训练得到的 MiniMind MoE 权重接入 mini-SGLang，实现：

- 使用标准 Hugging Face Qwen3-MoE 格式保存模型。
- 使用 mini-SGLang 完成离线推理和 OpenAI 兼容 API 服务。
- 保持 MiniMind 原始模型的推理语义。
- 验证权重转换、Attention、MoE、KV Cache 和调度链路。
- 对比 MiniMind 原生推理与 mini-SGLang 的性能。

第一版不包含量化、LoRA 直接加载、YaRN、专家并行和 Transformers 5 升级。

## 2. 总体方案

```text
MiniMind .pth
    -> 标准 Qwen3-MoE Hugging Face 目录
    -> Transformers 参考推理
    -> mini-SGLang 权重加载
    -> head_dim 96 到 128 的内核补齐
    -> 离线正确性验证
    -> API 服务验证
    -> 性能测试
```

不新增 `MiniMindForCausalLM`，不让 mini-SGLang 直接读取 `.pth`。

理由：

- MiniMind MoE 的模块结构和权重名称已经与 Qwen3-MoE 对齐。
- mini-SGLang 已支持 `Qwen3MoeForCausalLM`、MoE 权重堆叠和 fused MoE。
- 标准 Hugging Face 格式可同时用于 Transformers、SGLang 和 mini-SGLang。
- 直接支持 `.pth` 会重复实现配置、tokenizer 和权重转换逻辑。

## 3. 已确认的模型信息

实施时按以下顺序逐项完成：

- [ ] 固定环境、checkpoint 哈希和测试输入。
- [ ] 将 `.pth` 转换为 Qwen3-MoE safetensors。
- [ ] 建立 MiniMind 与 Transformers 参考结果。
- [ ] 实现逻辑 head 维度与内核 head 维度分离。
- [ ] 通过 RoPE、Attention、KV Cache 和 MoE 测试。
- [ ] 通过 mini-SGLang 端到端正确性测试。
- [ ] 启动 Shell 和 OpenAI 兼容 API。
- [ ] 完成单卡性能测试，再做 TP 功能验证。

当前权重使用以下结构：

| 配置 | 数值 |
| --- | ---: |
| `hidden_size` | 768 |
| `num_hidden_layers` | 8 |
| `vocab_size` | 6400 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 4 |
| `head_dim` | 96 |
| `intermediate_size` | 2432 |
| `num_local_experts` | 4 |
| `num_experts_per_tok` | 1 |
| `moe_intermediate_size` | 2432 |
| `norm_topk_prob` | true |
| `router_aux_loss_coef` | 0.0005 |
| `max_position_embeddings` | 32768 |
| `rope_theta` | 1000000 |
| `tie_word_embeddings` | true |
| 权重 dtype | FP16 |

主要权重：

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `pretrain_768_moe.pth` | 结构和基础模型验证 | `9c3d260d3bde3c438fc5536ba4c48a7a8ec6e511242f4aad1abe2b627deaedcf` |
| `kd_teacher_full_sft_v2_768_moe.pth` | 对话和 API 验证 | `ca95ef94b50a13e1f739ea8c5b432c2056f4fc1ac2b51a3d9b3f581daf90abb2` |

已确认：

- 两个 checkpoint 都可以 strict-load 到 MiniMind 和 Transformers `Qwen3MoeForCausalLM`。
- Q/K/V、Q/K Norm、专家 gate/up/down 和 router 的名称与 Qwen3-MoE 一致。
- mini-SGLang 现有权重加载器可以完成 QKV 合并、gate/up 合并和专家堆叠。
- 当前阻塞点不是模型结构或权重，而是 Attention 内核不能可靠处理 `head_dim=96`。

## 4. 第 0 步：固定基线

### 4.1 固定环境

MiniMind 转换和参考推理使用：

```bash
source ~/workspace/minimind/.venv/bin/activate
```

mini-SGLang 开发和推理使用：

```bash
source ~/workspace/mini-sglang/.venv/bin/activate
```

记录以下版本：

- Python
- PyTorch
- Transformers
- Safetensors
- FlashInfer
- CUDA Driver
- GPU 型号
- 两个仓库的 Git commit

### 4.2 固定测试输入

准备两类输入：

1. 固定 token IDs：用于排除 tokenizer 和 chat template 差异。
2. 固定对话文本：用于验证 tokenizer、chat template 和 API。

固定使用 greedy decoding：

- `temperature=0`
- 不采样
- 固定最大输出长度
- 测试时显式指定是否忽略 EOS

### 4.3 完成标准

- checkpoint 哈希与上表一致。
- 环境版本和测试输入已记录。
- MiniMind 原生实现能够完成基线推理。

## 5. 第 1 步：转换为标准 Qwen3-MoE

### 5.1 修改文件

修改 MiniMind 的 [`scripts/convert_model.py`](../../minimind/scripts/convert_model.py)。

使用现有 `convert_torch2transformers` 路线。不要使用生成 `MiniMindForCausalLM` 和 `.bin` 的 `convert_torch2transformers_minimind` 路线。

### 5.2 修改内容

增加以下命令行参数：

- `--torch-path`：输入 `.pth`。
- `--transformers-path`：输出 Hugging Face 目录。
- `--use-moe`：启用 MoE 配置。
- `--hidden-size`：默认 768。
- `--num-hidden-layers`：默认 8。

转换逻辑按以下顺序执行：

1. 根据参数构造 `MiniMindConfig`。
2. 根据 `use_moe` 构造 `Qwen3MoeConfig`。
3. 完整传递 Attention、RoPE、MoE 和 tied embedding 配置。
4. 使用 `torch.load(..., map_location="cpu", weights_only=True)` 加载权重。
5. 使用 `strict=True` 加载到 `Qwen3MoeForCausalLM`。
6. 使用 `safe_serialization=True` 保存模型。
7. 将 tokenizer 和 chat template 保存到同一目录。

tokenizer 源目录基于脚本文件定位，不依赖执行命令时的当前目录。

### 5.3 目标命令

```bash
cd ~/workspace/minimind
source .venv/bin/activate

python scripts/convert_model.py \
  --torch-path out/kd_teacher_full_sft_v2_768_moe.pth \
  --transformers-path out/hf/kd_teacher_full_sft_v2_768_moe \
  --use-moe
```

对 `pretrain_768_moe.pth` 执行相同转换，输出到独立目录。

### 5.4 输出要求

输出目录至少包含：

- `config.json`
- `model.safetensors` 或 safetensors 分片
- `tokenizer.json`
- `tokenizer_config.json`
- tokenizer 需要的其他文件

配置必须满足：

- `model_type == "qwen3_moe"`
- `architectures == ["Qwen3MoeForCausalLM"]`
- `head_dim == 96`
- `num_local_experts == 4`
- `num_experts_per_tok == 1`
- `rope_scaling == null`

### 5.5 完成标准

- 输出中没有 `.bin` 权重。
- Transformers `AutoModelForCausalLM.from_pretrained()` 可以加载输出目录。
- 转换过程没有 missing 或 unexpected keys。
- tokenizer 的 BOS、EOS、PAD ID 与 MiniMind 原始 tokenizer 一致。

## 6. 第 2 步：建立 Transformers 参考结果

在修改 mini-SGLang 前，先使用转换后的模型生成参考结果。

比较以下两套实现：

1. MiniMind `MiniMindForCausalLM` 加载原始 `.pth`。
2. Transformers `Qwen3MoeForCausalLM` 加载导出目录。

比较内容：

- 每层权重形状。
- 固定输入的最后一层 logits。
- 每个位置的 top-1 token。
- greedy decoding 的输出 token IDs。

允许 FP16 运算顺序造成小数值差异，不要求 logits 逐位相等。

完成标准：

- logits 全部有限。
- logits cosine similarity 不低于 `0.999`。
- 固定测试集的 top-1 token 一致。
- 短序列 greedy 输出一致。

## 7. 第 3 步：支持逻辑维度和内核维度分离

### 7.1 问题

MiniMind 使用 `head_dim=96`。

当前 [`rotary.py`](../python/minisgl/layers/rotary.py) 只允许固定 head size。移除断言后，当前 RTX 3090 和 FlashInfer 0.6.16 的 paged prefill 仍会产生 NaN。仅删除断言不能解决问题。

### 7.2 处理原则

定义两个维度：

- `logical_head_dim = 96`：模型真实维度。
- `kernel_head_dim = 128`：Attention 和 KV Cache 使用的维度。

计算流程：

```text
Q/K/V projection: 96
    -> Q/K Norm: 96
    -> RoPE: 96
    -> Q/K/V 尾部补零: 128
    -> Attention，scale = 1 / sqrt(96)
    -> 输出切片到前 96 维
    -> O projection: 96
```

补零不会改变有效的 QK 点积和 V 加权结果。Attention 缩放必须继续使用 96。

### 7.3 修改文件

#### `python/minisgl/models/config.py`

在 [`ModelConfig`](../python/minisgl/models/config.py) 增加只读 `kernel_head_dim`：

- 使用大于等于 `head_dim` 的下一档 2 的幂。
- 96 映射到 128。
- 64、128、256 等已有模型保持不变。

不要覆盖现有 `head_dim`。权重形状、Q/K Norm 和 RoPE 仍使用逻辑维度。

#### `python/minisgl/layers/rotary.py`

- 移除固定 head-size 断言。
- RoPE 继续使用逻辑维度 96。
- 不改变 RoPE 频率、排列方式或 `rope_theta`。

#### `python/minisgl/layers/attention.py`

在 [`AttentionLayer.forward`](../python/minisgl/layers/attention.py) 中：

1. 按逻辑维度拆分 Q/K/V。
2. 按逻辑维度执行 Q/K Norm。
3. 按逻辑维度执行 RoPE。
4. reshape 为按 head 分组的张量。
5. 最后一维补零到 `kernel_head_dim`。
6. 调用 Attention backend。
7. 将输出切回 `logical_head_dim`。
8. 按原始 `qo_attn_dim` flatten 后返回。

Linear 层和模型权重不做 padding。

#### `python/minisgl/kvcache/__init__.py`

创建 [`MHAKVCache`](../python/minisgl/kvcache/mha_pool.py) 时传入 `kernel_head_dim`。

#### `python/minisgl/engine/engine.py`

KV Cache 每页显存计算使用 `kernel_head_dim`，保证页数估算与实际分配一致。

#### `python/minisgl/attention/fi.py`

FlashInfer prefill 和 decode plan：

- `head_dim` 或 `head_dim_qk` 使用 `kernel_head_dim`。
- `sm_scale` 显式使用 `logical_head_dim ** -0.5`。
- CUDA Graph 使用相同 metadata 和 scale。

### 7.4 不修改的文件

- `python/minisgl/models/qwen3_moe.py`
- `python/minisgl/models/register.py`
- `python/minisgl/models/weight.py`
- MiniMind 模型权重

### 7.5 后端范围

第一版验收固定使用 `--attn fi`。

FA 和 TensorRT-LLM 后端继续使用逻辑 scale，通用 padding 应保持接口兼容，但不作为当前 RTX 3090 环境的第一版验收项。

## 8. 第 4 步：单元和算子验证

### 8.1 配置测试

验证：

- `head_dim=96` 得到 `kernel_head_dim=128`。
- `head_dim=64/128/256` 不发生额外 padding。

### 8.2 RoPE 测试

验证 head size 96：

- 输出全部有限。
- 输出形状不变。
- 与 MiniMind 非交错 RoPE 参考实现对齐。

### 8.3 Attention 测试

使用小规模张量比较 padded Attention 和 PyTorch SDPA：

- Q/K/V 有效维度为 96。
- 内核维度为 128。
- scale 固定为 `1/sqrt(96)`。
- 比较 prefill 和单 token decode。
- 检查 padding 区域为零。

### 8.4 KV Cache 测试

验证：

- 实际 cache 最后一维为 128。
- store/read 不改变前 96 维。
- 额外 32 维保持为零。
- 显存页数计算与实际分配一致。

### 8.5 MoE 测试

使用当前模型实际形状：

- experts = 4
- top-k = 1
- hidden = 768
- expert intermediate = 2432

将 fused MoE 与逐专家 PyTorch 参考实现比较，覆盖 router、gate/up 和 down projection。

### 8.6 完成标准

- 所有输出有限，无 NaN/Inf。
- Attention 和 MoE 误差处于 FP16 合理范围。
- 已支持模型的标准 head size 测试不回归。

## 9. 第 5 步：mini-SGLang 端到端验证

按以下顺序测试：

1. 关闭 CUDA Graph，单请求、短 prompt。
2. 开启 CUDA Graph，单请求 decode。
3. batch size 1、8、32。
4. prompt 长度 32、128、768。
5. chunked prefill。
6. naive cache。
7. radix cache 无命中。
8. radix cache 完整和部分前缀命中。
9. TP1。
10. TP2/TP4 功能验证。

每项比较 Transformers 参考结果和 mini-SGLang 结果：

- prefill logits 是否有限。
- 首 token 是否一致。
- greedy 输出 token IDs 是否一致。
- decode 多步后是否出现 NaN。
- KV Cache 命中前后输出是否一致。

完成标准：

- 固定测试集 top-1 一致。
- logits cosine similarity 不低于 `0.999`。
- CUDA Graph 开关不改变输出 token。
- naive 和 radix cache 不改变输出 token。
- TP1/TP2/TP4 在各自精度范围内输出一致。

## 10. 第 6 步：启动服务

### 10.1 Shell 验证

```bash
cd ~/workspace/mini-sglang
source .venv/bin/activate

python -m minisgl \
  --model ~/workspace/minimind/out/hf/kd_teacher_full_sft_v2_768_moe \
  --dtype float16 \
  --tp-size 1 \
  --attn fi \
  --moe-backend fused \
  --max-seq-len-override 2048 \
  --shell-mode
```

### 10.2 API 服务

```bash
python -m minisgl \
  --model ~/workspace/minimind/out/hf/kd_teacher_full_sft_v2_768_moe \
  --dtype float16 \
  --tp-size 1 \
  --attn fi \
  --moe-backend fused \
  --max-seq-len-override 2048 \
  --host 0.0.0.0 \
  --port 30000
```

验证接口：

- `/v1/models`
- `/v1/chat/completions`
- 非流式返回
- 流式返回
- 多轮对话的 chat template
- EOS 和最大输出长度

### 10.3 上下文长度说明

模型配置是 32768，tokenizer 配置可能是 131072，但训练脚本常用的实际序列长度明显更短。第一版固定 `--max-seq-len-override 2048`。

完成正确性验证后再测试 32768 的技术可运行性。技术可运行不等于长上下文质量有效。

## 11. 第 7 步：性能测试

### 11.1 对比对象

使用同一张 RTX 3090、同一组 token IDs 和 FP16：

1. MiniMind 原生实现。
2. Transformers Qwen3-MoE 导出模型。
3. mini-SGLang TP1。

TP2/TP4 单独测试，不与单卡加速结论混合。

### 11.2 测试组合

至少覆盖：

| 场景 | 输入长度 | 输出长度 | 并发/批量 |
| --- | ---: | ---: | ---: |
| 短对话 | 128 | 128 | 1 |
| 批量生成 | 128 | 128 | 8、32 |
| Prefill | 768 | 32 | 1、8 |
| Decode | 32 | 512 | 1、8 |
| 前缀复用 | 共享前缀 512 | 128 | 8、32 |

每项先预热，再重复多次。框架启动和模型加载时间不计入推理吞吐。

### 11.3 指标

记录：

- TTFT
- 平均 inter-token latency
- prefill tokens/s
- decode tokens/s
- 总吞吐 tokens/s
- 峰值 GPU 显存
- 不同并发下的请求完成时间

### 11.4 结果判断

该模型约有 1.98 亿唯一参数，每 token 激活参数约 6400 万。单卡计算量较小，TP 通信和服务调度开销可能占较高比例。

因此：

- TP1 作为主要性能配置。
- TP2/TP4 主要用于验证并行功能。
- 不预设必须达到固定加速倍数。
- 分别报告 prefill、decode、低并发和高并发结果。

## 12. 实施提交顺序

建议按以下边界提交：

1. MiniMind：参数化 Qwen3-MoE safetensors 转换。
2. mini-SGLang：引入 `kernel_head_dim` 和 Attention/KV Cache padding。
3. mini-SGLang：补充 RoPE、Attention、KV Cache 和 MoE 测试。
4. mini-SGLang：补充端到端验证和 benchmark 配置。
5. 文档：记录最终命令、测试数据和性能结果。

每一步通过对应测试后再进入下一步。不要同时修改转换、Attention 和调度逻辑后再统一排错。

## 13. 第一版明确不做

- 不新增 MiniMind 专用模型实现。
- 不新增 `.pth` 运行时加载器。
- 不直接加载 LoRA；需要时先合并到基础模型。
- 不增加 YaRN 或其他 RoPE scaling。
- 不做量化。
- 不升级到 Transformers 5。
- 不实现 expert parallel。
- 不把 TP2/TP4 作为性能默认值。

## 14. 参考资料

- [MiniMind README：模型转换和 SGLang 推理](https://github.com/jingyaogong/minimind/blob/89d674b8a517010f5561b6d8ab2dcbb58e2fb91b/README.md#L1613-L1673)
- [MiniMind 官方 Qwen3-MoE 转换代码](https://github.com/jingyaogong/minimind/blob/89d674b8a517010f5561b6d8ab2dcbb58e2fb91b/scripts/convert_model.py#L39-L96)
- [MiniMind-3-MoE 配置](https://huggingface.co/jingyaogong/minimind-3-moe/blob/main/config.json)
- [mini-SGLang Qwen3-MoE 模型](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/models/qwen3_moe.py)
- [mini-SGLang 权重加载](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/models/weight.py)
- [SGLang 新模型支持说明](https://github.com/sgl-project/sglang/blob/fd28242b683f367dbee47736a361cc694906d067/docs_new/docs/supported-models/support_new_models.mdx)
- [SGLang Attention backend 说明](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
