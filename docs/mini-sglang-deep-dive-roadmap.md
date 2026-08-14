# mini-SGLang 后续深入学习路线

本文基于 mini-SGLang 提交 `f675880`。目标不是继续罗列功能，而是说明在已经理解启动、模型初始化、Scheduler、Paged/Radix KV Cache、CUDA Graph 和 TP 之后，还应研究哪些问题。

文中严格区分三种状态：

- **已实现**：存在可执行主链和对应配置。
- **有限实现**：主链可用，但输入、模型或后端范围较窄。
- **未实现**：完整 SGLang 或 vLLM 支持，当前 mini-SGLang 没有相应状态和执行链。

## 1. 当前理解基线

建议先确认已经能够独立解释以下链路：

```text
HTTP request
  -> tokenize
  -> scheduler selects prefill/decode batch
  -> allocate page table entries
  -> prepare attention metadata
  -> model forward
  -> sample token
  -> update request/cache state
  -> incremental detokenize
  -> SSE response
```

同时应能解释：

- Prefill、extend prefill、decode 的输入形状和 KV 写入位置。
- Continuous batching、chunked prefill 与请求状态变化。
- Radix tree 的 prefix match、lock、eviction 与 page 回收。
- CUDA Graph 的静态地址、batch padding 和 replay 条件。
- TP 下 Q/K/V heads、MLP 中间维、KV Cache 和 collective 的切分。

如果这些内容仍不能落实到具体 tensor 和状态变化，应先补完已有主题，再进入下文。

## 2. P0：补完一次 token 的执行链

### 2.1 Attention backend 契约

**核心问题**

- Scheduler 中的 `Req`、`Batch` 如何转换为 `cu_seqlens_q`、`cu_seqlens_k`、sequence lengths 和 page table。
- backend 为什么先执行 `prepare_metadata()`，再由每层调用同一个 `forward()`。
- FA、FlashInfer、TensorRT-LLM kernel 对 KV layout、page size、workspace 和 CUDA Graph metadata 的要求有何不同。
- Prefill 和 decode 为什么可以使用不同 backend，而模型层不需要感知差异。

**代码入口**

- [attention/base.py](../python/minisgl/attention/base.py)：backend 和 metadata 的最小接口。
- [attention/__init__.py](../python/minisgl/attention/__init__.py)：单 backend 和 prefill/decode hybrid backend 的选择。
- [attention/fa.py](../python/minisgl/attention/fa.py)：将全局 token page table 转为 FlashAttention block table。
- [attention/fi.py](../python/minisgl/attention/fi.py)：FlashInfer plan/run、CPU pinned metadata 和 128 MiB workspace。
- [attention/trtllm.py](../python/minisgl/attention/trtllm.py)：TensorRT-LLM FMHA metadata。
- [layers/attention.py](../python/minisgl/layers/attention.py)：模型层到 backend 的唯一调用点。

**当前边界与工业对照**

- FlashInfer 路径当前只允许 `page_size=1`；TensorRT-LLM 路径要求 16、32 或 64。
- KV Cache 仅实现普通 MHA/GQA 布局，没有 MLA、滑动窗口、Mamba state 或混合 attention cache group。
- vLLM 的 Hybrid KV Cache Manager 需要同时管理不同 attention 类型和不同 cache group；这类状态是工业框架显著复杂于 mini-SGLang 的原因之一。[vLLM Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)
- 完整 SGLang 还需根据模型、GPU 架构、prefill/decode 阶段和量化格式选择 backend。[SGLang Attention Backend](https://docs.sglang.io/docs/advanced_features/attention_backend)

**实验**

在 `--graph 0`、greedy sampling 和相同 page size 下分别启动 `--attn fa` 与 `--attn fi`。对相同的短 prompt、prefix-hit prompt 和长 prompt 比较输出 token IDs、TTFT、TPOT 和显存。输出必须一致；性能差异才属于 backend 差异。

### 2.2 模型 forward、packed token 与融合边界

**核心问题**

- Prefill 输入为何通常是 `[sum(extend_len), hidden]`，而不是 padding 后的 `[batch, seq, hidden]`。
- QKV 和 Gate/Up 融合只是合并 GEMM，之后仍需按逻辑维度 split。
- GQA、RoPE、QK Norm、RMSNorm、residual 分别在哪个 shape 上执行。
- Prefill 后为何 LM Head 只读取每个请求最后一个 query token 的 hidden state。
- 哪些融合减少 kernel launch，哪些融合会改变内存读写和数值误差。

**代码入口**

- [models/qwen3.py](../python/minisgl/models/qwen3.py)：完整 dense decoder 主链。
- [models/utils.py](../python/minisgl/models/utils.py)：RopeAttn、GatedMLP 和 MoEMLP。
- [layers/linear.py](../python/minisgl/layers/linear.py)：QKV、column parallel、row parallel。
- [layers/norm.py](../python/minisgl/layers/norm.py)：RMSNorm 与 residual 融合。
- [layers/rotary.py](../python/minisgl/layers/rotary.py)：RoPE 构造和应用。
- [layers/embedding.py](../python/minisgl/layers/embedding.py)：prefill last-token 选择和 LM Head。

**当前边界与工业对照**

- mini-SGLang 使用紧凑、显式的模型 forward，便于观察 tensor 形状，但缺少通用 graph rewrite、量化方法注入和大量模型变体。
- vLLM 和完整 SGLang 的模型层还需同时兼容量化 linear、LoRA、PP、EP、speculative worker、不同 attention state 和 `torch.compile`。复杂度主要来自组合能力，而非 Transformer 数学本身。
- FlashAttention 的核心价值是减少 attention 中间矩阵的 HBM 读写，不是改变 attention 数学。[FlashAttention 论文](https://arxiv.org/abs/2205.14135)

**实验**

使用 NVTX/Nsight Systems 记录一次无 cache-hit prefill 和连续四步 decode。确认 prefill 的 token 维是 packed 的；每层主要序列应为 QKV GEMM、RoPE/attention、O GEMM、GateUp GEMM、activation、Down GEMM。

### 2.3 采样、停止和输出语义

**核心问题**

- 同一 batch 中 greedy、temperature、top-k、top-p 请求如何组成采样参数 tensor。
- 为什么采样前 logits 常转换为 FP32。
- `max_tokens`、EOS、`ignore_eos` 分别在哪一层终止请求。
- BPE token 不对应完整 Unicode 字符时，detokenizer 如何延迟输出。
- 客户端看到的 `finish_reason`、usage、stream chunk 是否准确反映内部状态。

**代码入口**

- [core.py](../python/minisgl/core.py)：`SamplingParams`、`Req` 和长度状态。
- [engine/sample.py](../python/minisgl/engine/sample.py)：greedy 与 FlashInfer sampling。
- [scheduler/scheduler.py](../python/minisgl/scheduler/scheduler.py)：EOS、长度终止和资源回收。
- [tokenizer/detokenize.py](../python/minisgl/tokenizer/detokenize.py)：增量 BPE/Unicode 输出。
- [server/api_server.py](../python/minisgl/server/api_server.py)：HTTP 参数和 OpenAI 风格响应。

**当前边界与工业对照**

- mini-SGLang 已实现 greedy、temperature、top-k 和 top-p。
- API schema 中存在 `n`、`stop`、presence/frequency penalty，但当前没有进入实际采样逻辑；usage 固定为 0，`finish_reason` 也未区分 EOS 和长度上限。
- 当前没有 logprobs、min-p、beam search、per-request seed、bad words 或 grammar mask。
- vLLM/SGLang 的 structured output 会在每一步根据 grammar 状态构造允许 token mask，并与普通采样、并发请求和 CUDA Graph 协同。[vLLM Structured Outputs](https://docs.vllm.ai/en/latest/examples/features/structured_outputs/)、[SGLang Structured Outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs)

**实验**

构造包含 greedy、temperature、top-k 和 top-p 的并发请求；分别测试 EOS、`ignore_eos=true`、客户端中断和包含中英文/emoji 的输出。记录内部 token IDs 与每个 SSE chunk，检查是否丢字符、重复输出或错误提前结束。

### 2.4 模型对象和权重加载

**核心问题**

- 为什么项目使用 `BaseOP` 而不是 `torch.nn.Module`。
- meta device 构造如何避免先分配完整模型权重。
- safetensors 如何逐 tensor 读取、按 TP rank 切片、再融合 QKV 和 Gate/Up。
- tied embedding、MoE expert stack 和复制参数如何进入 runtime state。
- 新模型的 HF config、checkpoint 名称、模型类和分片规则为何必须同时匹配。

**代码入口**

- [layers/base.py](../python/minisgl/layers/base.py)：轻量 state dict 递归。
- [models/config.py](../python/minisgl/models/config.py)：HF config 到内部模型配置。
- [models/register.py](../python/minisgl/models/register.py)：architecture 到模型类。
- [models/weight.py](../python/minisgl/models/weight.py)：流式读取、分片、融合和 expert stack。
- [engine/engine.py](../python/minisgl/engine/engine.py)：meta 构造与权重装载入口。

**当前边界与工业对照**

- loader 依赖 `.q_proj`、`.down_proj` 等命名模式，模型实现与 checkpoint 布局紧耦合。
- 每个 TP rank 独立读取原始 checkpoint tensor，再取本地分片；常驻显存较小，但存在重复 I/O 和单 tensor 峰值。
- vLLM/SGLang 还需要支持分片 checkpoint、远端存储、量化元数据、专家局部加载、LoRA 热加载和多种权重格式。
- 阅读新模型时应先以 `qwen3.py` 和 `qwen3_moe.py` 为稳定基线，再研究 `qwen3_5.py` 中更复杂的 attention/模型状态。

**实验**

对 Qwen3-0.6B 打印 checkpoint 原始 shape、TP1/TP2 runtime shape 和每 rank 参数量。验证 Q/K/V、Gate/Up、O/Down、embedding 的分片维度，并检查 tied LM Head 是否重复占用 runtime 参数。

## 3. P1：从模型执行进入服务系统

### 3.1 API、Tokenizer、ZMQ 和取消链路

**核心问题**

- FastAPI coroutine、tokenizer worker、scheduler rank 0、detokenizer worker 如何通过 uid 关联。
- ZMQ 中传输的是文本、token、控制消息还是 tensor。
- TP rank 间为何必须得到相同的请求顺序和 batch 状态。
- 客户端断开后 Abort 如何穿过各进程，并最终释放 page table 和 cache handle。
- 当 HTTP 输入速度高于模型处理速度时，系统在哪里排队，是否存在背压和队列上限。

**代码入口**

- [server/api_server.py](../python/minisgl/server/api_server.py)：异步请求、SSE、ack/event 和 abort。
- [tokenizer/server.py](../python/minisgl/tokenizer/server.py)：tokenizer/detokenizer worker 主循环。
- [tokenizer/tokenize.py](../python/minisgl/tokenizer/tokenize.py)：chat template 和 tokenization。
- [scheduler/io.py](../python/minisgl/scheduler/io.py)：rank 0 广播与结果返回。
- [message/](../python/minisgl/message)：跨进程消息类型和序列化。
- [utils/mp.py](../python/minisgl/utils/mp.py)：ZMQ queue 封装。

**当前边界与工业对照**

- 当前 OpenAI API 是兼容子集，不等于完整协议实现。
- 没有认证、租户配额、请求优先级、显式 admission control、速率限制或完善的 queue backpressure。
- 工业服务通常需要把 API server、router 和 model worker 分离，并依据请求成本、KV cache affinity、队列长度和故障状态路由。
- vLLM 的 DP 部署明确要求各副本独立 KV cache，并建议路由考虑 queue 和 prefix-cache 状态。[vLLM Data Parallel Deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)

**实验**

同时发起短请求、长请求和一个随后立即断开的流式请求。跟踪同一 uid 在 API、tokenizer、scheduler、detokenizer 的日志，确认 abort 后 table slot 和 KV page 可再次分配。

### 3.2 CUDA stream、event 和显存生命周期

**核心问题**

- Scheduler stream 与 Engine stream 分别负责什么，`wait_stream` 建立了哪条依赖。
- CPU metadata、pinned memory、non-blocking H2D 和 GPU forward 如何重叠。
- token D2H copy 为什么需要 CUDA event，CPU 在何处等待。
- FlashInfer plan 为什么在复用 pinned staging buffer 前等待上一次 event。
- 模型权重、KV pool、page table、graph buffers、logits、attention workspace 和 MoE scratch 各占多少显存。

**代码入口**

- [scheduler/scheduler.py](../python/minisgl/scheduler/scheduler.py)：双 stream overlap 和 result processing。
- [engine/engine.py](../python/minisgl/engine/engine.py)：Engine stream、forward 和 D2H event。
- [engine/graph.py](../python/minisgl/engine/graph.py)：固定 graph buffer 和内存池。
- [attention/fi.py](../python/minisgl/attention/fi.py)：workspace、pinned metadata 和 plan event。

**当前边界与工业对照**

- mini-SGLang 展示了 CPU 调度与 GPU forward 的一层 overlap，但没有抢占、异步 KV 传输、跨节点通信 overlap 或多级执行 pipeline。
- 工业框架需要处理多个异步结果同时在途、请求被取消或抢占后旧结果失效，以及通信/计算/copy stream 的精确生命周期。
- overlap 的收益取决于 CPU gap 是否足够大；它不会减少模型 FLOPs。

**实验**

对同一 workload 分别设置和取消 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`，用 Nsight Systems 比较 GPU kernel 间空隙、CPU metadata 时间、H2D/D2H overlap、TTFT 和 TPOT。避免只比较总吞吐。

### 3.3 自定义 kernel 与扩展机制

**核心问题**

- 哪些操作值得自写 kernel，哪些应复用 PyTorch、FlashAttention 或 FlashInfer。
- TVM-FFI 的 AOT 和 JIT 路径如何注册 Python 可调用函数。
- JIT cache key 如何由 element size、线程数和模板参数构成。
- kernel 如何继承当前 CUDA stream，错误的 stream 或生命周期会导致什么问题。
- 数值正确、越界正确和性能正确应如何分别验证。

**代码入口**

- [kernel/utils.py](../python/minisgl/kernel/utils.py)：TVM-FFI AOT/JIT 构建。
- [kernel/index.py](../python/minisgl/kernel/index.py)：vocab-parallel embedding indexing。
- [kernel/store.py](../python/minisgl/kernel/store.py)：KV scatter store。
- [kernel/radix.py](../python/minisgl/kernel/radix.py)：Radix key compare 的 C++ binding。
- [kernel/pynccl.py](../python/minisgl/kernel/pynccl.py)：自定义 NCCL wrapper。
- [kernel/triton/fused_moe.py](../python/minisgl/kernel/triton/fused_moe.py)：Triton grouped MoE GEMM。

**当前边界与工业对照**

- mini-SGLang 的自定义 kernel 数量少，适合完整追踪 Python 参数到 CUDA 指针。
- vLLM/SGLang 的 kernel 选择还受 dtype、quantization group、GPU compute capability、shape、CUDA Graph、TP/EP 和 backend availability 影响。
- 工业 kernel registry 的核心不是“实现更多 kernel”，而是正确选择、fallback、autotune 和组合测试。

**实验**

运行 `tests/kernel/`，并为 index/store 额外覆盖空输入、非对齐 token 数、不同 head_dim 和越界 index。正确性使用 PyTorch reference；性能使用 CUDA event，必须先 warmup 并排除首次 JIT 时间。

### 3.4 性能分析方法

**核心问题**

- TTFT、TPOT、ITL、E2E latency、request throughput 和 token throughput 分别描述什么。
- Prefill 是计算密集，decode 是权重/KV 访存和 launch/通信敏感，这一判断如何从 profile 得到。
- 并发、输入长度、输出长度、prefix sharing 和请求到达分布如何改变结果。
- warmup、CUDA Graph capture、JIT、模型下载为何不能进入稳态性能统计。
- 平均值为何不足，p50/p95/p99 分别揭示什么。

**代码入口**

- [benchmark/client.py](../python/minisgl/benchmark/client.py)：异步 workload 和逐 token 时间戳。
- [benchmark/perf.py](../python/minisgl/benchmark/perf.py)：CUDA event microbenchmark。
- [utils/torch_utils.py](../python/minisgl/utils/torch_utils.py)：NVTX 标注。

**当前边界与工业对照**

- mini-SGLang 有基础 benchmark 工具，但没有生产 metrics endpoint。
- vLLM 暴露 queue time、TTFT、decode time、preemption、KV usage 和 prefix hit 等指标；这些指标用于解释性能，而不仅是展示吞吐。[vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/)
- 完整 SGLang 支持 Prometheus、请求 dump/replay 和 crash dump/replay。[SGLang Observability](https://docs.sglang.io/docs/advanced_features/observability)

**实验**

固定模型和硬件，做以下最小矩阵：

| 变量 | 取值 |
|---|---|
| 输入/输出长度 | 128/128、2048/128、128/1024 |
| 并发 | 1、8、64 |
| prefix | 无共享、50% 共享、长公共前缀 |
| 优化 | graph on/off、overlap on/off、FA/FI |

每组报告 TTFT、TPOT、p99 E2E、输入/输出 token throughput、KV hit rate 和峰值显存。一次只解释一个变量。

## 4. P2：模型边界、正确性和失败状态

### 4.1 模型兼容与长上下文

**核心问题**

- `model_type`、`architectures`、head_dim、GQA、QK Norm、activation 和 RoPE scaling 如何共同确定模型语义。
- Llama、Qwen、Mistral checkpoint 名称相似时，哪些差异不能由通用 loader 自动推断。
- Llama 3 scaling、YaRN、滑动窗口、MLA 和混合 attention 分别需要哪些新状态。
- 长上下文的限制来自 RoPE 配置、KV 容量、attention kernel 还是 scheduler budget。

**代码入口**

- [models/config.py](../python/minisgl/models/config.py)：配置归一化。
- [models/register.py](../python/minisgl/models/register.py)：支持架构边界。
- [layers/rotary.py](../python/minisgl/layers/rotary.py)：位置编码实现。
- [models/](../python/minisgl/models)：各模型结构差异。

**当前边界与工业对照**

- 当前以 decoder-only dense/MoE MHA/GQA 模型为主；KV pool 也只实现 MHA layout。
- vLLM 支持 decoder、MoE、多模态、embedding、reward、Mamba 和 hybrid attention 等多种 runner/state；“支持 checkpoint”因此不仅是增加模型类。[vLLM 功能与模型范围](https://docs.vllm.ai/en/latest/)

**实验**

为每个支持架构选择一个小模型，与 Hugging Face eager 在相同 BF16/FP16、greedy 条件下比较前若干 token。再分别测试短上下文、接近训练长度和启用 rope scaling 的长上下文；不能只验证服务能启动。

### 4.2 正确性测试矩阵

**核心问题**

- 单 kernel parity 是否能代表端到端正确。
- eager/graph、FA/FI/TRTLLM、naive/radix、TP1/TP2、prefill/decode 的交叉状态如何覆盖。
- 随机采样和并发调度下，什么结果应完全相等，什么只应满足分布性质。
- abort、cache eviction、OOM 和进程异常后，资源不泄漏如何验证。

**当前测试入口**

- [tests/kernel/](../tests/kernel)：通信和基础 kernel。
- [tests/core/](../tests/core)：cache allocation 和 scheduler。
- [tests/models/](../tests/models)：当前 MoE 算子与端到端验证。

**当前边界与工业对照**

- 当前测试集中在 kernel 和少量核心结构，在线服务、backend parity、Graph、TP 和故障组合覆盖有限。
- 工业框架的大量代码用于维护功能组合的正确性，而不是增加新的 Transformer 算子。
- 建议将 Hugging Face eager 或已验证框架作为数值参考，将 token IDs、logits、KV 内容、page ownership 和资源计数分层比较。

**实验**

建立最小回归矩阵：

```text
phase:       prefill / extend / decode
cache:       naive / radix hit / radix miss / eviction
execution:   eager / cuda graph
attention:   fa / fi
parallel:    tp1 / tp2
termination: eos / length / abort
```

Greedy 比较 token IDs；算子级比较 logits 和 KV；每次请求结束后检查 table slot、free pages 和 radix lock/refcount。

### 4.3 资源不足和恢复语义

**核心问题**

- 当本轮所需 KV pages 多于 free pages 时，系统选择 eviction、等待、抢占还是失败。
- running request、waiting request 和 cached prefix 谁拥有资源，优先级如何定义。
- overlap scheduling 下已经在途的 forward 结果如何判定是否仍有效。
- rank、tokenizer 或 API 进程异常退出后，其他进程能否检测并恢复。

**代码入口**

- [scheduler/cache.py](../python/minisgl/scheduler/cache.py)：page allocation、cache 和 eviction 协调。
- [kvcache/naive_cache.py](../python/minisgl/kvcache/naive_cache.py)：无 eviction 的简单基线。
- [kvcache/radix_cache.py](../python/minisgl/kvcache/radix_cache.py)：prefix 节点、锁和 eviction。
- [scheduler/scheduler.py](../python/minisgl/scheduler/scheduler.py)：finished/abort 资源回收。

**当前边界与工业对照**

- 当前没有完整的请求抢占、CPU swap、recompute、优先级调度或 worker 容错。
- vLLM scheduler 包含 running/waiting、preemption 和 prefix cache reset 等更完整状态转换。[vLLM Scheduler API](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/)
- 生产系统还需要 admission control：在资源耗尽前拒绝、延迟或路由请求，而不是依赖 OOM。

**实验**

使用很小的 `--num-pages` 和多个长请求稳定触发压力。记录请求是否等待、prefix 是否被逐出、running request 是否失败、资源是否恢复。该实验应在独立服务进程中执行，并设置超时。

## 5. 工业级能力缺口

下表不是 mini-SGLang 的功能计划，而是理解 vLLM/SGLang 时需要补充的系统问题。

| 主题 | mini-SGLang 状态 | 工业框架增加的核心状态或约束 |
|---|---|---|
| 抢占与调度策略 | 未实现 | priority、fairness、preempt/recompute/swap、deadline、admission control |
| 分层 KV Cache | 未实现 | GPU/CPU/SSD/远端 KV、异步传输、cache coherence、跨实例复用 |
| 混合模型状态 | 未实现 | MLA、sliding window、Mamba state、不同 layer 的 cache group |
| 权重与 KV 量化 | 未实现 | scale/zero point、group size、量化 checkpoint、量化 GEMM/attention、精度回归 |
| 投机解码 | 未实现 | draft state、candidate tree/block、target verification、accept/reject、双 KV 状态 |
| 结构化输出 | 未实现 | grammar 状态机、per-request token mask、异步 grammar compile、采样联动 |
| LoRA serving | 未实现 | adapter cache、动态加载、batched multi-LoRA GEMM、租户隔离 |
| 多模态 | 未实现 | encoder 调度、图像/视频 cache、跨模态 position、encoder-decoder 资源分配 |
| DP/PP/EP/CP、多节点 | 未实现 | 多 process group、拓扑映射、负载均衡、stage/expert/token ownership |
| PD 分离 | 未实现 | prefill/decode 独立扩缩容、KV transfer、bootstrap、超时和失败清理 |
| 生产可观测性 | 未实现 | metrics、trace、request/crash dump、健康检查、SLO、容量告警 |
| 完整 API 和安全 | 有限实现 | validation、auth、quota、rate limit、usage、logprobs、tool/reasoning parser |

### 5.1 量化

量化不是简单地把权重保存为低精度。需要同时理解：

```text
checkpoint format
  -> quantization metadata
  -> loader dispatch
  -> quantized linear/MoE method
  -> hardware-specific kernel
  -> activation/KV dtype
  -> accuracy and performance regression
```

vLLM 支持 FP8、INT8、INT4、GPTQ/AWQ、GGUF、TorchAO 和量化 KV Cache等多种路径；每种路径的硬件和 kernel 覆盖不同。[vLLM Quantization](https://docs.vllm.ai/en/stable/features/quantization/index.html)

### 5.2 投机解码

投机解码由较便宜的 draft 过程一次提出多个候选 token，再由 target model 并行验证。关键指标是 accept length，而不是 draft model 自身速度。它新增 draft/target 双状态、验证 batch、接受/拒绝后的 KV 修正，并与 CUDA Graph 和 scheduler 产生新的组合约束。

完整 SGLang 支持 EAGLE、MTP、独立 draft model 和 n-gram 等路径。[SGLang Speculative Decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)；vLLM 也支持多种 speculative method。[vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)

### 5.3 PD 分离

Prefill 计算密集，decode 更依赖权重/KV 带宽和稳定的逐 token latency。PD 分离将两者部署为独立 worker，并把生成后的 KV 从 prefill worker 传给 decode worker。收益是独立扩缩容和减少长 prefill 对 decode 的干扰；代价是 KV 传输、路由、超时、失败清理和 cache locality。

完整 SGLang 使用 Mooncake、NIXL 等传输后端实现该链路。[SGLang PD Disaggregation](https://docs.sglang.io/docs/backend/pd_disaggregation)

### 5.4 生产服务

生产推理需要围绕 SLO 管理系统，而不是只保证 forward 正确。至少应观察：

- waiting/running request 数、queue time、TTFT、TPOT、ITL 和 p99 E2E。
- prefill/decode tokens、KV usage、prefix hit、eviction 和 preemption。
- 错误率、取消率、GPU/NCCL 错误、worker 健康和重启。
- 输入长度、输出长度、请求成本和租户配额。
- 可重放的请求/崩溃现场，同时避免记录敏感内容。

## 6. 推荐阅读顺序

### 阶段一：能够解释一个 token

顺序：Attention metadata → packed model forward → sampling/stop → detokenization → weight loading。

完成标准：给定一个 batch，能够写出每个主要 tensor 的 shape、device、owner、生命周期和下一次状态变化。

### 阶段二：能够定位正确性和性能问题

顺序：请求/ZMQ 链路 → CUDA stream/event → custom kernel → benchmark/profile → 回归矩阵。

完成标准：遇到输出错误、hang、OOM 或 latency spike 时，能够先判断问题属于控制面、资源状态、metadata、kernel、collective 还是输出层。

### 阶段三：能够阅读工业框架

顺序：preemption → hybrid/distributed KV → quantization → speculative/structured output → DP/PP/EP → PD 分离 → observability。

阅读 vLLM/SGLang 时，对每个功能固定回答四个问题：

1. 它解决什么瓶颈。
2. 它新增什么持久状态。
3. 它如何改变 scheduler 和资源所有权。
4. 它需要什么 kernel、通信和失败处理。

完成标准：不再按“功能数量”比较框架，而能按状态复杂度、性能瓶颈、正确性约束和运维成本比较。

## 7. 推荐资料

### 论文

- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)
- [NanoFlow: Towards Optimal Large Language Model Serving Throughput](https://arxiv.org/abs/2408.12757)

### 官方实现与文档

- [vLLM 文档](https://docs.vllm.ai/en/latest/)
- [vLLM Scheduler](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/)
- [vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/)
- [SGLang 文档](https://docs.sglang.io/)
- [SGLang Expert Parallelism](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/expert_parallelism.md)
- [SGLang Observability](https://docs.sglang.io/docs/advanced_features/observability)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)

## 8. 最小实践清单

按以下顺序完成四个实验即可形成较完整的系统认识：

1. **Backend parity**：FA/FI 在 prefill、prefix hit、decode 下比较 token、TTFT、TPOT 和显存。
2. **Overlap ablation**：用 Nsight 比较 overlap on/off 的 CPU gap、copy 和 kernel timeline。
3. **Failure test**：小 KV pool 下并发长请求，并插入 abort，检查 page/table/radix 状态回收。
4. **Industrial comparison**：在 vLLM 或完整 SGLang 上运行相同 workload，比较 metrics、抢占行为、API 完整度和 profile，而不是只比较 tokens/s。
