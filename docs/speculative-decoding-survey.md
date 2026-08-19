# 投机解码研究报告：从精确采样到 DFlash 2

> 状态截至 2026-08-19。DFlash 2 当前公开材料是技术博客、模型权重与运行时代码，没有独立论文或 arXiv。本文将论文结论、作者实验和代码事实分开表述。“分布无损”指给定 target、采样变换和数值精度下分布一致，不表示浮点结果逐 bit 相同。

## 1. 核心结论

投机解码用廉价草拟模型（drafter）提出多个未来 token，再由目标模型（target）一次并行验证。它减少串行 target 解码步数，通常不减少总 FLOPs。

端到端收益由三个相互独立的因素决定：

1. **候选质量**：target token 是否进入候选集合。
2. **候选组织**：能否从候选集合选出条件一致的序列。
3. **系统效率**：draft、验证、采样、KV Cache 和调度开销是否低于节省的串行 target 计算。

这解释了该方向的演进：

- EAGLE 改善 drafter 与 target 的特征对齐；
- MTP 将多 token 预测内生到 target 训练；
- DFlash 将草拟从自回归改为整块并行；
- DSpark 用轻量顺序校正和置信度调度控制后缀质量与验证浪费；
- DFlash 2 在保持重计算并行的前提下，用局部卷积提高候选覆盖，用路径选择器提高离散序列一致性。

DFlash 2 的关键认识不是“再做一次 diffusion”，而是：

> DFlash 的正确 token 经常已经位于每个位置的 top-k 集合中。问题主要是如何选择一条一致路径，以及如何防止候选集合在块尾退化。

## 2. 精确投机采样

### 2.1 线性验证

设 target 分布为 $p$，drafter 分布为 $q$，候选长度为 $\gamma$。给定已确认前缀，drafter 依次提出
$y_1,\ldots,y_\gamma$。

常见运行时还保留一个尚未写入 target KV 的末 token $a$，称为 anchor。target 验证输入为：

```text
[a, y_1, y_2, ..., y_gamma]
```

logits 的对齐关系是：

```text
row(a)       -> 验证 y_1
row(y_i)     -> 验证 y_(i+1)
row(y_gamma) -> 产生 bonus token
```

因此验证 $\gamma$ 个候选通常要计算 $\gamma+1$ 个 query token。候选并非不需要计算，而是把多次串行 decode 合并为一次并行前向计算。

### 2.2 Greedy 验证

对每个位置比较：

$$
y_i \stackrel{?}{=} \arg\max_x p_i(x).
$$

从左到右提交最长相等前缀。首次不等时提交 target argmax；若全部相等，再提交最后一行 logits 的 bonus token。结果与 target greedy decoding 一致。

### 2.3 随机采样

若 $y_i\sim q_i$，其接受概率为：

$$
A_i(y_i)=\min\left(1,\frac{p_i(y_i)}{q_i(y_i)}\right).
$$

首次拒绝发生在位置 $i$ 时，从残差分布采样修正 token：

$$
p_i'(x)=
\operatorname{norm}\left([p_i(x)-q_i(x)]_+\right).
$$

若所有候选均被接受，则从 $p_{\gamma+1}$ 采样 bonus token。该修正拒绝采样保持每一步条件分布，因此完整生成序列的联合分布与直接 target 采样一致。[Leviathan et al., 2023](https://proceedings.mlr.press/v202/leviathan23a.html) 和 [Chen et al., 2023](https://arxiv.org/abs/2302.01318) 独立给出了这一结果。

首次拒绝后必须丢弃整个 draft 后缀：后续 $q_{i+1}$ 条件于已被拒绝的 $y_i$，其条件前缀已经失效。

实现必须保留 drafter 实际采样所用的 $q_i$。$p_i$ 是 target 在当前前缀和输出约束下的目标分布，$q_i$ 是实际 proposal；二者本来就不相同，但都必须包含运行时真正施加于各自的 temperature、top-k、top-p、grammar 等变换。不能用 raw logits、错误归一化的稀疏分布或另一套 proposal 代替。仅比较两个模型采样出的 token 是否相同，不能保证分布无损。

### 2.4 速度条件

若各位置条件接受率近似为常数 $\alpha$，一轮平均输出 token 数为：

$$
\mathbb{E}[L]
=\sum_{i=0}^{\gamma}\alpha^i
=\frac{1-\alpha^{\gamma+1}}{1-\alpha}.
$$

单位置的理论接受率为：

$$
\alpha=\sum_x\min(p(x),q(x))=1-\operatorname{TV}(p,q).
$$

这是简化模型。真实接受率随位置下降，且不同请求差异明显。更实用的端到端近似为：

$$
T_{\text{token}}
=
\frac{
T_{\text{draft}}+
T_{\text{verify}}(B,V)+
T_{\text{sample}}+
T_{\text{runtime}}
}{
\mathbb{E}[L]
},
$$

其中 $B$ 是请求数，$V$ 是总验证 token 数。高并发时 target 已接近 compute-bound，增加 $V$ 可能使验证成本近似线性增长。因此接受长度提高，不必然带来同比吞吐提升。

报告实验时必须区分：

| 指标 | 含义 |
|---|---|
| accepted draft tokens | 通过验证的 draft token，不含修正或 bonus |
| acceptance length | 常见实现中的每轮输出长度，通常包含 target bonus；必须核对定义 |
| emitted tokens per cycle | 一轮最终提交的 token 数 |
| throughput | 单位时间内完成的输出 token，受 batch 和调度影响 |
| per-user speed | 单请求 decode 速度，与 aggregate throughput 不等价 |

## 3. 研究脉络与设计空间

### 3.1 关键发展

| 时间 | 工作 | 核心推进 |
|---|---|---|
| 2018 | [Blockwise Parallel Decoding](https://arxiv.org/abs/1811.03115) | 多 offset head 并行预测未来 token |
| 2022 | [SpecDec](https://arxiv.org/abs/2203.16487) | 专门训练 drafter 与并行 verifier |
| 2022--2023 | [Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)、[Speculative Sampling](https://arxiv.org/abs/2302.01318) | 建立分布无损的修正拒绝采样 |
| 2023 | [SpecInfer](https://arxiv.org/abs/2305.09781)、[REST](https://arxiv.org/abs/2311.08252) | token tree 与检索式候选 |
| 2024-01 | [Medusa](https://arxiv.org/abs/2401.10774)、[EAGLE](https://arxiv.org/abs/2401.15077) | 多预测头与 feature autoregression |
| 2024-02 | [Sequoia](https://arxiv.org/abs/2402.12374) | 按接受率和硬件代价优化候选树 |
| 2024-04 | [MTP](https://arxiv.org/abs/2404.19737) | 将多未来 token 预测作为训练目标 |
| 2024-12 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 顺序 MTP module 作为原生 drafter |
| 2025-03 | [EAGLE-3](https://arxiv.org/abs/2503.01840) | 多层特征融合与直接 token 目标 |
| 2026-02 | [DFlash](https://arxiv.org/abs/2602.06036) | 一次并行前向生成候选块 |
| 2026-07 | [DSpark](https://arxiv.org/abs/2607.05147) | 半自回归校正与置信度调度 |
| 2026-08 | [DFlash 2](https://inco.ai/blog/dflash2/) | 路径选择器与动态局部卷积；技术发布 |

### 3.2 四个分类轴

不同工作应沿独立维度比较，不能混成单一方法列表。

| 维度 | 主要选项 |
|---|---|
| 信息来源 | 独立小 LM、target hidden、原生 MTP、检索/ngram、自投机 |
| 块内依赖 | 独立多头、feature autoregression、顺序 token correction、局部卷积、并行块建模 |
| 候选结构 | 线性块、静态树、动态树 |
| 验证预算 | 固定长度、按置信度动态长度、按硬件 cost profile 动态长度 |

Ragged layout 不是候选拓扑，而是批处理布局：不同请求可以选择不同验证长度，再通过 `indptr` 和 sequence metadata 紧凑打包。

### 3.3 正确性等级

| 表述 | 准确含义 |
|---|---|
| 分布无损 | 使用精确 proposal 和修正拒绝采样，输出联合分布等于 target |
| Greedy 一致 | temperature=0 时 token 序列等于 target argmax |
| 质量近似不变 | 任务指标接近，但输出分布已经改变 |
| Relaxed acceptance | 用阈值、typical acceptance 等扩大接受范围 |

论文中的最大加速不能直接横向比较。模型、硬件、batch、采样、候选宽度、offloading 和 baseline 任一项变化，都可能改变结果。

## 4. 自回归与原生 drafter

### 4.1 EAGLE：在特征空间自回归

EAGLE 预测 target LM head 前的 feature，并将向前错一位的已选 token embedding 一同输入：

```text
target features: f_1, ..., f_t
shifted tokens:  t_2, ..., t_(t+1)
                       |
                 lightweight drafter
                       |
                  predict f_(t+1)
                       |
                target LM head -> q
```

shifted token 显式给出上一步实际选择，消除“feature 相似但离散 token 不同”的不确定性。EAGLE-2 根据 path confidence 动态分配树节点；EAGLE-3 融合 target 多层 feature，取消 feature regression，直接优化 draft token，并在训练中回灌 drafter 自己的 latent 以减轻 exposure shift。

EAGLE 的优势是 target 对齐强；代价是多步 feature autoregression、树构造和 tree attention verification。

### 4.2 MTP：训练目标不等于推理加速

并行 MTP head 在共享 trunk 上直接预测多个未来位置：

$$
h_t\rightarrow
\{q(x_{t+1}),q(x_{t+2}),\ldots,q(x_{t+D})\}.
$$

其远端 head 没有显式看到前面实际选择的 token，可能产生块内不一致。DeepSeek-V3 改用顺序 MTP module：训练时使用 teacher-forced future token embedding，推理时使用前一 MTP 步实际提出的 token，因此保留条件链，但 draft latency 随深度增长。

MTP 只有接入 target verification 才能加速。直接提交多个 MTP token 会改变 target 解码结果。

## 5. DFlash：并行块草拟

### 5.1 基本结构

DFlash 用轻量 block diffusion drafter 一次预测整个候选块。target 多层 hidden 被融合并投影为 drafter 各层 K/V，使每个 draft layer 都能读取正式上下文：

```text
target hidden from selected layers
              |
         fuse / project
              |
       persistent draft K/V

[anchor, MASK, MASK, ...]
              |
   block diffusion backbone
              |
   logits for all positions
```

与自回归 drafter 相比，重型 draft backbone 只执行一次，$T_{\text{draft}}$ 不再随候选长度近似线性增长。

### 5.2 DFlash 的误差来源

“DFlash 各位置完全独立”并不准确。MASK hidden 可以通过非 causal attention 交互。缺少的是：位置 $t$ 没有显式条件于位置 $t-1$ 最终选中的离散 token。

可将误差拆为三层：

1. **Candidate coverage**：正确 token 是否在当前位置的 top-k 中。
2. **Path selection**：候选集合中存在正确 token 时，能否选出条件一致的路径。
3. **Suffix coverage**：越靠近块尾，正确 token 是否仍在候选集合中。

普通 DFlash 每个位置独立取 top-1，同时受到 selection error 和 suffix decay。增加 backbone 深度可缓解后者，但会损害并行 drafter 的延迟优势。

### 5.3 DSpark：半自回归校正

DSpark 先用 DFlash backbone 产生整块 hidden 和基础 logits，再用轻量顺序模块引入实际 token 前缀：

- Markov head 仅条件于前一个 token；
- RNN head 条件于完整块内前缀；
- confidence head 预测在此前 token 均被接受条件下，当前位置继续存活的概率。

Markov head 用低秩转移近似完整 $V\times V$ 矩阵：

$$
B(x_{k-1},\cdot)=W_1[x_{k-1}]W_2.
$$

DSpark 的系统贡献同样重要：调度器根据每请求置信度和硬件 step-time profile 选择验证长度。它同时优化候选质量与高并发下无效验证 token 的机会成本。

## 6. DFlash 2

官方名称为 **DFlash 2**，也常写作 DFlash-2。它不是 SGLang 的 Spec V2 runtime，也不是第二轮 diffusion。其结构是：

```text
DFlash backbone
    |
    +-- grouped dynamic local convolution -> 改善块尾候选覆盖
    |
    +-- target LM head top-k
             |
       candidate lattice
             |
       short path walk -> 改善离散路径选择
             |
       target linear verification
```

首批公开权重面向 Qwen3.8-27B 和 Muse Glimmer 30B，配置中的 block size 分别为 8 和 16。block 包含一个 anchor，因此实际提出 7 或 15 个 draft token。

### 6.1 候选路径选择器

对每个位置保留 top-$K$ 候选。公开 checkpoint 使用 $K=16$。给定前驱 token $a$、当前候选 $b$ 和当前位置 hidden $h_t$，转移分数为：

$$
S_t(a,b)
=U_t(b)
+\left\langle
A(a)\odot H(h_t),B(b)
\right\rangle.
$$

- $U_t(b)$：DFlash 原始 unary logit；
- $A(a),B(b)$：前驱和后继 token 的低秩 codebook；
- $H(h_t)$：上下文门控；
- 公开 checkpoint 的 selector rank 为 256。

第一个位置以前一轮 target 验证后的 anchor 为前驱；之后以前一位置实际选中的 token 为前驱。

这不是对完整 lattice 做全局 Viterbi。实现从 anchor 开始逐位置执行局部 greedy 或 sampling：

```text
prev_idx = 0  # 首位置使用 anchor 对应的等价 predecessor row
for position t:
    scores = lattice[t, prev_idx, :]
    cur_idx = argmax(scores) or sample(scores)
    token = candidate_ids[t, cur_idx]
    prev_idx = cur_idx
```

`lattice` 的 predecessor 维是上一位置候选的局部索引，不是词表 token ID。所有 $K\times K$ 相邻转移分数可以并行计算；首位置的 anchor 被展开为等价 predecessor row。最终 walk 仍沿长度方向顺序执行，但每步只在 $K$ 个候选中选择，不需要再次运行 backbone 或完整 LM head。

### 6.2 为什么 selector 有效

作者在 Qwen3-4B、GSM8K、七个 draft 位置的分析中报告：

- 第一个位置 Recall@1 为 `85.4%`，Recall@16 为 `99.5%`；
- 块尾 Recall@1 为 `72.9%`，Recall@16 仍为 `87.8%`；
- top-1 的平均 acceptance length 为 `4.27`；
- top-16 oracle path 的对应值为 `6.79`。

二者差距说明大量错误来自 selection，而不是候选集合完全缺失正确 token。selector 的作用是从已有候选中恢复局部一致性。

该 oracle 不是可实现性能。它用于诊断 selection gap 与 coverage gap；这是一种实验归因方法，不是严格的概率恒等式。

### 6.3 动态局部卷积

selector 无法恢复未进入 top-k 的 token。DFlash 2 在每个 drafter layer 的 attention 和 MLP 子层前后加入 grouped dynamic causal depthwise convolution。公开模型使用两个 tap：

$$
\operatorname{Conv}(x)_t
=k_{t,0}\odot x_t
+k_{t,1}\odot x_{t-1}.
$$

系数由静态 base kernel 与输入相关修正组成：

$$
k_{t,j}=k^{\text{base}}_j+\Delta_j(x_t).
$$

每 16 个 channel 共享一组动态修正。卷积的作用范围仅在当前候选块：

- 第一个待预测 slot 读取 anchor representation；
- 后续 slot 读取并行块内前一个 slot 的 hidden representation；
- 所有位置同时计算；
- 跨 block 不保留卷积状态；
- 不改变 target verification。

卷积发生在 selector 之前，输入仍是并行 MASK slot；它不读取 path walk 已选 token。显式离散前驱条件只存在于 selector。卷积提供局部连续表示的归纳偏置，让 attention 更集中于读取长上下文。

作者在五层 Qwen3-4B drafter 上报告：

- convolution 增加约 `3%` 参数和 `0.7%` draft--verify cycle latency；
- 其块尾 Recall@1 接近十五层 DFlash；
- 十层额外 Transformer layer 的 cycle latency 增量为 `15.2%`。

这一结果支持“suffix decay 主要是局部依赖建模不足”的假设，但证据目前来自作者实验，尚缺不同模型和领域上的独立验证。

### 6.4 并行性的边界

DFlash 2 保持并行，准确含义是：

- block backbone：一次并行前向；
- LM head：一次处理所有位置；
- top-k 与 $K\times K$ lattice：并行；
- local convolution：所有位置并行；
- path walk：长度方向存在短顺序依赖；
- target verification：一次线性 causal 前向。

所以它消除的是昂贵模型层的自回归循环，而不是所有顺序控制流。

### 6.5 采样仍然精确

selector 在 temperature $>0$ 时定义稀疏 proposal：

$$
q_t(b\mid a,h_t),\qquad
b\in\operatorname{TopK}(U_t).
$$

运行时保存实际路径每一步的 $q_t$。target 给出 $p_t$ 后，仍使用标准接受概率和 residual distribution。top-k 外的 proposal 概率为零，但 target residual 可以采样这些 token，因此最终 target 分布不受 selector 支持集限制。

greedy 模式直接执行逐位置 target argmax 比较；随机模式才使用 $q_t$ 和修正拒绝采样。不能把 greedy 路径作为 point-mass proposal 代入随机验证公式。DFlash 2 改变的是 proposal，不改变两类 verifier 各自的正确性条件。

### 6.6 训练目标尚未公开

截至本文日期，官方博客、权重和仓库公开了推理结构，但没有公开完整训练 loss、数据配方与训练流程。仅从 inference codebook 和 checkpoint 形状，不能判断 selector 的全部训练方式或参数约束。

[SpecForge PR #772](https://github.com/sgl-project/SpecForge/pull/772) 与 [vLLM Speculators PR #1006](https://github.com/vllm-project/speculators/pull/1006) 给出了社区实验方案，而非官方 recipe。其共同思路是保留 DFlash unary objective，再加入带权 selector $K$ 分类交叉熵；若 gold token 不在 unary top-k 中，只在训练 loss 的候选集合中注入 gold，服务路径和严格 Recall@K 仍使用真实 top-k。后一个 PR 明确将该目标标为实验性实现。

这一区分对应两个不同目标：

1. unary backbone 提高正确 token 的 Recall@K；
2. selector 在候选已覆盖时学习条件路径。

训练评估应分别报告真实 Recall@K、teacher-forced selector accuracy 和自回归选路后的 acceptance。只报告训练候选集上的 selector accuracy 会掩盖 coverage error 与 exposure shift。

## 7. 方法对比

| 方法 | 重型 draft 关键路径 | 离散块内条件 | 候选结构 | 动态验证预算 | 主要代价 |
|---|---|---|---|---|---|
| EAGLE-3 | 多步 feature autoregression | 显式使用已选 token | 动态树 | 通常固定树预算 | 多步 drafter、tree attention |
| 原生 MTP | 多个顺序 MTP module | 显式 | 线性 | 通常固定 | checkpoint 耦合、顺序 module |
| DFlash | 一次并行 backbone | 无已选 token 条件 | 线性 | 固定 | selection error、suffix decay |
| DSpark | 并行 backbone + 轻量顺序校正 | Markov/RNN | 线性 ragged | confidence scheduling | 顺序校正、校准与 cost profile |
| DFlash 2 | 并行 backbone/conv + 短 path walk | selector 中局部一阶条件 | 线性 | 当前固定 | top-k、codebook、walk、dense $q$ |

DFlash 2 提高固定候选块的质量，DSpark 还控制每请求的 verification budget；二者可以组合，不是简单的新旧替代。

## 8. 代码级运行链路

### 8.1 公开实现

主要一手代码入口：

- [z-lab PyTorch/Transformers 实现](https://github.com/z-lab/dflash/blob/main/dflash/model.py)
- [z-lab MLX 实现](https://github.com/z-lab/dflash/blob/main/dflash/model_mlx.py)
- [SGLang 模型与 selector](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/dflash.py)
- [SGLang draft/verify worker](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/dflash_worker_v2.py)
- [SGLang Triton walk kernel](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/speculative/dflash.py)
- [vLLM DFlash 2 PR](https://github.com/vllm-project/vllm/pull/52816)
- [llama.cpp DFlash 2 PR](https://github.com/ggml-org/llama.cpp/pull/27342)

### 8.2 一轮完整执行

```text
初始化：target prefill
   -> 捕获已提交前缀的指定层 hidden
   -> 投影并建立初始 draft context/KV

稳态第 n 轮：
1. 构造 draft block
   -> [pending anchor, MASK, ..., MASK]
   -> 为 block 预留 draft/target 临时 KV slots

2. DFlash 2 forward
   -> dynamic local convolution
   -> parallel block hidden

3. candidate selection
   -> target LM head over all draft positions
   -> per-position top-k
   -> K x K adjacent transition lattice
   -> greedy/sample path walk
   -> 保存 selected tokens 与 proposal q

4. target verification
   -> [anchor, selected_1, ..., selected_gamma]
   -> 普通 linear causal attention
   -> 捕获本轮 target hidden
   -> greedy comparison 或 rejection sampling

5. commit / 跨轮反馈
   -> 提交 anchor + accepted drafts 的 target KV
   -> 将已提交位置的 target hidden 投影并写入 draft context/KV
   -> 丢弃 rejected suffix
   -> correction/bonus 成为下一轮 pending anchor
```

DFlash 2 只提出一条路径，所以 target 使用普通 causal mask，不需要 tree attention。prefill 仅在初始化执行一次；稳态每轮只有第 4 步的一次 target forward，本轮提交的 hidden 供下一轮 drafter 使用。

### 8.3 KV Cache

target KV 与 draft KV 属于不同模型状态：

- target verification 会为整个 `[anchor, drafts...]` 写临时 KV；
- 逻辑序列只提交 anchor 和已接受 draft；
- rejected suffix 的 KV 不进入下一轮有效长度；
- correction/bonus 由 logits 得到，尚无 KV；
- drafter 还需把已提交位置的 target hidden 投影为后续 draft context。

历史 KV 不会先 `concat` 成连续 tensor。Radix Cache 决定前缀命中及已有 KV slot 的复用；attention kernel 通过 page table 或 `req_to_token` 映射读取这些 slot。当前验证块只提供新的 query 和临时 slot 地址。

SGLang Spec V2 为 overlap 预留额外 lookahead slot，并分别维护 committed length 与临时 block。这里的 “V2” 是运行时架构，不是 DFlash 2 算法。

### 8.4 Selector 的工程实现

SGLang 先构造：

```text
candidate_ids: [B, L, K]
unary_logits:  [B, L, K]
lattice:       [B, L, K, K]
```

随后每个请求由一个 Triton program 完成 path walk。每一行的 $K$ 个分数保存在寄存器中，长度方向循环位于单个 kernel 内，避免每个位置发射一次 kernel。

PyTorch 和 MLX 参考实现则在 Python 中逐位置计算实际前驱对应的一行分数。数学含义相同，但系统开销不同。讨论“DFlash 2 latency”时必须注明具体 runtime。

### 8.5 随机验证的数据布局

selector 的 proposal 只在 top-k 上非零。SGLang 当前路径将稀疏 $q$ scatter 到形如 `[B, gamma, vocab]` 的 dense buffer，再复用通用 rejection kernel。

这简化了实现，但产生两个潜在问题：

- 大词表下临时显存以及首次分配或扩容的全量清零成本增加；
- 稀疏 proposal 的理论优势没有完全传递到 verifier。

buffer 在稳态被复用，每轮只需清理 top-k support，成本为 $O(B\gamma K)$，不是 $O(B\gamma V)$。直接在 `(candidate_ids, q_values)` 上完成接受概率和 residual sampling，仍是值得研究的 kernel 方向。

### 8.6 TP、LM head 与 CUDA Graph

DFlash 2 复用 target embedding 和 LM head。TP 下每个 vocab shard 先计算 local top-k，再 all-gather `TP × K` 个候选并做 global top-k，避免聚合完整词表 logits。

selector 的 predecessor/successor codebook 需要任意 token 行，因此 SGLang 在每个 TP rank 复制它们。对词表 $V$、selector dimension $d$，两张表规模为 $2Vd$。大词表下其存储和显存带宽不能忽略。

例如 Qwen3.8 的实现形状为 $V=248{,}320,d=256$：两张 codebook 共约 1.27 亿个 scalar，BF16 存储约 243 MiB，并在每个 TP rank 复制。作者博客中的 selector 参数增量口径不能直接解释为部署显存增量。

公开 SGLang 实现还具有以下约束：

- FlashInfer radix top-k 是关键优化；回退到 `torch.topk` 会显著降速；
- dense target LM head 时，top-k、lattice 和 walk 可折入 draft CUDA Graph；
- 截至本文日期，量化 target LM head 支持仍在独立 PR 中，不能默认与 dense 路径等价；
- sampled path 需要把真实 $q$ 带到 verifier。

这些细节说明 DFlash 2 的低理论 FLOPs 不自动等于低 runtime latency。

## 9. 已有实验

### 9.1 证据等级

| 来源 | 可支持的结论 | 局限 |
|---|---|---|
| [DFlash 论文](https://arxiv.org/abs/2602.06036) | DFlash 基线结构与原始实验 | 不包含 DFlash 2 |
| [DFlash 2 技术博客](https://inco.ai/blog/dflash2/) | 算法、消融和作者结果 | 非同行评审；训练 recipe 未完整公开 |
| [公开 checkpoint](https://huggingface.co/collections/z-lab/dflash-2) | 模型结构和可部署配置 | 当前仅少量 target |
| SGLang/vLLM/llama.cpp 实现与 PR | kernel、KV、TP 和 runtime 行为 | 版本快速变化；支持组合有限 |
| 社区首日测试 | 初步可复现性 | 样本少、硬件和配置不统一 |

### 9.2 匹配训练实验

作者在 Qwen3.5-4B 上统一训练 DFlash、DSpark 和 DFlash 2，并使用 thinking、temperature `1.0`、top-p `0.95`、top-k `20`、presence penalty `1.5` 和无损 rejection sampling。

五个数据集的平均 acceptance length 为：

| 方法 | Mean |
|---|---:|
| MTP | 4.54 |
| DFlash | 4.92 |
| DSpark | 5.49 |
| DFlash 2 | **5.97** |

DFlash 2 相对 DFlash 增加 `1.05` token，约 `21%`；selector 与 convolution 合计增加约 `1.3%` draft--verify cycle latency。

这是当前唯一匹配训练设置的公开对照。但 acceptance length 包含 verifier token，不等于 accepted draft count，也不能单独推出吞吐提升。

### 9.3 Qwen3.8-27B SGLang 并发基准

[SGLang 合入 PR 的 benchmark](https://github.com/sgl-project/sglang/pull/35371)使用单 H200、FlashAttention 3、block size `8`（1 个 anchor + 7 个 draft token）、temperature `1.0`、top-p `0.95`、top-k `20`、`xhigh` reasoning 和最多 `4096` 个新 token。

平均 acceptance length：

| MTP | 社区 DSpark | DFlash 2 |
|---:|---:|---:|
| 4.28 | 3.62 | **4.80** |

GSM8K throughput 的代表结果：

| 并发 | Autoregressive | DFlash 2 | Speedup |
|---:|---:|---:|---:|
| 1 | 68.9 tok/s | 236.1 tok/s | 3.43× |
| 8 | 467.2 tok/s | 1328.7 tok/s | 2.84× |
| 32 | 1329.8 tok/s | 1922.5 tok/s | 1.45× |

加速比从单请求的 `3.43×` 降至并发 32 的 `1.45×`。高并发 target verification 更接近 compute-bound，额外 draft token 会与有效 token 争用算力。

### 9.4 当前不能得出的结论

现有证据不足以证明：

- DFlash 2 在所有模型上优于 EAGLE-3、MTP 或完整 DSpark 系统；
- selector/convolution 在长上下文、代码、开放式高温采样上均稳定有效；
- 公开吞吐倍数可迁移到不同 GPU、量化格式、TP 或在线流量；
- 新增组件只带来博客所述的参数增量，而没有额外 codebook 和 runtime buffer 成本。

截至本文日期，[SGLang PR #35371](https://github.com/sgl-project/sglang/pull/35371) 已合入 main，但晚于当时最新 tag `v0.5.17`；vLLM 与 llama.cpp 支持仍位于开放 PR。稳定 release、量化 LM head 和模型组合需要逐版本核对。

## 10. 后续实验设计

本文不执行本地实验。下面给出可证伪的验证方案。

### 10.1 正确性

| 实验 | 对照 | 判据 |
|---|---|---|
| Greedy parity | target 原生 decoding | token IDs 完全一致 |
| Sampling distribution | target 直接采样 | 大样本频率或 KL/TV 在统计误差内 |
| 首 token/中途/全接受 | 人工构造 $p,q$ | correction、bonus 和后缀丢弃正确 |
| 稀疏 proposal | dense $q$ rejection | 输出分布一致 |
| eager vs CUDA Graph | 相同 seed/config | token 与 proposal metadata 一致 |
| TP 与量化 LM head | TP=1 dense baseline | top-k、path 和输出一致 |

### 10.2 问题分解实验

**实验 A：selector oracle gap**

- 指标：每位置 Recall@1、Recall@K、oracle acceptance、真实 selector acceptance。
- 自变量：$K$、selector rank、领域、温度。
- 假设：若 Recall@K 高而真实接受率低，瓶颈是 selector；若 Recall@K 本身快速下降，瓶颈是 backbone coverage。
- 可证伪条件：扩大 $K$ 不提高 oracle acceptance。

**实验 B：local convolution**

- 对照：DFlash 5L、5L+static conv、5L+dynamic conv、15L。
- 指标：分位置 conditional acceptance、draft latency、参数、显存。
- 假设：卷积主要改善块尾，而非第一个位置。
- 可证伪条件：收益在所有位置均匀，或来自模型容量而非局部结构。

**实验 C：selector 与 convolution 交互**

- 四组：baseline、selector-only、conv-only、full DFlash 2。
- 指标：Recall@K、selection gap、acceptance length、cycle latency。
- 目标：验证两组件是否分别作用于 selection 与 coverage，而不是重复解决同一误差。

**实验 D：训练目标与 exposure shift**

- 对照：unary-only、unary + selector CE（多组 loss 权重）、冻结或联合训练 backbone。
- 指标：严格 Recall@K、teacher-forced selector accuracy、自回归 selector acceptance。
- 目标：区分 candidate coverage 与 conditional selection 的收益。
- 可证伪条件：teacher-forced accuracy 提高，但自回归 acceptance 不变，说明训练候选注入或 teacher forcing 未转化为服务收益。

### 10.3 系统实验

| 维度 | 建议取值 | 主要问题 |
|---|---|---|
| block size | 4、8、12、16 | 接受长度增长能否覆盖 verify 成本 |
| concurrency | 1、8、32、饱和点 | 收益何时因 compute-bound 消失 |
| context | 短、中、长 | draft KV 与 target feature 成本 |
| sampling | greedy、低温、高温 | proposal 支持与拒绝率 |
| top-k/selector dimension | 多组网格 | 质量、top-k latency、codebook 显存 |
| precision | BF16、FP8、INT4/8 | LM head/top-k/selector 兼容性 |
| parallelism | TP=1/2/4/8 | `TP × K` 通信与 codebook 复制 |
| workload | chat、code、math、agent trace | domain shift 和序列熵 |

至少报告 draft、top-k、lattice、walk、target verify、rejection 和 KV commit 的分项 latency。只报告平均 acceptance length 无法解释系统收益。

### 10.4 独立复现的最小标准

1. 固定 target、prompt 模板、采样参数、输出上限和随机种子。
2. 使用相同 runtime commit 比较 autoregressive、DFlash、DFlash 2。
3. 同时给出 correctness、acceptance、per-user speed 和 aggregate throughput。
4. 区分冷启动、prefill、decode 与稳态服务。
5. 公布未筛选请求结果和方差，不只报告最佳任务。

## 11. 讨论与开放问题

### 11.1 接受率应被分解

单一 acceptance rate 混合了 candidate coverage、path selection 和 verification truncation 三类因素。三者不是可直接相加的概率事件，而是需要分别测量的诊断维度。

Recall@K 衡量 candidate coverage；oracle path 衡量 selector 上限；分位置曲线衡量 suffix decay；调度器验证长度衡量系统截断。分解后才能判断应该增加 backbone、改 selector，还是减少 verify budget。

### 11.2 离散条件不一定需要重模型自回归

EAGLE、MTP 和 DSpark 都通过某种顺序状态表达已选 token。DFlash 2 表明，若并行 backbone 已产生高 recall 候选，离散条件可以被压缩为低秩局部转移和短 path walk。

这是一种计算重排：

```text
昂贵的 V 维自回归预测
        ->
一次 V 维并行打分 + 多次 K 维局部选择
```

当 $K\ll V$ 时该重排有明显潜力。

### 11.3 Suffix decay 可能是结构问题

五层加局部卷积接近十五层 baseline，提示块尾退化未必要求更大模型。局部依赖有明确结构时，合适的归纳偏置可能比通用 Transformer 容量更有效。

需要进一步验证这一结论是否适用于：

- 长距离代码依赖；
- 多 token 数学表达式；
- 高熵开放式生成；
- 不同 tokenizer 粒度。

### 11.4 下一瓶颈是 verification budget

当 drafter 足够快且候选质量提高后，target 验证会成为主要成本。并发越高，这一问题越突出。未来方法需要联合优化：

$$
\text{proposal quality}
\times
\text{verification allocation}
\times
\text{runtime shape}.
$$

DFlash 2 提高第一项 proposal quality，但没有原生解决按请求动态验证长度。与 DSpark confidence scheduling、ragged CUDA Graph 和负载感知 cost model 结合，是直接方向。

### 11.5 稀疏 proposal 尚未被充分利用

selector 天然输出 top-k 稀疏 $q$，但当前部分实现将其展开为全词表 dense buffer。更合理的 verifier 可以：

- 直接计算 selected token 的 $q(y)$；
- 以稀疏修正表示 residual；
- 将 target top-k 与 proposal support 融合；
- 避免 `[B,L,V]` buffer 的初始化和清零。

难点是 residual distribution 仍可能在完整 target 词表上有质量，因此需要精确、可并行的补分布采样。

### 11.6 开放问题

- selector 能否使用更长依赖而不恢复昂贵自回归？
- 局部 greedy walk 与全局路径搜索的收益/延迟边界在哪里？
- top-k 能否按位置或请求动态变化？
- confidence 是否可以同时预测 candidate coverage 和 expected system gain？
- codebook 能否低比特量化、分片或按频率缓存？
- drafter 能否与 target 联合训练，使 Recall@K 而非 cross-entropy 成为直接目标？
- grammar、structured output 和 tool-call 约束应在候选生成还是验证阶段注入？

## 12. 资料与阅读顺序

### 12.1 核心论文

- [Speculative Decoding 综合综述](https://aclanthology.org/2024.findings-acl.456/)
- [Speculative Decoding and Beyond: An In-depth Survey](https://arxiv.org/abs/2502.19732)
- [COLING 2025 Speculative Decoding Tutorial](https://speculative-decoding.github.io/)
- [Google Research：Looking Back at Speculative Decoding](https://research.google/blog/looking-back-at-speculative-decoding/)
- [Hugging Face assisted generation 文档](https://huggingface.co/docs/transformers/main/en/generation_strategies#speculative-decoding)
- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [DistillSpec](https://arxiv.org/abs/2310.08461)、[MagicDec](https://arxiv.org/abs/2408.11049)
- [EAGLE-1](https://arxiv.org/abs/2401.15077)、[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840)
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DFlash](https://arxiv.org/abs/2602.06036)
- [DSpark](https://arxiv.org/abs/2607.05147)

### 12.2 DFlash 2 一手资料

- [DFlash 2: Keep Drafting Parallel](https://inco.ai/blog/dflash2/)
- [DFlash 官方仓库](https://github.com/z-lab/dflash)
- [DFlash 2 checkpoint collection](https://huggingface.co/collections/z-lab/dflash-2)
- [Qwen3.8-27B DFlash 2 model card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
- [Muse Glimmer DFlash 2 model card](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2)
- [SGLang DFlash 2 PR #35371](https://github.com/sgl-project/sglang/pull/35371)
- [vLLM DFlash 2 PR #52816](https://github.com/vllm-project/vllm/pull/52816)
- [llama.cpp DFlash 2 PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)
- [SpecForge DFlash 2 training PR #772](https://github.com/sgl-project/SpecForge/pull/772)
- [vLLM Speculators experimental training PR #1006](https://github.com/vllm-project/speculators/pull/1006)
- [DeepSpec：DFlash、DSpark、EAGLE-3 训练与评测](https://github.com/deepseek-ai/DeepSpec)
- [SpecForge：speculator 训练框架](https://github.com/sgl-project/SpecForge)

### 12.3 工业框架与可运行实现

- [EAGLE 官方实现](https://github.com/SafeAILab/EAGLE)
- [DeepSeek-V3 权重结构说明](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md)
- [SGLang speculative decoding 文档](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
- [SGLang speculative runtime 源码](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/speculative)
- [SGLang DSpark 系统实现说明](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/)
- [vLLM speculative decoding 文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [vLLM rejection sampler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py)
- [vLLM Speculators](https://docs.vllm.ai/projects/speculators/en/latest/)
- [TensorRT-LLM speculative decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md)
- [REST](https://github.com/FasterDecoding/REST)、[Lookahead Decoding](https://github.com/hao-ai-lab/LookaheadDecoding)、[教学版实现](https://github.com/romsto/Speculative-Decoding)

### 12.4 建议阅读顺序

1. 精确 speculative sampling：理解 proposal、接受概率和 residual。
2. EAGLE/MTP：理解离散 token 条件为什么重要。
3. DFlash：理解并行 block drafter 与 target feature injection。
4. DFlash 2 博客和 `model.py`：理解 selector 与 convolution。
5. SGLang worker/kernel：理解 KV、TP、CUDA Graph 和 rejection 的真实代价。

阅读任意实现时应回答五个问题：

1. 候选由什么信息产生？
2. 块内实际选中 token 如何影响后续候选？
3. target 如何验证并恢复精确分布？
4. KV 如何提交和回滚？
5. 在什么 batch、硬件和采样条件下端到端收益为正？
