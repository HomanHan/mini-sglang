# Mini-SGLang 代码解析

## Questions

- [x] 推理循环/数据流动过程是怎么样的？见图
- [ ] 具体模型（qwen3）推理的具体逻辑实现？
  - [ ] KV Cache 是怎么做的？自己在transformer block里手搓一个？
  - [x] Prefill Decode 是怎么做的？在 scheduler 中组织好 batch 输入给 forward 即可，优先 Prefill，prefill 后的结果筛选后进入 DecodeManager 的 `running_reqs` 候选队列中
  - [ ] 不同架构的模型是怎么适配的？MOE 怎么写的？
  - [ ] 不同的 AttentionBackend 是如何介入的？metadata是什么？

- [ ] RadixAttention 是如何实现的？
- [ ] PagedAttention 是如何实现的？
- [ ] SGLang-Router？router 往往被用在多 GPU/多节点/多实例的场景下（例如部署了多个 qwen-3-8B 的集群中），通过 Cache-Aware Routing 等方法将请求发给前缀匹配最优的 worker，也为实现 PD 分离提供基础
- [ ] TP 是怎么做的？在 kv_heads 维度做 shard；TP 下不做 KV 跨卡聚合；每卡只用本地 KV 参与注意力
- [x] Prefix Caching 是什么？

## Data Flow

[![Process overview diagram](https://camo.githubusercontent.com/0c0e17b5d59e7c341f1c42f7d12a35af9112df1f3430faa644bf93aa61ff80e8/68747470733a2f2f6c6d7379732e6f72672f696d616765732f626c6f672f6d696e6973676c2f64657369676e2e64726177696f2e706e67)](https://camo.githubusercontent.com/0c0e17b5d59e7c341f1c42f7d12a35af9112df1f3430faa644bf93aa61ff80e8/68747470733a2f2f6c6d7379732e6f72672f696d616765732f626c6f672f6d696e6973676c2f64657369676e2e64726177696f2e706e67)

1. **User** sends a request to the **API Server**.
2. **API Server** forwards it to the **Tokenizer**.
3. **Tokenizer** converts text to tokens and sends them to the **Scheduler (Rank 0)**.
4. **Scheduler (Rank 0)** broadcasts the request to all other Schedulers (if using multiple GPUs).
5. **All Schedulers** schedule the request and trigger their local **Engine** to compute the next token.
6. **Scheduler (Rank 0)** collects the output token and sends it to the **Detokenizer**.
7. **Detokenizer** converts the token to text and sends it back to the **API Server**.
8. **API Server** streams the result back to the **User**.

## Prefix Caching

为了解决跨 Request 共享前缀 KV Cache 需要重复计算的问题，只针对 Prefill 阶段。四个场景：

- few-slot learning
- multi-turn chat
- self-consistency：一个解码策略，通过多条推理路径，最后选择一致性最高的结果（投票）
- Tree of thought/Chain of thought

在 vLLM 通过哈希 hash 实现，其实是 PagedAttention 的一种拓展，在请求进来之后不一味分配新 Block，而是先找一次 Cached Block。

- Prefix Caching Aware Routing：多实例（多次部署 vLLM 分担流量）场景下要找前缀匹配最多的实例

### 将 Prefill 和 Decode 混合

为什么？Prefill 是 Compute-bound，Decode 是 Memory-bound，两者一起可以实现互补（batch 增大对 Prefill 无效，对 Decode 有效）

- Selective Batching：可以考虑只在部分 Layer 混合计算（FFN，Projection…），在 Attention 部分切分后调混合注意力 Kernel（Batch-Prefill-Decode Kernel）
- PP 容易有 Bubble

## Chunked Prefill

切开 Prefill 请求，在一个 Batch 中混入 prefill 和 decode。

后续的 Prefill chunk 需要依然能跨 chunk 访问到已经处理好的 KV Cache（RadixAttention，PagedAttention），还会调用 Prefix Attention 算子

- Prefill 优先：vLLM、Orca（混合 batch，但 prefill 先）
- Decode 优先：fast-transformer
- stall-free：精细调整每个 Batch 中 Prefill 含量，保证 TBT

‼️去看代码

## Reference

https://github.com/sgl-project/mini-sglang/

https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme-CN.md

sglang 源码学习笔记（一）- Cache、Req与Scheduler - 进击的Bruce的文章 - 知乎
https://zhuanlan.zhihu.com/p/17186885141

https://arxiv.org/pdf/2501.01005