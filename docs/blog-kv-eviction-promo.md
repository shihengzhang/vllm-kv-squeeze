# 你淘汰了 50% 的 KV，但显存几乎没省——PagedAttention 的块级陷阱

> 一个在多卡 GPU 上真机跑出来的反直觉结论，和一个叫 BlockKV 的解法。项目已开源：[vllm-kv-squeeze](https://github.com/shihengzhang/vllm-kv-squeeze)。

## 先说结论

我们给 7B 模型做 KV 缓存淘汰时，发现了一个很反直觉的现象：

**同样是「淘汰 50% 的 token」，H2O 只释放了 0 个块，而 BlockKV 释放了 12 个块。**

（真实 vLLM 引擎、真实注意力分数、960 token 上下文，实测数据）

也就是说——你辛辛苦苦做了一堆 token 级淘汰策略，结果显存**根本没省下来**。这不是 bug，是 PagedAttention 的机制决定的。

## 为什么：KV 是按「块」管理的，不是按 token

vLLM 的 PagedAttention 把 KV 缓存切成固定大小的块（默认 16 个 token 一块），分配和释放都以**块**为单位。一个块只有当里面**所有 token** 都被淘汰时，才能真正释放回空闲池。

问题来了：

- **H2O / SnapKV** 这类方法，保留的是散落在全文各处的「heavy hitter」（重要 token）。结果几乎每个块里都还沾着至少一个保留 token → **一个块都释放不了**。
- **StreamingLLM** 只留开头的 sink + 末尾的窗口，是**连续**的 → 中间整块整块地空出来 → 释放了一堆块，但中间的信息全丢了。

这就是那个著名的矛盾：**「散点保留」保质量但不省内存，「连续保留」省内存但伤质量。** 而 token 级的淘汰率，和块级的实际释放率，根本是两回事。

## 我们的解法：BlockKV

把选择单位从 **token 换成 block**：

1. 按块聚合重要性分数（块内 token 分数求和）；
2. 保留 sink + 窗口 + 分数最高的 **top-K 整块**（淘汰的都是整块，能真释放）；
3. 再加一个小比例的「token 救援」——从被淘汰的块里捞回少数极高分的零散 token（掩码保留），把块级粒度误伤的关键信息补回来。

结果就是：**同预算下释放 3 倍块数，perplexity 几乎不变**（3.343 → 3.385，差 1.3%）。

## 真机数据（GPU）

7B @ 8.4K 上下文：

| 策略 | 保留 token | 释放块数 | perplexity |
|---|---|---|---|
| 全量（不淘汰） | 663 | 0 | 3.157 |
| H2O | 331 (50%) | **3** | 3.343 |
| **BlockKV** | 318 (48%) | **9** | **3.385** |
| StreamingLLM | 68 (10%) | 36 | 3.432 |

外推到 128K 上下文：7B 的 KV 缓存是 **7.00 GiB**，50% 淘汰省 **~3.5 GiB**，decode 提速 **1.34×**。

更关键的是，我们把「单阶段分数捕获」也打通了——prefill 时 hook 最后一层，补算一层 `Q@K^T`（约 4% 开销，不需要二次前向），让这套东西能真正接进生产环境的 vLLM 引擎里，边推理边释放显存。

## 代码已经开源

- 仓库：[github.com/shihengzhang/vllm-kv-squeeze](https://github.com/shihengzhang/vllm-kv-squeeze)（Apache-2.0）
- 4 个策略：StreamingLLM / H2O / SnapKV / **BlockKV**
- 5 个可复现 demo + 真机集成文档
- 32 个单元测试全过

如果你也在做长上下文推理、或者被 KV 缓存的显存压得难受，欢迎 star、提 issue、或者直接跑一下 `examples/demo_blockkv.py` 看那个对比。

**省显存就是省卡，省卡就是省钱。**
