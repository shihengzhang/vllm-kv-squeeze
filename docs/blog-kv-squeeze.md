# 给 vLLM 的 KV Cache 装个"遗忘开关"：vllm-kv-squeeze

> 长上下文推理里，KV Cache 是第一成本。而大部分 KV，模型其实根本没在"看"。

自回归生成有一个众所周知的矛盾：**预填充是算力密集，解码是访存密集**。每生成一个 token，模型都要把整段历史的 Key/Value 从显存里读出来、重算一遍注意力。当上下文到 32K、128K 甚至 1M 时，KV Cache 的显存占用和带宽压力会先于算力把你卡死。

我们做了一个叫 [vllm-kv-squeeze](https://github.com/shihengzhang/vllm-kv-squeeze) 的小项目：把几种"KV 淘汰"策略做成干净、可插拔的库，目标是在不重训模型的前提下，把长上下文的显存和延迟压下来。这篇文章讲清楚三件事：**为什么能省、为什么难做、我们怎么设计的**。

## 一、反直觉的发现：注意力大部分是浪费的

如果你把预填充阶段的注意力权重累加起来，会看到两个稳定的现象：

1. **Attention sink（注意力汇）**：开头那几个 token（哪怕是 BOS）会吃掉异常多的注意力。这不是语义重要，而是 softmax 需要"泄压"——把多余的概率质量堆到少数几个位置上。丢掉它们，输出立刻崩。
2. **Heavy hitter（重击者）**：中间绝大多数 token 的累计注意力几乎为 0，真正被反复读的只有一小撮。**丢掉中间的"沉默的大多数"，对生成质量的影响小得惊人。**

基于这两点，学术界给了几套成熟的淘汰策略：

| 策略 | 思路 | 要不要评分 |
|---|---|---|
| **StreamingLLM** | 只留 sink + 最近滑动窗口 | 不要 |
| **H2O** | sink + 窗口 + 中间累计注意力最高的"重击者" | 要 |
| **SnapKV** | 对重要性做局部 max-pool（聚类）再选 top-K | 要 |

实验上，把 KV 砍到 50% 甚至 20%，在很多长文本任务上的 perplexity 几乎不动。这就是"能省"的依据。

## 二、但这件事在 vLLM 里特别难做

这些论文的原版实现几乎都跑在 Hugging Face `transformers` 上，因为那里 `past_key_values` 是一个普通张量，想删哪个 token 直接切片就完事。

**vLLM 不行。** 它用的是 **PagedAttention**：KV 被切成固定大小的 block（默认 16 个 token），每个序列持有一张"逻辑块 → 物理块"的映射表（block table），注意力是在 block 上算的。这带来两个硬约束：

1. **粒度是 block，不是 token**。你不能"删掉第 128 个 token 的 KV"——只能决定"这个 block 还要不要"。而一个 block 里通常既混着要留的 token 也混着要丢的 token，删错了就丢信息。
2. **要省显存，必须把 block 归还到池子里**。只是把注意力 mask 掉（Level 1）能省算力，但显存一分不动；真正省显存，得改 block table、把 free 的 block 交还给 `KVCacheManager`（Level 2）。

更要命的是，H2O/SnapKV 这类方法是**两遍的**：得先跑一次前向拿到注意力分数，才知道该删谁，然后再继续解码。把它插进 vLLM 高度流水线化的 prefill 里，不是改一行代码的事。

官方至今没有原生的 KV 淘汰 API，社区里只有 SnapKV-vLLM、PyramidKV 这些容易跟丢上游的 fork。**这就是缺口，也是我们做这个项目的理由。**

## 三、我们的设计：把"算法"和"工程"拆开

核心原则一句话：**策略层要纯、要可测；vLLM 集成要薄、要可验证。**

```
src/vllm_kv_squeeze/
├── policy/              # 纯 PyTorch，无状态，不依赖 vLLM
│   ├── base.py          # EvictionPolicy / EvictionResult
│   ├── h2o.py
│   ├── snapkv.py
│   └── streaming_llm.py
├── hooks/
│   ├── v1.py            # vLLM V1 集成脚手架 + 掩码 helper
│   └── block_eviction.py  # 纯逻辑：token 决策 → block 释放计划
└── utils/
    └── budget.py        # 比例/绝对值预算
```

### 策略层：一个干净的最小接口

```python
from vllm_kv_squeeze import H2OPolicy, resolve_budget

seq_len = 2048
scores = torch.rand(seq_len)            # 每 token 的重要性（注意力累计）
budget = resolve_budget(seq_len, 0.5)   # 保留 50%

policy = H2OPolicy(sink_size=4, window_size=256)
result = policy(scores, seq_len, budget)
print(result.kept, result.evicted)      # 1024 1024
```

所有策略都是**无状态、纯函数**，`select` 不修改输入。这让单元测试能精确断言"保留了哪些、淘汰了哪些"，也是整个库的质量底线（现在 16+ 个测试全绿）。

### 块级规划器：Level 2 的核心

token 级决策出来了，怎么映射到 block？规则只有一条，但这条规则决定了正确性：

> **一个 block 只有当它覆盖的所有 token 都被淘汰时，才能被释放；只要有一个 token 要留，整个 block 就得留**（其中被淘汰的 token 用掩码跳过）。

```python
from vllm_kv_squeeze import plan_block_eviction

plan = plan_block_eviction(result, block_size=16, block_ids=block_table)
print(plan.kept_block_ids)   # 新的、更短的 block table
print(plan.freed_block_ids)  # 归还给 KVCacheManager 的块
```

这段逻辑完全不含 vLLM import，能独立单测——块内部分保留、末块不满、`block_size=1`、越界输入，这些边界情况都有测试兜底。

### vLLM 集成：薄薄一层，把 TODO 写得明明白白

真正动 vLLM 内部的那部分（改 `AttentionMetadata.block_tables`、调 `KVCacheManager.free_blocks`），我们没有硬编一套"自以为能跑"的代码，而是把它做成一个**带验证清单的脚手架**——因为 vLLM V1 的 API 每个版本都在漂，写死等于骗人。你对着本机版本核对三处，就能接通：

1. 用 `plan.freed_block_ids` 调 `free_blocks(...)` 归还显存；
2. 把序列的 block table 重写为 `plan.kept_block_ids`；
3. 同步修正 `slot_mapping` / `seq_lens`。

## 四、现状与下一步

**已经能用的**：三种策略 + 预算解析 + 掩码式淘汰（Level 1）+ 块级规划器（Level 2 的算法核心），全部带测试，CI 覆盖 Python 3.10/3.11/3.12。

**还差的两块硬骨头**：

1. **Level 2 的 vLLM 接线**——把规划器接到真实的 `KVCacheManager`，跑出"同显存下上下文长度 ×2 / batch ×2"的真数据。这是项目从"玩具"到"能用"的那一步。
2. **质量回归**——用 perplexity 和长文本任务证明"砍一半 KV，输出几乎不变"，这是别人敢在生产用的信任来源。

## 五、如果你也关心这件事

- 仓库：[vllm-kv-squeeze](https://github.com/shihengzhang/vllm-kv-squeeze)（Apache-2.0）
- 想一起做 Level 2 接线、补 benchmark、或者加新策略（比如 DMC、Quest），欢迎提 issue 或 PR。

> 推理加速的下半场，拼的不是再多叠一个 kernel，而是**搞清楚哪些计算根本不该做**。KV 淘汰是这个问题里最干净、也最被低估的一个切口。
