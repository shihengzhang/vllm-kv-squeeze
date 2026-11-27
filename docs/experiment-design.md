# 实验设计：精确注意力 vs 近似的块级 KV 淘汰

> 可证伪假设：**用单阶段精确注意力做块级淘汰，能在「释放显存 vs 质量」前沿上，严格优于 PagedEviction 的静态代理（‖V‖/‖K‖）和 KV-Compress 的观察窗口近似，代价只多 ~4% 计算。**

---

## 1. 核心设计原则：只变「打分函数」

三个方法共用**完全相同的块级淘汰机制**，唯一区别是「如何给 token/块打分」。这样任何前沿差异都只能归因于打分函数。

**共享机制（固定不变）：**
- 保留 sink（前 S token）+ 最近窗口（后 W token）；
- 中间 token 按「块」聚合分数（块分 = 块内 token 分的均值）；
- 按预算保留 top-K 块，其余整块释放（对齐 PagedAttention 块粒度）。

**唯一变量 = 三种打分函数（注意力的「保真度」从低到高）：**

| 方法 | 打分函数 | 注意力保真度 |
|---|---|---|
| **Proxy**（= PagedEviction） | `s_i = ‖V_i‖₂ / ‖K_i‖₂` | 0（完全不用 query） |
| **Window**（= KV-Compress/SnapKV） | `s_i = mean_{最后W个query} attention(q_j, k_i)` | 部分（只看 W 个 query） |
| **Exact**（本文） | `s_i = Σ_q attention(q, k_i)`（最后一层单阶段 Q@K^T） | 完整（所有 query） |

---

## 2. 评测设置

- **模型**：Qwen2.5-7B-Instruct（已下载）；可选补 Llama-3.1-8B-Instruct
- **引擎**：vLLM 0.19.0 + bf16，block size 16
- **上下文长度**：4K / 8K / 16K / 32K
- **淘汰预算**：25% / 50% / 75%（token 预算，转成块预算）
- **打分开销**：Proxy≈0，Window≈O(W)，Exact≈一层 Q@K^T（~4%）

## 3. 指标

| 维度 | 指标 |
|---|---|
| 显存 | 实际释放的块数 / GiB（不是理论 token 淘汰率） |
| 质量 | LongBench（主）；PG19 perplexity（辅） |
| 开销 | 单阶段打分带来的额外 prefill 时间（%） |

## 4. 关键图（论文的 Figure 1）

**「释放显存（GiB） vs 质量（LongBench 分 / ppl）」的帕累托前沿，三条曲线：**

```
质量 ↑
  │        ● Exact（本文，目标）
  │       ●
  │      ●  Window
  │     ●●
  │    ●●   Proxy
  └──────────────→ 释放显存
```

**成功判据**：Exact 曲线在 Proxy 和 Window 曲线**之上**（同显存下质量更高），且差距在多个长度/预算下一致。

## 5. 额外需要报的数（诚实项）

1. **Exact 的开销到底是多少**：测真实 prefill 时间增量（目标 <5%，否则「精确注意力便宜」的卖点不成立）；
2. **sink/window 消融**：S、W 大小对三条曲线的影响；
3. **block size 消融**：16 / 32 / 64 下结论是否稳健。

## 6. 实施步骤（复用已有资产）

1. ✅ 单阶段分数捕获：vllm-kv-squeeze 已实现（hook 最后一层 + Q@K^T）；
2. ✅ 块级淘汰 + 释放：BlockKV 已实现（live free_evicted_blocks）；
3. ➕ 新增：把打分函数抽象成三个可插拔实现（Proxy / Window / Exact）；
4. ➕ 新增：统一的「预算 → top-K 块 → 释放 → 评测」流水线；
5. ➕ 新增：LongBench 评测 + 前沿曲线绘图脚本。

## 7. 风险与诚实边界

- **主要风险**：Exact 可能**不**明显优于 Window（因为最后一层注意力和观察窗口高度相关）。若如此，结论降级为「Proxy 不够好，但 Exact 与 Window 等价」，gap 变弱——需要提前想好这种情况下的贡献定位（比如改为「Exact 在长上下文下比 Window 更稳」）；
- **定位**：系统/实证短文（workshop / HotInfra / arXiv preprint），非主会算法；
- **原创性边界**：块级淘汰本身不新（KV-Compress/PagedEviction），新的是「精确注意力 + 块级淘汰」的**正面受控对比**和「注意力保真度 ↔ 前沿」的关系刻画。

---

## 决策点

开始动手前，请确认两个选择：

1. **评测基准**：只用 LongBench（快、够用），还是加 PG19/RULER（更硬但更慢）？
2. **模型**：先只 Qwen2.5-7B（快），还是同时跑 Llama-3.1-8B（更普适）？
