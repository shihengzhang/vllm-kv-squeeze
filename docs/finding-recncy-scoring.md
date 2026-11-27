# 块级 KV 淘汰打分函数的受控实验：一份诚实的负结果

**结论（论文级严谨性：2 模型 × 30 段上下文 × 配对 t 检验）**：早期观察到的「近因衰减加权打分最优」**不成立**，不可泛化。

## 实验设置

- 模型：Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct
- 上下文：中文维基百科 30 段（每段 2000 token + 128 token continuation）
- 受控变量：仅打分函数不同，块级淘汰机制（BlockKV，block=16，sink+window+top-K）完全一致
- 指标：perplexity（越低越好）+ 配对 t 检验

## 四种打分函数

| 名称 | 定义 | 对应方法 |
|---|---|---|
| proxy | `‖V_i‖/‖K_i‖`（query 无关） | PagedEviction |
| window | 最后 64 个 query 的注意力之和 | SnapKV / KV-Compress |
| exact | 所有 query 的注意力之和 | H2O 式累计注意力 |
| recency | 指数近因衰减加权注意力 | 本文假设 |

## 结果

**Qwen2.5-7B（配对 t 检验，n=30）**

| 对比 | mean diff | t | 显著性 |
|---|---|---|---|
| recency vs window | +0.108 | 1.94 | n.s.（边缘） |
| recency vs proxy | +0.591 | 2.31 | * |
| recency vs exact | +0.181 | 1.52 | n.s. |
| exact vs window | -0.073 | -0.53 | n.s. |

**Llama-3.1-8B（配对 t 检验，n=30）**

| 对比 | mean diff | t | 显著性 |
|---|---|---|---|
| recency vs window | +0.004 | 0.26 | n.s. |
| recency vs proxy | +0.063 | 0.98 | n.s. |
| recency vs exact | +0.066 | 1.00 | n.s. |
| exact vs window | -0.062 | -0.89 | n.s. |

## 三个诚实的结论

1. **「近因衰减加权最优」被推翻**：在 Qwen 上仅边缘显著（t=1.94），在 Llama 上完全不显著（t=0.26）；
2. **「注意力打分 > 静态代理」不跨模型成立**：Qwen 上显著（t=2.31），Llama 上消失（t=0.98）——是模型特定现象，不是普适规律；
3. **query 聚合方式（window/exact/recency）基本可互换**：三者在两模型上两两都不显著。

## 教训（留给下一个方向）

- **单模型、单语料的正结果必须警惕**：Qwen 单模型的「发现」在跨模型验证下消失；
- **配对 t 检验 + 跨模型是底线**，不是加分项——没有它，噪声会被当成发现；
- 这一轮的**受控实验脚本**（`examples/experiment_frontier.py`，4 打分 × 5 预算 × 30 段 × 配对检验）是可复用的资产，下个方向直接套用。

## 状态

此方向为**负结果**，不再投入。脚本与数据留档，供后续对照。
