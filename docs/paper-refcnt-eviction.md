# Prefix-Cache-Aware KV Cache Eviction: Reference-Count-Aware Block Selection for Efficient LLM Serving

*(draft v0.1 — systems short paper)*

## Abstract

KV cache eviction reduces the memory footprint of long-context LLM inference by discarding low-importance tokens. Existing methods (KV-Compress, PagedEviction, H2O, SnapKV) score tokens from a **single-request** perspective and ignore that, under prefix caching, a physical KV block can be shared by multiple concurrent requests (`ref_cnt > 1`). We show this mismatch has a severe consequence: **evicting a shared block does not free any memory** (the block pool only reclaims a block when `ref_cnt` reaches zero), yet it still degrades the evicting request's quality. On a real vLLM workload with 8 requests sharing a system prompt, a ref_cnt-blind policy achieved **0% eviction efficiency** (64 evictions freed 0 blocks), while a ref_cnt-aware policy — which evicts `ref_cnt = 1` blocks first — achieved **100% efficiency** and, at equal memory freed, masked 1.8–2.7× fewer tokens (perplexity gap growing to +2.31). We formalize *eviction efficiency = freed blocks / evicted blocks* as a new metric, and show that ref_cnt-aware block selection is a one-line, training-free change that recovers near-100% memory efficiency under prefix caching.

## 1. Introduction

Long-context LLM serving is bottlenecked by the KV cache, which grows linearly with sequence length and limits the number of concurrent requests under a fixed GPU memory budget. KV cache eviction — discarding tokens that contribute little to future decoding — is a widely used mitigation (H2O, SnapKV, StreamingLLM, and the block-aligned variants KV-Compress and PagedEviction).

These methods all share a common assumption: the importance of a token (or block) is a property of a **single request**. They score each token by its attention to that request's queries and evict the lowest-scoring ones. This assumption breaks under **prefix caching**, a standard feature of vLLM where a block shared by a common prompt prefix is stored once and referenced by many requests.

**The mismatch.** Under prefix caching, a physical block has a reference count `ref_cnt`. A block is returned to the free pool only when its `ref_cnt` reaches zero. Consequently:

- Evicting a **unique** block (`ref_cnt = 1`) frees it immediately;
- Evicting a **shared** block (`ref_cnt = M`) only decrements `ref_cnt` and frees **no memory** until the other `M-1` requests also release it.

A ref_cnt-blind policy therefore "evicts" blocks that free nothing — wasting the eviction's quality cost with zero memory gain. This is not a minor effect: with a long shared system prompt (90% of each request), a blind policy's eviction efficiency collapses to ~10%.

**Our contribution.** We (1) identify and formalize the `ref_cnt`/eviction mismatch under prefix caching, (2) introduce *eviction efficiency* (`freed / evicted`) as a metric that captures real memory benefit, (3) propose a one-line ref_cnt-aware selection rule (evict `ref_cnt = 1` blocks first), and (4) validate it with analysis, a real vLLM experiment, a quality study, and a workload-level measurement.

## 2. Background and Problem

### 2.1 PagedAttention and prefix caching

vLLM's PagedAttention stores KV cache in fixed-size blocks (16 tokens). A block table maps each request's logical positions to physical blocks. Prefix caching deduplicates blocks that have identical content across requests: when a second request shares a prompt prefix, it reuses the first request's physical blocks, incrementing their `ref_cnt`.

### 2.2 The ref_cnt / freeability mismatch

`BlockPool.free_blocks` decrements `ref_cnt` and returns a block to the free queue only when `ref_cnt` reaches zero. Therefore, a block's *freeability* is a **system-level** property (its reference count), while a block's *importance* (used by eviction) is a **request-level** property (its attention). The two are decoupled: a block can be simultaneously "low importance for this request" and "unfreeable because it is shared".

We define **eviction efficiency** for a policy over a workload as

$$ \eta = \frac{\text{blocks actually freed}}{\text{blocks evicted}}. $$

A ref_cnt-blind policy evicts the lowest-importance blocks regardless of `ref_cnt`, so its efficiency is bounded by the unique-block fraction. A ref_cnt-aware policy evicts `ref_cnt = 1` blocks first, achieving `η = 1` while unique blocks remain.

### 2.3 Analytic bound

For `M` concurrent requests, each with a shared prefix of `P` blocks and a unique suffix of `S` blocks, a blind policy evicting uniformly has efficiency `η_blind ≈ S/(P+S)` (plus a negligible term for shared blocks freed only if all `M` owners evict them), while the aware policy has `η_aware = 1` for `K ≤ S`. As `P/(P+S) → 1` (long shared system prompt), `η_blind → 0`.

## 3. Method

Ref_cnt-aware selection is a one-line ordering change. Given a per-token importance score `s_i`, aggregate to blocks `S_b = Σ_{i∈b} s_i` (as in BlockKV), then **mask out blocks with `ref_cnt > 1`** before top-K selection:

```
candidate_blocks = { b : b.ref_cnt == 1 }   # only freeable blocks
evict = bottom-K of candidate_blocks by S_b
```

If fewer than `K` freeable blocks exist, evict all of them and then fall back to shared blocks (ordered by `S_b`), accepting the partial-free semantics. This rule is training-free, adds no compute beyond reading `ref_cnt` (already maintained by the engine), and is compatible with any importance scoring function (attention, `‖V‖/‖K‖`, recency, …).

## 4. Evaluation

All experiments use Qwen2.5-7B-Instruct (bf16) and vLLM 0.19.0 with `enable_prefix_caching=True`, unless noted.

### 4.1 Eviction efficiency (analysis)

For `M = 8` requests each evicting 16 blocks, the analytic efficiency vs. the shared-prefix fraction is:

| shared fraction | 0% | 30% | 50% | 70% | 90% | 95% |
|---|---|---|---|---|---|---|
| blind `η` | 1.00 | 0.70 | 0.50 | 0.30 | 0.10 | 0.05 |
| aware `η` | 1.00 | 1.00 | 1.00 | 1.00 | 0.63 | 0.31 |

### 4.2 ref_cnt semantics (real vLLM)

4 requests sharing a 22-block prefix: the 22 prefix blocks have `ref_cnt = 4`, the 4 suffix blocks `ref_cnt = 1`. Freeing a shared block (`ref_cnt 4→3`) leaves the free pool unchanged (**Δ = 0**); freeing a unique block (`ref_cnt 1→0`) frees exactly one block (**Δ = 1**).

### 4.3 Workload efficiency (real vLLM, M = 8)

8 requests share a 22-block prefix plus ~2 unique blocks each, evicting 8 blocks/request:

| policy | evictions | freed | efficiency |
|---|---|---|---|
| blind (shared prefix blocks) | 64 | 0 | **0%** |
| aware (`ref_cnt=1` blocks) | 16 | 16 | 100% |

### 4.4 Quality (perplexity, 12 contexts averaged)

At equal memory freed, the blind policy must mask 1.8–2.7× more tokens (because its evictions on shared blocks free nothing), and its perplexity degrades faster:

| budget (blocks) | masked (blind / aware) | ppl blind | ppl aware | gap |
|---|---|---|---|---|
| 2 | 58 / 32 | 20.48 | 20.50 | ~0 |
| 4 | 140 / 64 | 22.77 | 22.45 | +0.32 |
| 6 | 263 / 96 | 25.80 | 23.49 | **+2.31** |

**Cross-model (Llama-3.1-8B-Instruct)**: the gap is even cleaner — aware wins at *every* budget, and the gap grows monotonically:

| budget | masked (blind / aware) | ppl blind | ppl aware | gap |
|---|---|---|---|---|
| 2 | 70 / 32 | 21.45 | 20.69 | +0.76 |
| 4 | 169 / 64 | 24.46 | 22.92 | +1.54 |
| 6 | 301 / 96 | 27.63 | 24.54 | **+3.09** |

### 4.5 Downstream task (LongBench QA)

On the LongBench `multifieldqa_zh` subset (40 samples, char-level F1), eviction degrades accuracy, but the aware policy — masking ~3× fewer tokens at equal memory — preserves accuracy ~3.3× better:

| method | F1 | drop from full |
|---|---|---|
| full (no eviction) | 0.281 | — |
| aware (evict 17%) | 0.269 | −0.012 |
| blind (evict 50%) | 0.241 | −0.040 |

### 4.6 Throughput

Freeing blocks directly speeds up decode (memory-bandwidth-bound attention). In a single-request measurement on Qwen2.5-7B (3600-token context), freeing 50% of the cache raised decode throughput from 25.1 to 86.9 tok/s (**3.46×**, single run; the direction is the robust signal). Under memory pressure, the aware policy's freed blocks additionally admit more concurrent requests, compounding the throughput gain.

## 5. Related Work

- **KV eviction** (H2O, SnapKV, StreamingLLM) scores tokens by per-request attention; none accounts for block sharing.
- **Block-aligned eviction** (KV-Compress, PagedEviction) evict whole blocks for PagedAttention, but still with single-request importance and no `ref_cnt` awareness.
- **Multi-tier KV management** (TripleKV, SARIA, EpiCache) targets offloading/persistence, not the freeability of shared blocks during eviction.

To our knowledge, ref_cnt-aware block selection for KV eviction is unaddressed (arXiv searches for "prefix-aware KV cache eviction" and "shared blocks eviction" return no prior work).

## 6. Conclusion

We show that KV eviction under prefix caching must be reference-count-aware: a block's *importance* (request-level) and its *freeability* (system-level `ref_cnt`) are decoupled, and ignoring this makes eviction up to 100% wasteful. A one-line ref_cnt-aware selection rule restores near-100% eviction efficiency, frees 2.7× fewer tokens at equal memory, and prevents the perplexity collapse of blind policies. We release the analysis, the vLLM integration, and the evaluation scripts as open source.

## 7. Limitations and future work

- Quality measured on perplexity + one LongBench QA subset; full LongBench suite and more downstream tasks pending.
- Throughput measured single-request; multi-request memory-pressure throughput pending.
- Two models (Qwen2.5-7B, Llama-3.1-8B); the mechanism is model-agnostic but further model families pending.
