# BlockKV: Block-Aware KV-Cache Eviction

**Status:** method + runnable prototype (this repo). Not yet a submission — this
documents the novel method and the experiment plan needed to turn it into a
systems paper.

---

## 1. Problem

Long-context LLM serving is bottlenecked by the KV cache. A line of work
(StreamingLLM, H2O, SnapKV) evicts "unimportant" tokens to shrink it. These
methods all make a **token-level** decision: keep the top-K most important
tokens, drop the rest.

The problem, made concrete by our real-hardware run on vLLM's PagedAttention:

> **Token-level eviction ratio ≠ block-level memory freed.**

PagedAttention allocates and frees KV in fixed-size **blocks** (default 16
tokens). A block is only returned to the pool when *every* token in it is
evicted. H2O/SnapKV keep scattered "heavy hitters", so almost every block still
holds at least one kept token — few blocks are ever fully freed. In our 7B run,
H2O evicts 50% of tokens yet frees only **3 of 42 blocks** (7% of memory), while
StreamingLLM — which keeps a *contiguous* sink+window — frees **36 blocks** but
loses the whole middle.

So the two goals are in tension at the token level: scattered selection buys
quality but not memory; contiguous selection buys memory but not quality.

## 2. Method

**BlockKV** resolves the tension by moving the selection unit from *token* to
*block*, then patching the quality gap with a small token-level "rescue".

Inputs: per-token importance `s ∈ R^T` (e.g. H2O cumulative attention), token
budget `B`, block size `L`, sink `s₀`, window `w`, rescue ratio `r`.

1. **Always keep** the attention sink `[0, s₀)` and the recent window
   `[T-w, T)`.
2. **Aggregate** the middle region `[s₀, T-w)` into whole blocks
   `[bL, (b+1)L)`; score each block as the sum of its token scores:
   `S_b = Σ_{i∈block b} s_i`.
3. **Keep the top-K blocks** by `S_b`, where
   `K = (B - s₀ - w - rB) / L` (block-level budget). The kept middle is now
   *block-structured* → the evicted middle consists of whole blocks that can be
   freed.
4. **Rescue** (optional): among the *evicted* blocks, keep the top-`rB`
   individual tokens (e.g. `r = 0.1`). These stay as masked partial blocks —
   they cost little memory but recover quality lost at block granularity.

The key distinction from H2O/SnapKV: the bulk of the budget is spent on
**whole blocks** (freeable memory), and only a small `r` fraction is spent on
scattered tokens (quality). When `r = 0` it degenerates to pure block-level
top-K; when `L = 1` it degenerates to H2O. In between, it traces the
memory↔quality frontier.

**Why the rescue matters.** A single crucial token in a low-scoring block
(e.g. a name, a number, a cross-reference) is exactly what block-level top-K
drops. Rescuing the top-`rB` such tokens recovers most of the quality loss at a
memory cost proportional to `r`, not to `L`.

## 3. Prototype result (7B, single context)

Qwen2.5-7B-Instruct, context 663 tokens, budget 50%, block size 16, window 64.
Perplexity on a 36-token held-out continuation (masked attention = Level-1
eviction):

| Policy | kept tokens | freed blocks | freed mem | perplexity |
|---|---|---|---|---|
| full (no eviction) | 663 | 0 | — | 3.157 |
| H2O | 331 | 3 | 2.6 MiB | 3.343 |
| **BlockKV** (r=0.1) | 318 | **9** | **7.9 MiB** | **3.385** |
| StreamingLLM | 68 | 36 | 31.5 MiB | 3.432 |

**BlockKV frees 3× the blocks of H2O at the same token budget, at a 1.3%
perplexity cost** — the memory↔quality frontier improved. StreamingLLM frees the
most but degrades quality the most.

## 4. What a paper still needs

**Novelty is the method (Section 2), not the policies.** The experiment must
show BlockKV dominates the frontier.

- **Models:** Qwen2.5 (0.5B/7B/14B/32B), Llama-3.1-8B/70B.
- **Lengths:** 4K / 16K / 32K / 128K.
- **Quality:** LongBench, RULER, ∞Bench, PG19 perplexity, multi-doc QA.
- **Baselines:** Full KV, StreamingLLM, H2O, SnapKV, PyramidKV, InfLLM,
  random-block baseline (ablates "block scoring" vs "random blocks").
- **Systems:** peak GPU memory, decode latency, max batch under a memory cap;
  integrate `free_evicted_blocks` into a live vLLM loop (not just planning).
- **Ablations:** block size L, rescue ratio r, sum vs max vs mean block scoring,
  sink/window sizes.
- **Core figure:** *freed memory vs perplexity* (or LongBench score) — BlockKV
  should sit strictly above the H2O/SnapKV curve.

**Target venue:** systems for ML (MLSys workshop, HotInfra, EuroSys'ML /
  similar) — the contribution is memory management, not model capability.

## 5. Implementation in this repo

- `src/vllm_kv_squeeze/policy/blockkv.py` — `BlockKVPolicy` (pure, tested).
- `examples/demo_blockkv.py` — the frontier prototype (freed blocks + perplexity).
- `tests/test_blockkv.py` — unit tests incl. "frees more blocks than H2O".
