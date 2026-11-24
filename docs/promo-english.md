# English promotion copy

Ready-to-post copy for the three channels. The hook in every case is the same
counterintuitive finding: **you evicted 50% of the KV cache and freed almost no
memory.**

---

## 1. Hacker News — "Show HN"

**Title:** Show HN: vllm-kv-squeeze — KV-cache eviction for vLLM that actually frees memory

We found something counterintuitive while building a KV-cache eviction plugin for vLLM:

**Evicting 50% of tokens does not free 50% of memory.** With H2O (a popular method), evicting 50% of tokens freed **0 blocks**.

Why: PagedAttention frees memory in whole blocks (16 tokens), not tokens. H2O keeps scattered "heavy hitters", so every block still holds at least one kept token → nothing gets freed.

Our fix, **BlockKV**, selects whole blocks instead of scattered tokens, plus a small "rescue" of exceptional tokens. Same 50% budget → **3× more blocks freed**, perplexity basically unchanged (3.343 → 3.385).

Measured on real hardware (GPU, Qwen2.5-7B, inside a real vLLM engine):
- H2O @ 50%: **0 blocks freed**
- BlockKV @ 50%: **12 blocks freed**

Single-pass score capture (~4% overhead, no second forward), so it runs in production, not just in a notebook.

Repo: https://github.com/shihengzhang/vllm-kv-squeeze (Apache-2.0)

---

## 2. Reddit — r/LocalLLaMA

**Title:** [Project] KV cache eviction that actually frees memory — the block-level trap

I've been building a KV-eviction plugin for vLLM and hit a counterintuitive result worth sharing.

**Evicting 50% of tokens barely frees any memory with H2O/SnapKV.** PagedAttention frees whole 16-token blocks, and those methods keep scattered heavy-hitters — so almost every block still has one kept token, and nothing gets freed.

TL;DR:
- Token-level eviction ≠ block-level memory freed
- H2O @ 50% budget: **freed 0 blocks**
- BlockKV (block-aware eviction): **freed 12 blocks**, same budget, perplexity +1.3%

Also got single-pass score capture working (hook the last layer during prefill, ~4% overhead), so it runs inside a real vLLM engine — not just at the planning layer.

Repo: https://github.com/shihengzhang/vllm-kv-squeeze

---

## 3. X / Twitter — thread

You evicted 50% of your KV cache — and freed almost no memory.

Why: PagedAttention frees whole 16-token blocks, not tokens. H2O keeps scattered "heavy hitters", so every block still has ≥1 kept token → 0 blocks freed.

Our fix, **BlockKV**: keep whole blocks + rescue a few exceptional tokens.
→ 3× more blocks freed at the same budget, perplexity basically unchanged.

Measured on GPU (Qwen2.5-7B, real vLLM engine):
H2O: 0 blocks freed
BlockKV: 12 blocks freed

Open source (Apache-2.0): https://github.com/shihengzhang/vllm-kv-squeeze
