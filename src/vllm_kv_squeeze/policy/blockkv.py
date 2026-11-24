"""Block-aware KV eviction (BlockKV).

The observation that motivates this policy (from the real-hardware run): under
PagedAttention, memory is freed in *blocks* (default 16 tokens), not tokens. A
token-level top-K policy such as H2O keeps scattered "heavy hitters", so almost
every block still contains at least one kept token — few blocks are ever fully
evicted, and the token-level 50% eviction barely translates into freed memory.

BlockKV flips the selection unit from *token* to *block*:

1. Aggregate per-token importance into per-block scores (sum of token scores).
2. Keep the attention sink + recent window (always), plus the top-K *middle
   blocks* by block score, so the evicted region consists of whole blocks that
   can be returned to the pool.
3. Optionally "rescue" a small number of exceptional high-score tokens from the
   evicted blocks (kept as masked partial blocks) to recover quality at a tiny
   memory cost.

This makes the keep-set *block-structured*: for the same token budget it frees
strictly more blocks than a scattered token-level policy, which is exactly the
"memory-savings vs quality" win the project's README documents.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import EvictionPolicy, EvictionResult, make_result


class BlockKVPolicy(EvictionPolicy):
    """Keep whole blocks (not scattered tokens) to maximize freed memory.

    ``scores[i]`` is the per-token importance (e.g. H2O cumulative attention).
    The policy keeps the sink, the recent window, the highest-scoring middle
    *blocks*, and — if ``rescue_ratio > 0`` — a few top individual tokens from
    the evicted blocks (these stay as masked partial blocks).
    """

    def __init__(
        self,
        sink_size: int = 4,
        window_size: int = 256,
        block_size: int = 16,
        rescue_ratio: float = 0.1,
    ) -> None:
        super().__init__(sink_size=sink_size, window_size=window_size)
        if block_size < 1:
            raise ValueError("block_size must be >= 1, got %r" % block_size)
        if not 0.0 <= rescue_ratio <= 1.0:
            raise ValueError("rescue_ratio must be in [0, 1], got %r" % rescue_ratio)
        self.block_size = block_size
        self.rescue_ratio = rescue_ratio

    def select(
        self,
        scores: Optional[torch.Tensor],
        seq_len: int,
        budget: int,
    ) -> EvictionResult:
        budget = self._clip_budget(budget, seq_len)
        if budget == seq_len:
            return make_result(torch.arange(seq_len, dtype=torch.int64), seq_len)

        if scores is None:
            raise ValueError("BlockKVPolicy requires per-token importance scores")
        if scores.numel() != seq_len:
            raise ValueError(f"scores has {scores.numel()} elements, expected {seq_len}")

        L = self.block_size
        sink = min(self.sink_size, budget)
        window = min(self.window_size, budget - sink)
        remaining = budget - sink - window

        sink_idx = torch.arange(sink, dtype=torch.int64)
        window_idx = torch.arange(seq_len - window, seq_len, dtype=torch.int64)

        # Middle region, aligned to whole blocks.
        mid_start = sink
        mid_end = seq_len - window
        first_block = (mid_start + L - 1) // L          # ceil(mid_start / L)
        last_block = mid_end // L                        # floor(mid_end / L)
        n_mid_blocks = last_block - first_block

        # If the middle has no whole blocks, fall back to token-level top-K.
        if n_mid_blocks <= 0:
            middle_scores = scores[mid_start:mid_end]
            k = min(remaining, middle_scores.numel())
            topk = torch.topk(middle_scores, k=k)
            keep = torch.cat([sink_idx, mid_start + topk.indices.to(torch.int64), window_idx])
            return make_result(keep, seq_len)

        # Per-block scores (sum of token importance within each block).
        block_scores = torch.empty(n_mid_blocks, dtype=scores.dtype, device=scores.device)
        for i, b in enumerate(range(first_block, last_block)):
            block_scores[i] = scores[b * L : (b + 1) * L].sum()

        # Split the remaining budget between whole blocks and rescued tokens.
        rescue_budget = int(remaining * self.rescue_ratio)
        block_budget = remaining - rescue_budget
        k_blocks = min(block_budget // L, n_mid_blocks)

        keep_parts = [sink_idx]
        kept_block_ids: set[int] = set()

        if k_blocks > 0:
            topk = torch.topk(block_scores, k=k_blocks)
            kept_blocks = (first_block + topk.indices.to(torch.int64)).tolist()
            kept_block_ids = set(kept_blocks)
            for b in sorted(kept_blocks):
                keep_parts.append(torch.arange(b * L, (b + 1) * L, dtype=torch.int64))

        keep_parts.append(window_idx)

        # Rescue a few exceptional tokens from the evicted middle blocks.
        if rescue_budget > 0 and k_blocks < n_mid_blocks:
            evicted_tokens = [
                t
                for b in range(first_block, last_block)
                if b not in kept_block_ids
                for t in range(b * L, min((b + 1) * L, mid_end))
            ]
            if evicted_tokens:
                ev = torch.tensor(evicted_tokens, dtype=torch.int64, device=scores.device)
                ev_scores = scores[ev]
                m = min(rescue_budget, ev_scores.numel())
                topm = torch.topk(ev_scores, k=m)
                keep_parts.append(ev[topm.indices.to(torch.int64)])

        keep = torch.cat(keep_parts)
        return make_result(keep, seq_len)
