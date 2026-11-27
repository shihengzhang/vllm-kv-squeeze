"""Reference-count-aware block eviction for prefix caching.

Under vLLM prefix caching a physical KV block can be shared by multiple
concurrent requests (``ref_cnt > 1``). ``BlockPool.free_blocks`` only reclaims a
block when its ``ref_cnt`` reaches zero, so **evicting a shared block frees no
memory** (yet still costs the evicting request quality).

This policy makes block selection ``ref_cnt``-aware: when choosing which middle
blocks to keep, it prioritizes *shared* blocks (``ref_cnt > 1``, which are both
more valuable and unfreeable anyway) and evicts *freeable* blocks (``ref_cnt ==
1``) first. With this ordering, eviction efficiency
(``freed blocks / evicted blocks``) stays at ~100% instead of collapsing to the
unique-block fraction.

If ``block_ref_cnt`` is ``None``, every block is treated as freeable and the
policy reduces to a plain block-level top-K (the :class:`BlockKVPolicy` rule).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .base import EvictionPolicy, EvictionResult, make_result


class RefCntAwarePolicy(EvictionPolicy):
    """Keep shared (ref_cnt > 1) blocks and high-score freeable blocks.

    ``scores[i]`` is the per-token importance. ``block_ref_cnt[j]`` is the
    reference count of middle block ``j`` (in order). Blocks are kept in the
    priority order: shared blocks (by descending score), then freeable blocks
    (by descending score); the rest are evicted as whole blocks.
    """

    def __init__(
        self,
        sink_size: int = 4,
        window_size: int = 256,
        block_size: int = 16,
    ) -> None:
        super().__init__(sink_size=sink_size, window_size=window_size)
        if block_size < 1:
            raise ValueError("block_size must be >= 1, got %r" % block_size)
        self.block_size = block_size

    def select(
        self,
        scores: Optional[torch.Tensor],
        seq_len: int,
        budget: int,
        block_ref_cnt: Optional[Sequence[int]] = None,
    ) -> EvictionResult:
        budget = self._clip_budget(budget, seq_len)
        if budget == seq_len:
            return make_result(torch.arange(seq_len, dtype=torch.int64), seq_len)
        if scores is None or scores.numel() != seq_len:
            raise ValueError(f"scores must be a [seq_len] tensor, got {None if scores is None else scores.numel()}")

        L = self.block_size
        sink = min(self.sink_size, budget)
        window = min(self.window_size, budget - sink)
        remaining = budget - sink - window

        sink_idx = torch.arange(sink, dtype=torch.int64)
        window_idx = torch.arange(seq_len - window, seq_len, dtype=torch.int64)

        mid_start = sink
        mid_end = seq_len - window
        first_block = (mid_start + L - 1) // L
        last_block = mid_end // L
        n_mid_blocks = last_block - first_block

        if n_mid_blocks <= 0:  # no whole middle block: fall back to token top-K
            middle_scores = scores[mid_start:mid_end]
            k = min(remaining, middle_scores.numel())
            keep = torch.cat([sink_idx, mid_start + torch.topk(middle_scores, k).indices.to(torch.int64), window_idx])
            return make_result(keep, seq_len)

        block_scores = torch.empty(n_mid_blocks, dtype=scores.dtype, device=scores.device)
        for i, b in enumerate(range(first_block, last_block)):
            block_scores[i] = scores[b * L:(b + 1) * L].sum()

        if block_ref_cnt is None:
            block_ref_cnt = torch.ones(n_mid_blocks, dtype=torch.int64)
        rc = torch.as_tensor(block_ref_cnt, dtype=torch.int64, device=scores.device)
        if rc.numel() != n_mid_blocks:
            raise ValueError(f"block_ref_cnt has {rc.numel()} entries, expected {n_mid_blocks}")

        shared_mask = rc > 1
        freeable_mask = ~shared_mask
        shared_idx = torch.nonzero(shared_mask, as_tuple=False).flatten()
        freeable_idx = torch.nonzero(freeable_mask, as_tuple=False).flatten()

        # keep priority: shared (score desc) then freeable (score desc)
        if shared_idx.numel():
            shared_idx = shared_idx[torch.argsort(block_scores[shared_idx], descending=True)]
        if freeable_idx.numel():
            freeable_idx = freeable_idx[torch.argsort(block_scores[freeable_idx], descending=True)]
        keep_order = torch.cat([shared_idx, freeable_idx])

        k_blocks = min(remaining // L, n_mid_blocks)
        kept_block_ids = set((first_block + keep_order[:k_blocks]).tolist())

        keep_parts = [sink_idx]
        for b in sorted(kept_block_ids):
            keep_parts.append(torch.arange(b * L, (b + 1) * L, dtype=torch.int64))
        keep_parts.append(window_idx)
        return make_result(torch.cat(keep_parts), seq_len)
