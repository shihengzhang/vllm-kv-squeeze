"""Heavy-Hitter Oracle (H2O) eviction policy."""

from __future__ import annotations

from typing import Optional

import torch

from .base import EvictionPolicy, EvictionResult, make_result


class H2OPolicy(EvictionPolicy):
    """Keep the attention sink, a recent window, and the highest-scoring
    "heavy hitter" tokens in between, as described in the H2O paper.

    ``scores[i]`` is the cumulative attention weight received by token ``i``
    (typically the softmax attention summed over all query positions). The
    policy is "oracle" in the paper's sense: it needs those scores before it
    can decide, so it is a two-pass method in practice.
    """

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
            raise ValueError("H2OPolicy requires per-token importance scores")
        if scores.numel() != seq_len:
            raise ValueError(
                f"scores has {scores.numel()} elements, expected {seq_len}"
            )

        sink = min(self.sink_size, budget)
        window = min(self.window_size, budget - sink)
        num_heavy = budget - sink - window

        sink_idx = torch.arange(sink, dtype=torch.int64)
        window_idx = torch.arange(seq_len - window, seq_len, dtype=torch.int64)

        if num_heavy > 0:
            middle_start, middle_end = sink, seq_len - window
            if middle_end > middle_start:
                middle_scores = scores[middle_start:middle_end]
                k = min(num_heavy, middle_scores.numel())
                topk = torch.topk(middle_scores, k=k)
                heavy_idx = middle_start + topk.indices.to(dtype=torch.int64)
            else:
                heavy_idx = torch.empty(0, dtype=torch.int64)
            keep = torch.cat([sink_idx, heavy_idx, window_idx])
        else:
            keep = torch.cat([sink_idx, window_idx])

        return make_result(keep, seq_len)
