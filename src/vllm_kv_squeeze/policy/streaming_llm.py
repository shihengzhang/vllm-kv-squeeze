"""StreamingLLM policy: attention sink + sliding recent window."""

from __future__ import annotations

from typing import Optional

import torch

from .base import EvictionPolicy, EvictionResult, make_result


class StreamingLLMPolicy(EvictionPolicy):
    """Drop everything except the first ``sink_size`` tokens (the attention
    sink) and the most recent ``window_size`` tokens.

    Unlike H2O this needs no per-token scores, so it is a single-pass method and
    the cheapest to compute.
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

        sink = min(self.sink_size, budget)
        window = min(self.window_size, budget - sink)

        sink_idx = torch.arange(sink, dtype=torch.int64)
        window_idx = torch.arange(seq_len - window, seq_len, dtype=torch.int64)
        keep = torch.cat([sink_idx, window_idx])

        return make_result(keep, seq_len)
