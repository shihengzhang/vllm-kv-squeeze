"""SnapKV-inspired policy: local max-pooling of importance before top-K."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import EvictionPolicy, EvictionResult, make_result


class SnapKVPolicy(EvictionPolicy):
    """Select important KV positions via a local max-pooling (clustering) of
    per-token importance, then top-K — the selection step of SnapKV.

    SnapKV's full algorithm derives importance from the *observation window*
    (the last ``window_size`` queries) and clusters nearby keys; pass those
    already-pooled scores here and this policy applies the clustering + top-K
    selection. Set ``pooling_kernel_size=1`` to disable clustering.
    """

    def __init__(
        self,
        sink_size: int = 4,
        window_size: int = 256,
        pooling_kernel_size: int = 7,
    ) -> None:
        super().__init__(sink_size=sink_size, window_size=window_size)
        if pooling_kernel_size < 1:
            raise ValueError("pooling_kernel_size must be >= 1")
        self.pooling_kernel_size = pooling_kernel_size

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
            raise ValueError("SnapKVPolicy requires per-token importance scores")
        if scores.numel() != seq_len:
            raise ValueError(
                f"scores has {scores.numel()} elements, expected {seq_len}"
            )

        sink = min(self.sink_size, budget)
        window = min(self.window_size, budget - sink)
        num_selected = budget - sink - window

        sink_idx = torch.arange(sink, dtype=torch.int64)
        window_idx = torch.arange(seq_len - window, seq_len, dtype=torch.int64)

        if num_selected <= 0:
            return make_result(torch.cat([sink_idx, window_idx]), seq_len)

        middle_start, middle_end = sink, seq_len - window
        if middle_end <= middle_start:
            return make_result(torch.cat([sink_idx, window_idx]), seq_len)

        middle_scores = scores[middle_start:middle_end]

        # Local max-pooling clusters nearby positions (SnapKV "pooling" step).
        pooled = middle_scores
        if self.pooling_kernel_size > 1 and middle_scores.numel() > 1:
            x = middle_scores.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
            out = F.max_pool1d(
                x,
                kernel_size=self.pooling_kernel_size,
                stride=1,
                padding=self.pooling_kernel_size // 2,
            ).squeeze()
            # Keep length == L only for odd kernels; otherwise fall back.
            if out.numel() == middle_scores.numel():
                pooled = out

        k = min(num_selected, pooled.numel())
        topk = torch.topk(pooled, k=k)
        heavy_idx = middle_start + topk.indices.to(dtype=torch.int64)

        keep = torch.cat([sink_idx, heavy_idx, window_idx])
        return make_result(keep, seq_len)
