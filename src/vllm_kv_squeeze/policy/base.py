"""Core abstractions for KV-cache eviction policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EvictionResult:
    """The outcome of an eviction decision for a single KV sequence.

    Both tensors are 1-D ``int64`` tensors of token positions, sorted in
    ascending order. ``keep_indices`` and ``evict_indices`` partition the full
    ``[0, seq_len)`` range.
    """

    keep_indices: torch.Tensor
    evict_indices: torch.Tensor

    @property
    def kept(self) -> int:
        return int(self.keep_indices.numel())

    @property
    def evicted(self) -> int:
        return int(self.evict_indices.numel())


def make_result(keep_indices: torch.Tensor, seq_len: int) -> EvictionResult:
    """Build a valid :class:`EvictionResult` from an unsorted keep-set.

    Evicted positions are the complement of the keep-set within ``[0, seq_len)``.
    """
    keep = keep_indices.to(dtype=torch.int64).unique(sorted=True)
    mask = torch.ones(seq_len, dtype=torch.bool)
    mask[keep] = False
    full = torch.arange(seq_len, dtype=torch.int64)
    evict = full[mask]
    return EvictionResult(keep, evict)


class EvictionPolicy(ABC):
    """Maps per-token importance scores to a keep/evict decision.

    A policy is stateless and pure: ``select`` must not mutate its inputs.
    Subclasses implement one strategy (H2O, StreamingLLM, SnapKV, ...) while the
    shared ``sink_size`` / ``window_size`` knobs encode the widely observed
    "attention sink + recent window" inductive bias.
    """

    def __init__(self, sink_size: int = 4, window_size: int = 256) -> None:
        if sink_size < 0:
            raise ValueError("sink_size must be >= 0, got %r" % sink_size)
        if window_size < 0:
            raise ValueError("window_size must be >= 0, got %r" % window_size)
        self.sink_size = sink_size
        self.window_size = window_size

    @abstractmethod
    def select(
        self,
        scores: Optional[torch.Tensor],
        seq_len: int,
        budget: int,
    ) -> EvictionResult:
        """Decide which token positions to keep for a KV sequence.

        Args:
            scores: Optional ``[seq_len]`` float tensor of per-token importance.
                Policies that do not need scores (e.g. StreamingLLM) may receive
                ``None``.
            seq_len: Current KV sequence length (number of tokens).
            budget: Maximum number of tokens to keep. Must satisfy
                ``0 < budget <= seq_len``.

        Returns:
            An :class:`EvictionResult` whose ``keep_indices`` has length at most
            ``budget``.
        """

    def __call__(
        self,
        scores: Optional[torch.Tensor],
        seq_len: int,
        budget: int,
    ) -> EvictionResult:
        return self.select(scores, seq_len, budget)

    def _clip_budget(self, budget: int, seq_len: int) -> int:
        if budget <= 0:
            raise ValueError("budget must be > 0, got %r" % budget)
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0, got %r" % seq_len)
        return min(budget, seq_len)
