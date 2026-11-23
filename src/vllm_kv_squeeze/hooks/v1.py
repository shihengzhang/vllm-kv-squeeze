"""vLLM V1 integration points.

The policy layer (``vllm_kv_squeeze.policy``) and the block planner
(``vllm_kv_squeeze.hooks.block_eviction``) are self-contained and fully
testable. Wiring them into vLLM requires touching version-specific internals,
so this module documents exactly where to hook. The parts that call into vLLM
are marked ``# TODO(verify)`` and must be checked against your installed vLLM
version (vLLM 0.8+ uses the "V1" engine; paths below follow V1).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..policy import EvictionPolicy, EvictionResult
from .block_eviction import BlockEvictionPlan, plan_block_eviction


def build_attention_bias(
    result: EvictionResult,
    seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an additive attention bias that blocks attention to evicted tokens.

    Returns a ``[seq_len]`` tensor of ``0.0`` for kept positions and ``-inf``
    for evicted positions. Add it to the attention logits before softmax to skip
    evicted KV positions without freeing their blocks.

    This is the "Level 1" mask-based path: it reduces FLOPs/latency but does not
    free memory.
    """
    bias = torch.zeros(seq_len, dtype=dtype)
    bias[result.evict_indices.to(dtype=torch.long)] = float("-inf")
    return bias


class EvictionHook:
    """Coordinates two-pass KV eviction inside the V1 engine.

    The intended flow, mirroring H2O/SnapKV:

    1. During prefill, capture per-token attention scores (cumulative softmax
       attention for H2O, or last-window pooled attention for SnapKV).
    2. Run the chosen :class:`EvictionPolicy` to get an :class:`EvictionResult`.
    3. Run :meth:`plan` to decide which physical blocks to free (Level 2), and
       optionally inject :func:`build_attention_bias` to skip the evicted tokens
       that remain inside partially-kept blocks (Level 1).

    Block path (the real memory win) touches V1 internals:

    * ``vllm.v1.core.kv_cache_manager.KVCacheManager`` owns the block pool and
      exposes ``free_blocks(...)`` / ``free(...)``.
    * Each sequence's physical blocks live in ``AttentionMetadata.block_tables``
      (a list of per-sequence ``[num_blocks]`` int64 tensors).
    * Evicting tokens means removing their blocks from the table and returning
      them; paged attention already supports arbitrary block-id lists, so a
      non-contiguous keep-set is fine.

    TODO(verify): confirm the exact method names and tensor layouts against the
    installed vLLM; they drift between releases.
    """

    def __init__(self, policy: EvictionPolicy, block_size: int = 16) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be > 0, got %r" % block_size)
        self.policy = policy
        self.block_size = block_size

    def decide(
        self,
        scores: Optional[torch.Tensor],
        seq_len: int,
        budget: int,
    ) -> EvictionResult:
        return self.policy(scores, seq_len, budget)

    def plan(
        self,
        block_ids: torch.Tensor,
        result: EvictionResult,
        seq_len: Optional[int] = None,
    ) -> BlockEvictionPlan:
        """Compute the pure block-level free/keep plan (see module above)."""
        return plan_block_eviction(result, self.block_size, block_ids, seq_len)

    def apply_block_eviction(
        self,
        kv_cache_manager,
        block_ids: torch.Tensor,
        result: EvictionResult,
        seq_len: Optional[int] = None,
    ) -> BlockEvictionPlan:
        """Plan and (when wired) apply block frees for one sequence.

        The planning is pure and always runs; the vLLM mutation below is a
        scaffold — do not call it without verifying the API against your
        installed vLLM.
        """
        plan = self.plan(block_ids, result, seq_len)

        # TODO(verify) against installed vLLM V1:
        #   1. rewrite this sequence's block table to ``plan.kept_block_ids``
        #      (update AttentionMetadata.block_tables and slot_mapping/seq_lens);
        #   2. return the freed blocks:
        #        kv_cache_manager.free_blocks(plan.freed_block_ids)
        raise NotImplementedError(
            "Block-level eviction is planned but not yet applied: verify the "
            "vLLM V1 KVCacheManager API against your installed version, then "
            "rewrite the block table and call free_blocks(plan.freed_block_ids)."
        )
