"""vLLM V1 integration — written against vLLM **0.34.0**.

The policy layer (``vllm_kv_squeeze.policy``) and the block planner
(``vllm_kv_squeeze.hooks.block_eviction``) are self-contained and fully tested.
The functions below do the actual vLLM mutation and were written by reading the
0.34.0 source (``vllm/v1/core/{kv_cache_manager,single_type_kv_cache_manager,
block_pool}.py``); they are **API-verified against that version** but have not
been executed inside a live serving loop (this repo's CI runs without a GPU).

Key 0.34.0 facts these functions rely on:

* Each ``SingleTypeKVCacheManager`` keeps ``req_to_blocks[request_id]`` as a
  ``list[KVCacheBlock]`` — the per-request block table, in logical order.
* ``KVCacheBlock`` has ``block_id``, ``ref_cnt``, ``is_null``, ``block_hash``.
* ``BlockPool.free_blocks(iterable[KVCacheBlock])`` decrements ``ref_cnt`` and
  returns blocks whose ref count reaches zero to the free queue.
* Replacing a block-table entry with ``manager._null_block`` is exactly how
  sliding-window / R-SWA mark skipped positions (see
  ``SingleTypeKVCacheManager._remove_blocks_in_range``).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

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
    evicted KV positions that remain inside partially-kept blocks (Level 1).
    """
    bias = torch.zeros(seq_len, dtype=dtype)
    bias[result.evict_indices.to(dtype=torch.long)] = float("-inf")
    return bias


def free_evicted_blocks(
    kv_cache_manager,
    request_id: str,
    evicted_block_indices: Sequence[int],
) -> None:
    """Free fully-evicted blocks for one request, in place (vLLM 0.34.0 V1).

    ``evicted_block_indices`` are **logical positions** into the request's block
    table (as returned by
    :attr:`BlockEvictionPlan.evicted_block_indices
    <vllm_kv_squeeze.hooks.BlockEvictionPlan.evicted_block_indices>`). Blocks are
    freed in reverse order (tail first), matching vLLM's own convention.

    Call this from the scheduler *after* the eviction decision is made and
    *before* the next step builds ``AttentionMetadata.block_tables``, so the
    nulled-out positions are not read by the next attention pass.
    """
    evicted = sorted(int(i) for i in evicted_block_indices)
    if not evicted:
        return

    # In 0.34.0 the coordinator exposes one SingleTypeKVCacheManager per
    # KV-cache group; each holds its own req_to_blocks table.
    managers = kv_cache_manager.coordinator.single_type_managers
    for manager in managers:
        blocks = manager.req_to_blocks[request_id]
        to_free = []
        for idx in reversed(evicted):
            idx = min(idx, len(blocks) - 1)
            blk = blocks[idx]
            if not blk.is_null:
                to_free.append(blk)
                blocks[idx] = manager._null_block
        if to_free:
            manager.block_pool.free_blocks(to_free)


class EvictionHook:
    """Coordinates two-pass KV eviction inside the V1 engine.

    The intended flow, mirroring H2O/SnapKV:

    1. During prefill, capture per-token attention scores (cumulative softmax
       attention for H2O, or last-window pooled attention for SnapKV).
    2. Run the chosen :class:`EvictionPolicy` to get an :class:`EvictionResult`.
    3. Run :meth:`apply_block_eviction` to free fully-evicted blocks (Level 2),
       and optionally inject :func:`build_attention_bias` to skip evicted tokens
       that remain inside partially-kept blocks (Level 1).

    The policy/planning parts are pure and tested; :meth:`apply_block_eviction`
    is the vLLM 0.34.0 mutation (verified against source, see module docstring).
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
        """Compute the pure block-level free/keep plan."""
        return plan_block_eviction(result, self.block_size, block_ids, seq_len)

    def apply_block_eviction(
        self,
        kv_cache_manager,
        request_id: str,
        block_ids: torch.Tensor,
        result: EvictionResult,
        seq_len: Optional[int] = None,
    ) -> BlockEvictionPlan:
        """Plan and apply block frees for one request (vLLM 0.34.0 V1).

        ``block_ids`` is the request's current block table as a tensor of
        physical block ids (e.g. ``torch.tensor(kv_cache_manager
        .get_block_ids(request_id)[0])`` for a single full-attention group).
        """
        plan = self.plan(block_ids, result, seq_len)
        free_evicted_blocks(kv_cache_manager, request_id, plan.evicted_block_indices)
        return plan
