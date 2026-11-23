"""Pure block-level eviction planning for paged KV caches.

This module contains the algorithmic core of the "Level 2" integration: given a
token-level keep/evict decision, decide which *physical blocks* can be returned
to the pool. It is deliberately free of any vLLM import so it can be unit-tested
in isolation; the actual vLLM mutation (rewrite block table + ``free_blocks``)
lives in :mod:`vllm_kv_squeeze.hooks.v1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..policy import EvictionResult


@dataclass
class BlockEvictionPlan:
    """Which physical blocks to keep and which to free after eviction."""

    kept_block_ids: torch.Tensor  # [K] physical block ids, in logical order
    freed_block_ids: torch.Tensor  # [F] physical block ids to return to the pool

    @property
    def kept_blocks(self) -> int:
        return int(self.kept_block_ids.numel())

    @property
    def freed_blocks(self) -> int:
        return int(self.freed_block_ids.numel())


def plan_block_eviction(
    result: EvictionResult,
    block_size: int,
    block_ids: torch.Tensor,
    seq_len: Optional[int] = None,
) -> BlockEvictionPlan:
    """Plan block frees from a token-level keep/evict decision.

    A physical block is freed only when *every* token it covers is evicted.
    Partially-kept blocks are retained; their evicted tokens are skipped via the
    attention mask (see :func:`vllm_kv_squeeze.hooks.build_attention_bias`)
    instead of being freed.

    Args:
        result: Token-level keep/evict decision for one sequence.
        block_size: Tokens per KV block (vLLM's ``block_size``, typically 16).
        block_ids: The sequence's current block table — physical block ids in
            logical order (``AttentionMetadata.block_tables`` entry).
        seq_len: KV sequence length in tokens. Defaults to
            ``result.kept + result.evicted``.

    Returns:
        A :class:`BlockEvictionPlan` whose ``kept_block_ids`` is the new,
        shorter block table and whose ``freed_block_ids`` are the blocks to hand
        back to the pool.
    """
    if seq_len is None:
        seq_len = result.kept + result.evicted
    if block_size <= 0:
        raise ValueError("block_size must be > 0, got %r" % block_size)
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0, got %r" % seq_len)

    num_logical = (seq_len + block_size - 1) // block_size
    if block_ids.numel() < num_logical:
        raise ValueError(
            f"block table has {block_ids.numel()} blocks but seq_len={seq_len} "
            f"with block_size={block_size} needs {num_logical}"
        )
    block_ids = block_ids[:num_logical]

    token_kept = torch.zeros(seq_len, dtype=torch.bool)
    token_kept[result.keep_indices.to(dtype=torch.long)] = True

    keep_mask = torch.zeros(num_logical, dtype=torch.bool)
    for b in range(num_logical):
        start = b * block_size
        end = min(start + block_size, seq_len)
        keep_mask[b] = bool(token_kept[start:end].any())

    return BlockEvictionPlan(
        kept_block_ids=block_ids[keep_mask],
        freed_block_ids=block_ids[~keep_mask],
    )
