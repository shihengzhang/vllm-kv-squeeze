"""vLLM V1 integration points.

The policy layer above is self-contained (pure PyTorch) and fully testable.
Wiring it into vLLM requires touching version-specific internals, so this module
is a *scaffold*: it documents exactly where to hook and ships helper functions
that are pure and testable, while the parts that call into vLLM are marked
``# TODO(verify)`` and must be checked against your installed vLLM version
(vLLM 0.8+ uses the "V1" engine; paths below follow V1).
"""

from __future__ import annotations

import torch

from ..policy import EvictionPolicy, EvictionResult


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
    3. Either (mask path) inject :func:`build_attention_bias` into decode
       attention, or (block path) rewrite the sequence's block table and return
       the freed blocks to the pool.

    Block path (the real memory win) touches V1 internals:

    * ``vllm.v1.core.kv_cache_manager.KVCacheManager`` owns the block pool and
      exposes ``free_blocks(...)`` / ``free(...)``.
    * Each sequence's physical blocks live in ``AttentionMetadata.block_tables``
      (a list of per-sequence ``[num_blocks]`` int64 tensors).
    * Evicting a token means removing the blocks that held it from the table and
      returning them; paged attention already supports arbitrary block-id lists,
      so non-contiguous keep-sets are fine.

    TODO(verify): confirm the exact method names and tensor layouts against the
    installed vLLM; they drift between releases.
    """

    def __init__(self, policy: EvictionPolicy) -> None:
        self.policy = policy

    def decide(
        self,
        scores: torch.Tensor | None,
        seq_len: int,
        budget: int,
    ) -> EvictionResult:
        return self.policy(scores, seq_len, budget)

    def apply_block_eviction(self, kv_cache_manager, seq_id, result: EvictionResult) -> None:
        """Free the blocks backing evicted tokens for one sequence.

        Scaffold only — see the module docstring. Do not run without verifying
        the vLLM API against your installed version.
        """
        # TODO(verify): map token indices -> block ids via the block size and the
        # sequence's block table, then call, e.g.:
        #   kv_cache_manager.free_blocks(evicted_block_ids)
        raise NotImplementedError(
            "Block-level eviction is not wired up yet: verify the vLLM V1 "
            "KVCacheManager API against your installed version, then map "
            "evicted token indices to block ids and free them."
        )
