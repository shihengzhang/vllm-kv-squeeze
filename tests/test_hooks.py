"""Tests for the mask-based integration helper and block-free mutation."""

from types import SimpleNamespace

import torch

from vllm_kv_squeeze.hooks import build_attention_bias, free_evicted_blocks
from vllm_kv_squeeze.policy import StreamingLLMPolicy


def test_attention_bias_blocks_evicted():
    policy = StreamingLLMPolicy(sink_size=2, window_size=2)
    result = policy.select(None, seq_len=8, budget=4)
    bias = build_attention_bias(result, seq_len=8)

    assert torch.equal(bias[result.keep_indices], torch.zeros(result.kept))
    assert torch.isneginf(bias[result.evict_indices]).all()


def test_attention_bias_dtype():
    policy = StreamingLLMPolicy(sink_size=1, window_size=1)
    result = policy.select(None, seq_len=4, budget=2)
    bias = build_attention_bias(result, seq_len=4, dtype=torch.float16)
    assert bias.dtype == torch.float16


def _block(block_id, is_null=False):
    return SimpleNamespace(block_id=block_id, is_null=is_null)


def _fake_manager(blocks, freed):
    null = _block(-1, is_null=True)
    manager = SimpleNamespace(
        req_to_blocks={"r": blocks},
        _null_block=null,
        block_pool=SimpleNamespace(free_blocks=lambda blks: freed.extend(blks)),
    )
    kvcm = SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[manager]))
    return manager, kvcm


def test_free_evicted_blocks_nulls_and_frees_in_reverse():
    b0, b1, b2, b3 = _block(0), _block(1), _block(2), _block(3)
    freed = []
    manager, kvcm = _fake_manager([b0, b1, b2, b3], freed)

    free_evicted_blocks(kvcm, "r", [1, 2])

    assert manager.req_to_blocks["r"] == [b0, manager._null_block, manager._null_block, b3]
    assert freed == [b2, b1]  # reverse (tail-first) order


def test_free_evicted_blocks_noop_when_empty_or_null():
    freed = []
    manager, kvcm = _fake_manager([_block(0)], freed)

    free_evicted_blocks(kvcm, "r", [])
    assert freed == []

    # Evicting an already-null block is a no-op.
    manager.req_to_blocks["r"] = [manager._null_block]
    free_evicted_blocks(kvcm, "r", [0])
    assert freed == []
