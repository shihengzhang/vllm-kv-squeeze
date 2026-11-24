"""Tests for the BlockKV block-aware eviction policy."""

import pytest
import torch

from vllm_kv_squeeze.hooks.block_eviction import plan_block_eviction
from vllm_kv_squeeze.policy import BlockKVPolicy, H2OPolicy


def test_no_eviction_when_budget_equals_seq_len():
    policy = BlockKVPolicy()
    result = policy.select(None, seq_len=64, budget=64)
    assert result.kept == 64
    assert result.evicted == 0


def test_requires_scores():
    policy = BlockKVPolicy()
    with pytest.raises(ValueError):
        policy.select(None, seq_len=64, budget=32)


def test_partition_is_complete_and_disjoint():
    policy = BlockKVPolicy(sink_size=4, window_size=16, block_size=16, rescue_ratio=0.1)
    scores = torch.rand(128)
    result = policy.select(scores, seq_len=128, budget=64)
    assert result.kept + result.evicted == 128
    assert len(set(result.keep_indices.tolist()) & set(result.evict_indices.tolist())) == 0


def test_middle_keep_set_is_block_aligned():
    # With rescue disabled, every kept middle token belongs to a fully-kept block.
    policy = BlockKVPolicy(sink_size=4, window_size=16, block_size=16, rescue_ratio=0.0)
    scores = torch.rand(128)
    result = policy.select(scores, seq_len=128, budget=64)
    keep = set(result.keep_indices.tolist())
    # Middle region is blocks [1, 7) = tokens [16, 112).
    for b in range(1, 7):
        block_tokens = set(range(b * 16, (b + 1) * 16))
        kept_in_block = block_tokens & keep
        # Either the whole block is kept, or none of it is (except sink/window overlap).
        assert kept_in_block == set() or kept_in_block == block_tokens


def test_frees_more_blocks_than_h2o_on_scattered_scores():
    # One high-score token per middle block: H2O keeps scattered tokens (few full
    # blocks freed); BlockKV keeps whole blocks (more blocks freed).
    seq_len = 128
    block_size = 16
    scores = torch.zeros(seq_len)
    for b in range(2, 7):  # middle blocks (avoid sink block 0/1 and window block 7)
        scores[b * block_size + 3] = 10.0  # one important token per block

    budget = 64
    h2o = H2OPolicy(sink_size=4, window_size=16).select(scores, seq_len, budget)
    bkv = BlockKVPolicy(sink_size=4, window_size=16, block_size=block_size, rescue_ratio=0.0).select(
        scores, seq_len, budget
    )

    block_ids = torch.arange(seq_len // block_size)
    h2o_plan = plan_block_eviction(h2o, block_size, block_ids, seq_len)
    bkv_plan = plan_block_eviction(bkv, block_size, block_ids, seq_len)

    assert bkv_plan.freed_blocks > h2o_plan.freed_blocks


def test_rescue_recovers_top_tokens_from_evicted_blocks():
    # A lone very-important token sits in a low-scoring block; rescue should keep it.
    policy = BlockKVPolicy(sink_size=4, window_size=16, block_size=16, rescue_ratio=0.2)
    seq_len = 128
    scores = torch.zeros(seq_len)
    scores[2 * 16 + 5] = 1000.0  # token 37, in middle block 2
    result = policy.select(scores, seq_len=seq_len, budget=64)
    assert 37 in result.keep_indices.tolist()
