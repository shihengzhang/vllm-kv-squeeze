"""Tests for the eviction policy layer."""

import pytest
import torch

from vllm_kv_squeeze.policy import (
    H2OPolicy,
    SnapKVPolicy,
    StreamingLLMPolicy,
)


def test_streaming_llm_keeps_sink_and_window():
    policy = StreamingLLMPolicy(sink_size=4, window_size=10)
    result = policy.select(None, seq_len=100, budget=14)
    expected_keep = torch.tensor(list(range(4)) + list(range(90, 100)))
    assert torch.equal(result.keep_indices, expected_keep)
    assert result.kept == 14
    assert result.evicted == 86


def test_h2o_keeps_top_heavy_hitters():
    policy = H2OPolicy(sink_size=2, window_size=3)
    scores = torch.zeros(20)
    scores[5] = 100.0
    scores[6] = 50.0
    scores[7] = 25.0
    result = policy.select(scores, seq_len=20, budget=7)
    # sink [0, 1], window [17, 18, 19], plus 2 heavy hitters from the middle.
    assert torch.equal(result.keep_indices[:2], torch.tensor([0, 1]))
    assert torch.equal(result.keep_indices[-3:], torch.tensor([17, 18, 19]))
    middle = result.keep_indices[2:-3]
    assert set(middle.tolist()) == {5, 6}


def test_h2o_requires_scores_when_evicting():
    policy = H2OPolicy()
    with pytest.raises(ValueError):
        policy.select(None, seq_len=10, budget=5)


def test_h2o_scores_wrong_length():
    policy = H2OPolicy()
    with pytest.raises(ValueError):
        policy.select(torch.zeros(7), seq_len=10, budget=5)


def test_no_eviction_when_budget_equals_seq_len():
    for policy in (H2OPolicy(), SnapKVPolicy(), StreamingLLMPolicy()):
        result = policy.select(None, seq_len=10, budget=10)
        assert result.kept == 10
        assert result.evicted == 0


def test_snapkv_selects_top_scoring():
    policy = SnapKVPolicy(sink_size=2, window_size=3, pooling_kernel_size=1)
    scores = torch.zeros(20)
    scores[4] = 10.0
    scores[9] = 8.0
    result = policy.select(scores, seq_len=20, budget=7)
    middle = result.keep_indices[2:-3]
    assert set(middle.tolist()) == {4, 9}


def test_partition_is_complete_and_disjoint():
    policy = H2OPolicy(sink_size=4, window_size=8)
    scores = torch.rand(64)
    result = policy.select(scores, seq_len=64, budget=32)
    assert result.kept + result.evicted == 64
    assert len(set(result.keep_indices.tolist()) & set(result.evict_indices.tolist())) == 0
    assert set(result.keep_indices.tolist()) | set(result.evict_indices.tolist()) == set(
        range(64)
    )


def test_sink_size_zero_is_allowed():
    policy = H2OPolicy(sink_size=0, window_size=5)
    scores = torch.rand(20)
    result = policy.select(scores, seq_len=20, budget=10)
    assert result.kept == 10
