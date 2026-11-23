"""Tests for pure block-level eviction planning."""

import pytest
import torch

from vllm_kv_squeeze.hooks import plan_block_eviction
from vllm_kv_squeeze.policy import make_result


def _result(keep, seq_len):
    return make_result(torch.tensor(keep, dtype=torch.int64), seq_len)


def test_frees_only_fully_evicted_blocks():
    # seq_len=10, block_size=2 -> blocks [0..4] cover tokens [0,1][2,3][4,5][6,7][8,9]
    result = _result([0, 1, 4, 5], seq_len=10)
    block_ids = torch.tensor([10, 11, 12, 13, 14])
    plan = plan_block_eviction(result, block_size=2, block_ids=block_ids)

    assert torch.equal(plan.kept_block_ids, torch.tensor([10, 12]))
    assert torch.equal(plan.freed_block_ids, torch.tensor([11, 13, 14]))
    assert plan.kept_blocks == 2
    assert plan.freed_blocks == 3


def test_partial_block_is_kept():
    # seq_len=10, block_size=4 -> blocks cover [0..3][4..7][8..9]
    result = _result([0, 9], seq_len=10)
    block_ids = torch.tensor([20, 21, 22])
    plan = plan_block_eviction(result, block_size=4, block_ids=block_ids)

    # block 1 (tokens 4..7) is fully evicted; blocks 0 and 2 are partial -> kept
    assert torch.equal(plan.kept_block_ids, torch.tensor([20, 22]))
    assert torch.equal(plan.freed_block_ids, torch.tensor([21]))


def test_last_partial_block():
    # keep everything except the last token -> the last (partial) block is kept
    result = _result(list(range(9)), seq_len=10)
    block_ids = torch.tensor([0, 1, 2])
    plan = plan_block_eviction(result, block_size=4, block_ids=block_ids)
    assert torch.equal(plan.freed_block_ids, torch.empty(0, dtype=torch.int64))
    assert plan.freed_blocks == 0


def test_no_eviction_frees_nothing():
    result = _result(list(range(10)), seq_len=10)
    block_ids = torch.tensor([0, 1, 2])
    plan = plan_block_eviction(result, block_size=4, block_ids=block_ids)
    assert plan.freed_blocks == 0
    assert torch.equal(plan.kept_block_ids, block_ids)


def test_block_size_one():
    # each token is its own block
    result = _result([0, 2, 4], seq_len=5)
    block_ids = torch.tensor([100, 101, 102, 103, 104])
    plan = plan_block_eviction(result, block_size=1, block_ids=block_ids)
    assert torch.equal(plan.kept_block_ids, torch.tensor([100, 102, 104]))
    assert torch.equal(plan.freed_block_ids, torch.tensor([101, 103]))


def test_seq_len_inferred_from_result():
    result = _result([0, 1], seq_len=4)
    plan = plan_block_eviction(result, block_size=2, block_ids=torch.tensor([7, 8]))
    assert torch.equal(plan.kept_block_ids, torch.tensor([7]))
    assert torch.equal(plan.freed_block_ids, torch.tensor([8]))


def test_invalid_block_size():
    result = _result([0], seq_len=2)
    with pytest.raises(ValueError):
        plan_block_eviction(result, block_size=0, block_ids=torch.tensor([0]))


def test_too_few_blocks_raises():
    result = _result([0], seq_len=10)
    with pytest.raises(ValueError):
        plan_block_eviction(result, block_size=4, block_ids=torch.tensor([0, 1]))
