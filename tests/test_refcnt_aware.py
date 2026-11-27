"""Tests for the ref_cnt-aware block eviction policy."""

import pytest
import torch

from vllm_kv_squeeze.policy import RefCntAwarePolicy


def test_shared_blocks_kept_over_low_score_freeable():
    # 4 middle blocks (block_size=4, seq_len=16, no sink/window):
    #   block 0: ref_cnt=1, high score  -> freeable, keep (high score)
    #   block 1: ref_cnt=8, low score   -> shared,   KEEP (shared priority)
    #   block 2: ref_cnt=1, low score   -> freeable, evict
    #   block 3: ref_cnt=1, low score   -> freeable, evict
    scores = torch.tensor([10.0, 10, 10, 10,    # block 0 (high)
                           1.0, 1, 1, 1,        # block 1 (shared, low)
                           2.0, 2, 2, 2,        # block 2 (freeable, low)
                           3.0, 3, 3, 3])       # block 3 (freeable, low)
    ref_cnt = [1, 8, 1, 1]
    policy = RefCntAwarePolicy(sink_size=0, window_size=0, block_size=4)
    # budget = 8 tokens = 2 blocks: keep block 0 (freeable high) + block 1 (shared)
    res = policy.select(scores, 16, 8, block_ref_cnt=ref_cnt)
    assert set(res.keep_indices.tolist()) == set(range(8))  # blocks 0 and 1


def test_freeable_evicted_before_shared():
    # shared block with LOWER score is still kept over a freeable block with higher score
    scores = torch.tensor([5.0, 5, 5, 5,    # block 0: freeable, score 5
                           1.0, 1, 1, 1])   # block 1: shared, score 1 (lower!)
    ref_cnt = [1, 8]
    policy = RefCntAwarePolicy(sink_size=0, window_size=0, block_size=4)
    res = policy.select(scores, 8, 4, block_ref_cnt=ref_cnt)  # keep 1 block
    # shared block (block 1) must be kept despite lower score
    assert set(res.keep_indices.tolist()) == {4, 5, 6, 7}


def test_no_refcnt_treats_all_freeable():
    # without ref_cnt, the policy is plain top-K block selection
    scores = torch.tensor([9.0, 9, 9, 9,    # block 0 (highest)
                           1.0, 1, 1, 1])   # block 1 (lowest)
    policy = RefCntAwarePolicy(sink_size=0, window_size=0, block_size=4)
    res = policy.select(scores, 8, 4)  # keep 1 block = block 0
    assert set(res.keep_indices.tolist()) == {0, 1, 2, 3}


def test_refcnt_length_mismatch_raises():
    scores = torch.tensor([1.0] * 16)
    policy = RefCntAwarePolicy(sink_size=0, window_size=0, block_size=4)
    with pytest.raises(ValueError):
        policy.select(scores, 16, 8, block_ref_cnt=[1, 1])  # only 2 ref_cnt for 4 blocks
