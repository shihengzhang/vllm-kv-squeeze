"""Tests for the mask-based integration helper."""

import torch

from vllm_kv_squeeze.hooks import build_attention_bias
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
