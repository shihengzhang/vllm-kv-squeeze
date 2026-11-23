"""Micro-benchmarks for the policy layer (selection latency).

Run with: python benchmarks/benchmark_eviction.py
"""

import time

import torch

from vllm_kv_squeeze import H2OPolicy, SnapKVPolicy, StreamingLLMPolicy


def bench(policy, scores, seq_len, budget, iters=100):
    for _ in range(10):  # warmup
        policy(scores, seq_len, budget)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        policy(scores, seq_len, budget)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    seq_len = 32768
    budget = seq_len // 2
    scores = torch.rand(seq_len)

    policies = {
        "StreamingLLM": StreamingLLMPolicy(sink_size=4, window_size=1024),
        "H2O": H2OPolicy(sink_size=4, window_size=1024),
        "SnapKV": SnapKVPolicy(sink_size=4, window_size=1024, pooling_kernel_size=7),
    }

    print(f"{'policy':<14} {'select (ms)':>12} {'kept':>8} {'evicted':>8}")
    for name, policy in policies.items():
        ms = bench(policy, scores, seq_len, budget) * 1e3
        result = policy(scores, seq_len, budget)
        print(f"{name:<14} {ms:>11.3f} {result.kept:>8} {result.evicted:>8}")


if __name__ == "__main__":
    main()
