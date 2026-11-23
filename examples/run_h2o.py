"""Minimal runnable demo of H2O eviction on synthetic importance scores.

No vLLM required — this exercises the policy layer end to end.

Run with: python examples/run_h2o.py
"""

import torch

from vllm_kv_squeeze import H2OPolicy
from vllm_kv_squeeze.utils.budget import resolve_budget


def main() -> None:
    seq_len = 2048
    scores = torch.rand(seq_len)   # stand-in for per-token attention importance
    scores[:4] += 10.0             # the attention sink is "always important"

    budget = resolve_budget(seq_len, 0.5)  # keep 50%
    policy = H2OPolicy(sink_size=4, window_size=256)
    result = policy(scores, seq_len, budget)

    print(f"seq_len = {seq_len}")
    print(f"budget  = {budget} (kept {result.kept}, evicted {result.evicted})")
    print(f"kept head: {result.keep_indices[:8].tolist()}")
    print(f"kept tail: {result.keep_indices[-8:].tolist()}")
    print(f"evicted:   {result.evict_indices[:8].tolist()} ...")

    # Real vLLM wiring lives in vllm_kv_squeeze/hooks/v1.py (scaffold).


if __name__ == "__main__":
    main()
