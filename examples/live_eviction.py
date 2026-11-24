"""Live KV eviction inside a running vLLM engine — single-pass score capture.

This is the production integration: it runs the vLLM V1 engine **in-process**
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`), hooks the last decoder layer's input
during prefill, then computes the last layer's attention scores with a single
`Q @ K^T` (one layer, ~4% overhead — not a full second forward), runs BlockKV,
and calls :func:`free_evicted_blocks` on the engine's real `KVCacheManager`.

Run:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 \
        python examples/live_eviction.py
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from vllm_kv_squeeze.hooks.block_eviction import plan_block_eviction
from vllm_kv_squeeze.hooks.v1 import free_evicted_blocks
from vllm_kv_squeeze.policy import BlockKVPolicy

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK_SIZE = 16


def single_pass_scores(engine, seq_len: int) -> torch.Tensor:
    """Last-layer per-token importance, captured in a single prefill pass.

    Hooks the last decoder layer's input hidden states during prefill, then
    recomputes only that layer's ``Q @ K^T`` (the H2O/BlockKV score source).
    """
    mr = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    last = mr.model.model.layers[-1]
    sa = last.self_attn

    cap = {}

    def hook(module, args):
        cap["h"] = args[1].detach()  # vLLM DecoderLayer.forward(positions, hidden_states, residual)

    last.register_forward_pre_hook(hook)
    # prefill runs here; the hook captures the last layer's input.
    engine.step()
    H = cap["h"]  # [S, hidden], bf16

    h_norm = last.input_layernorm(H)
    qkv, _ = sa.qkv_proj(h_norm)
    qkv = qkv.float()
    q, k, _ = qkv.split([sa.q_size, sa.kv_size, sa.kv_size], dim=-1)
    q = q.view(seq_len, sa.num_heads, sa.head_dim)
    k = k.view(seq_len, sa.num_kv_heads, sa.head_dim)
    pos = torch.arange(seq_len, device=q.device)
    q, k = sa.rotary_emb(pos, q, k)
    q = q.transpose(0, 1)  # [H, S, d]
    k = k.transpose(0, 1).repeat_interleave(sa.num_heads // sa.num_kv_heads, dim=0)
    w = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(sa.head_dim)
    w = torch.softmax(w, dim=-1)
    return w.sum(dim=(0, 1))  # [S] attention received per key


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    ids = tok("大语言模型的推理部署是一个典型的系统工程问题。" * 80,
              add_special_tokens=False).input_ids
    seq_len = len(ids)

    args = EngineArgs(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.5,
                      max_model_len=8192)
    engine = LLMEngine.from_engine_args(args)
    kcm = engine.engine_core.engine_core.scheduler.kv_cache_manager
    mgr = kcm.coordinator.single_type_managers[0]

    req_id = engine.add_request("req-1", ids, SamplingParams(max_tokens=16, temperature=0))

    scores = single_pass_scores(engine, seq_len)

    res = BlockKVPolicy(window_size=256, block_size=BLOCK_SIZE, rescue_ratio=0.1).select(
        scores.cpu(), seq_len, seq_len // 2)
    n_blocks = len(mgr.req_to_blocks[req_id])
    plan = plan_block_eviction(res, BLOCK_SIZE, torch.arange(n_blocks), seq_len)
    free_before = mgr.block_pool.get_num_free_blocks()
    free_evicted_blocks(kcm, req_id, plan.evicted_block_indices)
    print(f"BlockKV (single-pass scores): freed "
          f"{mgr.block_pool.get_num_free_blocks() - free_before} blocks")

    for _ in range(20):
        out = engine.step()
        if out and all(o.finished for o in out):
            break
    text = out[-1].outputs[0].text if out and out[-1].outputs else "(finished)"
    print(f"decode: {text[:40]!r}")
    print("LIVE_EVICTION_OK")


if __name__ == "__main__":
    main()
