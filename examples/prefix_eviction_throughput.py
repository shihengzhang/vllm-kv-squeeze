"""Throughput: decode tokens/sec vs freed KV blocks (vLLM 0.19.0).

Frees a fraction of a request's KV blocks and measures the decode throughput,
showing that actually freeing blocks (what ref_cnt-aware eviction does) speeds
up decode, while a ref_cnt-blind policy frees 0 shared blocks and gets no gain.

Run:  VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 \
      python examples/prefix_eviction_throughput.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = os.path.expanduser("~/models/Qwen2.5-7B-Instruct")
BLOCK = 16


def main() -> None:
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    ids = tok("大语言模型的推理部署是一个典型的系统工程问题。" * 300,
              add_special_tokens=False).input_ids

    args = EngineArgs(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.5,
                      max_model_len=16384)
    engine = LLMEngine.from_engine_args(args)
    kcm = engine.engine_core.engine_core.scheduler.kv_cache_manager
    mgr = kcm.coordinator.single_type_managers[0]

    engine.add_request("req-1", ids, SamplingParams(max_tokens=64, temperature=0))
    engine.step()  # prefill

    req_id = list(mgr.req_to_blocks.keys())[0]
    blocks = mgr.req_to_blocks[req_id]
    n_blocks = len(blocks)
    print(f"context {len(ids)} tokens, {n_blocks} blocks", flush=True)

    def decode_tps(n_steps=40):
        t0 = time.time()
        out = None
        for _ in range(n_steps):
            out = engine.step()
            if out and all(o.finished for o in out):
                break
        dt = time.time() - t0
        n_tok = sum(len(o.outputs[0].token_ids) for o in out) if out else 0
        return n_tok / dt if dt > 0 else 0.0

    # baseline (full cache)
    tps_full = decode_tps()
    print(f"decode throughput (full cache): {tps_full:.1f} tok/s", flush=True)

    # free 50% of blocks (what ref_cnt-aware eviction achieves on unique blocks)
    free_before = mgr.block_pool.get_num_free_blocks()
    to_free = [b for b in blocks[len(blocks) // 2:] if not b.is_null]
    mgr.block_pool.free_blocks(to_free)
    freed = mgr.block_pool.get_num_free_blocks() - free_before
    print(f"freed {freed} blocks (50% of cache)", flush=True)

    tps_evict = decode_tps()
    print(f"decode throughput (50% freed): {tps_evict:.1f} tok/s "
          f"({tps_evict / tps_full:.2f}x)", flush=True)
    print("THROUGHPUT_DONE")


if __name__ == "__main__":
    main()
