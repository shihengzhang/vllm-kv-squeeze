"""Live verification: ref_cnt semantics + eviction efficiency under prefix caching.

Launches the vLLM V1 engine in-process with prefix caching, adds M requests that
share a common prefix (plus unique suffixes), prefills, then inspects the real
``KVCacheBlock.ref_cnt`` to confirm: shared-prefix blocks have ref_cnt == M, and
evicting a shared block does NOT free memory while evicting a unique block does.

Run:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 \
        python examples/prefix_eviction_live.py
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
M = 4  # number of requests sharing the prefix


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    # long shared prefix (system prompt) + short unique suffixes
    prefix = tok("大语言模型的推理部署是一个典型的系统工程问题。" * 30,
                 add_special_tokens=False).input_ids
    suffixes = [tok(f"这是第{i}个独立的问题。", add_special_tokens=False).input_ids
                for i in range(M)]

    args = EngineArgs(model=MODEL, enforce_eager=True, gpu_memory_utilization=0.4,
                      max_model_len=8192, enable_prefix_caching=True)
    engine = LLMEngine.from_engine_args(args)
    kcm = engine.engine_core.engine_core.scheduler.kv_cache_manager
    mgr = kcm.coordinator.single_type_managers[0]

    for i in range(M):
        engine.add_request(f"req-{i}", prefix + suffixes[i],
                           SamplingParams(max_tokens=100, temperature=0))

    # prefill only (one step); requests keep decoding, so blocks stay resident
    engine.step()

    # inspect block ref_cnt per request (use the real keys vLLM assigns)
    print(f"prefix={len(prefix)} tokens ({len(prefix)//16} blocks), "
          f"suffix={len(suffixes[0])} tokens each\n")
    print("request   blocks   ref_cnt list")
    print("-" * 60)
    all_blocks = {}
    for req_id, blocks in mgr.req_to_blocks.items():
        rcs = [b.ref_cnt for b in blocks]
        for b in blocks:
            all_blocks.setdefault(b.block_id, b)
        print(f"{req_id}   {len(blocks):3d}     {rcs}")

    # count shared (ref_cnt == M) vs unique (ref_cnt == 1) physical blocks
    shared = sum(1 for b in all_blocks.values() if b.ref_cnt == M)
    unique = sum(1 for b in all_blocks.values() if b.ref_cnt == 1)
    print(f"\nphysical blocks: {len(all_blocks)}  shared(ref_cnt={M}): {shared}  unique(ref_cnt=1): {unique}")

    free0 = mgr.block_pool.get_num_free_blocks()
    # try freeing a SHARED block (ref_cnt == M) from one request's table
    shared_blk = next((b for b in all_blocks.values() if b.ref_cnt == M), None)
    if shared_blk is not None:
        mgr.block_pool.free_blocks([shared_blk])
        free1 = mgr.block_pool.get_num_free_blocks()
        print(f"\nfreed shared block (ref_cnt {M}->{M-1}): free pool {free0} -> {free1} "
              f"(delta {free1 - free0})")
    # try freeing a UNIQUE block (ref_cnt == 1)
    unique_blk = next((b for b in all_blocks.values() if b.ref_cnt == 1), None)
    if unique_blk is not None:
        free2 = mgr.block_pool.get_num_free_blocks()
        mgr.block_pool.free_blocks([unique_blk])
        free3 = mgr.block_pool.get_num_free_blocks()
        print(f"freed unique block (ref_cnt 1->0): free pool {free2} -> {free3} "
              f"(delta {free3 - free2})")

    print("\nPREFIX_LIVE_DONE")


if __name__ == "__main__":
    main()
