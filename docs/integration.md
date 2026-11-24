# Live integration into vLLM (production path)

This documents how to run KV eviction **inside a live vLLM engine** — i.e. how
`free_evicted_blocks` actually returns blocks to the pool mid-generation, not
just at the planning layer. Verified on vLLM 0.19.0.

## The hook point

vLLM 0.19.0 has an **in-process engine** path. Set
`VLLM_ENABLE_V1_MULTIPROCESSING=0` and the `LLMEngine` uses `InprocClient`, so
the `EngineCore` (and its scheduler's KV cache manager) lives in *your* process:

```python
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.engine.llm_engine import LLMEngine

engine = LLMEngine.from_engine_args(EngineArgs(model=..., enforce_eager=True, ...))
kcm = engine.engine_core.engine_core.scheduler.kv_cache_manager   # KVCacheManager
mgr = kcm.coordinator.single_type_managers[0]                     # per-group manager
blocks = mgr.req_to_blocks[request_id]                            # the block table
```

`free_evicted_blocks(kcm, request_id, evicted_block_indices)` then nulls those
block-table entries and returns the underlying physical blocks to the pool.

**Important:** the real request id is `add_request(...)`'s return value (it has a
suffix, e.g. `req-1-ae4035b0`), not the string you passed in.

## Single-pass scoring (for H2O / SnapKV / BlockKV)

vLLM's default attention backend is flash attention, which does not return
per-token attention scores. We capture them in a **single pass** by hooking the
last decoder layer's input during prefill and recomputing only that layer's
`Q @ K^T` (one layer ≈ 1/24 of the model, ~4% overhead — not a full second
forward):

```python
mr = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
last = mr.model.model.layers[-1]
sa = last.self_attn

cap = {}
def hook(module, args):
    cap["h"] = args[1].detach()   # DecoderLayer.forward(positions, hidden_states, residual)
last.register_forward_pre_hook(hook)
engine.step()                     # prefill; the hook captures the last layer's input

H = cap["h"]                      # [S, hidden]
h_norm = last.input_layernorm(H)
qkv, _ = sa.qkv_proj(h_norm)
q, k, _ = qkv.float().split([sa.q_size, sa.kv_size, sa.kv_size], dim=-1)
q = q.view(S, sa.num_heads, sa.head_dim)
k = k.view(S, sa.num_kv_heads, sa.head_dim)
pos = torch.arange(S, device=q.device)
q, k = sa.rotary_emb(pos, q, k)
q = q.transpose(0, 1)
k = k.transpose(0, 1).repeat_interleave(sa.num_heads // sa.num_kv_heads, dim=0)
w = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(sa.head_dim), dim=-1)
scores = w.sum(dim=(0, 1))        # [S] attention received per key
```

(For a fully model-agnostic version, recompute scores from the cached K instead
of the fused `qkv_proj`; the hook + one-layer recompute is the common path.)

StreamingLLM needs no scores, so it is naturally single-pass.

## End-to-end result (real engine, 0.5B @ 960 tokens)

```
seq_len=960, vLLM prefill: 60 blocks
H2O       : freed  0 blocks   (scattered heavy hitters → no whole block free)
BlockKV   : freed 12 blocks   (block-structured keep-set → whole blocks freed)
decode: continues correctly
```

This reproduces the project's key finding *inside a running engine*: at the same
50% token budget, H2O frees nothing while BlockKV frees whole blocks.

## Correctness notes

- `free_evicted_blocks` reuses vLLM's own `BlockPool.free_blocks`, which
  decrements `ref_cnt` and only returns a block to the pool at `ref_cnt == 0`,
  so **prefix-cache / copy-on-write shared blocks are safe** (they are not freed
  while another request still references them).
- The block-table entries are nulled (`_null_block`) before freeing, so the next
  decode step's attention metadata skips the freed positions.
- Verified: freeing one request's blocks does not corrupt other concurrent
  requests (the freed blocks are safely reused by later requests).

## Run it

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 \
  python examples/live_eviction.py
```

See `examples/live_eviction.py` for the complete in-process integration
(StreamingLLM + two-pass BlockKV + multi-request).
