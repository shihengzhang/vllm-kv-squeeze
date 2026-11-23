# vllm-kv-squeeze

> Pluggable KV-cache eviction for [vLLM](https://github.com/vllm-project/vllm). Drop the tokens the model no longer needs, and turn the freed memory into longer context or larger batches.

The KV cache is the single biggest memory cost of long-context LLM serving, and most of it is wasted: a handful of "attention sink" and recent tokens carry the useful signal, while the middle is largely forgettable. `vllm-kv-squeeze` packages several eviction strategies as a small, pure, well-tested policy layer, and documents exactly how to wire them into vLLM's paged KV cache.

## Why this exists

- **vLLM has no native KV eviction API.** PagedAttention stores KV in *blocks*, not tokens, so you cannot simply zero out a token's cache — you have to drop whole blocks from the sequence's block table and return them to the pool.
- **Existing methods are scattered in forks** (SnapKV-vLLM, PyramidKV-vLLM, …) that drift from upstream. This repo aims to be a clean, version-tracked policy layer plus a clearly documented integration path.

## Supported policies

| Policy | Needs scores? | Idea |
|---|---|---|
| `StreamingLLMPolicy` | no | Keep the attention sink + a recent sliding window; drop the rest. |
| `H2OPolicy` | yes | Keep sink + window + the highest cumulative-attention "heavy hitters" in between. |
| `SnapKVPolicy` | yes | Local max-pooling (clustering) of importance before top-K selection. |

See the papers: [StreamingLLM](https://arxiv.org/abs/2309.17453), [H2O](https://arxiv.org/abs/2306.14048), [SnapKV](https://arxiv.org/abs/2404.14469).

## Install

The policy layer needs only PyTorch. vLLM is an optional extra (it is heavy and pins many dependencies).

```bash
pip install -e .          # policy layer only
pip install -e ".[vllm]"  # + vLLM for integration work
pip install -e ".[dev]"   # + pytest/ruff/pre-commit for contributing
```

## Quick start (no vLLM required)

```python
import torch
from vllm_kv_squeeze import H2OPolicy, resolve_budget

seq_len = 2048
scores = torch.rand(seq_len)            # stand-in for per-token importance
budget = resolve_budget(seq_len, 0.5)   # keep 50%

policy = H2OPolicy(sink_size=4, window_size=256)
result = policy(scores, seq_len, budget)

print(result.kept, result.evict_indices[:8].tolist())
```

## Architecture

```
src/vllm_kv_squeeze/
├── policy/          # pure, stateless, testable eviction policies
│   ├── base.py      # EvictionPolicy / EvictionResult / make_result
│   ├── h2o.py
│   ├── snapkv.py
│   └── streaming_llm.py
├── hooks/
│   ├── v1.py            # vLLM V1 integration scaffold + attention-bias helper
│   └── block_eviction.py  # pure planner: token keep/evict -> block free plan
└── utils/
    └── budget.py    # ratio/absolute budget resolution
```

Two integration levels are planned (see `hooks/v1.py`):

1. **Mask path (Level 1)** — block attention to evicted tokens. Saves FLOPs and latency; the KV stays resident. Implemented (`build_attention_bias`).
2. **Block-eviction path (Level 2)** — rewrite the sequence's block table and return freed blocks to the pool. This is the real memory win. The *planning* logic is implemented (`plan_block_eviction`, fully tested); the vLLM mutation itself is a documented scaffold.

## Roadmap

- [x] Policy layer: StreamingLLM, H2O, SnapKV + budget helpers
- [x] Pure block-eviction planner (token keep/evict → block free plan)
- [ ] vLLM V1 block-eviction wiring (verify against installed vLLM API)
- [ ] Perplexity / output-quality regression tests vs. no-eviction baseline
- [ ] End-to-end throughput / memory benchmarks
- [ ] Version CI tracking the latest vLLM release

## Contributing

```bash
pip install -e ".[dev]"
pre-commit install
pytest
ruff check src tests
```

## License

[Apache-2.0](LICENSE)
