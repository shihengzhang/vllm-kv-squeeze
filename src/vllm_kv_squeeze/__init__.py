"""vllm-kv-squeeze: pluggable KV-cache eviction for vLLM."""

from .hooks import (
    BlockEvictionPlan,
    EvictionHook,
    build_attention_bias,
    free_evicted_blocks,
    plan_block_eviction,
)
from .policy import (
    EvictionPolicy,
    EvictionResult,
    H2OPolicy,
    SnapKVPolicy,
    StreamingLLMPolicy,
    make_result,
)
from .utils import resolve_budget

__version__ = "0.1.0"

__all__ = [
    "BlockEvictionPlan",
    "EvictionHook",
    "EvictionPolicy",
    "EvictionResult",
    "H2OPolicy",
    "SnapKVPolicy",
    "StreamingLLMPolicy",
    "build_attention_bias",
    "free_evicted_blocks",
    "make_result",
    "plan_block_eviction",
    "resolve_budget",
]
