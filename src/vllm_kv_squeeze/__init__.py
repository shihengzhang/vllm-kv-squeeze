"""vllm-kv-squeeze: pluggable KV-cache eviction for vLLM."""

from .hooks import EvictionHook, build_attention_bias
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
    "EvictionHook",
    "EvictionPolicy",
    "EvictionResult",
    "H2OPolicy",
    "SnapKVPolicy",
    "StreamingLLMPolicy",
    "build_attention_bias",
    "make_result",
    "resolve_budget",
]
