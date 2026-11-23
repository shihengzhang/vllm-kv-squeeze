"""vLLM integration hooks."""

from .block_eviction import BlockEvictionPlan, plan_block_eviction
from .v1 import EvictionHook, build_attention_bias, free_evicted_blocks

__all__ = [
    "BlockEvictionPlan",
    "EvictionHook",
    "build_attention_bias",
    "free_evicted_blocks",
    "plan_block_eviction",
]
