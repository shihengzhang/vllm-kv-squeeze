"""Helpers for resolving KV budgets from ratios or absolute counts."""

from __future__ import annotations

from typing import Optional, Union


def resolve_budget(
    seq_len: int,
    budget: Optional[Union[int, float]] = None,
    default_ratio: float = 0.5,
) -> int:
    """Turn a budget spec into an absolute token count.

    Args:
        seq_len: Current KV sequence length.
        budget: Either an absolute token count (``int``) or a keep ratio
            (``float`` in ``(0, 1]``). ``None`` falls back to ``default_ratio``.
        default_ratio: Ratio used when ``budget`` is ``None``.

    Returns:
        The resolved token count, clamped to ``[1, seq_len]``.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0, got %r" % seq_len)

    if budget is None:
        budget = default_ratio

    if isinstance(budget, float):
        if not 0.0 < budget <= 1.0:
            raise ValueError("ratio budget must be in (0, 1], got %r" % budget)
        return max(1, int(seq_len * budget))

    if isinstance(budget, int) and not isinstance(budget, bool):
        if budget <= 0:
            raise ValueError("absolute budget must be > 0, got %r" % budget)
        return min(budget, seq_len)

    raise TypeError("budget must be an int, a float in (0, 1], or None")
