"""Tests for budget resolution."""

import pytest

from vllm_kv_squeeze.utils.budget import resolve_budget


def test_ratio_budget():
    assert resolve_budget(1000, 0.5) == 500
    assert resolve_budget(1000, 0.25) == 250
    assert resolve_budget(3, 0.5) == 1  # clamps to >= 1


def test_absolute_budget_clamped():
    assert resolve_budget(100, 40) == 40
    assert resolve_budget(100, 500) == 100  # clamps to seq_len


def test_default_ratio():
    assert resolve_budget(100) == 50
    assert resolve_budget(100, default_ratio=0.8) == 80


def test_invalid_ratios():
    with pytest.raises(ValueError):
        resolve_budget(100, 0.0)
    with pytest.raises(ValueError):
        resolve_budget(100, 1.5)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        resolve_budget(0, 0.5)
    with pytest.raises(ValueError):
        resolve_budget(100, -1)
    with pytest.raises(TypeError):
        resolve_budget(100, "half")


def test_bool_is_rejected():
    with pytest.raises(TypeError):
        resolve_budget(100, True)
