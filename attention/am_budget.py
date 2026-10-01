from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Mapping


def load_head_budget(
    path: str,
    num_layers: int,
    num_kv_heads: int,
) -> Dict[str, float]:
    budget_path = Path(path)
    with budget_path.open() as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"AM head budget must be a JSON object: {budget_path}")

    expected = {
        f"L{layer_idx}H{head_idx}"
        for layer_idx in range(num_layers)
        for head_idx in range(num_kv_heads)
    }
    actual = set(raw)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"AM head budget keys do not match {num_layers}x{num_kv_heads}: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )

    proportions: Dict[str, float] = {}
    for key, value in raw.items():
        proportion = float(value)
        if not math.isfinite(proportion) or proportion < 0:
            raise ValueError(
                f"AM head budget proportion for {key} must be finite and "
                f"nonnegative, got {value!r}"
            )
        proportions[key] = proportion

    total = sum(proportions.values())
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(
            f"AM head budget proportions must sum to 1, got {total:.10f}"
        )
    return proportions


def compute_head_budget(
    proportions: Mapping[str, float],
    layer_idx: int,
    head_idx: int,
    *,
    target_ratio: float,
    context_len: int,
    num_layers: int,
    num_kv_heads: int,
) -> int:
    key = f"L{layer_idx}H{head_idx}"
    total_heads = num_layers * num_kv_heads
    budget = int(
        proportions[key]
        * total_heads
        * float(target_ratio)
        * int(context_len)
    )
    return min(int(context_len), max(0, budget))
