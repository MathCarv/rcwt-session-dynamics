"""Statistical helpers for RCWT call-level summaries."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np


def stable_seed(*parts: object) -> int:
    """Return a deterministic 32-bit seed for a sequence of labels."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def stratified_bootstrap_mean_ci(
    strata: Mapping[str, Sequence[float]],
    *,
    seed: int,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
) -> tuple[float, float, float]:
    """Estimate a mean and percentile interval by resampling calls per stratum."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    arrays = [np.asarray(values, dtype=float) for values in strata.values()]
    if not arrays or any(values.size == 0 for values in arrays):
        raise ValueError("every bootstrap stratum must contain at least one value")

    point = float(np.mean(np.concatenate(arrays)))
    rng = np.random.default_rng(seed)
    bootstrap_sums = np.zeros(n_resamples, dtype=float)
    sample_count = 0
    for values in arrays:
        sampled = rng.choice(values, size=(n_resamples, values.size), replace=True)
        bootstrap_sums += np.sum(sampled, axis=1)
        sample_count += values.size
    bootstrap_means = bootstrap_sums / sample_count

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return point, float(low), float(high)
