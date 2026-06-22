"""Helpers: proportional attention, budget-as-heat allocation, and a cost model."""
from __future__ import annotations

import numpy as np

__all__ = ["proportional_attention", "allocate_budget", "estimate_cost"]


def proportional_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    multiplicity: np.ndarray | None = None,
) -> np.ndarray:
    """Softmax attention with multiplicity weighting.

    A merged representative standing for ``m`` original tokens has its logit boosted by
    ``log(m)``, so it participates in attention as if those ``m`` tokens were still present.
    This is what keeps dense (redundant) regions from losing their attention mass after
    compression.

    Parameters
    ----------
    query : (d,) array
    keys, values : (M, d) arrays   (use the compressed ``representatives`` for both)
    multiplicity : (M,) array or None

    Returns
    -------
    (d,) attention output.
    """
    query = np.asarray(query, dtype=float)
    keys = np.asarray(keys, dtype=float)
    values = np.asarray(values, dtype=float)
    d = keys.shape[1]
    scores = keys @ query / np.sqrt(d)
    if multiplicity is not None:
        scores = scores + np.log(np.asarray(multiplicity, dtype=float) + 1e-9)
    scores -= scores.max()
    w = np.exp(scores)
    w /= w.sum()
    return w @ values


def allocate_budget(
    marginal,
    total: float,
    adjacency: np.ndarray,
    steps: int = 500,
    dt: float = 0.02,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Budget-as-heat: split a fixed token budget across pipeline stages.

    Drives the per-stage budgets to the *equimarginal* (water-filling) optimum -- the point
    where one more token buys the same marginal value at every stage -- via a Laplacian
    consensus flow that uses only neighbouring stages' marginals. Total budget is conserved.

    Parameters
    ----------
    marginal : callable
        ``marginal(b) -> (S,) array`` giving each stage's marginal value at budget ``b``.
        For a concave value ``v_s(b) = alpha_s * log(1 + b)`` this is ``alpha_s / (1 + b)``.
    total : float
        The fixed total budget B.
    adjacency : (S, S) array
        Symmetric 0/1 (or weighted) connectivity between pipeline stages.
    steps, dt : int, float
        Integration controls.
    init : (S,) array or None
        Initial allocation (defaults to uniform).

    Returns
    -------
    (S,) array of per-stage budgets.
    """
    A = np.asarray(adjacency, dtype=float)
    S = A.shape[0]
    deg = A.sum(1)
    Lg = np.diag(deg) - A  # combinatorial Laplacian of the pipeline graph
    b = np.full(S, total / S) if init is None else np.array(init, dtype=float)
    eps = 1e-6
    for _ in range(steps):
        m = marginal(b)
        # finite-difference local curvature as a positive, per-stage step size
        gp = np.abs(marginal(b + eps) - m) / eps + 1e-9
        b = b + dt * (Lg @ m) / gp
        b = np.clip(b, 1e-6, None)
        b *= total / b.sum()  # project back onto the budget simplex (conserve B)
    return b


def estimate_cost(
    n_context_tokens: int,
    kept_fraction: float,
    input_price_per_mtok: float = 3.0,
    cache_read_multiplier: float = 0.1,
    batch_multiplier: float = 0.5,
    reuses: int = 1,
) -> dict:
    """Rough model of the per-call context input cost under compression + provider levers.

    All multipliers are user-overridable; defaults reflect commonly published rates
    (cache reads ~0.1x base input, batch ~0.5x). This is a planning estimate, not a
    measurement -- always confirm against your provider's current pricing.
    """
    price = input_price_per_mtok / 1e6
    full = n_context_tokens * price
    kept = max(1, int(round(kept_fraction * n_context_tokens)))
    compressed = kept * price
    # static compressed preamble: paid once, then cache-read on subsequent calls
    cached = (kept * price + (reuses - 1) * kept * price * cache_read_multiplier) / max(1, reuses)
    cached_batched = cached * batch_multiplier
    return {
        "full_context_usd_per_call": full,
        "compressed_usd_per_call": compressed,
        "compressed_cached_usd_per_call": cached,
        "compressed_cached_batched_usd_per_call": cached_batched,
        "reduction_vs_full": full / max(cached_batched, 1e-12),
    }
