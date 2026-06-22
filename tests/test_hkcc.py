"""Tests for HKCC. Run with: pytest -q"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

from hkcc import (
    HKCC, proportional_attention, allocate_budget, estimate_cost,
    build_knn_graph, normalized_laplacian, heat_diffuse, heat_residual, normalize_rows,
)

rng = np.random.default_rng(0)


def clustered(n_clusters=6, per=30, outliers=15, d=32, spread=0.12):
    """A context with redundant clusters + a few distinctive outliers."""
    centers = normalize_rows(rng.normal(size=(n_clusters, d)))
    pts = [centers[c] + spread * rng.normal(size=(per, d)) for c in range(n_clusters)]
    pts.append(normalize_rows(rng.normal(size=(outliers, d))))
    return normalize_rows(np.vstack(pts))


def test_chebyshev_matches_exact_exponential():
    X = clustered()
    L = normalized_laplacian(build_knn_graph(X, k=8)).toarray()
    approx = heat_diffuse(L, X, tau=1.0, order=24)
    exact = expm(-1.0 * L) @ X
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    assert rel < 1e-3


def test_sparse_and_dense_agree():
    X = clustered()
    Ls = normalized_laplacian(build_knn_graph(X, k=8))
    a = heat_diffuse(Ls, X, tau=1.0, order=20)
    b = heat_diffuse(Ls.toarray(), X, tau=1.0, order=20)
    assert np.allclose(a, b, atol=1e-9)


def test_convex_hull_safety():
    """Every representative must lie within the coordinate-wise range of the originals."""
    X = clustered()
    r = HKCC(budget=0.1).compress(X)
    lo, hi = X.min(0) - 1e-9, X.max(0) + 1e-9
    assert np.all(r.representatives >= lo) and np.all(r.representatives <= hi)


def test_compress_shapes_and_budget():
    X = clustered()
    r = HKCC(budget=0.1).compress(X)
    M = r.representatives.shape[0]
    assert M == max(1, round(0.1 * len(X)))
    assert r.multiplicity.sum() == len(X)          # every token accounted for
    assert r.assignment.shape[0] == len(X)
    r2 = HKCC(budget=20).compress(X)               # integer budget
    assert r2.representatives.shape[0] == 20


def test_residual_separates_outliers():
    X = clustered()
    r = HKCC(budget=0.1).compress(X)
    # the last 15 rows are outliers (distinctive) -> higher mean residual
    assert r.residual[-15:].mean() > r.residual[:-15].mean()


def test_merge_beats_prune_under_attention():
    """The core empirical claim: re-expressing redundant mass >> deleting it."""
    X = clustered()
    M = 18
    r = HKCC(budget=M).compress(X)
    queries = normalize_rows(rng.normal(size=(200, X.shape[1])))

    def fidelity(keys, values, mult):
        full = np.array([proportional_attention(q, X, X) for q in queries])
        comp = np.array([proportional_attention(q, keys, values, mult) for q in queries])
        full /= np.linalg.norm(full, axis=1, keepdims=True) + 1e-9
        comp /= np.linalg.norm(comp, axis=1, keepdims=True) + 1e-9
        return float(np.mean(np.sum(full * comp, axis=1)))

    merge = fidelity(r.representatives, r.representatives, r.multiplicity)
    prune_idx = r.anchor_indices
    prune = fidelity(X[prune_idx], X[prune_idx], np.ones(M))
    assert merge > 0.95
    assert merge > prune + 0.2  # a large, decisive gap


def test_allocate_budget_equalizes_marginals():
    S = 6
    alpha = rng.uniform(0.5, 2.5, S)
    A = np.diag(np.ones(S - 1), 1) + np.diag(np.ones(S - 1), -1)  # path graph
    b = allocate_budget(lambda b: alpha / (1.0 + b), total=20.0, adjacency=A)
    m = alpha / (1.0 + b)
    assert m.std() < 1e-2
    assert abs(b.sum() - 20.0) < 1e-6  # budget conserved


def test_estimate_cost_monotone():
    c = estimate_cost(6000, kept_fraction=0.08, reuses=100)
    assert c["compressed_usd_per_call"] < c["full_context_usd_per_call"]
    assert c["reduction_vs_full"] > 1.0
