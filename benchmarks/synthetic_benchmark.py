"""Reproduce the headline synthetic result: merge >> prune under attention.

    python benchmarks/synthetic_benchmark.py

Prints attention-output fidelity vs. kept fraction for four strategies. This mirrors
Figure 2 / Table 1 of the paper, in a few seconds, with no external data.
"""
import numpy as np
from hkcc import HKCC, proportional_attention, normalize_rows

rng = np.random.default_rng(7)


def synth(n_clusters=8, per=40, outliers=20, d=64, spread=0.12):
    centers = normalize_rows(rng.normal(size=(n_clusters, d)))
    pts = [centers[c] + spread * rng.normal(size=(per, d)) for c in range(n_clusters)]
    pts.append(normalize_rows(rng.normal(size=(outliers, d))))
    return normalize_rows(np.vstack(pts))


def fidelity(X, keys, values, mult, queries):
    full = np.array([proportional_attention(q, X, X) for q in queries])
    comp = np.array([proportional_attention(q, keys, values, mult) for q in queries])
    full = full / (np.linalg.norm(full, axis=1, keepdims=True) + 1e-9)
    comp = comp / (np.linalg.norm(comp, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(np.sum(full * comp, axis=1)))


fracs = [0.05, 0.08, 0.12, 0.20, 0.35]
trials = 5
print(f"{'kept':>6} | {'random':>8} {'stride':>8} {'prune':>8} {'HKCC':>8}")
print("-" * 48)
for f in fracs:
    acc = {"random": [], "stride": [], "prune": [], "hkcc": []}
    for t in range(trials):
        X = synth()
        n, d = X.shape
        M = max(2, int(round(f * n)))
        Q = normalize_rows(rng.normal(size=(300, d)))
        r = HKCC(budget=M).compress(X)
        # HKCC (residual + diffused merge)
        acc["hkcc"].append(fidelity(X, r.representatives, r.representatives, r.multiplicity, Q))
        # prune only (keep anchors, drop the rest)
        ai = r.anchor_indices
        acc["prune"].append(fidelity(X, X[ai], X[ai], np.ones(M), Q))
        # random / stride
        ridx = rng.choice(n, M, replace=False)
        acc["random"].append(fidelity(X, X[ridx], X[ridx], np.ones(M), Q))
        sidx = np.linspace(0, n - 1, M).astype(int)
        acc["stride"].append(fidelity(X, X[sidx], X[sidx], np.ones(M), Q))
    print(f"{f:>6.0%} | {np.mean(acc['random']):>8.3f} {np.mean(acc['stride']):>8.3f} "
          f"{np.mean(acc['prune']):>8.3f} {np.mean(acc['hkcc']):>8.3f}")
print("\nHKCC stays near-lossless where prune-only collapses: the diffused merge is the point.")
