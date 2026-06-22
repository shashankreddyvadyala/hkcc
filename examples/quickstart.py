"""Quickstart: compress a context and check it preserves attention output.

    python examples/quickstart.py
"""
import numpy as np
from hkcc import HKCC, proportional_attention, estimate_cost

rng = np.random.default_rng(0)

# Pretend these are 800 context-token embeddings from your model (here: 6 topics + noise).
d = 64
centers = rng.normal(size=(6, d))
X = np.vstack([centers[i] + 0.12 * rng.normal(size=(130, d)) for i in range(6)]
              + [rng.normal(size=(20, d))])
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# Compress to ~10% of the tokens.
result = HKCC(budget=0.1, tau=1.0).compress(X)
print(f"compressed {len(X)} tokens -> {result.n_tokens} representatives "
      f"({result.kept_fraction:.0%} kept)")

# Verify it still gives the model nearly the same thing under attention.
queries = X[rng.choice(len(X), 300, replace=False)]
cos = []
for q in queries:
    full = proportional_attention(q, X, X)
    comp = proportional_attention(q, result.representatives,
                                  result.representatives, result.multiplicity)
    cos.append(np.dot(full, comp) / (np.linalg.norm(full) * np.linalg.norm(comp) + 1e-9))
print(f"attention-output fidelity: {np.mean(cos):.4f}")

# What it might save (planning estimate; check your provider's real pricing).
c = estimate_cost(n_context_tokens=6000, kept_fraction=result.kept_fraction, reuses=100)
print(f"effective context cost reduction vs full prompt: ~{c['reduction_vs_full']:.0f}x")
