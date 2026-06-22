"""Multimodal benchmark: image patch tokens + the fidelity-target knob.

    python benchmarks/multimodal_benchmark.py

Demonstrates that HKCC compresses image/vision tokens (not just text), that the
position-aware (spatial) graph helps on grid data, and that the fidelity target hits
a quality floor automatically. Synthetic data, a few seconds, no downloads.
"""
import numpy as np
from hkcc import HKCC, compress_vision_tokens, proportional_attention, normalize_rows

rng = np.random.default_rng(7)


def synth_image(H=32, W=32, d=48):
    """A realistic fake feature map: smooth multi-channel gradients (like a real ViT
    feature field) + a distinct object + scattered high-frequency detail patches."""
    coords = np.stack(np.divmod(np.arange(H * W), W), axis=1).astype(float)
    yy, xx = coords[:, 0] / H, coords[:, 1] / W
    X = np.zeros((H * W, d))
    for _ in range(16):  # sum of low-frequency 2-D waves -> smooth but multi-dimensional
        fx, fy = rng.uniform(0.5, 4.0, 2)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.normal(size=d)
        X += np.outer(np.sin(2 * np.pi * (fx * xx + fy * yy) + phase), amp)
    obj = (coords[:, 0] >= 6) & (coords[:, 0] < 12) & (coords[:, 1] >= 20) & (coords[:, 1] < 26)
    X[obj] += 3.0 * rng.normal(size=d)
    detail = rng.choice(H * W, 70, replace=False)
    X[detail] += 3.0 * rng.normal(size=(70, d))
    X += 0.08 * rng.normal(size=X.shape)
    return normalize_rows(X), coords, (H, W)


def fidelity(X, reps, mult, Q):
    full = np.array([proportional_attention(q, X, X) for q in Q])
    comp = np.array([proportional_attention(q, reps, reps, mult) for q in Q])
    full = full / (np.linalg.norm(full, axis=1, keepdims=True) + 1e-9)
    comp = comp / (np.linalg.norm(comp, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(np.sum(full * comp, axis=1)))


X, coords, (H, W) = synth_image()
N, d = X.shape
Q = X[rng.choice(N, 250, replace=False)]   # in-distribution queries (patches attend to patches)
print(f"image feature map: {H}x{W} = {N} patch tokens, dim {d}\n")

print("attention fidelity vs kept fraction:")
print(f"{'kept':>6} | {'appearance-only':>16} | {'+ spatial graph':>16}")
print("-" * 48)
for f in [0.05, 0.10, 0.20, 0.35]:
    M = max(2, int(round(f * N))) 
    a = HKCC(budget=M, spatial_weight=0.0).compress(X)
    s = compress_vision_tokens(X, grid_hw=(H, W), budget=M, spatial_weight=0.4)
    fa = fidelity(X, a.representatives, a.multiplicity, Q)
    fs = fidelity(X, s.representatives, s.multiplicity, Q)
    print(f"{f:>6.0%} | {fa:>16.3f} | {fs:>16.3f}")

print("\nfidelity-target mode (set a floor, let it pick the budget):")
for floor in [0.99, 0.999]:
    r = compress_vision_tokens(X, grid_hw=(H, W), target_fidelity=floor, spatial_weight=0.4)
    speedup = (1.0 / r.kept_fraction) ** 2  # attention is O(N^2): keep fraction f -> f^2 work
    print(f"  floor {floor:<6} -> keep {r.kept_fraction:>5.1%}  "
          f"(achieved {r.fidelity:.4f});  attention compute ~{speedup:.0f}x lower")

print("\nNotes:")
print("  - HKCC compresses vision tokens, not just text; the spatial graph is an option for")
print("    grid data (here it ties appearance-only -- it helps most on strongly local redundancy).")
print("  - Budgets depend on how redundant the data is; this smooth field compresses heavily.")
print("  - Attention is quadratic in token count, so the kept fraction squared bounds the")
print("    attention-compute saving -- a real, superlinear latency win at large N.")
