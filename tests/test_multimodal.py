"""Tests for the multimodal + fidelity-target features. Run: pytest -q"""
import numpy as np

from hkcc import (
    HKCC, compress_documents, compress_vision_tokens, embed, chunk_text,
)

rng = np.random.default_rng(1)


def clustered(nc=6, per=30, out=15, d=32, spread=0.12):
    cen = rng.normal(size=(nc, d)); cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    pts = [cen[c] + spread * rng.normal(size=(per, d)) for c in range(nc)]
    pts.append(rng.normal(size=(out, d)))
    X = np.vstack(pts)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_target_fidelity_meets_floor_and_compresses():
    X = clustered()
    r = HKCC(target_fidelity=0.99).compress(X)
    assert r.fidelity is not None and r.fidelity >= 0.99
    assert r.kept_fraction < 1.0          # it actually compressed


def test_target_fidelity_monotone_in_floor():
    X = clustered()
    lo = HKCC(target_fidelity=0.95).compress(X).kept_fraction
    hi = HKCC(target_fidelity=0.999).compress(X).kept_fraction
    assert hi >= lo                       # a higher floor keeps at least as many tokens


def test_spatial_graph_runs_and_merges():
    # a 16x16 patch grid: flat background + a small distinct square
    H = W = 16
    d = 24
    bg = np.tile(rng.normal(size=(1, d)), (H * W, 1)) + 0.02 * rng.normal(size=(H * W, d))
    coords = np.stack(np.divmod(np.arange(H * W), W), axis=1).astype(float)
    fg = (coords[:, 0] < 4) & (coords[:, 1] < 4)
    bg[fg] = rng.normal(size=(fg.sum(), d))      # distinct foreground patch
    X = bg / np.linalg.norm(bg, axis=1, keepdims=True)
    r = compress_vision_tokens(X, grid_hw=(H, W), budget=0.2)
    assert r.representatives.shape[0] == round(0.2 * H * W)
    assert r.multiplicity.sum() == H * W
    # the distinct foreground should be over-represented among anchors
    anchor_is_fg = fg[r.anchor_indices]
    assert anchor_is_fg.mean() > fg.mean()


def test_compress_documents_returns_real_chunks_in_order():
    docs = (["The capital of France is Paris."] * 8
            + ["Photosynthesis converts light into chemical energy."] * 8
            + ["The mitochondria is the powerhouse of the cell."])
    out = compress_documents(docs, budget=0.25)
    assert all(isinstance(c, str) for c in out.kept_chunks)        # real text, not centroids
    assert len(out.kept_chunks) < out.n_in                        # condensed
    assert list(out.kept_indices) == sorted(out.kept_indices)     # original order preserved


def test_embed_offline_is_deterministic():
    a = embed(["hello world", "token compression"])
    b = embed(["hello world", "token compression"])
    assert np.allclose(a, b)
    assert a.shape[0] == 2


def test_chunk_text_respects_max_chars():
    text = "\n\n".join([f"Paragraph number {i} with some filler words." for i in range(40)])
    chunks = chunk_text(text, max_chars=120)
    assert all(len(c) <= 120 for c in chunks)
    assert len(chunks) > 1
