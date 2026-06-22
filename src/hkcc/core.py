"""Core graph + heat-diffusion primitives for HKCC.

Everything here is dependency-light (numpy + scipy) and works on plain embedding
arrays, so the library never assumes a particular model or framework.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

__all__ = [
    "normalize_rows",
    "build_knn_graph",
    "normalized_laplacian",
    "heat_diffuse",
    "heat_residual",
]


def normalize_rows(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """L2-normalize each row (so cosine similarity == dot product)."""
    X = np.asarray(X, dtype=float)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def build_knn_graph(
    X: np.ndarray,
    k: int = 10,
    sigma: float | None = None,
    spatial_coords: np.ndarray | None = None,
    spatial_weight: float = 0.5,
) -> sp.csr_matrix:
    """Symmetric k-nearest-neighbour affinity on cosine geometry.

    Tokens are L2-normalized, so Euclidean kNN on the unit sphere matches cosine
    nearest neighbours. Edge weights use a Gaussian kernel whose bandwidth defaults
    to the median neighbour distance (a robust, scale-free choice).

    For grid-structured tokens (e.g. image patches), pass ``spatial_coords`` of shape
    ``(N, 2)`` and a ``spatial_weight`` in [0, 1]: the graph is then built on a blend of
    appearance and position, so neighbouring patches that also look alike are joined.
    This is the anisotropic, position-aware variant used for vision tokens.

    Returns a sparse, symmetric affinity matrix ``W`` (zero diagonal).
    """
    Xn = normalize_rows(X)
    if spatial_coords is not None and spatial_weight > 0:
        c = np.asarray(spatial_coords, dtype=float)
        c = (c - c.min(0)) / (np.ptp(c, axis=0) + 1e-9)  # scale coords to [0, 1]
        feats = np.hstack([Xn * np.sqrt(1.0 - spatial_weight),
                           c * np.sqrt(spatial_weight)])
    else:
        feats = Xn
    n = feats.shape[0]
    k = min(k, n - 1)
    tree = cKDTree(feats)
    dist, idx = tree.query(feats, k=k + 1)  # +1 because the first hit is self
    dist, idx = dist[:, 1:], idx[:, 1:]
    d2 = dist ** 2
    if sigma is None:
        sigma = float(np.median(d2)) + 1e-12
    w = np.exp(-d2 / sigma)
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    W = sp.csr_matrix((w.ravel(), (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)  # symmetrize (keep the larger of w_ij, w_ji)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def normalized_laplacian(W: sp.spmatrix) -> sp.csr_matrix:
    """Symmetric normalized Laplacian L = I - D^{-1/2} W D^{-1/2} (spectrum in [0, 2])."""
    W = sp.csr_matrix(W)
    deg = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(deg + 1e-12)
    Dinv = sp.diags(dinv)
    return (sp.eye(W.shape[0]) - Dinv @ W @ Dinv).tocsr()


def _chebyshev_coeffs(tau: float, order: int, lmax: float = 2.0, m: int = 200) -> tuple[np.ndarray, float]:
    """Chebyshev coefficients of exp(-tau * lambda) on [0, lmax]."""
    a = lmax / 2.0
    theta = np.pi * (np.arange(m) + 0.5) / m
    x = np.cos(theta)
    fx = np.exp(-tau * a * (x + 1.0))
    c = np.array([(2.0 / m) * np.sum(fx * np.cos(j * theta)) for j in range(order + 1)])
    return c, a


def heat_diffuse(L, X: np.ndarray, tau: float, order: int = 24, lmax: float = 2.0) -> np.ndarray:
    """Apply the graph heat semigroup: return ``exp(-tau L) X`` without forming the matrix.

    Uses a Chebyshev polynomial expansion, so the cost is ``order`` sparse mat-vecs,
    i.e. O(order * nnz(L) * d) -- linear in the number of tokens for a kNN graph.

    Works with either a dense ndarray ``L`` or any scipy sparse matrix.
    """
    X = np.asarray(X, dtype=float)
    c, a = _chebyshev_coeffs(tau, order, lmax)
    n = L.shape[0]
    eye = sp.eye(n) if sp.issparse(L) else np.eye(n)
    Lhat = (L - a * eye) / a  # rescale spectrum to [-1, 1]
    t_prev = X
    t_cur = Lhat @ X
    out = 0.5 * c[0] * t_prev + c[1] * t_cur
    for j in range(2, order + 1):
        t_next = 2.0 * (Lhat @ t_cur) - t_prev
        out = out + c[j] * t_next
        t_prev, t_cur = t_cur, t_next
    return out


def heat_residual(X: np.ndarray, X_diffused: np.ndarray) -> np.ndarray:
    """Per-token heat residual rho_i = || x_i - (exp(-tau L) X)_i ||_2.

    Large residual => the token is high-frequency relative to its neighbours
    (distinctive); small residual => it is well-predicted by its neighbours (redundant).
    """
    return np.linalg.norm(np.asarray(X) - np.asarray(X_diffused), axis=1)
