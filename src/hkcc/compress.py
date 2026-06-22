"""The HKCC compressor: turn a long context into a few representative tokens.

Works on any set of embedding vectors -- text tokens, document chunks, or image patches.
For grid-structured tokens (image patches), pass ``coords`` so the graph respects position.
Set ``target_fidelity`` to compress as hard as possible while staying above a quality floor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    build_knn_graph, normalized_laplacian, heat_diffuse, heat_residual, normalize_rows,
)
from .helpers import proportional_attention

__all__ = ["HKCC", "CompressionResult"]


@dataclass
class CompressionResult:
    """Output of :meth:`HKCC.compress`.

    representatives : (M, d)   the compressed context -- send these to the model.
    multiplicity    : (M,)     how many original tokens each representative stands for.
    assignment      : (N,)     which representative each original token folded into.
    anchor_indices  : (M,)     indices of the distinctive tokens chosen as anchors.
    residual        : (N,)     the heat-residual saliency score for every original token.
    kept_fraction   : float    M / N.
    tau             : float     the diffusion time used.
    fidelity        : float | None  estimated attention-output fidelity (if measured).
    """

    representatives: np.ndarray
    multiplicity: np.ndarray
    assignment: np.ndarray
    anchor_indices: np.ndarray
    residual: np.ndarray
    kept_fraction: float
    tau: float
    fidelity: "float | None" = None

    @property
    def n_tokens(self) -> int:
        return len(self.representatives)


class HKCC:
    """Heat-Kernel Context Compression.

    Parameters
    ----------
    budget : float or int
        Target size. A float in (0, 1] is a *fraction* of tokens to keep; an int is an
        absolute count. Ignored when ``target_fidelity`` is set.
    target_fidelity : float or None
        If set (e.g. 0.99), ignore ``budget`` and keep the *fewest* tokens whose estimated
        attention-output fidelity stays at or above this floor. This is a self-consistent
        estimate from probe queries, not a guarantee on downstream accuracy.
    tau : float
        Diffusion time -- the single compression-scale knob. 1.0 is robust.
    k : int
        Neighbours per token in the similarity graph.
    spatial_weight : float
        For grid tokens (images): blend appearance and position in [0, 1]. 0 = appearance only.
    cheby_order : int
        Chebyshev order for the diffusion (~20-24 is plenty at the default scale).

    Example
    -------
    >>> import numpy as np
    >>> from hkcc import HKCC
    >>> HKCC(budget=0.1).compress(np.random.randn(500, 64)).representatives.shape
    (50, 64)
    """

    def __init__(
        self,
        budget=0.1,
        target_fidelity=None,
        tau=1.0,
        k=10,
        spatial_weight=0.0,
        cheby_order=24,
        fidelity_probes=96,
        min_keep=0.02,
    ):
        if isinstance(budget, float) and not (0.0 < budget <= 1.0):
            raise ValueError("float budget must be in (0, 1]; use an int for an absolute count")
        if target_fidelity is not None and not (0.0 < target_fidelity <= 1.0):
            raise ValueError("target_fidelity must be in (0, 1]")
        self.budget = budget
        self.target_fidelity = target_fidelity
        self.tau = 1.0 if tau is None else float(tau)
        self.k = int(k)
        self.spatial_weight = float(spatial_weight)
        self.cheby_order = int(cheby_order)
        self.fidelity_probes = int(fidelity_probes)
        self.min_keep = float(min_keep)

    def _target_m(self, n):
        if isinstance(self.budget, float):
            return max(1, int(round(self.budget * n)))
        return min(int(self.budget), n)

    def _assemble(self, X, U, rho, M):
        anchors = np.argsort(-rho)[:M]
        Un = normalize_rows(U)
        sim = Un @ normalize_rows(U[anchors]).T
        assign_local = np.argmax(sim, axis=1)
        assignment = anchors[assign_local]
        reps = np.zeros((M, X.shape[1]))
        mult = np.zeros(M)
        for a in range(M):
            members = np.where(assign_local == a)[0]
            if members.size == 0:
                members = np.array([anchors[a]])
            reps[a] = X[members].mean(axis=0)
            mult[a] = members.size
        return anchors, assignment, reps, mult

    def compress(self, X, coords=None):
        """Compress an (N, d) array of token embeddings to M representatives.

        coords : optional (N, 2) positions for grid tokens (image patches).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_tokens, dim)")
        n, d = X.shape

        if self.target_fidelity is None and self._target_m(n) >= n:
            return CompressionResult(X.copy(), np.ones(n), np.arange(n), np.arange(n),
                                     np.zeros(n), 1.0, self.tau, fidelity=1.0)

        W = build_knn_graph(X, k=self.k, spatial_coords=coords, spatial_weight=self.spatial_weight)
        L = normalized_laplacian(W)
        U = heat_diffuse(L, X, self.tau, order=self.cheby_order)
        rho = heat_residual(X, U)

        if self.target_fidelity is None:
            M = self._target_m(n)
            anchors, assignment, reps, mult = self._assemble(X, U, rho, M)
            return CompressionResult(reps, mult, assignment, anchors, rho, M / n, self.tau)

        # fidelity-target mode -- probe with in-distribution queries (the tokens that
        # actually get attended), which is more discriminative than random directions.
        rng = np.random.default_rng(0)
        qidx = rng.choice(n, size=min(self.fidelity_probes, n), replace=False)
        queries = X[qidx]
        full = np.array([proportional_attention(q, X, X) for q in queries])
        full = full / (np.linalg.norm(full, axis=1, keepdims=True) + 1e-9)

        cands = sorted(set(int(round(f * n)) for f in
                           np.geomspace(max(self.min_keep, 1.0 / n), 0.6, 14)))
        cands = [max(2, m) for m in cands if 1 <= m < n]
        cands = sorted(set(cands))
        best = None
        for M in cands:
            anchors, assignment, reps, mult = self._assemble(X, U, rho, M)
            comp = np.array([proportional_attention(q, reps, reps, mult) for q in queries])
            comp = comp / (np.linalg.norm(comp, axis=1, keepdims=True) + 1e-9)
            fid = float(np.mean(np.sum(full * comp, axis=1)))
            best = (reps, mult, assignment, anchors, M, fid)
            if fid >= self.target_fidelity:
                break
        reps, mult, assignment, anchors, M, fid = best
        return CompressionResult(reps, mult, assignment, anchors, rho, M / n, self.tau, fidelity=fid)
