"""High-level, modality-aware entry points.

Two honest modes:

* **Documents / text** -> HKCC works as a *redundancy-aware selector*. You can't detokenize
  a centroid back into words, so we keep the real anchor chunks and drop the redundant ones.
  Great for condensing retrieved RAG context.

* **Vision / embedding tokens** -> HKCC works as a *merger*. Image patches are just vectors,
  so the centroid representatives are usable directly as compressed visual tokens (pass the
  multiplicity to attention). This is the setting closest to the token-merging literature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compress import HKCC, CompressionResult
from .io_adapters import load_document, chunk_text, embed as default_embed

__all__ = ["compress_documents", "compress_vision_tokens", "DocumentCompression"]


@dataclass
class DocumentCompression:
    kept_chunks: list  # the surviving chunks, in original order (real text)
    kept_indices: np.ndarray
    n_in: int
    result: CompressionResult

    @property
    def kept_fraction(self) -> float:
        return len(self.kept_chunks) / max(self.n_in, 1)


def compress_documents(
    sources,
    budget=0.2,
    target_fidelity=None,
    embedder=None,
    max_chars: int = 500,
    **hkcc_kwargs,
) -> DocumentCompression:
    """Condense a set of documents/strings down to the chunks that carry distinct content.

    Parameters
    ----------
    sources : list[str]
        File paths or raw strings (auto-detected). Files are parsed by :func:`load_document`.
    budget, target_fidelity, **hkcc_kwargs
        Passed to :class:`HKCC`.
    embedder : callable or None
        ``embedder(list[str]) -> (N, d) array``. Defaults to the bundled embedder.

    Returns
    -------
    DocumentCompression with the surviving real chunks (in original order).
    """
    embedder = embedder or default_embed
    chunks: list[str] = []
    import os
    for s in sources:
        if isinstance(s, str) and os.path.exists(s):
            chunks.extend(chunk_text(load_document(s), max_chars=max_chars))
        else:
            chunks.extend(chunk_text(str(s), max_chars=max_chars))
    if len(chunks) <= 1:
        X = embedder(chunks) if chunks else np.zeros((0, 1))
        res = CompressionResult(X, np.ones(len(chunks)), np.arange(len(chunks)),
                                np.arange(len(chunks)), np.zeros(len(chunks)), 1.0, 1.0, 1.0)
        return DocumentCompression(chunks, np.arange(len(chunks)), len(chunks), res)

    X = np.asarray(embedder(chunks), dtype=float)
    res = HKCC(budget=budget, target_fidelity=target_fidelity, **hkcc_kwargs).compress(X)
    kept = np.sort(res.anchor_indices)                      # keep real chunks, original order
    return DocumentCompression([chunks[i] for i in kept], kept, len(chunks), res)


def compress_vision_tokens(
    patch_embeddings: np.ndarray,
    grid_hw=None,
    budget=0.25,
    target_fidelity=None,
    spatial_weight: float = 0.4,
    **hkcc_kwargs,
) -> CompressionResult:
    """Compress image/vision patch tokens into fewer merged tokens.

    Parameters
    ----------
    patch_embeddings : (N, d) array
        Patch/token embeddings from any vision encoder (e.g. a ViT).
    grid_hw : (H, W) or None
        If given, patches are laid out on an H x W grid and the graph is made
        position-aware (neighbouring patches that also look alike are merged), which suits
        the spatial redundancy of images.
    spatial_weight : float
        How much position counts vs appearance, in [0, 1].

    Returns
    -------
    CompressionResult whose ``representatives`` are the merged visual tokens; pass
    ``multiplicity`` to attention so large flat regions keep their weight.
    """
    X = np.asarray(patch_embeddings, dtype=float)
    coords = None
    if grid_hw is not None:
        H, W = grid_hw
        if H * W != X.shape[0]:
            raise ValueError(f"grid {H}x{W} != {X.shape[0]} patches")
        yy, xx = np.divmod(np.arange(H * W), W)
        coords = np.stack([yy, xx], axis=1).astype(float)
    return HKCC(budget=budget, target_fidelity=target_fidelity,
                spatial_weight=spatial_weight if coords is not None else 0.0,
                **hkcc_kwargs).compress(X, coords=coords)
