"""HKCC -- Heat-Kernel Context Compression.

A training-free, modality-agnostic context compressor for token-efficient LLM inference.
It puts tokens on a similarity graph, runs the graph heat equation, keeps the distinctive
tokens, and folds the redundant ones into multiplicity-weighted centroids. The same method
applies to text tokens, document chunks, and image/vision patches -- under the hood it only
ever sees embedding vectors.

Quickstart
----------
>>> import numpy as np
>>> from hkcc import HKCC, proportional_attention
>>> X = np.random.randn(800, 64)                       # any embeddings (N, d)
>>> r = HKCC(target_fidelity=0.99).compress(X)         # compress to a quality floor
>>> out = proportional_attention(X[0], r.representatives,
...                              r.representatives, r.multiplicity)

Documents and images
---------------------
>>> from hkcc import compress_documents, compress_vision_tokens
>>> kept = compress_documents(["a.pdf", "b.md"], budget=0.2).kept_chunks
>>> vis  = compress_vision_tokens(patch_embeddings, grid_hw=(24, 24), budget=0.25)
"""
from .core import (
    build_knn_graph, normalized_laplacian, heat_diffuse, heat_residual, normalize_rows,
)
from .compress import HKCC, CompressionResult
from .helpers import proportional_attention, allocate_budget, estimate_cost
from .io_adapters import load_document, chunk_text, embed, Embedder
from .multimodal import compress_documents, compress_vision_tokens, DocumentCompression

__version__ = "0.2.0"

__all__ = [
    "HKCC", "CompressionResult",
    "compress_documents", "compress_vision_tokens", "DocumentCompression",
    "proportional_attention", "allocate_budget", "estimate_cost",
    "load_document", "chunk_text", "embed", "Embedder",
    "build_knn_graph", "normalized_laplacian", "heat_diffuse", "heat_residual", "normalize_rows",
]
