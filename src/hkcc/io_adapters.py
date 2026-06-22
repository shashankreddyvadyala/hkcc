"""Turn documents of any type into chunks + embeddings, ready for HKCC.

HKCC itself only ever sees vectors, so "supporting a document type" just means: parse it
to text, chunk it, and embed the chunks. Plain-text formats are handled natively; PDF and
DOCX use optional parsers if installed. The embedder is pluggable -- bring your own, or use
the bundled sentence-transformers wrapper (with a deterministic offline fallback).
"""
from __future__ import annotations

import os
import re

import numpy as np

__all__ = ["load_document", "chunk_text", "embed", "Embedder"]

_TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".java",
            ".c", ".cpp", ".go", ".rs", ".json", ".csv", ".tsv", ".log", ".sql", ".yaml", ".yml"}


def _strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return re.sub(r"[ \t]*\n[ \t]*", "\n", s)


def load_document(path: str) -> str:
    """Read a file to plain text.

    Native: .txt .md .html .py .json .csv (and similar). Optional: .pdf (needs ``pypdf``),
    .docx (needs ``python-docx``). Raises a clear error if an optional parser is missing.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".html", ".htm"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return _strip_html(f.read())
    if ext in _TEXT_EXT:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("PDF support needs `pip install pypdf`") from e
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    if ext == ".docx":
        try:
            import docx
        except ImportError as e:
            raise ImportError("DOCX support needs `pip install python-docx`") from e
        return "\n\n".join(p.text for p in docx.Document(path).paragraphs)
    # last resort: try as utf-8 text
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, max_chars: int = 500, overlap: int = 0) -> list[str]:
    """Split text into chunks at paragraph boundaries, capped at ``max_chars``."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            # a single huge paragraph: hard-split it
            while len(p) > max_chars:
                chunks.append(p[:max_chars])
                p = p[max_chars - overlap:]
            buf = p
    if buf:
        chunks.append(buf)
    return chunks or [text[:max_chars]]


class Embedder:
    """Default text embedder.

    Uses sentence-transformers (``all-MiniLM-L6-v2``) when available. Falls back to a
    deterministic hashing embedding so pipelines still run offline / in CI -- good enough
    to exercise plumbing, not for production quality.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.dim = dim
        self._st = None
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(model)
            self.dim = self._st.get_sentence_embedding_dimension()
        except Exception:
            self._st = None

    def __call__(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            return np.asarray(self._st.encode(list(texts), normalize_embeddings=True))
        vecs = np.empty((len(texts), self.dim))
        for i, t in enumerate(texts):
            r = np.random.default_rng(abs(hash(t)) % (2 ** 32))
            vecs[i] = r.normal(size=self.dim)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


_DEFAULT: Embedder | None = None


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of strings with the default (lazily-built) :class:`Embedder`."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Embedder()
    return _DEFAULT(texts)
