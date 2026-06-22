# HKCC — Heat-Kernel Context Compression

[![CI](https://github.com/svadyala/hkcc/actions/workflows/ci.yml/badge.svg)](https://github.com/svadyala/hkcc/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/hkcc.svg)](https://pypi.org/project/hkcc/) [![Python](https://img.shields.io/pypi/pyversions/hkcc.svg)](https://pypi.org/project/hkcc/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Training-free context compression for token-efficient LLM inference — text, documents, and images.**

Long contexts are mostly redundant tokens you pay for on every call. HKCC puts your tokens on a
similarity graph, runs the **graph heat equation** over their embeddings, keeps the *distinctive*
tokens, and folds the *redundant* ones into multiplicity-weighted centroids — so you send far
fewer tokens while the model sees almost the same thing.

It works on anything you can embed: **text tokens, document chunks, or image/vision patches.**
Under the hood HKCC only ever sees vectors, so the same method spans modalities.

```bash
pip install hkcc
```

```python
import numpy as np
from hkcc import HKCC, proportional_attention

X = np.random.randn(800, 64)                  # any embeddings (N, d)
result = HKCC(target_fidelity=0.99).compress(X)   # compress to a quality floor

result.representatives    # (M, d)  -> send these to the model
result.multiplicity       # (M,)    -> pass to proportional_attention
result.fidelity           # estimated attention fidelity it achieved
```

---

## Set a quality floor instead of guessing a budget

You can't promise a lossy compressor never loses anything — so don't. Set a **fidelity target**
and HKCC keeps the fewest tokens that stay above it:

```python
HKCC(target_fidelity=0.99).compress(X)   # as small as possible, fidelity >= 0.99
HKCC(budget=0.1).compress(X)             # or just keep 10% if you prefer a fixed budget
```

The fidelity is estimated from in-distribution probe queries — a self-consistent estimate, not a
guarantee on your downstream metric. Validate on your task (harness included).

---

## Documents — any type

```python
from hkcc import compress_documents

out = compress_documents(["report.pdf", "notes.md", "page.html"], budget=0.2)
out.kept_chunks        # the surviving real chunks, in original order
```

For text, HKCC acts as a **redundancy-aware selector** (you can't detokenize a centroid back into
words, so it keeps real chunks and drops near-duplicates) — ideal for condensing retrieved RAG
context. Plain-text formats work out of the box; `pip install pypdf python-docx` adds PDF/DOCX.

## Images — vision tokens

```python
from hkcc import compress_vision_tokens

# patch_embeddings: (N, d) from any ViT-style encoder; grid_hw makes the graph position-aware
vis = compress_vision_tokens(patch_embeddings, grid_hw=(24, 24), target_fidelity=0.99)
vis.representatives     # merged visual tokens, usable directly
```

Image patches are just vectors, so HKCC **merges** them (this is the setting closest to the
token-merging literature). Because attention is quadratic in token count, cutting vision tokens
gives a **superlinear** latency win — keep 20% of tokens and attention does ~25× less work.

---

## Benchmarks (synthetic)

Text, attention-output fidelity vs. kept fraction (`benchmarks/synthetic_benchmark.py`):

| kept | random | stride | prune-only | **HKCC** |
|-----:|-------:|-------:|-----------:|---------:|
|   5% |  0.751 |  0.846 |      0.085 | **1.000** |
|   8% |  0.818 |  0.871 |      0.269 | **1.000** |
|  12% |  0.864 |  0.919 |      0.672 | **1.000** |

`prune-only` keeps the same distinctive tokens HKCC does but *deletes* the rest, and collapses —
the diffused merge is doing the work.

Vision, fidelity-target on a 32×32 patch field (`benchmarks/multimodal_benchmark.py`):

| floor | tokens kept | achieved | attention compute |
|------:|------------:|---------:|------------------:|
| 0.999 |       21.1% |   0.9994 |        ~22× lower |

---

## Why it holds up

- **A score from physics.** The heat residual is the per-token magnitude of high-frequency
  content — a principled saliency signal, set by one diffusion time.
- **Nothing invented.** Every representative is a convex combination of real tokens, so a
  compressed token can never take a value outside the original context. Auditable by construction.
- **Linear time.** Diffusion runs through a Chebyshev approximation — never the matrix exponential.

Full derivation, spectral analysis, and experiments: [`paper/hkcc.pdf`](paper/hkcc.pdf).

---

## Status — read before depending on it

Research-stage. The numbers above are on **synthetic data** with an attention-output proxy; they
isolate the mechanism but are **not** a real-LLM benchmark yet. The honest next step is wiring it
into a real retrieval-QA / vision pipeline and reporting task quality against measured token
reduction. A ready-to-run harness is in [`examples/rag_validation.py`](examples/rag_validation.py).
If you run it, **please open an issue/PR with your numbers.**

## Develop

```bash
git clone https://github.com/svadyala/hkcc && cd hkcc
pip install -e ".[dev]"
pytest -q
```

## Cite

```bibtex
@misc{vadyala2026hkcc,
  title  = {Heat-Kernel Context Compression: Continuous-Time Diffusion for Token-Efficient
            Language-Model Inference},
  author = {Vadyala, Shashank Reddy},
  year   = {2026},
}
```

## License

MIT.
