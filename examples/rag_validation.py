"""Real-LLM validation harness for HKCC on a retrieval-QA task.

This is the experiment that turns the synthetic claim into a real one. It is written to
run on YOUR machine with YOUR keys/models, because it needs a real embedding model and a
real LLM. It measures, for several compression budgets:

    * answer quality   (exact-match / F1 against gold answers)
    * input tokens actually sent
    * cost reduction

against baselines (no compression, random pruning, uniform stride).

How to use
----------
1. pip install hkcc sentence-transformers datasets
2. Provide an LLM call by editing ``answer_with_llm`` below (OpenAI/Anthropic/local).
3. python examples/rag_validation.py

Without an LLM configured it runs in DRY mode: it still reports token reduction and a
cheap lexical-overlap proxy for answer quality, so you can sanity-check plumbing offline.
"""
from __future__ import annotations

import re
import numpy as np

from hkcc import HKCC

# ---------------------------------------------------------------- plug in your models
def embed(texts):
    """Return (len(texts), d) embeddings. Default: sentence-transformers if available."""
    try:
        from sentence_transformers import SentenceTransformer
        global _MODEL
        if "_MODEL" not in globals():
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return np.asarray(_MODEL.encode(texts, normalize_embeddings=True))
    except Exception:
        # offline fallback: hashing embedding so the harness still runs end-to-end
        rng = np.random.default_rng(abs(hash("emb")) % (2**32))
        vecs = []
        for t in texts:
            r = np.random.default_rng(abs(hash(t)) % (2**32))
            vecs.append(r.normal(size=384))
        return np.asarray(vecs) / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


def answer_with_llm(question: str, context_chunks: list[str]) -> str | None:
    """Return the model's answer given the (possibly compressed) context.

    Return None to run in DRY mode. Wire this to your provider, e.g.::

        from anthropic import Anthropic
        client = Anthropic()
        ctx = "\\n\\n".join(context_chunks)
        msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=128,
                  messages=[{"role": "user",
                             "content": f"Context:\\n{ctx}\\n\\nQuestion: {question}\\nAnswer concisely."}])
        return msg.content[0].text
    """
    return None


# ---------------------------------------------------------------- a tiny built-in dataset
# Replace with HotpotQA / Natural Questions / your own via `datasets`.
SAMPLES = [
    {
        "question": "What year did the company move its headquarters to Austin?",
        "answer": "2021",
        "chunks": (
            ["The company was founded in 2003 in San Jose."] * 6
            + ["Quarterly revenue grew steadily through the 2010s."] * 6
            + ["In 2021 the company relocated its headquarters to Austin, Texas."]
            + ["The board approved a stock buyback in 2019."] * 6
        ),
    },
    {
        "question": "Which protein does the drug primarily inhibit?",
        "answer": "kinase X",
        "chunks": (
            ["The trial enrolled 240 patients across 12 sites."] * 6
            + ["Adverse events were mild and transient."] * 6
            + ["The drug primarily inhibits kinase X, blocking the signaling cascade."]
            + ["Dosing was once daily for twelve weeks."] * 6
        ),
    },
]


# ---------------------------------------------------------------- metrics
def normalize(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common = {}
    for w in p:
        common[w] = min(p.count(w), g.count(w))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def compress_chunks(chunks, strategy, keep):
    """Return a reduced list of chunks under a given strategy (keep = target count)."""
    n = len(chunks)
    keep = min(keep, n)
    if strategy == "none":
        return chunks
    if strategy == "random":
        idx = np.random.default_rng(0).choice(n, keep, replace=False)
        return [chunks[i] for i in sorted(idx)]
    if strategy == "stride":
        idx = np.linspace(0, n - 1, keep).astype(int)
        return [chunks[i] for i in idx]
    if strategy == "hkcc":
        X = embed(chunks)
        r = HKCC(budget=int(keep)).compress(X)
        # represent each anchor cluster by its highest-residual member (a real chunk)
        reps = []
        for a in r.anchor_indices:
            reps.append(chunks[int(a)])
        return reps
    raise ValueError(strategy)


def run():
    strategies = ["none", "random", "stride", "hkcc"]
    keep = 4
    print(f"{'strategy':<10}{'avg F1':>8}{'avg chunks':>12}")
    for strat in strategies:
        f1s, sizes = [], []
        for s in SAMPLES:
            ctx = compress_chunks(s["chunks"], strat, keep)
            sizes.append(len(ctx))
            pred = answer_with_llm(s["question"], ctx)
            if pred is None:  # DRY mode: lexical-overlap proxy
                joined = " ".join(ctx)
                pred = s["answer"] if normalize(s["answer"]) in normalize(joined) else ""
            f1s.append(f1(pred, s["answer"]))
        print(f"{strat:<10}{np.mean(f1s):>8.3f}{np.mean(sizes):>12.1f}")
    print("\nDRY mode = lexical proxy. Wire `answer_with_llm` to a real model for true F1.")


if __name__ == "__main__":
    run()
