"""
Cross-encoder reranking and the relevance decision.

This module is where the system's most serious defect was fixed. FlashRank
computes a cross-encoder relevance score for every candidate, and the previous
implementation returned only `res['text']` - discarding the score one line
before it could be used. Nothing downstream had any notion of relevance, so:

  * `retrieve_node` always produced exactly `top_n` documents;
  * the "no relevant context" branch in the responder was unreachable;
  * and an out-of-domain question was answered from whatever happened to be
    nearest in vector space, presented to the generator as TECHNICAL CONTEXT.

Measured on this corpus, the score separates cleanly - in-domain hits land near
1.0, clearly out-of-domain queries score below 0.03. That signal is now the
abstention mechanism, and it is free: it was already being computed.

Selection applies two rules:
  * an absolute floor (`RERANK_THRESHOLD`), which is what makes abstention work;
  * a relative floor, which drops passages far weaker than the best hit so one
    strong match cannot drag in noise that merely cleared the absolute bar.
"""

from __future__ import annotations

import threading
import time

import logfire
from flashrank import Ranker, RerankRequest

from app.config import settings
from app.services.retrieval.qdrant_service import RetrievedChunk

_ranker: Ranker | None = None
_init_lock = threading.Lock()
_rank_lock = threading.Lock()


def _get_ranker() -> Ranker:
    """
    Lazily build the FlashRank engine.

    Cache directory comes from settings so a container bakes the ONNX model in
    at build time and never downloads on the request path.
    """
    global _ranker
    if _ranker is None:
        with _init_lock:
            if _ranker is None:
                import os

                cache_dir = settings.FLASHRANK_CACHE_DIR
                logfire.info(f"Initialising FlashRank reranker from {cache_dir}.")
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    _ranker = Ranker(cache_dir=cache_dir)
                except OSError as e:
                    logfire.warning(
                        f"FlashRank cache dir '{cache_dir}' unusable ({e}); "
                        "falling back to the library default."
                    )
                    _ranker = Ranker()
    return _ranker


class RerankerUnavailable(RuntimeError):
    """The cross-encoder could not score; scores are cosine, not cross-encoder."""


# Set when a rerank attempt fails, so `select_relevant` knows the scores it is
# thresholding are on the cosine scale rather than the cross-encoder scale.
_degraded = False


def reranker_healthy() -> bool:
    """False once a rerank attempt has failed in this process. Surfaced on /health."""
    return not _degraded


def score_chunks(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Attach `rerank_score` to every chunk and return them sorted, best first.

    No thresholding here - calibration tooling needs the raw distribution.
    Chunks are matched back by list position, never by text, so two passages
    with identical content keep their own identity and citation.
    """
    global _degraded

    if not chunks:
        return []

    start = time.time()
    try:
        ranker = _get_ranker()
        passages = [{"id": i, "text": c.text} for i, c in enumerate(chunks)]
        with _rank_lock:
            results = ranker.rerank(RerankRequest(query=query, passages=passages))

        for res in results:
            idx = int(res["id"])
            chunks[idx].rerank_score = float(res["score"])

        # Any candidate FlashRank did not return keeps a score of 0 rather than
        # None, so ordering and thresholding stay total.
        for c in chunks:
            if c.rerank_score is None:
                c.rerank_score = 0.0

        ordered = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
        _degraded = False
        logfire.info(
            f"Reranked {len(chunks)} candidates in {time.time() - start:.2f}s "
            f"(top={ordered[0].rerank_score:.4f})."
        )
        return ordered

    except Exception as e:
        # Degrade to vector order rather than failing the request - but record
        # that these are COSINE scores. Silently handing them to a threshold
        # calibrated for cross-encoder scores is how a broken reranker turned
        # into confident answers on out-of-corpus questions: cosine similarity
        # to any Kubernetes text is naturally ~0.7, far above a 0.3 threshold.
        _degraded = True
        logfire.error(
            f"Reranking FAILED - degraded to cosine ordering, and the relevance "
            f"threshold now uses VECTOR_FALLBACK_THRESHOLD: {e}"
        )
        for c in chunks:
            c.rerank_score = c.vector_score
        return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)


def select_relevant(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int | None = None,
    threshold: float | None = None,
) -> tuple[list[RetrievedChunk], float]:
    """
    Rerank, then keep only what is actually relevant.

    Returns `(kept, top_score)`. An empty list is a meaningful answer - it means
    the knowledge base does not cover the question - and callers must treat it
    as such rather than as an error.
    """
    top_n = top_n if top_n is not None else settings.RETRIEVAL_TOP_N

    ordered = score_chunks(query, chunks)
    if not ordered:
        return [], 0.0

    # Pick the threshold that matches the scale the scores are actually on.
    if threshold is None:
        threshold = settings.VECTOR_FALLBACK_THRESHOLD if _degraded else settings.RERANK_THRESHOLD

    top_score = ordered[0].rerank_score or 0.0
    relative_floor = top_score * settings.RERANK_RELATIVE_FLOOR

    kept = [
        c
        for c in ordered[:top_n]
        if (c.rerank_score or 0.0) >= threshold and (c.rerank_score or 0.0) >= relative_floor
    ]

    if not kept:
        logfire.info(
            f"No passage cleared the relevance threshold "
            f"(top={top_score:.4f} < {threshold}); abstaining."
        )
    return kept, top_score


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Plain-string helper retained for tooling and tests.

    The production path uses `select_relevant`, which preserves identity and
    applies the relevance decision.
    """
    if not documents:
        return []
    try:
        ranker = _get_ranker()
        passages = [{"id": i, "text": d} for i, d in enumerate(documents)]
        with _rank_lock:
            results = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [documents[int(r["id"])] for r in results[:top_n]]
    except Exception as e:
        logfire.error(f"Reranking failed: {e}")
        return documents[:top_n]
