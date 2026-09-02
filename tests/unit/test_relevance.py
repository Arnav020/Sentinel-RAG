"""
Relevance and abstention.

This is the regression guard for the most serious defect the audit found: the
system had no notion of relevance at all, so it returned a fixed number of
passages for every question and answered "how do I bake sourdough bread" from
an operating-system-kernel textbook.

These tests fake the cross-encoder, so they assert the *decision logic* without
needing the ONNX model or a network call.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.services.retrieval import ranking_service as rs
from app.services.retrieval.qdrant_service import RetrievedChunk


def chunk(idx: int, text: str = "body", source: str = "doc.html") -> RetrievedChunk:
    return RetrievedChunk(
        id=f"id-{idx}",
        text=text,
        body=text,
        source=source,
        title="Doc",
        source_url="",
        doc_topic="workloads",
        heading_path="Doc > Section",
        kind="prose",
        chunk_index=idx,
        vector_score=0.7,
    )


@pytest.fixture(autouse=True)
def _reset_degraded(monkeypatch):
    monkeypatch.setattr(rs, "_degraded", False)


def fake_ranker(scores: list[float], monkeypatch):
    """Patch score_chunks to assign the given scores in candidate order."""

    def _score(query, chunks):
        for c, s in zip(chunks, scores, strict=False):
            c.rerank_score = s
        return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)

    monkeypatch.setattr(rs, "score_chunks", _score)


class TestSelectRelevant:
    def test_keeps_passages_above_threshold(self, monkeypatch):
        fake_ranker([0.99, 0.95, 0.90], monkeypatch)
        kept, top = rs.select_relevant("q", [chunk(i) for i in range(3)])
        assert len(kept) == 3
        assert top == pytest.approx(0.99)

    def test_abstains_when_nothing_clears_threshold(self, monkeypatch):
        """The bread-and-France case: strong candidates exist, none are relevant."""
        fake_ranker([0.02, 0.01, 0.0], monkeypatch)
        kept, top = rs.select_relevant(
            "how do I bake sourdough bread", [chunk(i) for i in range(3)]
        )
        assert kept == []
        assert top == pytest.approx(0.02)

    def test_drops_only_the_weak_passages(self, monkeypatch):
        fake_ranker([0.98, 0.85, 0.05, 0.001], monkeypatch)
        kept, _ = rs.select_relevant("q", [chunk(i) for i in range(4)])
        assert len(kept) == 2
        assert all((c.rerank_score or 0) >= settings.RERANK_THRESHOLD for c in kept)

    def test_relative_floor_drops_far_weaker_neighbours(self, monkeypatch):
        monkeypatch.setattr(settings, "RERANK_THRESHOLD", 0.0)
        monkeypatch.setattr(settings, "RERANK_RELATIVE_FLOOR", 0.5)
        fake_ranker([1.0, 0.9, 0.1], monkeypatch)
        kept, _ = rs.select_relevant("q", [chunk(i) for i in range(3)])
        assert len(kept) == 2, "0.1 is below half of the 1.0 top hit"

    def test_respects_top_n(self, monkeypatch):
        fake_ranker([0.99] * 10, monkeypatch)
        kept, _ = rs.select_relevant("q", [chunk(i) for i in range(10)], top_n=3)
        assert len(kept) == 3

    def test_empty_candidates(self, monkeypatch):
        kept, top = rs.select_relevant("q", [])
        assert kept == [] and top == 0.0

    def test_scores_are_attached_to_chunks(self, monkeypatch):
        fake_ranker([0.9, 0.8], monkeypatch)
        kept, _ = rs.select_relevant("q", [chunk(0), chunk(1)])
        assert all(c.rerank_score is not None for c in kept)


class TestDegradedReranker:
    def test_failure_marks_degraded_and_uses_vector_threshold(self, monkeypatch):
        """
        A broken cross-encoder must not have its cosine fallback scored against
        a cross-encoder threshold. Cosine similarity to any in-domain text is
        naturally ~0.7, far above a 0.3 rerank threshold, so the silent fallback
        turned a dead reranker into confident answers on everything.
        """

        class Boom:
            def rerank(self, *_a, **_k):
                raise RuntimeError("onnx model missing")

        monkeypatch.setattr(rs, "_get_ranker", lambda: Boom())
        monkeypatch.setattr(settings, "VECTOR_FALLBACK_THRESHOLD", 0.8)

        c = chunk(0)
        c.vector_score = 0.72  # would clear RERANK_THRESHOLD=0.3
        kept, top = rs.select_relevant("q", [c])

        assert rs.reranker_healthy() is False
        assert kept == [], "cosine 0.72 must not clear the 0.8 vector-fallback threshold"
        assert top == pytest.approx(0.72)

    def test_healthy_by_default(self):
        assert rs.reranker_healthy() is True


class TestIdentityPreservation:
    def test_duplicate_text_keeps_separate_identity(self, monkeypatch):
        """
        Two passages with identical text previously collapsed to one citation,
        because the handoff between retrieval and reranking was keyed on text.
        """
        fake_ranker([0.9, 0.8], monkeypatch)
        a = chunk(0, text="identical text", source="a.html")
        b = chunk(1, text="identical text", source="b.html")
        kept, _ = rs.select_relevant("q", [a, b])
        assert {c.source for c in kept} == {"a.html", "b.html"}
        assert len({c.id for c in kept}) == 2
