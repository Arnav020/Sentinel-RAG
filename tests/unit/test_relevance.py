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


class TestEntailmentGate:
    """
    The gate that closes what the relevance threshold provably cannot: a
    cross-encoder scores topical relatedness, so an in-domain question the
    corpus does not cover can clear the threshold on shared vocabulary alone.
    """

    @pytest.mark.parametrize(
        "verdict,answers",
        [
            ("YES", True),
            ("yes", True),
            ("Yes, passage 2 states it", True),
            ("NO", False),
            ("no.", False),
            ("**NO**", False),
            ("NO - the passages only mention the topic", False),
            ("", True),  # unparseable must fail open, not silently refuse
            ("maybe?", True),
        ],
    )
    def test_verdict_parsing(self, verdict, answers):
        from app.services.retrieval.entailment import _verdict_says_yes

        assert _verdict_says_yes(verdict) is answers

    def test_disabled_gate_passes_without_calling_a_model(self, monkeypatch):
        from app.services.retrieval import entailment as ent

        called = []
        monkeypatch.setattr(settings, "ENTAILMENT_GATE_ENABLED", False)
        monkeypatch.setattr(ent, "complete", lambda *a, **k: called.append(1))

        answers, checked = ent.context_answers_question("q", [chunk(0)])
        assert (answers, checked) == (True, False)
        assert called == [], "a disabled gate must not spend a model call"

    def test_empty_context_is_not_checked(self, monkeypatch):
        from app.services.retrieval import entailment as ent

        monkeypatch.setattr(settings, "ENTAILMENT_GATE_ENABLED", True)
        assert ent.context_answers_question("q", []) == (True, False)

    def test_rejects_when_model_says_no(self, monkeypatch):
        from app.gateway import client as gw
        from app.services.retrieval import entailment as ent

        monkeypatch.setattr(settings, "ENTAILMENT_GATE_ENABLED", True)
        monkeypatch.setattr(
            ent, "complete", lambda *a, **k: gw.LLMResponse(content="NO", model="stub")
        )
        assert ent.context_answers_question("q", [chunk(0)]) == (False, True)

    def test_accepts_when_model_says_yes(self, monkeypatch):
        from app.gateway import client as gw
        from app.services.retrieval import entailment as ent

        monkeypatch.setattr(settings, "ENTAILMENT_GATE_ENABLED", True)
        monkeypatch.setattr(
            ent, "complete", lambda *a, **k: gw.LLMResponse(content="YES", model="stub")
        )
        assert ent.context_answers_question("q", [chunk(0)]) == (True, True)

    def test_fails_open_when_model_unreachable(self, monkeypatch):
        """
        An outage here must not turn into mass false abstentions. The generator's
        decline instruction is still downstream, so degrading to "allow" is the
        same posture as the topical filter, for the same reason.
        """
        from app.gateway import client as gw
        from app.services.retrieval import entailment as ent

        def boom(*a, **k):
            raise gw.LLMError("gate down")

        monkeypatch.setattr(settings, "ENTAILMENT_GATE_ENABLED", True)
        monkeypatch.setattr(ent, "complete", boom)
        assert ent.context_answers_question("q", [chunk(0)]) == (True, False)

    def test_only_top_k_passages_are_sent(self, monkeypatch):
        from app.gateway import client as gw
        from app.services.retrieval import entailment as ent

        seen = {}

        def capture(messages, **k):
            seen["user"] = messages[1]["content"]
            return gw.LLMResponse(content="YES", model="stub")

        monkeypatch.setattr(settings, "ENTAILMENT_GATE_ENABLED", True)
        monkeypatch.setattr(settings, "ENTAILMENT_TOP_K", 2)
        monkeypatch.setattr(ent, "complete", capture)

        ent.context_answers_question("q", [chunk(i, text=f"body{i}") for i in range(5)])
        assert "body0" in seen["user"] and "body1" in seen["user"]
        assert "body2" not in seen["user"], "sent more than ENTAILMENT_TOP_K passages"


class TestMultiQueryRerank:
    """
    The cross-encoder is phrasing-sensitive enough to bury the right passage.

    Measured on the live corpus: the passage stating "Only a RestartPolicy equal
    to Never or OnFailure is allowed" scored 0.0003 against "What restart
    policies can a Kubernetes Job use?" (rank 65 of 78) and 0.9997 against "Job
    pod template restartPolicy Never OnFailure allowed" (rank 2). Reranking
    against both phrasings and keeping the maximum is what makes that question
    answerable.
    """

    @staticmethod
    def _per_query_ranker(table: dict[str, list[float]], monkeypatch):
        """Fake a cross-encoder whose scores depend on how the query is phrased."""

        def _score(query, chunks):
            for c, s in zip(chunks, table[query], strict=False):
                c.rerank_score = s
            return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)

        monkeypatch.setattr(rs, "score_chunks", _score)

    def test_keyword_phrasing_rescues_a_passage_the_prose_form_buries(self, monkeypatch):
        self._per_query_ranker(
            {
                "what restart policies can a Job use": [0.99, 0.0003],
                "Job restartPolicy Never OnFailure": [0.10, 0.9997],
            },
            monkeypatch,
        )
        chunks = [chunk(0, "pod failure policy"), chunk(1, "Only a RestartPolicy ...")]
        kept, top = rs.select_relevant(
            "what restart policies can a Job use",
            chunks,
            extra_queries=["Job restartPolicy Never OnFailure"],
        )
        assert top == pytest.approx(0.9997)
        assert kept[0].text.startswith("Only a RestartPolicy")

    def test_takes_the_max_not_the_last_score(self, monkeypatch):
        """A weak alternative phrasing must not overwrite a strong primary score."""
        self._per_query_ranker({"primary": [0.95], "weak alt": [0.01]}, monkeypatch)
        kept, top = rs.select_relevant("primary", [chunk(0)], extra_queries=["weak alt"])
        assert top == pytest.approx(0.95)
        assert len(kept) == 1

    def test_off_topic_stays_off_topic_under_both_phrasings(self, monkeypatch):
        """Taking the max must not weaken abstention."""
        self._per_query_ranker({"bake bread": [0.02], "sourdough starter": [0.03]}, monkeypatch)
        kept, top = rs.select_relevant(
            "bake bread", [chunk(0)], extra_queries=["sourdough starter"]
        )
        assert kept == []
        assert top == pytest.approx(0.03)

    def test_duplicate_and_empty_phrasings_are_ignored(self, monkeypatch):
        self._per_query_ranker({"q": [0.9]}, monkeypatch)
        kept, top = rs.select_relevant("q", [chunk(0)], extra_queries=["q", "", "   "])
        assert top == pytest.approx(0.9)
        assert len(kept) == 1


class TestGateWindowMatchesGeneratorContext:
    """
    The answer-presence gate must judge the same passages the generator gets.

    It did not: ENTAILMENT_TOP_K was 3 while RETRIEVAL_TOP_N was 5, so the gate
    was asked whether the answer was present in a context two passages smaller
    than the one about to be used. Measured on the 33 questions the gate
    rejected, widening it to 5 recovered 4 of 5 false abstentions and kept all
    28 correct rejections - the questions it had been refusing had their
    evidence in passages 4 and 5.
    """

    def test_gate_window_is_not_narrower_than_the_generator_context(self):
        assert settings.ENTAILMENT_TOP_K >= settings.RETRIEVAL_TOP_N

    def test_validate_rejects_a_narrower_gate_window(self, monkeypatch):
        # validate() is a classmethod reading cls.X, so the class is what must
        # be patched - setting the attribute on the instance is invisible to it.
        cls = type(settings)
        monkeypatch.setattr(cls, "ENTAILMENT_GATE_ENABLED", True)
        monkeypatch.setattr(cls, "ENTAILMENT_TOP_K", 3)
        monkeypatch.setattr(cls, "RETRIEVAL_TOP_N", 5)
        with pytest.raises(ValueError, match="smaller than"):
            settings.validate()

    def test_the_widening_was_kept_free(self):
        """5 x 660 is the same character budget the gate read at 3 x 1100."""
        budget = settings.ENTAILMENT_TOP_K * settings.ENTAILMENT_MAX_CHARS
        assert budget <= 3300


class TestEntailmentBudgetAllocation:
    """
    The gate's character budget is shared, not per-passage.

    A flat cap truncated the passages carrying the answer while short passages
    left their allowance unused. Measured: five passages of 654/1081/252/972/671
    characters under a uniform 660 cap sliced one answer sentence 43 characters
    in, turning two answerable questions into refusals.
    """

    @staticmethod
    def _alloc(lengths):
        from app.services.retrieval.entailment import allocate_budget

        return [len(b) for b in allocate_budget(["x" * n for n in lengths])]

    def test_short_passages_donate_to_long_ones(self):
        kept = self._alloc([654, 1081, 252, 972, 671])
        assert kept[2] == 252, "a short passage must never be padded or cut"
        assert kept[1] > 660, "the long passage must receive the donated budget"
        assert kept[1] > 617, "the A005 answer sentence must survive"

    def test_nothing_is_truncated_when_everything_fits(self):
        lengths = [907, 287, 870, 318, 467]  # the A064 case, 2849 total
        assert self._alloc(lengths) == lengths

    def test_the_total_budget_is_respected(self):
        budget = settings.ENTAILMENT_TOP_K * settings.ENTAILMENT_MAX_CHARS
        for lengths in ([5000, 5000, 5000, 5000, 5000], [654, 1081, 252, 972, 671], [9000]):
            assert sum(self._alloc(lengths)) <= budget

    def test_an_even_split_when_all_passages_are_oversized(self):
        kept = self._alloc([5000] * 5)
        assert max(kept) - min(kept) <= 1, "equally oversized passages share equally"

    def test_empty_input_is_handled(self):
        from app.services.retrieval.entailment import allocate_budget

        assert allocate_budget([]) == []
