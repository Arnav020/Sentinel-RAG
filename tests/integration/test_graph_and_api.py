"""
Graph routing and the HTTP contract, with retrieval and generation stubbed.

The behaviours asserted here are the ones the audit found were either unreachable
or wrong end-to-end:

  * abstention actually reaching the user as a refusal to answer;
  * a retrieval outage being distinguishable from "not covered";
  * conversational turns skipping retrieval entirely;
  * errors surfacing as 5xx rather than HTTP 200 with an apology, which made an
    outage invisible to monitoring;
  * session ownership being enforced.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agents.state import CONVERSATIONAL
from app.services.retrieval.qdrant_service import RetrievalError, RetrievedChunk


def make_chunk(idx: int = 0, score: float = 0.95) -> RetrievedChunk:
    c = RetrievedChunk(
        id=f"id-{idx}",
        text="[Jobs > Restart policies]\nA Job supports OnFailure and Never.",
        body="A Job supports OnFailure and Never.",
        source="kubernetes/jobs.html",
        title="Jobs",
        source_url="https://kubernetes.io/docs/concepts/workloads/controllers/job/",
        doc_topic="workloads",
        heading_path="Jobs > Restart policies",
        kind="prose",
        chunk_index=idx,
        vector_score=0.8,
    )
    c.rerank_score = score
    return c


@pytest.fixture
def graph_env(monkeypatch):
    """Stub retrieval + generation; return a builder to configure each case."""
    import app.agents.nodes.planner as planner_mod
    import app.agents.nodes.responder as responder_mod
    import app.agents.nodes.retriever as retriever_mod
    from app.gateway import client as gw

    def build(
        *,
        plan_reply="kubernetes job restart policy",
        chunks=None,
        retrieval_error=False,
        answer="A Job supports OnFailure or Never [1].",
    ):
        def fake_search(query, limit=None, doc_topics=None):
            if retrieval_error:
                raise RetrievalError("qdrant down")
            return list(chunks or [])

        def fake_select(query, candidates, top_n=None, threshold=None):
            kept = [c for c in candidates if (c.rerank_score or 0) >= 0.3]
            top = max((c.rerank_score or 0) for c in candidates) if candidates else 0.0
            return kept, top

        monkeypatch.setattr(retriever_mod, "search", fake_search)
        monkeypatch.setattr(retriever_mod, "select_relevant", fake_select)

        def fake_complete(messages, *, model=None, feature="rag", **kw):
            reply = plan_reply if feature == "planner" else answer
            return gw.LLMResponse(content=reply, model=model or "stub")

        monkeypatch.setattr(planner_mod, "complete", fake_complete)
        monkeypatch.setattr(responder_mod, "complete", fake_complete)

        from app.agents.graph import build_graph

        return build_graph()

    return build


def run(agent, question: str) -> dict:
    state = {
        "messages": [{"role": "user", "content": question}],
        "current_query": question,
        "retrieved": [],
        "citations": [],
        "plan": [],
        "status": "start",
    }
    return agent.invoke(state, config={"configurable": {"thread_id": "t1"}})


class TestGraphRouting:
    def test_answers_when_passages_are_relevant(self, graph_env):
        agent = graph_env(chunks=[make_chunk(0, 0.95), make_chunk(1, 0.88)])
        out = run(agent, "What restart policies can a Job use?")
        assert out["abstained"] is False
        assert "OnFailure" in out["final_answer"]
        assert len(out["citations"]) == 2

    def test_abstains_when_nothing_is_relevant(self, graph_env):
        """The regression guard for the flagship defect."""
        agent = graph_env(chunks=[make_chunk(0, 0.01), make_chunk(1, 0.0)])
        out = run(agent, "How do I bake sourdough bread?")
        assert out["abstained"] is True
        assert out["citations"] == []
        assert "could not find" in out["final_answer"].lower()

    def test_abstention_does_not_call_the_generator(self, graph_env, monkeypatch):
        import app.agents.nodes.responder as responder_mod

        calls = []
        agent = graph_env(chunks=[make_chunk(0, 0.0)])
        original = responder_mod.complete
        monkeypatch.setattr(
            responder_mod,
            "complete",
            lambda *a, **k: (calls.append(1), original(*a, **k))[1],
        )
        run(agent, "How do I bake sourdough bread?")
        assert calls == [], "abstention must not spend a generation call"

    def test_retrieval_outage_is_distinct_from_not_covered(self, graph_env):
        agent = graph_env(retrieval_error=True)
        out = run(agent, "What restart policies can a Job use?")
        answer = out["final_answer"].lower()
        assert "could not reach" in answer or "problem on my side" in answer
        assert "could not find anything in the documentation" not in answer

    def test_conversational_skips_retrieval(self, graph_env):
        agent = graph_env(plan_reply="CONVERSATIONAL", answer="Hello! How can I help?")
        out = run(agent, "hi there")
        assert out["current_query"] == CONVERSATIONAL
        assert out.get("citations", []) == []

    def test_planner_normalises_noisy_token(self, graph_env):
        agent = graph_env(plan_reply='"CONVERSATIONAL."', answer="Hi!")
        out = run(agent, "hello")
        assert out["current_query"] == CONVERSATIONAL

    def test_citations_carry_provenance(self, graph_env):
        agent = graph_env(chunks=[make_chunk(0, 0.95)])
        out = run(agent, "What restart policies can a Job use?")
        cite = out["citations"][0]
        assert cite["section"] == "Jobs > Restart policies"
        assert cite["url"].startswith("https://kubernetes.io/")
        assert cite["score"] == pytest.approx(0.95)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "guard", lambda q, history="": (False, None))
    # raise_server_exceptions=False so the app's own exception handler produces
    # the response, which is exactly what we need to assert on. With the default
    # TestClient re-raises and the handler's 500 is never observed.
    return TestClient(main_mod.app, raise_server_exceptions=False)


class TestApiContract:
    def test_health_reports_subsystems(self, client, monkeypatch):
        import app.main as main_mod

        monkeypatch.setattr(main_mod, "collection_stats", lambda: {"points": 10, "dim": 768})
        body = client.get("/health").json()
        assert set(body) >= {"status", "guardrails", "corpus", "auth_required", "models"}

    def test_health_is_degraded_when_corpus_is_down(self, client, monkeypatch):
        import app.main as main_mod

        def boom():
            raise RetrievalError("down")

        monkeypatch.setattr(main_mod, "collection_stats", boom)
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["corpus"]["available"] is False

    def test_empty_question_is_rejected(self, client):
        assert client.post("/query", json={"q": ""}).status_code == 422

    def test_oversized_question_is_rejected(self, client):
        assert client.post("/query", json={"q": "x" * 5000}).status_code == 422

    def test_unknown_session_is_rejected(self, client):
        r = client.post("/query", json={"q": "hello", "session_id": "not-a-real-session"})
        assert r.status_code == 404

    def test_session_roundtrip(self, client, monkeypatch):
        import app.main as main_mod

        monkeypatch.setattr(
            main_mod.rag_agent,
            "invoke",
            lambda *a, **k: {"final_answer": "hi", "plan": [], "citations": [], "status": "ok"},
        )
        sid = client.post("/session").json()["session_id"]
        r = client.post("/query", json={"q": "hello", "session_id": sid})
        assert r.status_code == 200

    def test_internal_error_is_5xx_not_200(self, client, monkeypatch):
        """
        An outage must look like an outage. The previous handler returned HTTP
        200 with an apology, so monitoring saw a 0% error rate and the eval
        harness scored the apology as a bad answer instead of an incident.
        """
        import app.main as main_mod

        def boom(*a, **k):
            raise RuntimeError("graph exploded")

        monkeypatch.setattr(main_mod.rag_agent, "invoke", boom)
        r = client.post("/query", json={"q": "What is a Job?"})
        assert r.status_code == 500
        assert "correlation_id" in r.json()

    def test_scope_endpoint_lists_topics(self, client, monkeypatch):
        from app.services import scope as scope_mod

        scope_mod.reset_for_tests(
            scope_mod.Scope(subject="Kubernetes", topics={"workloads": 5}, points=5, available=True)
        )
        body = client.get("/scope").json()
        assert body["subject"] == "Kubernetes"
        assert body["topics"][0]["key"] == "workloads"
        scope_mod.reset_for_tests(None)
