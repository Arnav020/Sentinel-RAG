"""
Session isolation, rate limiting, bounded history, and configuration validation.

The session tests exist because `thread_id` was previously a client-supplied
string keying the conversation store, so any caller could read another user's
conversation by guessing or reusing its id - and every client that omitted the
field shared one thread called "default_user".
"""

from __future__ import annotations

import pytest

from app.agents.state import format_history, latest_user_message, recent_history
from app.api.security import RateLimiter, SessionStore
from app.config import Settings, settings


class TestSessionStore:
    def test_ids_are_unguessable(self):
        store = SessionStore()
        ids = {store.create("owner").id for _ in range(200)}
        assert len(ids) == 200
        assert all(len(i) >= 20 for i in ids)

    def test_owner_cannot_be_impersonated(self):
        store = SessionStore()
        session = store.create("key:alice")
        assert store.get(session.id, "key:alice") is not None
        assert store.get(session.id, "key:mallory") is None

    def test_unknown_session_is_rejected(self):
        store = SessionStore()
        assert store.get("made-up-id", "key:alice") is None

    def test_expired_session_is_evicted(self, monkeypatch):
        import app.api.security as sec

        store = SessionStore()
        session = store.create("key:alice")
        monkeypatch.setattr(sec, "SESSION_TTL_SECONDS", -1)
        assert store.get(session.id, "key:alice") is None

    def test_capacity_is_bounded(self, monkeypatch):
        import app.api.security as sec

        monkeypatch.setattr(sec, "MAX_SESSIONS", 10)
        store = SessionStore()
        for _ in range(50):
            store.create("key:alice")
        assert store.count() <= 10

    def test_turns_are_counted(self):
        store = SessionStore()
        s = store.create("o")
        for _ in range(3):
            store.get(s.id, "o")
        assert store.get(s.id, "o").turns == 4


class TestRateLimiter:
    def test_allows_up_to_limit(self):
        rl = RateLimiter(per_minute=5)
        assert all(rl.check("id")[0] for _ in range(5))

    def test_blocks_beyond_limit(self):
        rl = RateLimiter(per_minute=3)
        for _ in range(3):
            rl.check("id")
        allowed, _ = rl.check("id")
        assert allowed is False

    def test_identities_are_independent(self):
        rl = RateLimiter(per_minute=2)
        rl.check("a"), rl.check("a")
        assert rl.check("a")[0] is False
        assert rl.check("b")[0] is True

    def test_disabled_when_zero(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
        rl = RateLimiter(per_minute=0)
        assert all(rl.check("id")[0] for _ in range(100))


class TestBoundedHistory:
    def _messages(self, n: int) -> list[dict]:
        out = []
        for i in range(n):
            out.append({"role": "user", "content": f"question {i}"})
            out.append({"role": "assistant", "content": f"answer {i}"})
        return out

    def test_turn_count_is_bounded(self, monkeypatch):
        monkeypatch.setattr(settings, "HISTORY_MAX_TURNS", 4)
        monkeypatch.setattr(settings, "HISTORY_MAX_CHARS", 100000)
        history = recent_history(self._messages(50))
        assert len(history) <= 4

    def test_character_budget_is_bounded(self, monkeypatch):
        monkeypatch.setattr(settings, "HISTORY_MAX_TURNS", 100)
        monkeypatch.setattr(settings, "HISTORY_MAX_CHARS", 120)
        msgs = [{"role": "user", "content": "x" * 50} for _ in range(20)]
        history = recent_history(msgs)
        assert sum(len(m["content"]) for m in history) <= 120

    def test_keeps_the_most_recent(self, monkeypatch):
        monkeypatch.setattr(settings, "HISTORY_MAX_TURNS", 2)
        # exclude_last drops the in-flight message, so the newest retained entry
        # is the user turn immediately before it.
        history = recent_history(self._messages(10))
        assert "question 9" in history[-1]["content"]
        assert "answer 8" in history[0]["content"]
        assert not any("answer 9" in m["content"] for m in history)

    def test_excludes_the_current_message(self):
        msgs = [{"role": "user", "content": "old"}, {"role": "user", "content": "current"}]
        assert "current" not in format_history(msgs)

    def test_latest_user_message(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert latest_user_message(msgs) == "second"

    def test_empty_history(self):
        assert format_history([]) == ""
        assert latest_user_message([]) == ""


class TestConfigValidation:
    def test_rejects_same_family_judge(self, monkeypatch):
        """
        A judge from the generator's own family is self-evaluation, and the whole
        credibility of the reported numbers rests on that not happening.
        """
        monkeypatch.setattr(Settings, "GENERATION_MODEL", "openai/gpt-oss-120b")
        monkeypatch.setattr(Settings, "JUDGE_MODEL", "openai/gpt-oss-20b")
        with pytest.raises(ValueError, match="same family"):
            Settings.validate()

    def test_accepts_independent_judge(self, monkeypatch):
        monkeypatch.setattr(Settings, "GENERATION_MODEL", "openai/gpt-oss-120b")
        monkeypatch.setattr(Settings, "JUDGE_MODEL", "qwen/qwen3.8-27b")
        Settings.validate()

    def test_rejects_overlap_larger_than_chunk(self, monkeypatch):
        monkeypatch.setattr(Settings, "CHUNK_SIZE", 100)
        monkeypatch.setattr(Settings, "CHUNK_OVERLAP", 100)
        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            Settings.validate()

    def test_rejects_top_n_above_candidates(self, monkeypatch):
        monkeypatch.setattr(Settings, "RETRIEVAL_TOP_N", 50)
        monkeypatch.setattr(Settings, "RETRIEVAL_CANDIDATES", 20)
        with pytest.raises(ValueError, match="RETRIEVAL_TOP_N"):
            Settings.validate()

    def test_rejects_out_of_range_threshold(self, monkeypatch):
        monkeypatch.setattr(Settings, "RERANK_THRESHOLD", 1.5)
        with pytest.raises(ValueError, match="RERANK_THRESHOLD"):
            Settings.validate()

    def test_reports_missing_secrets(self, monkeypatch):
        monkeypatch.setattr(Settings, "QDRANT_URL", None)
        assert "QDRANT_CLUSTER_ENDPOINT" in Settings.missing_required()
