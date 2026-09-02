"""
Authentication, conversation sessions, and rate limiting.

Three problems this replaces:

  * `thread_id` was a client-supplied string keying the conversation store, so
    any caller could set another user's thread id and have the model read that
    conversation back to them. Every client that omitted the field shared one
    thread called "default_user".
  * There was no authentication on an endpoint that costs several upstream model
    calls per request, and no rate limit, so a single caller could drain an
    organisation-wide quota.
  * Sessions were never evicted, so memory grew for the lifetime of the process.

Conversation ids are now unguessable server-issued capability tokens, bound to
the caller's identity when auth is enabled, and expired on a TTL.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from fastapi import Header, HTTPException, Request, status

from app.config import settings

SESSION_TTL_SECONDS = 60 * 60 * 4
MAX_SESSIONS = 10_000


@dataclass(slots=True)
class Session:
    id: str
    owner: str
    created_at: float
    last_seen: float
    turns: int = 0


@dataclass(slots=True)
class _Bucket:
    hits: deque = field(default_factory=deque)


class SessionStore:
    """In-memory session registry with TTL and a hard capacity bound."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, owner: str) -> Session:
        now = time.time()
        session = Session(id=secrets.token_urlsafe(24), owner=owner, created_at=now, last_seen=now)
        with self._lock:
            self._evict_locked(now)
            if len(self._sessions) >= MAX_SESSIONS:
                oldest = min(self._sessions.values(), key=lambda s: s.last_seen)
                self._sessions.pop(oldest.id, None)
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str, owner: str) -> Session | None:
        now = time.time()
        with self._lock:
            self._evict_locked(now)
            session = self._sessions.get(session_id)
            if session is None:
                return None
            # Constant-time comparison: a session id is a bearer capability.
            if not hmac.compare_digest(session.owner, owner):
                return None
            session.last_seen = now
            session.turns += 1
            return session

    def _evict_locked(self, now: float) -> None:
        cutoff = now - SESSION_TTL_SECONDS
        stale = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
        for sid in stale:
            self._sessions.pop(sid, None)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


class RateLimiter:
    """Fixed-cost sliding window, keyed on caller identity."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, identity: str) -> tuple[bool, int]:
        if not settings.RATE_LIMIT_ENABLED or self.per_minute <= 0:
            return True, self.per_minute
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            bucket = self._buckets.setdefault(identity, _Bucket())
            while bucket.hits and bucket.hits[0] < cutoff:
                bucket.hits.popleft()
            if len(bucket.hits) >= self.per_minute:
                return False, 0
            bucket.hits.append(now)
            # Opportunistic cleanup so idle identities do not accumulate.
            if len(self._buckets) > 5000:
                for key in [k for k, b in self._buckets.items() if not b.hits]:
                    self._buckets.pop(key, None)
            return True, self.per_minute - len(bucket.hits)


sessions = SessionStore()
rate_limiter = RateLimiter(settings.RATE_LIMIT_PER_MINUTE)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def identify(request: Request, api_key: str | None) -> str:
    """
    Stable identity for rate limiting and session ownership.

    An API key identifies a caller precisely; without one we fall back to the
    peer address, which is weaker but still bounds a single source.
    """
    if api_key:
        return f"key:{_hash(api_key)}"
    client = request.client.host if request.client else "unknown"
    return f"ip:{_hash(client)}"


async def require_auth(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """
    Validate the caller and apply the rate limit. Returns the caller identity.

    When no API_KEYS are configured, auth is disabled - correct for local
    development and CI - and /health reports that so a deployment cannot be
    unprotected without it being visible.
    """
    if settings.AUTH_REQUIRED and (
        not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in settings.API_KEYS)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key header is required.",
        )

    identity = identify(request, x_api_key)
    allowed, remaining = rate_limiter.check(identity)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({settings.RATE_LIMIT_PER_MINUTE}/minute).",
            headers={"Retry-After": "60"},
        )
    request.state.rate_remaining = remaining
    return identity
