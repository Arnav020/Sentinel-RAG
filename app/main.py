"""
Sentinel-RAG API.

A Kubernetes documentation assistant. The scope is the corpus: whatever is
indexed is what it answers, and when nothing indexed is relevant it says so
rather than reaching for the nearest passage.
"""

from __future__ import annotations

import sys
import uuid

if sys.platform == "win32":  # emoji in log messages vs the cp1252 console
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import logfire

from app.config import settings

# Configure tracing before importing anything that emits spans, so no module's
# startup work is lost.
logfire.configure(
    service_name="sentinel-rag",
    send_to_logfire=bool(settings.LOGFIRE_ENABLED),
    token=settings.LOGFIRE_TOKEN,
)

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agents.graph import rag_agent
from app.agents.state import CONVERSATIONAL
from app.api.security import require_auth, sessions
from app.guardrails import guard, initialize_rails, rails_ready
from app.services import scope as scope_service
from app.services.retrieval.qdrant_service import RetrievalError, collection_stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup work. `@app.on_event` is removed in current FastAPI.

    Neither step is fatal: the API should come up and report itself degraded on
    /health rather than crash-looping, because a crash-looping container tells an
    operator far less than one that says exactly which subsystem is down.
    """
    try:
        initialize_rails()
    except Exception as e:
        logfire.error(f"Guardrails failed to initialise - gate is DOWN: {e}")
    try:
        scope_service.refresh()
    except Exception as e:
        logfire.error(f"Could not derive scope from corpus: {e}")
    yield


app = FastAPI(
    title="Sentinel-RAG",
    description="Kubernetes documentation assistant with grounded answers and explicit abstention.",
    version="2.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None,
        description="Conversation id from POST /session. Omit for a one-shot question.",
    )


class SessionResponse(BaseModel):
    session_id: str


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Report failures as failures.

    The previous handler caught everything and returned HTTP 200 with an apology
    string, so monitoring saw a 0% error rate during an outage and the eval
    harness scored the apology as a low-faithfulness answer instead of an
    incident.
    """
    correlation_id = str(uuid.uuid4())
    logfire.error(f"Unhandled error [{correlation_id}] on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal error while processing the request.",
            "correlation_id": correlation_id,
        },
    )


@app.get("/health")
def health():
    """Liveness plus a readiness breakdown, so degradation is visible not silent."""
    corpus: dict = {"available": False}
    try:
        corpus = {"available": True, **collection_stats()}
    except RetrievalError as e:
        corpus["error"] = str(e)[:200]

    scope = scope_service.get_scope()
    degraded = not rails_ready() or not corpus["available"] or scope.points == 0

    return {
        "status": "degraded" if degraded else "ok",
        "guardrails": rails_ready(),
        "corpus": corpus,
        "scope_topics": len(scope.topics),
        "auth_required": settings.AUTH_REQUIRED,
        "rate_limit_per_minute": settings.RATE_LIMIT_PER_MINUTE
        if settings.RATE_LIMIT_ENABLED
        else None,
        "active_sessions": sessions.count(),
        "models": {
            "generation": settings.GENERATION_MODEL,
            "planner": settings.PLANNER_MODEL,
            "embedding": settings.EMBEDDING_MODEL,
        },
    }


@app.get("/scope")
def get_scope():
    """What the assistant can answer, derived from what is actually indexed."""
    return scope_service.get_scope().as_dict()


@app.post("/session", response_model=SessionResponse)
def create_session(identity: str = Depends(require_auth)):
    """
    Issue a conversation id.

    Server-issued and unguessable, and bound to the caller's identity, so one
    caller cannot read another's conversation by choosing its id.
    """
    return SessionResponse(session_id=sessions.create(identity).id)


@app.post("/query")
def query(request: QueryRequest, identity: str = Depends(require_auth)):
    q = request.q.strip()
    if not q:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    # Resolve the conversation. An unknown or foreign session id is rejected
    # rather than silently starting a new thread, so a client bug cannot look
    # like working memory.
    if request.session_id:
        session = sessions.get(request.session_id, identity)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unknown or expired session_id. Request a new one from POST /session.",
            )
        thread_id = session.id
    else:
        thread_id = f"oneshot:{uuid.uuid4()}"

    config = {"configurable": {"thread_id": thread_id}}

    # Gate 1: guardrails. History is included so a multi-turn injection cannot
    # slip past by splitting itself across messages.
    history = ""
    if request.session_id:
        try:
            snapshot = rag_agent.get_state(config)
            prior = (snapshot.values or {}).get("messages", []) if snapshot else []
            history = "\n".join(str(m.get("content", "")) for m in prior[-4:])
        except Exception:
            history = ""

    rail_fired, rail_response = guard(q, history=history)
    if rail_fired:
        logfire.info(f"Blocked by guardrails | thread={thread_id[:12]}")
        return {
            "question": q,
            "answer": rail_response,
            "abstained": False,
            "blocked": True,
            "thought_process": ["Guardrails: blocked", "Retrieval: skipped"],
            "status": "Blocked by guardrails.",
            "sources": [],
        }

    # Gate 2: the RAG graph.
    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "retrieved": [],
        "citations": [],
        "plan": [],
        "status": "starting",
    }
    final = rag_agent.invoke(initial_state, config=config)

    return {
        "question": q,
        "answer": final.get("final_answer"),
        "abstained": bool(final.get("abstained")),
        "blocked": False,
        "thought_process": final.get("plan", []),
        "status": final.get("status"),
        "sources": final.get("citations", []),
        "top_score": final.get("top_score", 0.0),
        "answered_by": final.get("answered_by"),
        "cache_hit": final.get("cache_hit", False),
        "conversational": final.get("current_query") == CONVERSATIONAL,
    }


# ---------------------------------------------------------------------------
# Web client. Mounted LAST: FastAPI resolves routes in declaration order, so
# every API route above still wins over this catch-all static mount.
# ---------------------------------------------------------------------------
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
else:
    logfire.warning(f"Web client directory not found at {_WEB_DIR} - API-only mode.")
