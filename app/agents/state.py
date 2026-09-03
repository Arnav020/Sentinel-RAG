"""
Graph state.

`documents` used to be a list of pre-formatted strings, which threw away the
score, the source identity and the section a passage came from - so nothing
downstream could make a relevance decision or produce a verifiable citation.
Retrieved passages are now carried as typed objects all the way to the response.

Conversation history is bounded here. It previously grew without limit inside an
in-process checkpointer and was rendered into every prompt, so a long thread
inflated cost on every turn and eventually exceeded the context window.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.config import settings
from app.services.retrieval.qdrant_service import RetrievedChunk

# Sentinel written into `current_query` when the planner decides the turn needs
# no retrieval. Not a real query, and never sent to the vector store.
CONVERSATIONAL = "__CONVERSATIONAL__"


class AgentState(TypedDict, total=False):
    # operator.add appends rather than replaces, so history accumulates.
    messages: Annotated[list[dict], operator.add]
    current_query: str
    search_terms: str
    retrieved: list[RetrievedChunk]
    citations: list[dict]
    top_score: float
    abstained: bool
    entailment_checked: bool
    plan: list[str]
    status: str
    final_answer: str
    answered_by: str
    cache_hit: bool
    error: str


def recent_history(messages: list[dict], exclude_last: bool = True) -> list[dict]:
    """
    The trailing window of a conversation, bounded by turn count and characters.

    Both bounds matter: turn count keeps the prompt structurally sane, and the
    character cap stops a single very long assistant answer from consuming the
    whole budget.
    """
    history = messages[:-1] if exclude_last and messages else list(messages)
    window = history[-settings.HISTORY_MAX_TURNS :]

    out: list[dict] = []
    budget = settings.HISTORY_MAX_CHARS
    for msg in reversed(window):
        content = str(msg.get("content", ""))
        if len(content) > budget:
            break
        budget -= len(content)
        out.append(msg)
    return list(reversed(out))


def format_history(messages: list[dict], exclude_last: bool = True) -> str:
    lines = []
    for msg in recent_history(messages, exclude_last=exclude_last):
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def latest_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def to_citation(chunk: RetrievedChunk) -> dict[str, Any]:
    """Client-facing provenance for one retrieved passage."""
    return {
        "source": chunk.source,
        "title": chunk.title or chunk.source,
        "section": chunk.heading_path,
        "url": chunk.source_url,
        "topic": chunk.doc_topic,
        "kind": chunk.kind,
        "chunk_index": chunk.chunk_index,
        "score": round(chunk.rerank_score or 0.0, 4),
        "vector_score": round(chunk.vector_score, 4),
        "char_start": chunk.extra.get("char_start"),
        "excerpt": chunk.body[:600],
    }
