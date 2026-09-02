"""
Qdrant access layer.

Returns rich, typed results rather than dicts of strings: every downstream stage
(reranking, thresholding, citation) needs the point id and score, and the
previous string-keyed handoff meant two chunks with identical text collapsed to
one citation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_query


class RetrievalError(Exception):
    """
    The vector DB search itself failed (outage, auth, missing collection).

    Distinct from a search that succeeds and finds nothing: callers must not
    answer as though the topic simply is not covered when the index is down.
    """


@dataclass(slots=True)
class RetrievedChunk:
    """One candidate passage, carried intact through the whole pipeline."""

    id: str
    text: str  # embedded form: heading prefix + body
    body: str  # source text exactly as it appears in the document
    source: str  # document filename
    title: str
    source_url: str
    doc_topic: str
    heading_path: str
    kind: str
    chunk_index: int
    vector_score: float
    rerank_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Human-readable provenance for the answer's source list."""
        where = f"{self.title or self.source}"
        if self.heading_path:
            where += f" - {self.heading_path}"
        return where


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT,
        )
    return _client


def _to_chunk(point) -> RetrievedChunk:
    p = point.payload or {}
    return RetrievedChunk(
        id=str(point.id),
        text=p.get("text", ""),
        body=p.get("body", p.get("text", "")),
        source=p.get("source", "unknown"),
        title=p.get("title", ""),
        source_url=p.get("source_url", ""),
        doc_topic=p.get("doc_topic", ""),
        heading_path=p.get("heading_path", ""),
        kind=p.get("kind", "prose"),
        chunk_index=int(p.get("chunk_index", -1)),
        vector_score=float(point.score),
    )


def search(
    query: str,
    limit: int | None = None,
    doc_topics: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    Dense search over the knowledge base.

    `doc_topics` applies a payload filter; it exists for scoped experiments and
    is not used on the default request path, where scope is enforced by the
    relevance threshold rather than by pre-filtering.

    Raises RetrievalError on infrastructure failure.
    """
    limit = limit or settings.RETRIEVAL_CANDIDATES
    try:
        vector = embed_query(query)
        query_filter = None
        if doc_topics:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_topic", match=models.MatchAny(any=list(doc_topics))
                    )
                ]
            )

        response = get_client().query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [_to_chunk(p) for p in response.points]
    except Exception as e:
        logfire.error(f"Qdrant search failed: {e}")
        raise RetrievalError(str(e)) from e


def collection_stats() -> dict[str, Any]:
    """Point count and per-topic breakdown; drives /health and the scope text."""
    try:
        client = get_client()
        info = client.get_collection(settings.QDRANT_COLLECTION)
        return {
            "collection": settings.QDRANT_COLLECTION,
            "points": info.points_count,
            "status": str(info.status),
            "dim": info.config.params.vectors.size,
        }
    except Exception as e:
        raise RetrievalError(str(e)) from e


def distinct_payload_values(field: str, limit: int = 20000) -> dict[str, int]:
    """
    Count distinct values of a payload field by scrolling the collection.

    Used to derive the assistant's advertised scope from what is actually
    indexed, instead of from four hand-maintained prose lists that had drifted
    away from the corpus.
    """
    counts: dict[str, int] = {}
    offset = None
    seen = 0
    client = get_client()
    while seen < limit:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=min(1000, limit - seen),
            offset=offset,
            with_payload=[field],
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            value = (p.payload or {}).get(field)
            if value:
                counts[str(value)] = counts.get(str(value), 0) + 1
        seen += len(points)
        if offset is None:
            break
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# Backwards-compatible alias used by older call sites and notebooks.
def search_enterprise_knowledge(query: str, limit: int | None = None):
    return search(query, limit=limit)
