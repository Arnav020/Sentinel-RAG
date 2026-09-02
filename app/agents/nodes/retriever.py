"""
Retriever: search, rerank, and decide whether anything is actually relevant.

The relevance decision is the important part. Previously this node returned a
fixed five passages whatever the question, so the downstream "no context" branch
could never fire and an out-of-domain question was answered from whatever
happened to be nearest in vector space.

Three outcomes are now distinguishable, and they are not the same thing:
  * passages found          -> answer from them
  * nothing relevant found  -> abstain, and say so
  * search itself failed    -> report an infrastructure error
"""

from __future__ import annotations

import logfire

from app.agents.state import AgentState, to_citation
from app.config import settings
from app.services.retrieval.qdrant_service import RetrievalError, search
from app.services.retrieval.ranking_service import select_relevant


def retrieve_node(state: AgentState) -> dict:
    query = state["current_query"]
    plan = list(state.get("plan", []))

    with logfire.span("Retrieval", query=query[:120]):
        try:
            candidates = search(query, limit=settings.RETRIEVAL_CANDIDATES)
        except RetrievalError as e:
            logfire.error(f"Retrieval infrastructure failure: {e}")
            return {
                "retrieved": [],
                "citations": [],
                "top_score": 0.0,
                "abstained": False,
                "error": "retrieval_unavailable",
                "status": "Knowledge base search failed - the retrieval service is unavailable.",
                "plan": [*plan, "Retrieval: service error"],
            }

        logfire.info(f"Retrieved {len(candidates)} candidates.")
        kept, top_score = select_relevant(query, candidates)

        for chunk in kept:
            chunk.extra["char_start"] = chunk.chunk_index

        if not kept:
            return {
                "retrieved": [],
                "citations": [],
                "top_score": top_score,
                "abstained": True,
                "status": "No sufficiently relevant passage found in the knowledge base.",
                "plan": [
                    *plan,
                    f"Retrieval: {len(candidates)} candidates scanned",
                    f"Relevance: best score {top_score:.3f} below threshold "
                    f"{settings.RERANK_THRESHOLD} - abstaining",
                ],
            }

        sources = {c.source for c in kept}
        return {
            "retrieved": kept,
            "citations": [to_citation(c) for c in kept],
            "top_score": top_score,
            "abstained": False,
            "status": f"Found {len(kept)} relevant passage(s) across {len(sources)} document(s).",
            "plan": [
                *plan,
                f"Retrieval: {len(candidates)} candidates scanned",
                f"Relevance: {len(kept)} passage(s) above threshold (best {top_score:.3f})",
            ],
        }
