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

from app.agents.state import AgentState, latest_user_message, to_citation
from app.config import settings
from app.services.retrieval.entailment import context_answers_question
from app.services.retrieval.qdrant_service import RetrievalError, search
from app.services.retrieval.ranking_service import select_relevant


def retrieve_node(state: AgentState) -> dict:
    query = state["current_query"]
    plan = list(state.get("plan", []))

    # The answer-presence gate must judge the USER's question, not the planner's
    # search query. Those are different things: the planner emits a keyword
    # phrase tuned for embedding similarity ("Namespace definition first in
    # manifest"), and asking whether a passage "answers" a keyword phrase is not
    # a well-formed question. Wiring the gate to `current_query` made it reject
    # 12 of 83 answerable questions whose gold passage was sitting at rank 1.
    #
    # It also means a lossy planner rewrite - "recursive read-only mounts"
    # became "ReadOnlyRootFilesystem feature gate", a different feature - is
    # judged against what the user actually asked.
    user_question = latest_user_message(state.get("messages", [])) or query

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

        # Widen with the planner's keyword form, and with the user's own words if
        # they differ. Dense retrieval is phrasing-sensitive: the passage stating
        # "Only a RestartPolicy equal to Never or OnFailure is allowed" ranks
        # 332nd for "what restart policies can a Job use" but 29th for the
        # identifier spelling. One extra vector search per alternative phrasing
        # is cheap, no LLM is involved, and the cross-encoder decides which
        # framing actually won.
        alternates = [t for t in (state.get("search_terms"), user_question) if t]
        seen = {c.id for c in candidates}
        for alt in alternates:
            if alt.strip().lower() == query.strip().lower():
                continue
            try:
                for extra in search(alt, limit=settings.RETRIEVAL_CANDIDATES):
                    if extra.id not in seen:
                        candidates.append(extra)
                        seen.add(extra.id)
            except RetrievalError as e:
                # The primary search already succeeded; a failed widening is not
                # a reason to fail the turn.
                logfire.warning(f"Secondary retrieval on {alt[:40]!r} failed: {e}")

        logfire.info(f"Retrieved {len(candidates)} candidates.")
        # Rerank against every phrasing, not just the prose question: the
        # cross-encoder is phrasing-sensitive enough that the passage answering
        # the question can score 0.0003 one way and 0.9997 the other. See
        # `score_chunks_multi`.
        kept, top_score = select_relevant(query, candidates, extra_queries=alternates)

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

        # Second gate: relevance is not the same as answer presence. The
        # cross-encoder cannot tell "this passage is about the same area" from
        # "this passage answers the question", so an in-domain question the
        # corpus does not cover can clear the threshold on vocabulary alone.
        # Judge the planner's self-contained question, not the raw message: a
        # follow-up like "could you elaborate?" is not answerable by any passage
        # on its own, so gating on it would refuse every follow-up.
        answers, checked = context_answers_question(query, kept)
        if not answers:
            return {
                "retrieved": [],
                "citations": [],
                "top_score": top_score,
                "abstained": True,
                "status": "Retrieved passages are related but do not answer the question.",
                "plan": [
                    *plan,
                    f"Retrieval: {len(candidates)} candidates scanned",
                    f"Relevance: {len(kept)} passage(s) above threshold (best {top_score:.3f})",
                    "Answer presence: context is topically related but does not "
                    "contain the answer - abstaining",
                ],
            }

        sources = {c.source for c in kept}
        return {
            "retrieved": kept,
            "citations": [to_citation(c) for c in kept],
            "top_score": top_score,
            "abstained": False,
            "entailment_checked": checked,
            "status": f"Found {len(kept)} relevant passage(s) across {len(sources)} document(s).",
            "plan": [
                *plan,
                f"Retrieval: {len(candidates)} candidates scanned",
                f"Relevance: {len(kept)} passage(s) above threshold (best {top_score:.3f})",
                "Answer presence: confirmed" if checked else "Answer presence: not checked",
            ],
        }
