"""
Deterministic retrieval and abstention evaluation.

No LLM, no judge, no token cost, no run-to-run variance. That matters twice
over: these numbers are the only ones that can gate every pull request, and they
measure things an LLM judge systematically hides.

Metrics:

  * **hit@k / MRR / nDCG@5** against the gold chunk id recorded at dataset build
    time - identity, not text matching, so duplicate passages cannot be confused.
  * **Document-level hit@k**, the weaker claim the old README reported.
  * **Gold-body containment.** What fraction of the gold passage's sentences
    actually reach the generator. This is the metric that exposed the old
    chunker losing ~19% of the evidence its own reference answers needed, while
    RAGAS faithfulness of 0.846 absorbed the loss without showing it.
  * **Abstention.** Whether the relevance threshold correctly declines
    unanswerable and off-domain questions, and correctly does NOT decline
    answerable ones. The system's most serious defect was invisible to the old
    suite because every question in it was answerable.

Scope, stated plainly: this tier searches and reranks the dataset question
alone. Production does more - the planner also emits a keyword phrasing, and
both phrasings are searched and reranked with the best score kept per passage
(see `ranking_service.score_chunks_multi`), which is worth real points on
questions whose answer is written in identifiers rather than prose. Reproducing
that here would put an LLM call on the path and forfeit exactly the properties
this tier exists for: zero cost, zero variance, runnable on every pull request.

So read these numbers as a **floor on the retriever**, not as production
performance. The behaviour tier runs the full graph, planner included, and is
the number to quote for end-to-end quality.
"""

from __future__ import annotations

import re

from app.config import settings
from app.services.retrieval.qdrant_service import search
from app.services.retrieval.ranking_service import select_relevant
from evals.stats import mrr, ndcg_at_k, summarise_binary, summarise_scores


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?:;\n])\s+", text)
    return [re.sub(r"\s+", " ", p).strip().lower() for p in parts if len(p.strip()) > 25]


def containment(gold_body: str, retrieved_texts: list[str]) -> float:
    """Fraction of the gold passage's sentences present in the retrieved context."""
    sents = _sentences(gold_body)
    if not sents:
        return 1.0
    blob = re.sub(r"\s+", " ", " ".join(retrieved_texts)).lower()
    return sum(1 for s in sents if s in blob) / len(sents)


def evaluate_item(item: dict) -> dict:
    """Run one item through real retrieval and score it. Pure, no side effects."""
    question = item["question"]
    candidates = search(question, limit=settings.RETRIEVAL_CANDIDATES)
    kept, top_score = select_relevant(question, candidates)

    # Reranked order over all candidates, so ranks are comparable across items
    # regardless of where the threshold happens to fall.
    from app.services.retrieval.ranking_service import score_chunks

    ordered = score_chunks(question, list(candidates))

    row: dict = {
        "id": item.get("id"),
        "category": item.get("category"),
        "question_type": item.get("question_type"),
        "doc_topic": item.get("doc_topic", ""),
        "top_score": round(top_score, 4),
        "kept": len(kept),
        "abstained": len(kept) == 0,
        "candidates": len(candidates),
    }

    if item.get("category") == "answerable":
        gold_id = item.get("gold_chunk_id")
        gold_source = item.get("gold_source")

        chunk_ranks = [i + 1 for i, c in enumerate(ordered) if c.id == gold_id]
        doc_ranks = [i + 1 for i, c in enumerate(ordered) if c.source == gold_source]
        chunk_rank = chunk_ranks[0] if chunk_ranks else None
        doc_rank = doc_ranks[0] if doc_ranks else None

        relevances = [1 if c.id == gold_id else 0 for c in ordered]

        row.update(
            {
                "chunk_rank": chunk_rank,
                "doc_rank": doc_rank,
                "hit_at_1": chunk_rank == 1,
                "hit_at_3": chunk_rank is not None and chunk_rank <= 3,
                "hit_at_5": chunk_rank is not None and chunk_rank <= 5,
                "hit_at_k": chunk_rank is not None,
                "doc_hit_at_1": doc_rank == 1,
                "doc_hit_at_5": doc_rank is not None and doc_rank <= 5,
                "ndcg_at_5": round(ndcg_at_k(relevances, 5), 4),
                "containment": round(
                    containment(item.get("gold_body", ""), [c.body for c in kept]), 4
                ),
                "gold_survived_threshold": any(c.id == gold_id for c in kept),
            }
        )
    return row


def aggregate(rows: list[dict]) -> dict:
    """Summarise per-item rows into reportable metrics with intervals."""
    answerable = [r for r in rows if r["category"] == "answerable"]
    should_abstain = [r for r in rows if r["category"] in ("unanswerable_in_domain", "off_domain")]

    out: dict = {"n": {"answerable": len(answerable), "should_abstain": len(should_abstain)}}

    if answerable:
        out["retrieval"] = [
            summarise_binary("hit@1", sum(r["hit_at_1"] for r in answerable), len(answerable)),
            summarise_binary("hit@3", sum(r["hit_at_3"] for r in answerable), len(answerable)),
            summarise_binary("hit@5", sum(r["hit_at_5"] for r in answerable), len(answerable)),
            summarise_binary(
                f"hit@{settings.RETRIEVAL_CANDIDATES}",
                sum(r["hit_at_k"] for r in answerable),
                len(answerable),
            ),
            summarise_binary(
                "doc_hit@1", sum(r["doc_hit_at_1"] for r in answerable), len(answerable)
            ),
            summarise_binary(
                "doc_hit@5", sum(r["doc_hit_at_5"] for r in answerable), len(answerable)
            ),
            summarise_scores("ndcg@5", [r["ndcg_at_5"] for r in answerable]),
            summarise_scores("gold_containment@k", [r["containment"] for r in answerable]),
            {"metric": "mrr", "value": round(mrr([r["chunk_rank"] for r in answerable]), 4)},
        ]
        # False abstention: an answerable question the threshold refused.
        false_abstain = sum(1 for r in answerable if r["abstained"])
        out["abstention_on_answerable"] = summarise_binary(
            "false_abstention_rate", false_abstain, len(answerable)
        )
        out["gold_survived_threshold"] = summarise_binary(
            "gold_survived_threshold",
            sum(r["gold_survived_threshold"] for r in answerable),
            len(answerable),
        )

    if should_abstain:
        correct = sum(1 for r in should_abstain if r["abstained"])
        out["abstention_on_unanswerable"] = summarise_binary(
            "correct_abstention_rate", correct, len(should_abstain)
        )
        for cat in ("unanswerable_in_domain", "off_domain"):
            subset = [r for r in should_abstain if r["category"] == cat]
            if subset:
                out[f"abstention_{cat}"] = summarise_binary(
                    f"correct_abstention_{cat}",
                    sum(1 for r in subset if r["abstained"]),
                    len(subset),
                )

    # Per-stratum retrieval, so a weak topic cannot hide inside a global mean.
    by_topic: dict[str, list[dict]] = {}
    for r in answerable:
        by_topic.setdefault(r.get("doc_topic") or "unknown", []).append(r)
    out["by_topic"] = {
        topic: summarise_binary("hit@5", sum(x["hit_at_5"] for x in rs), len(rs))
        for topic, rs in sorted(by_topic.items())
    }

    by_type: dict[str, list[dict]] = {}
    for r in answerable:
        by_type.setdefault(r.get("question_type") or "unknown", []).append(r)
    out["by_question_type"] = {
        qtype: summarise_binary("hit@5", sum(x["hit_at_5"] for x in rs), len(rs))
        for qtype, rs in sorted(by_type.items())
    }

    return out


def run(items: list[dict], progress=None) -> tuple[list[dict], dict]:
    rows = []
    total = len(items)
    for i, item in enumerate(items):
        if item.get("category") in ("adversarial", "conversational"):
            continue  # these exercise the gate and the planner, not retrieval
        rows.append(evaluate_item(item))
        if progress and (i % 10 == 0 or i == total - 1):
            progress(i + 1, total)
    return rows, aggregate(rows)
