"""
Run dataset items through the real pipeline and capture what it did.

In-process by default rather than over HTTP: the same guardrails, graph and
retrieval code runs, but there is no server to start, which is what lets the
behavioural checks run in CI. `--http` targets a live server when you want to
include the API layer.

Two correctness properties this harness has and the old one did not:

  * **No ground-truth fallback.** The previous version assigned
    `actual_contexts = relevant_contexts` whenever a call failed, and the scorer
    fell back the same way, so gold context could reach the judge as though the
    system had retrieved it. A failure is now recorded as a failure.
  * **Cache is bypassed.** Generation runs with the gateway cache off, so a
    repeat run measures the system rather than the cache.
"""

from __future__ import annotations

import time
import uuid

import logfire

from app.agents.graph import build_graph
from app.agents.state import CONVERSATIONAL
from app.guardrails import guard, initialize_rails, rails_ready


def _ensure_rails() -> None:
    if not rails_ready():
        initialize_rails()


def run_item(item: dict, agent=None, with_guardrails: bool = True) -> dict:
    """
    Execute one item end to end.

    Returns a record with the answer, the citations actually used, whether the
    system abstained or blocked, and timing. On failure it records the error
    rather than substituting anything.
    """
    question = item["question"]
    agent = agent or build_graph()
    record: dict = {
        "id": item.get("id"),
        "category": item.get("category"),
        "question": question,
        "answer": "",
        "contexts": [],
        "citations": [],
        "abstained": False,
        "blocked": False,
        "conversational": False,
        "error": "",
        "latency_s": 0.0,
        "top_score": 0.0,
    }

    start = time.time()
    try:
        if with_guardrails:
            _ensure_rails()
            fired, refusal = guard(question)
            if fired:
                record.update(
                    {"blocked": True, "answer": refusal or "", "latency_s": time.time() - start}
                )
                return record

        state = {
            "messages": [{"role": "user", "content": question}],
            "current_query": question,
            "retrieved": [],
            "citations": [],
            "plan": [],
            "status": "starting",
        }
        config = {"configurable": {"thread_id": f"eval:{uuid.uuid4()}"}}
        final = agent.invoke(state, config=config)

        record.update(
            {
                "answer": final.get("final_answer") or "",
                "citations": final.get("citations", []),
                "contexts": [c.get("excerpt", "") for c in final.get("citations", [])],
                "abstained": bool(final.get("abstained")),
                "conversational": final.get("current_query") == CONVERSATIONAL,
                "top_score": float(final.get("top_score") or 0.0),
                "answered_by": final.get("answered_by", ""),
                "cache_hit": bool(final.get("cache_hit")),
                "error": final.get("error", ""),
                "plan": final.get("plan", []),
            }
        )
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        logfire.error(f"Pipeline failed on {item.get('id')}: {record['error']}")

    record["latency_s"] = round(time.time() - start, 3)
    return record


def run_items(
    items: list[dict],
    pace_seconds: float = 1.0,
    with_guardrails: bool = True,
    progress=None,
) -> list[dict]:
    """Run a list of items sequentially, paced to stay under per-model TPM."""
    agent = build_graph()
    records = []
    for i, item in enumerate(items):
        records.append(run_item(item, agent=agent, with_guardrails=with_guardrails))
        if progress:
            progress(i + 1, len(items), records[-1])
        if i < len(items) - 1:
            time.sleep(pace_seconds)
    return records
