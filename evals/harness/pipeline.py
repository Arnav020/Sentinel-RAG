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
from app.config import settings
from app.gateway import daily_quota_exhausted, tokens_used_since
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


class DailyQuotaExhausted(RuntimeError):
    """A model ran out of daily tokens mid-run; the rest of the run is worthless."""


# Groq's per-minute token allowance, and the fraction of it we aim to occupy.
# The headroom matters: an item's cost is only known after it runs, so the
# budget check is always one item stale, and a burst that lands exactly on the
# limit is a 429 rather than a wait.
TPM_LIMIT = 8000
TPM_HEADROOM = 0.75
_THROTTLE_TIMEOUT = 90.0

# Seed costs, measured on this corpus (prompt + completion, per item). These
# only prime the throttle; from the first item onward it reads what the models
# actually spent, so a wrong seed self-corrects within one item.
_SEED_COST = 1500


def _throttle(models: list[str], expected: int = _SEED_COST) -> float:
    """
    Block until every model has room in its rolling per-minute budget.

    A flat delay cannot do this job. Items cost wildly different amounts - a
    blocked question spends 500 tokens, an answered one nearly 3,000 - so a
    pace tuned for the average runs 50% over on the expensive ones. Measured on
    a 145-item run, a 1.5s pace produced 236 rate-limit errors before the daily
    cap was even reached.

    Waiting on measured spend instead means the run costs only what it must:
    fast through cheap items, slow through expensive ones.
    """
    waited = 0.0
    ceiling = TPM_LIMIT * TPM_HEADROOM
    while waited < _THROTTLE_TIMEOUT:
        if all(tokens_used_since(m, 60) + expected <= ceiling for m in models):
            return waited
        time.sleep(1.0)
        waited += 1.0
    return waited


def run_items(
    items: list[dict],
    pace_seconds: float = 1.0,
    with_guardrails: bool = True,
    progress=None,
) -> list[dict]:
    """
    Run a list of items sequentially, throttled by measured token spend.

    Two limits are enforced, and they are not the same problem:

      * **Per minute** - handled by `_throttle`, which waits on what the models
        have actually spent in the last 60 seconds rather than on a fixed sleep.
      * **Per day** - cannot be waited out, so it aborts. Pushing on through it
        is what made an earlier run so misleading: 73 consecutive failures, then
        a summary reporting "answerable 0.373" as though it were a measurement
        of quality rather than of an exhausted budget.

    Records gathered before an abort are returned and are perfectly good; the
    caller marks the run partial rather than scoring it.
    """
    agent = build_graph()
    models = [
        settings.GENERATION_MODEL,
        settings.PLANNER_MODEL,
        settings.ENTAILMENT_MODEL,
        settings.TOPIC_FILTER_MODEL,
    ]
    models = list(dict.fromkeys(models))

    records: list[dict] = []
    throttled = 0.0
    for i, item in enumerate(items):
        throttled += _throttle(models)
        records.append(run_item(item, agent=agent, with_guardrails=with_guardrails))
        if progress:
            progress(i + 1, len(items), records[-1])

        exhausted = daily_quota_exhausted()
        if exhausted:
            error = DailyQuotaExhausted(
                f"Daily token budget exhausted for: {', '.join(sorted(exhausted))}. "
                f"Stopped after {len(records)} of {len(items)} items. "
                "Groq's per-day window is rolling, so budget returns gradually "
                "over the next 24h; re-run then."
            )
            # Carry the completed records so the caller can keep them.
            error.records = records
            raise error

        if i < len(items) - 1 and pace_seconds:
            time.sleep(pace_seconds)

    if throttled:
        logfire.info(f"Throttled {throttled:.0f}s in total to stay under per-minute limits.")
    return records
