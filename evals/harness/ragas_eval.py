"""
RAGAS scoring of generation quality.

Only the answerable slice is scored: faithfulness against an abstention is
meaningless, and including abstentions would let a system that refuses
everything post excellent grounding numbers.

Judge independence is enforced in `Settings.validate()` - the judge must be a
different model family from the generator, or the system grades its own lineage.
Here the generator is `openai/gpt-oss-120b` and the judge is `qwen/qwen3.8-27b`.

Metrics, and why each is here:
  * Faithfulness       - is every claim supported by the retrieved passages?
                         The primary grounding metric.
  * ResponseGroundedness - a second, differently-framed grounding check, so a
                         single metric's blind spot is visible.
  * ContextPrecision   - were the retrieved passages relevant to the reference?
  * ContextRecall      - did retrieval cover what the reference needed?
  * AnswerCorrectness  - agreement with the reference answer.
  * AnswerRelevancy    - reported as a DIAGNOSTIC only. It works by generating
                         questions from the answer and comparing embeddings, so
                         it largely measures on-topic fluency; a confident wrong
                         answer scores well.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from pathlib import Path

import logfire

from app.config import settings
from evals.stats import summarise_scores

# ragas imports ChatVertexAI from a langchain-community path that upstream
# removed. Installing the shim before importing ragas is the documented
# workaround; it is pinned in requirements-eval.txt so it cannot drift silently.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        from langchain_google_vertexai import ChatVertexAI

        _shim = types.ModuleType("langchain_community.chat_models.vertexai")
        _shim.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = _shim
    except ImportError:  # pragma: no cover - surfaced clearly at call time
        pass

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Generous on purpose. RAGAS asks the judge for structured JSON, and a reasoning
# model emits its reasoning first. At 2,500 the reply was truncated mid-object
# and the provider rejected it with `json_validate_failed` - which surfaces as
# "InstructorRetryException" and looks like a judge that cannot follow a schema,
# not like a token budget. At 6,000 the same samples score cleanly.
#
# It cannot simply be maximised either: the provider sizes a request as
# prompt + max_tokens against the per-minute ceiling, so an over-large value
# turns every call into a 429.
JUDGE_MAX_TOKENS = 6000


def _build_judge():
    if settings.JUDGE_KEY_IS_SHARED:
        logfire.warning(
            "JUDGE_GROQ is not set - the judge is sharing the serving API key, so "
            "eval traffic competes with answer generation for quota. Model-level "
            "independence (a different family) is unaffected."
        )
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory

    client = AsyncOpenAI(api_key=settings.JUDGE_API_KEY, base_url=GROQ_BASE_URL, max_retries=5)
    llm = llm_factory(
        settings.JUDGE_MODEL,
        provider="openai",
        client=client,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2", use_api=False
    )
    return llm, embeddings


def _metrics(
    llm, embeddings, only: tuple[str, ...] | None = None
) -> list[tuple[str, object, callable]]:
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        ResponseGroundedness,
    )

    everything = [
        (
            "faithfulness",
            Faithfulness(llm=llm),
            lambda s: {
                "user_input": s["question"],
                "response": s["answer"],
                "retrieved_contexts": s["contexts"],
            },
        ),
        (
            "response_groundedness",
            ResponseGroundedness(llm=llm),
            lambda s: {"response": s["answer"], "retrieved_contexts": s["contexts"]},
        ),
        (
            "context_precision",
            ContextPrecision(llm=llm),
            lambda s: {
                "user_input": s["question"],
                "retrieved_contexts": s["contexts"],
                "reference": s["reference"],
            },
        ),
        (
            "context_recall",
            ContextRecall(llm=llm),
            lambda s: {
                "user_input": s["question"],
                "retrieved_contexts": s["contexts"],
                "reference": s["reference"],
            },
        ),
        (
            "answer_correctness",
            AnswerCorrectness(llm=llm, embeddings=embeddings),
            lambda s: {
                "user_input": s["question"],
                "response": s["answer"],
                "reference": s["reference"],
            },
        ),
        (
            "answer_relevancy",
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            lambda s: {"user_input": s["question"], "response": s["answer"]},
        ),
    ]
    if not only:
        return everything
    chosen = [m for m in everything if m[0] in only]
    unknown = set(only) - {m[0] for m in everything}
    if unknown:
        raise SystemExit(f"unknown metric(s): {sorted(unknown)}")
    return chosen


def _prepare(items: list[dict], records: list[dict]) -> list[dict]:
    """
    Answerable items that produced a real, grounded answer.

    No ground-truth fallback: a sample with no retrieved context is excluded and
    counted, never scored against the gold passage as though the system had
    found it. That substitution is what made the previous numbers unusable.
    """
    by_id = {r["id"]: r for r in records}
    prepared = []
    for item in items:
        if item.get("category") != "answerable":
            continue
        record = by_id.get(item["id"])
        if record is None or record.get("error"):
            continue
        if record.get("abstained") or record.get("blocked"):
            continue
        contexts = [c for c in record.get("contexts", []) if c and c.strip()]
        answer = (record.get("answer") or "").strip()
        if not contexts or not answer:
            continue
        prepared.append(
            {
                "id": item["id"],
                "question": item["question"],
                "reference": item["reference"],
                "answer": answer,
                "contexts": contexts,
                "doc_topic": item.get("doc_topic", ""),
                "question_type": item.get("question_type", ""),
            }
        )
    return prepared


# The judge allows 8,000 tokens per minute and a context-heavy metric call costs
# roughly 2,600 tokens, so fewer than four calls per minute actually fit.
JUDGE_TPM = 8000
EST_TOKENS_PER_CALL = 2600
RATE_LIMIT_BACKOFF = (20, 45, 90, 150)

# The provider also enforces a DAILY token cap (200,000 per model on this tier),
# which is the binding constraint on how much can be judged - not wall time.
# Budget roughly: 6 metrics x ~2 calls x ~2,600 tokens = ~14k tokens per sample,
# so a full 6-metric pass fits about 14 samples per day. Choose the metric set
# and sample size together; `coverage` and `n` are reported so a reduced run is
# never mistaken for a full one.
JUDGE_TPD = 200_000

# Metrics worth the budget when it is scarce, in priority order. Faithfulness is
# the primary grounding claim; context precision and recall measure retrieval as
# the judge sees it. AnswerRelevancy is excluded by default: it generates
# questions from the answer and compares embeddings, so it largely measures
# on-topic fluency and a confident wrong answer scores well.
CORE_METRICS = ("faithfulness", "context_precision", "context_recall")


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


async def _score_one(metric, payload: dict) -> float:
    """
    Score one sample, retrying through rate limits.

    RAGAS wraps the client in `instructor`, which exhausts its own retries and
    then raises, so backing off has to happen here. Without it a transient quota
    stall silently became a permanent, whole-metric failure.
    """
    last: Exception | None = None
    for attempt, wait in enumerate((0, *RATE_LIMIT_BACKOFF)):
        if wait:
            await asyncio.sleep(wait)
        try:
            result = await metric.ascore(**payload)
            value = float(result.value)
            if value != value:  # NaN
                raise ValueError("judge returned NaN")
            return value
        except Exception as e:
            last = e
            if not _is_rate_limit(e):
                raise
            logfire.warning(f"rate limited, backing off (attempt {attempt + 1})")
    raise last if last else RuntimeError("scoring failed")


async def _score_all(
    samples: list[dict],
    pace_seconds: float,
    out_dir: Path,
    metric_names: tuple[str, ...] | None = None,
) -> dict:
    llm, embeddings = _build_judge()
    per_item: dict[str, dict[str, float]] = {s["id"]: {} for s in samples}
    summary: dict = {}

    # Pace from the token budget rather than a flat guess. A flat 1s put four of
    # six metrics roughly 15x over the judge's TPM ceiling and every sample
    # failed with HTTP 429 - which at least reports "ALL SAMPLES FAILED" rather
    # than a wrong number, but wastes the run either way.
    pace = max(pace_seconds, 60.0 * EST_TOKENS_PER_CALL / JUDGE_TPM)
    print(f"  pacing {pace:.1f}s between judge calls ({JUDGE_TPM} TPM budget)")

    for name, metric, build_input in _metrics(llm, embeddings, metric_names):
        values: list[float] = []
        failures = 0
        print(f"\n  [{name}] scoring {len(samples)} samples")
        for n, sample in enumerate(samples, 1):
            try:
                value = await _score_one(metric, build_input(sample))
                values.append(value)
                per_item[sample["id"]][name] = round(value, 4)
            except Exception as e:
                failures += 1
                logfire.error(f"{name} failed on {sample['id']}: {type(e).__name__}: {e}")
            print(
                f"    {n}/{len(samples)} scored={len(values)} failed={failures}",
                end="\r",
                flush=True,
            )
            await asyncio.sleep(pace)

        if values:
            entry = summarise_scores(name, values)
            entry["scored"] = len(values)
            entry["failed"] = failures
            entry["coverage"] = round(len(values) / len(samples), 3)
            summary[name] = entry
            print(
                f"\n    {name}: {entry['value']:.3f} "
                f"[{entry['ci_low']:.3f}, {entry['ci_high']:.3f}] "
                f"n={entry['n']} coverage={entry['coverage']}"
            )
        else:
            summary[name] = {
                "metric": name,
                "error": "every sample failed to score",
                "failed": failures,
            }
            print(f"\n    {name}: ALL SAMPLES FAILED")

        (out_dir / "ragas_per_item.json").write_text(
            json.dumps(per_item, indent=2), encoding="utf-8"
        )
        (out_dir / "ragas_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Per-stratum faithfulness, so a weak topic cannot hide inside the mean.
    by_topic: dict[str, list[float]] = {}
    for sample in samples:
        value = per_item[sample["id"]].get("faithfulness")
        if value is not None:
            by_topic.setdefault(sample["doc_topic"] or "unknown", []).append(value)
    summary["faithfulness_by_topic"] = {
        topic: summarise_scores("faithfulness", vals) for topic, vals in sorted(by_topic.items())
    }
    return summary


def _stratified_sample(samples: list[dict], size: int, seed: int = 11) -> list[dict]:
    """
    Proportional sample across doc_topic.

    Subsampling here is a budget decision, not a quality one: RAGAS issues
    several judge calls per metric per sample, and the provider allows 1000
    requests per model per day. Sampling proportionally keeps every topic
    represented, and the reported `n` plus confidence interval make the reduced
    precision explicit rather than hiding it behind a bare mean.
    """
    if size <= 0 or size >= len(samples):
        return samples

    import random
    from collections import defaultdict

    by_topic: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_topic[s.get("doc_topic") or "unknown"].append(s)

    rng = random.Random(seed)
    picked: list[dict] = []
    for _topic, group in sorted(by_topic.items()):
        rng.shuffle(group)
        quota = max(1, round(size * len(group) / len(samples)))
        picked.extend(group[:quota])
    rng.shuffle(picked)
    return picked[:size]


def score_records(
    items: list[dict],
    records: list[dict],
    out_dir: Path,
    pace_seconds: float = 1.0,
    sample_size: int = 0,
    metric_names: tuple[str, ...] | None = None,
) -> dict:
    samples = _prepare(items, records)
    answerable = sum(1 for i in items if i.get("category") == "answerable")
    scorable = len(samples)
    print(
        f"  {scorable} of {answerable} answerable items are scorable "
        f"(excluded: abstained, blocked, errored, or no context)"
    )
    if sample_size:
        samples = _stratified_sample(samples, sample_size)
        print(f"  stratified subsample: scoring {len(samples)} of {scorable}")
    if not samples:
        raise SystemExit("No scorable samples. Run --tier behaviour first.")

    started = time.time()
    summary = asyncio.run(_score_all(samples, pace_seconds, out_dir, metric_names))
    summary["scored_samples"] = len(samples)
    summary["scorable_samples"] = scorable
    summary["answerable_total"] = answerable
    summary["subsampled"] = bool(sample_size and len(samples) < scorable)
    summary["metrics_run"] = list(metric_names) if metric_names else "all"
    summary["judge_model"] = settings.JUDGE_MODEL
    summary["generation_model"] = settings.GENERATION_MODEL
    summary["wall_seconds"] = round(time.time() - started, 1)
    return summary
