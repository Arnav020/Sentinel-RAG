"""
Phase 2 — RAGAS + Tool Correctness metrics.

Judged by an independent model family (see JUDGE_MODEL) against the *full*
retrieved context the generator actually used, one sample at a time with
cooldowns sized to the judge's TPM ceiling.

Pass `result_cb(name, df)` to persist each experiment as it finishes — a full
run takes ~1h, and without it a failure at experiment 4 discards everything.
"""


import os
import sys
import types
import asyncio
import logfire
import pandas as pd
from openai import AsyncOpenAI

# The installed ragas version unconditionally imports ChatVertexAI from
# langchain_community.chat_models.vertexai at module load time. Newer
# langchain-community releases dropped that path — Vertex AI integrations
# moved to the standalone langchain-google-vertexai package — so the bare
# `from ragas.llms import llm_factory` below raises ModuleNotFoundError
# without this shim, even though langchain-google-vertexai is installed.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    from langchain_google_vertexai import ChatVertexAI
    _vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_shim.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_shim

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas import SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Judge model choice matters for the credibility of these numbers:
#   * Independent family. The system under test generates with Llama 3.3 70B, so
#     judging with a Llama model would be self-evaluation — a model grading its
#     own output is a bias a reviewer can fairly challenge. gpt-oss is a
#     different lineage.
#   * Its own rate-limit budget, so eval runs never starve answer generation.
#
# 20b rather than 120b: same lineage and same independence argument, but a
# separate daily token budget (120b's 200k TPD is easily exhausted by a few full
# passes). Verified equivalent before switching — on the same worst-case samples
# both judges returned identical Faithfulness scores (1.0 and 0.889), so this is
# a quota decision, not a change in how strictly the system is graded.
#
# All metrics must be scored by the SAME judge: mixing judges across metrics
# would make the numbers incomparable.
JUDGE_MODEL = "openai/gpt-oss-20b"

COOLDOWN_STANDARD = 62
# Judge TPM is 8,000. Passing the *full* retrieved context (see below) costs
# ~3k tokens/call and Faithfulness issues 2 calls, so budget ~6k tokens/sample:
# 60s * 6000/8000 = 45s is the minimum safe spacing.
COOLDOWN_MINI = 45
GENERAL_BATCH_SIZE = 1  # one sample at a time: abatch_score fires calls concurrently per sample,
                         # so batch>1 stacks multiple samples' async calls inside the same second

# Context is NOT truncated any more, and this is the single most important fix
# in this file. The previous 300-chars x 2-chunks limit existed to fit an
# 8B judge's 6,000 TPM ceiling, but chunks are ~1,400 chars, so the judge only
# ever saw each document's *header* — never the passage the answer came from.
# Every grounding metric therefore scored near zero regardless of system quality
# (measured: context_precision 0.000 across every sample, while independent
# retrieval diagnostics showed the correct document ranked #1 for 15/15
# questions). The judge must see exactly the context the generator saw, or the
# metric is measuring the harness rather than the system.
CONTEXT_TRUNCATE = None  # no truncation
CONTEXT_LIMIT = 5        # matches rerank top_n in app/agents/nodes/retriever.py


# RAGAS asks the judge for structured JSON. With full (untruncated) contexts and
# full-length answers, Faithfulness enumerates many statements, so the JSON reply
# is long — under the default output cap it gets cut mid-object and Groq rejects
# it (`IncompleteOutputException` / `json_validate_failed`, measured 0/4 success).
#
# But max_tokens cannot simply be maximised: Groq sizes a request as
# prompt + max_tokens and rejects it outright above the model's TPM ceiling
# (max_tokens=16000 produced `413 ... Requested 16959, Limit 8000`). So this is a
# budget split, not a free knob:
#     worst-case prompt (5 full contexts + longest answer) ~2,122 tokens
#     2,122 + 4,000 = 6,122  <  8,000 ceiling
# Verified on the three largest samples: 3000/4000/5000 all score identically and
# without truncation, so 4000 keeps output headroom while staying well clear of
# the ceiling. `reasoning_effort=low` stops the reasoning preamble eating it.
JUDGE_MAX_TOKENS = 4000
JUDGE_REASONING_EFFORT = "low"


def _build_judge():
    api_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL, max_retries=5)
    llm = llm_factory(
        JUDGE_MODEL,
        provider="openai",
        client=client,
        max_tokens=JUDGE_MAX_TOKENS,
        reasoning_effort=JUDGE_REASONING_EFFORT,
    )
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )
    return llm, embeddings

async def _cooldown(seconds: int, label: str, status_cb=None):
    msg = f"⏳ {seconds}s cooldown after {label} (Groq TPM buffer)..."
    if status_cb:
        status_cb(msg)
    for _ in range(seconds // 10):
        await asyncio.sleep(10)
    if status_cb:
        status_cb(f"✅ Ready — starting next experiment.")
        
        
def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns only samples with actual_response populated, passing through the
    full retrieved context the generator actually used (see CONTEXT_TRUNCATE).
    """
    valid = []
    for s in golden_dataset["rag_samples"]:
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        raw_contexts = s.get("actual_contexts") or s.get("relevant_contexts") or []
        contexts = raw_contexts[:CONTEXT_LIMIT]
        if CONTEXT_TRUNCATE is not None:
            contexts = [c[:CONTEXT_TRUNCATE] for c in contexts]
        valid.append({**s, "actual_contexts": contexts})
    return valid


def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    """Build the per-question frame, dropping samples whose scoring failed (None)."""
    rows = []
    for s, r in zip(samples, scores):
        if r is None:
            continue
        rows.append({"question": s["question"][:65], metric_key: round(float(r.value), 3)})
    if not rows:
        raise RuntimeError(f"Every sample failed to score for '{metric_key}'.")
    return pd.DataFrame(rows)


async def _batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = "") -> list:
    """
    Runs abatch_score one chunk at a time with cooldowns, keeping each burst
    under the judge's TPM ceiling.

    A sample that fails to score yields None rather than aborting the run: a full
    pass takes ~80 minutes, so one bad sample must not discard the other 14.
    Failures are surfaced via status_cb and logged, never silently dropped.
    """
    all_scores = []
    batches = [inputs[i : i + GENERAL_BATCH_SIZE] for i in range(0, len(inputs), GENERAL_BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        if b_idx > 0:
            await _cooldown(COOLDOWN_MINI, f"{label} batch {b_idx}", status_cb)
        try:
            scores = await metric.abatch_score(batch)
        except Exception as e:
            msg = f"⚠️ {label} batch {b_idx + 1}/{len(batches)} failed to score: {type(e).__name__}: {e}"
            logfire.error(msg)
            if status_cb:
                status_cb(msg)
            scores = [None] * len(batch)
        all_scores.extend(scores)
    return all_scores

# Each LLM-scored experiment: (key, human label, metric factory, input builder).
# Declarative so a resumed run can skip finished metrics without duplicating the
# scoring/cooldown/persistence logic five times.
_EXPERIMENTS = [
    (
        "faithfulness", "Faithfulness",
        lambda llm, emb: Faithfulness(llm=llm),
        lambda s: {"user_input": s["question"], "response": s["actual_response"],
                   "retrieved_contexts": s["actual_contexts"]},
    ),
    (
        "answer_relevancy", "Answer Relevancy",
        lambda llm, emb: AnswerRelevancy(llm=llm, embeddings=emb),
        lambda s: {"user_input": s["question"], "response": s["actual_response"]},
    ),
    (
        "context_precision", "Context Precision",
        lambda llm, emb: ContextPrecision(llm=llm),
        lambda s: {"user_input": s["question"], "reference": s["reference"],
                   "retrieved_contexts": s["actual_contexts"]},
    ),
    (
        "context_recall", "Context Recall",
        lambda llm, emb: ContextRecall(llm=llm),
        lambda s: {"user_input": s["question"], "reference": s["reference"],
                   "retrieved_contexts": s["actual_contexts"]},
    ),
    (
        "answer_correctness", "Answer Correctness",
        lambda llm, emb: AnswerCorrectness(llm=llm, embeddings=emb),
        lambda s: {"user_input": s["question"], "response": s["actual_response"],
                   "reference": s["reference"]},
    ),
]


def _tool_correctness_df(samples: list) -> pd.DataFrame:
    """Jaccard overlap of tools called vs expected. No LLM, so it always runs."""
    rows = []
    for s in samples:
        called = set(s.get("actual_tools_called") or [])
        expected = set(s.get("expected_tools") or [])
        union = len(called | expected)
        score = len(called & expected) / union if union > 0 else 0.0
        rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
    return pd.DataFrame(rows)


async def run_all_metrics(golden_dataset: dict, status_cb=None, result_cb=None,
                          skip_metrics=None) -> dict:
    """
    Score the enriched dataset. Returns {metric_name: DataFrame}.

    status_cb(message)      - live progress updates
    result_cb(name, df)     - called as each experiment finishes, so results are
                              persisted incrementally rather than only at the end
    skip_metrics            - metric keys to skip because they are already scored

    `skip_metrics` matters for more than convenience: one full pass costs roughly
    240k judge tokens while the free-tier daily cap is 200k, so a complete run
    cannot fit in a single day from scratch. Resuming with the finished metrics
    skipped is what makes the suite completable.
    """
    judge_llm, ragas_embeddings = _build_judge()
    samples = _prep_samples(golden_dataset)
    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    skip = set(skip_metrics or ())
    results = {}
    total = len(_EXPERIMENTS) + 1  # + tool correctness

    with logfire.span("🧪 Eval Phase 2 — All Metrics", total_samples=len(samples)):
        ran_any = False
        for idx, (key, label, make_metric, build_input) in enumerate(_EXPERIMENTS, start=1):
            if key in skip:
                if status_cb:
                    status_cb(f"⏭️  Exp {idx}/{total} — {label}: already scored, skipping.")
                continue

            if ran_any:
                # Only cool down between experiments we actually ran.
                await _cooldown(COOLDOWN_STANDARD, "previous experiment", status_cb)
            ran_any = True

            if status_cb:
                status_cb(f"🧪 Exp {idx}/{total} — {label} ({len(samples)} samples)...")
            with logfire.span(f"🧪 Exp {idx} — {label}"):
                inputs = [build_input(s) for s in samples]
                scores = await _batched_score(
                    make_metric(judge_llm, ragas_embeddings), inputs, samples, status_cb, label
                )
                df = _score_df(key, samples, scores)
                results[key] = df
                if result_cb:
                    result_cb(key, df)
                logfire.info(f"🧪 {label} done", avg=round(df[key].mean(), 3),
                             scored=len(df), total=len(samples))

        # ── Tool Correctness (no LLM — Jaccard) ──────────────────────────────
        if "tool_correctness" in skip:
            if status_cb:
                status_cb(f"⏭️  Exp {total}/{total} — Tool Correctness: already scored, skipping.")
        else:
            if status_cb:
                status_cb(f"⚡ Exp {total}/{total} — Tool Correctness (zero LLM calls)...")
            with logfire.span(f"🧪 Exp {total} — Tool Correctness"):
                df = _tool_correctness_df(samples)
                results["tool_correctness"] = df
                if result_cb:
                    result_cb("tool_correctness", df)
                logfire.info("🧪 Tool Correctness done", avg=round(df["tool_correctness"].mean(), 3))

        if status_cb:
            status_cb(f"✅ Scoring complete ({len(results)} metric(s) produced this run).")

    return results
