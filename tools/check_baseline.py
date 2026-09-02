"""
Compare retrieval quality against the committed baseline and fail on regression.

This is what makes evaluation a gate rather than a report. It is deterministic
(no LLM, no judge, no sampling), so a failure means the system changed, not that
the grader was in a different mood.

Two modes:

  --fixture   ingest tests/fixtures/corpus into a scratch collection and check
              the small fixture baseline. Runs in CI against a local Qdrant in
              about two minutes and needs no credentials beyond the container.

  (default)   check the full golden dataset against the production baseline.
              Run locally or nightly.

Thresholds are one-sided with an explicit tolerance: a metric may improve
freely, but may not drop more than the tolerance below its baseline.

    python -m tools.check_baseline
    python -m tools.check_baseline --update      # re-record after a real change
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import logfire

logfire.configure(service_name="baseline-check", send_to_logfire=False)

import contextlib

from app.config import settings

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "evals" / "baseline.json"
FIXTURE_BASELINE_PATH = ROOT / "evals" / "baseline_fixture.json"
DATASET_PATH = ROOT / "evals" / "dataset" / "golden_dataset.json"
FIXTURE_CORPUS = ROOT / "tests" / "fixtures" / "corpus"

# Metrics that gate, with how far each may fall before the build fails.
# Tolerances are sized to the sampling noise at these n, not chosen for comfort.
GATED = {
    "hit@1": 0.08,
    "hit@5": 0.06,
    "doc_hit@5": 0.05,
    "ndcg@5": 0.06,
    "gold_containment@k": 0.06,
    "correct_abstention_rate": 0.05,
    "false_abstention_rate": -0.05,  # negative: LOWER is better, so cap increases
}

# Fixture questions with the document each should retrieve, plus out-of-corpus
# questions that must be declined. Deliberately small and hand-checked.
FIXTURE_CASES = [
    ("What does the kube-scheduler do?", "components"),
    ("How does the TTL controller clean up finished Jobs?", "ttlafterfinished"),
    ("What is the CronJob schedule syntax?", "cron-jobs"),
    ("How do I debug a pod stuck in Pending?", "debug-pods"),
    ("What is a Kubernetes controller?", "controller"),
    ("How does workload autoscaling work?", "autoscaling"),
]
FIXTURE_NEGATIVES = [
    "How do I bake sourdough bread?",
    "What is the capital of France?",
    "Write a poem about the sea.",
    "Who won the 2022 football World Cup?",
    "xyzzy plugh frobnicate",
]


def _flatten(summary: dict) -> dict[str, float]:
    """Pull the gated metrics out of a retrieval summary into a flat dict."""
    flat: dict[str, float] = {}
    for entry in summary.get("retrieval", []):
        if isinstance(entry, dict) and "metric" in entry and "value" in entry:
            flat[entry["metric"]] = entry["value"]
    for key, name in (
        ("abstention_on_unanswerable", "correct_abstention_rate"),
        ("abstention_on_answerable", "false_abstention_rate"),
    ):
        entry = summary.get(key)
        if isinstance(entry, dict) and "value" in entry:
            flat[name] = entry["value"]
    return flat


def measure_full() -> dict[str, float]:
    from evals.harness import retrieval_eval

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    _rows, summary = retrieval_eval.run(dataset["items"])
    return _flatten(summary)


def measure_fixture() -> dict[str, float]:
    """Ingest the fixture corpus, then measure routing and abstention on it."""
    import uuid

    from app.ingestion.processor import run as ingest
    from app.services.retrieval.qdrant_service import get_client, search
    from app.services.retrieval.ranking_service import select_relevant

    name = f"ci_{uuid.uuid4().hex[:10]}"
    settings.QDRANT_COLLECTION = name
    exit_code = ingest(str(FIXTURE_CORPUS), wipe=True)
    if exit_code != 0:
        raise SystemExit("fixture ingestion failed")

    try:
        routed = 0
        for question, slug in FIXTURE_CASES:
            kept, _ = select_relevant(
                question, search(question, limit=settings.RETRIEVAL_CANDIDATES)
            )
            if kept and any(slug in c.source for c in kept):
                routed += 1

        answered_on_negatives = 0
        for question in FIXTURE_NEGATIVES:
            kept, _ = select_relevant(
                question, search(question, limit=settings.RETRIEVAL_CANDIDATES)
            )
            if kept:
                answered_on_negatives += 1

        return {
            "fixture_routing_accuracy": routed / len(FIXTURE_CASES),
            "fixture_false_answer_rate": answered_on_negatives / len(FIXTURE_NEGATIVES),
        }
    finally:
        with contextlib.suppress(Exception):
            get_client().delete_collection(name)


FIXTURE_GATED = {
    "fixture_routing_accuracy": 0.17,  # one case may regress before failing
    "fixture_false_answer_rate": -0.001,  # must stay at zero
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true", help="use the CI fixture corpus")
    ap.add_argument("--update", action="store_true", help="rewrite the baseline")
    args = ap.parse_args()

    settings.validate()
    path = FIXTURE_BASELINE_PATH if args.fixture else BASELINE_PATH
    gated = FIXTURE_GATED if args.fixture else GATED

    print(f"Measuring ({'fixture' if args.fixture else 'full dataset'})...")
    current = measure_fixture() if args.fixture else measure_full()

    if args.update or not path.exists():
        payload = {
            "metrics": {k: round(v, 4) for k, v in current.items()},
            "config": {
                "embedding_model": settings.EMBEDDING_MODEL,
                "rerank_threshold": settings.RERANK_THRESHOLD,
                "retrieval_candidates": settings.RETRIEVAL_CANDIDATES,
                "retrieval_top_n": settings.RETRIEVAL_TOP_N,
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
            },
            "tolerances": gated,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        action = "Updated" if path.exists() else "Created"
        print(f"{action} baseline -> {path}")
        for k, v in current.items():
            print(f"  {k:32} {v:.4f}")
        return 0

    baseline = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    print(f"\n{'metric':34} {'baseline':>9} {'current':>9} {'delta':>9}  status")
    print("-" * 78)

    failures = []
    for metric, tolerance in gated.items():
        if metric not in baseline or metric not in current:
            continue
        base, now = baseline[metric], current[metric]
        delta = now - base
        if tolerance >= 0:
            ok = delta >= -tolerance  # higher is better
        else:
            ok = delta <= -tolerance  # lower is better
        status = "ok" if ok else "REGRESSION"
        if not ok:
            failures.append(f"{metric}: {base:.4f} -> {now:.4f} (delta {delta:+.4f})")
        print(f"{metric:34} {base:9.4f} {now:9.4f} {delta:+9.4f}  {status}")

    if failures:
        print("\nRETRIEVAL REGRESSION DETECTED:")
        for f in failures:
            print(f"  {f}")
        print(
            "\nIf this change is intended, re-record with:\n"
            "  python -m tools.check_baseline --update"
        )
        return 1

    print("\nNo regression against the committed baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
