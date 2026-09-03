"""
Evaluation orchestrator.

Writes every run to `evals/runs/<timestamp>/` as a committed artifact, because
the previous suite kept results only in Streamlit session state: `save_results`
existed and was called from nowhere, the resumability the README described was
implemented but never wired into its only caller, and no result file existed
anywhere in the repository. Numbers you cannot re-open are not evidence.

Tiers, ordered by cost:

    --tier retrieval   deterministic, no LLM, ~2 min   -> gates every PR
    --tier behaviour   full pipeline, LLM              -> nightly
    --tier ragas       judged generation quality       -> nightly / on demand
    --tier all         everything

Each artifact records the git SHA, the corpus fingerprint, every model id and
the resolved package versions, so a number can always be traced back to the
system that produced it.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import logfire

from app.config import settings

logfire.configure(
    service_name="sentinel-rag-evals",
    send_to_logfire=bool(settings.LOGFIRE_ENABLED),
    token=settings.LOGFIRE_TOKEN,
)

RUNS_DIR = Path(__file__).resolve().parent / "runs"
DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "golden_dataset.json"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _package_versions() -> dict[str, str]:
    from importlib.metadata import version

    out = {}
    for pkg in (
        "ragas",
        "sentence-transformers",
        "qdrant-client",
        "flashrank",
        "nemoguardrails",
        "langgraph",
    ):
        try:
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = "not installed"
    return out


def _corpus_fingerprint() -> dict:
    from app.services.retrieval.qdrant_service import (
        RetrievalError,
        collection_stats,
        distinct_payload_values,
    )

    try:
        stats = collection_stats()
        topics = distinct_payload_values("doc_topic")
        return {**stats, "topics": topics}
    except RetrievalError as e:
        return {"error": str(e)[:200]}


def build_metadata(dataset: dict) -> dict:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "corpus": _corpus_fingerprint(),
        "dataset": {
            "path": str(DATASET_PATH),
            "total": dataset["meta"]["total"],
            "counts": dataset["meta"]["counts"],
            "generator_model": dataset["meta"].get("generator_model"),
            "verifier_model": dataset["meta"].get("verifier_model"),
        },
        "config": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "generation_model": settings.GENERATION_MODEL,
            "planner_model": settings.PLANNER_MODEL,
            "judge_model": settings.JUDGE_MODEL,
            "rerank_threshold": settings.RERANK_THRESHOLD,
            "retrieval_candidates": settings.RETRIEVAL_CANDIDATES,
            "retrieval_top_n": settings.RETRIEVAL_TOP_N,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "use_gateway": settings.USE_GATEWAY,
        },
        "packages": _package_versions(),
    }


def run_retrieval(dataset: dict, out_dir: Path) -> dict:
    from evals.harness import retrieval_eval

    print("\n=== Tier 1: retrieval + abstention (deterministic, no LLM) ===")
    started = time.time()

    def progress(done, total):
        print(f"  {done}/{total}", end="\r", flush=True)

    rows, summary = retrieval_eval.run(dataset["items"], progress=progress)
    summary["duration_seconds"] = round(time.time() - started, 1)
    (out_dir / "retrieval_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  done in {summary['duration_seconds']}s")
    return summary


def run_behaviour(dataset: dict, out_dir: Path, pace: float) -> dict:
    from evals.harness import behaviour_eval
    from evals.harness.pipeline import DailyQuotaExhausted, run_items

    print("\n=== Tier 2: end-to-end behaviour (full pipeline, LLM) ===")
    started = time.time()
    items = dataset["items"]

    def progress(done, total, record):
        state = (
            "ERR"
            if record.get("error")
            else "BLK"
            if record.get("blocked")
            else "ABS"
            if record.get("abstained")
            else "ans"
        )
        print(f"  {done}/{total} [{state}] {record['question'][:52]:52}", end="\r", flush=True)

    aborted = ""
    try:
        records = run_items(items, pace_seconds=pace, progress=progress)
    except DailyQuotaExhausted as e:
        # Keep what was gathered, but never score it. A partial run graded as
        # though it were complete reports exhausted quota as a quality
        # regression, which is worse than reporting nothing.
        aborted = str(e)
        records = getattr(e, "records", [])
        print(f"\n\n  ABORTED: {aborted}")

    # Grade the hard negatives: retrieval scores them as relevant, so only the
    # generator can decline. Needs a judge call per item.
    # Judge the answer text wherever the system answered something it was
    # supposed to decline. Without this the metric would score the retrieval
    # flag rather than what the user actually receives.
    needs_decline_check = {"decline_in_answer", "abstain", "refuse_or_abstain"}
    by_id = {i["id"]: i for i in items}
    for record in records:
        item = by_id.get(record["id"], {})
        if (
            item.get("expected_behaviour") in needs_decline_check
            and not record.get("error")
            and not record.get("blocked")
            and not record.get("abstained")
        ):
            declined, judged = behaviour_eval.answer_declines(
                record["question"], record.get("answer", "")
            )
            record["declined"] = declined
            record["decline_judged"] = judged
            time.sleep(pace)

    if records:
        (out_dir / "behaviour_records.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if aborted:
        return {
            "incomplete": aborted,
            "items_completed": len(records),
            "items_total": len(items),
            "duration_seconds": round(time.time() - started, 1),
        }

    summary = behaviour_eval.run(items, records)
    summary["duration_seconds"] = round(time.time() - started, 1)
    print(f"\n  done in {summary['duration_seconds']}s")
    return summary


def run_ragas(
    dataset: dict,
    out_dir: Path,
    pace: float,
    sample_size: int = 0,
    metric_names: tuple[str, ...] | None = None,
) -> dict:
    from evals.harness.ragas_eval import score_records

    print("\n=== Tier 3: RAGAS generation quality (judged) ===")
    records_path = out_dir / "behaviour_records.json"
    if not records_path.exists():
        raise SystemExit("RAGAS needs behaviour records - run --tier behaviour first.")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    started = time.time()
    summary = score_records(
        dataset["items"],
        records,
        out_dir=out_dir,
        pace_seconds=pace,
        sample_size=sample_size,
        metric_names=metric_names,
    )
    summary["duration_seconds"] = round(time.time() - started, 1)
    return summary


def print_summary(results: dict) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    def show(label, entry, indent=2):
        pad = " " * indent
        if not isinstance(entry, dict):
            return
        if "value" not in entry:
            print(f"{pad}{label:38} {entry.get('error', '-')}")
            return
        # Point estimates without an interval (e.g. MRR) print bare rather than
        # crashing the report - a summary that fails to render loses the run.
        if "ci_low" in entry and "ci_high" in entry:
            ci = f"[{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]"
            n = entry.get("n", "-")
            extra = f"  coverage={entry['coverage']}" if "coverage" in entry else ""
            print(f"{pad}{label:38} {entry['value']:.3f}  95% CI {ci:16} n={n}{extra}")
        else:
            print(f"{pad}{label:38} {entry['value']:.3f}")

    for tier, summary in results.items():
        if not isinstance(summary, dict):
            continue
        print(f"\n[{tier}]")
        for key, value in summary.items():
            if key in ("duration_seconds", "n", "by_topic", "by_question_type"):
                continue
            if isinstance(value, list):
                print(f"  {key}:")
                for entry in value:
                    show(entry.get("metric", "?"), entry, indent=4)
            elif isinstance(value, dict) and "value" in value:
                show(key, value)
            elif isinstance(value, dict) and key == "behaviour":
                print("  behaviour:")
                for name, entry in value.items():
                    show(name, entry, indent=4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tier", choices=["retrieval", "behaviour", "ragas", "all"], default="retrieval"
    )
    ap.add_argument("--dataset", default=str(DATASET_PATH))
    ap.add_argument("--pace", type=float, default=1.0, help="seconds between LLM calls")
    ap.add_argument("--out", default="", help="reuse an existing run directory")
    ap.add_argument("--limit", type=int, default=0, help="only the first N items (smoke runs)")
    ap.add_argument(
        "--ragas-sample",
        type=int,
        default=0,
        help="score a stratified subsample of N answerable items (0 = all)",
    )
    ap.add_argument(
        "--ragas-metrics",
        default="",
        help=(
            "comma-separated metric names, empty for all. The judge's DAILY "
            "token cap is usually the binding constraint, not wall time: a full "
            "six-metric pass costs roughly 14k judge tokens per sample."
        ),
    )
    args = ap.parse_args()

    settings.validate()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        dataset["items"] = dataset["items"][: args.limit]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else RUNS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge into an existing summary rather than replacing it. Tiers are run
    # separately - the deterministic one is cheap, the judged ones are paced
    # around a daily token cap - so `--out <existing dir>` is the normal way to
    # add a tier to a run. Rebuilding the dict from scratch silently destroyed
    # the retrieval block when the behaviour tier was added to the same run.
    summary_path = out_dir / "summary.json"
    results: dict = {}
    if summary_path.exists():
        try:
            results = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"Merging into existing run: {sorted(k for k in results if k != 'meta')}")
        except json.JSONDecodeError:
            print(f"warning: {summary_path} is unreadable; starting a fresh summary")
            results = {}
    results["meta"] = build_metadata(dataset)
    tiers = ["retrieval", "behaviour", "ragas"] if args.tier == "all" else [args.tier]

    for tier in tiers:
        if tier == "retrieval":
            results["retrieval"] = run_retrieval(dataset, out_dir)
        elif tier == "behaviour":
            results["behaviour"] = run_behaviour(dataset, out_dir, args.pace)
        elif tier == "ragas":
            metrics = tuple(m.strip() for m in args.ragas_metrics.split(",") if m.strip())
            # Same best-of merge the artifacts use. Replacing the block
            # wholesale destroyed a completed faithfulness measurement when a
            # later, quota-starved run scored a different metric into the same
            # directory - the tier-level merge above only protects tiers from
            # each other, not metrics within a tier.
            from evals.harness.ragas_eval import merge_summaries

            results["ragas"] = merge_summaries(
                results.get("ragas", {}),
                run_ragas(dataset, out_dir, args.pace, args.ragas_sample, metrics or None),
            )
        # Persist after every tier: a long run must not lose finished work.
        summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(results)
    print(f"\nArtifact: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
