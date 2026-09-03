"""
Measure what approximate search costs, without needing a bigger corpus.

At 2,069 points the collection sits below Qdrant's `indexing_threshold`, so
searches are an exact brute-force scan. Every retrieval number in this repo was
therefore measured at a recall ceiling that will not hold as the corpus grows:
past the threshold Qdrant switches to approximate HNSW traversal and recall
drops by some unknown amount.

"Unknown" is the problem. Lowering the threshold forces HNSW on the corpus we
already have, so the exact-vs-approximate delta becomes a measured number rather
than a caveat in a README.

    python -m tools.scale_test --mode exact        # report current index state
    python -m tools.scale_test --mode hnsw         # force HNSW, wait, report
    python -m tools.scale_test --mode hnsw --measure   # ...and run the suite
    python -m tools.scale_test --mode exact --measure  # restore, re-measure

The collection is not rebuilt and no vectors are re-embedded - only the
optimizer's indexing threshold changes.

IMPORTANT, and learned the hard way: this is only reversible in one direction.
`indexing_threshold` controls when Qdrant *builds* an index, not whether it uses
one. Raising it again leaves an already-built HNSW index in place and searches
keep using it. So the exact arm must be measured BEFORE the hnsw arm, and going
back to genuinely exact search means recreating the collection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import logfire

logfire.configure(service_name="scale-test", send_to_logfire=False)

from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.qdrant_service import get_client

OUT = Path(__file__).resolve().parent.parent / "evals" / "runs" / "scale_test.json"

# Below the collection's on-disk vector size, so the optimizer builds an HNSW
# index. Above it, Qdrant keeps the plain (exact) index.
HNSW_THRESHOLD_KB = 1000
EXACT_THRESHOLD_KB = 20000


def index_state(client) -> dict:
    info = client.get_collection(settings.QDRANT_COLLECTION)
    return {
        "points": info.points_count,
        "indexed_vectors": info.indexed_vectors_count,
        "indexing_threshold_kb": info.config.optimizer_config.indexing_threshold,
        "hnsw_m": info.config.hnsw_config.m,
        "hnsw_ef_construct": info.config.hnsw_config.ef_construct,
        "status": str(info.status),
        "mode": "approximate (HNSW)" if info.indexed_vectors_count else "exact (full scan)",
    }


def set_mode(client, mode: str, wait_seconds: int = 300) -> dict:
    threshold = HNSW_THRESHOLD_KB if mode == "hnsw" else EXACT_THRESHOLD_KB
    print(f"Setting indexing_threshold to {threshold} KB ...")
    client.update_collection(
        collection_name=settings.QDRANT_COLLECTION,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=threshold),
    )

    if mode == "exact" and index_state(client)["indexed_vectors"]:
        print(
            "  NOTE: an HNSW index is already built. Raising the threshold does not\n"
            "  remove it - searches keep using it. The exact arm must be measured\n"
            "  first, or the collection recreated, to get a true full scan."
        )

    want_indexed = mode == "hnsw"
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        state = index_state(client)
        indexed = bool(state["indexed_vectors"])
        if indexed == want_indexed and state["status"] == "green":
            print(
                f"  index settled: {state['mode']} "
                f"({state['indexed_vectors']}/{state['points']} vectors indexed)"
            )
            return state
        print(
            f"  waiting... status={state['status']} "
            f"indexed={state['indexed_vectors']}/{state['points']}",
            end="\r",
            flush=True,
        )
        time.sleep(5)

    print("\n  timed out waiting for the optimizer to settle")
    return index_state(client)


def measure() -> dict:
    """Run the deterministic retrieval suite and return the headline metrics."""
    from evals.harness import retrieval_eval

    dataset_path = (
        Path(__file__).resolve().parent.parent / "evals" / "dataset" / "golden_dataset.json"
    )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    def progress(done, total):
        print(f"  scoring {done}/{total}", end="\r", flush=True)

    _rows, summary = retrieval_eval.run(dataset["items"], progress=progress)
    print()

    flat = {}
    for entry in summary.get("retrieval", []):
        if isinstance(entry, dict) and "metric" in entry and "value" in entry:
            flat[entry["metric"]] = entry["value"]
    for key, name in (
        ("abstention_on_unanswerable", "correct_abstention_rate"),
        ("abstention_on_answerable", "false_abstention_rate"),
    ):
        if key in summary:
            flat[name] = summary[key]["value"]
    return flat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exact", "hnsw"], required=True)
    ap.add_argument(
        "--measure", action="store_true", help="run the retrieval suite after switching"
    )
    ap.add_argument("--wait", type=int, default=300)
    args = ap.parse_args()

    settings.validate()
    client = get_client()

    print(f"collection: {settings.QDRANT_COLLECTION}")
    print(f"before: {json.dumps(index_state(client), indent=2)}\n")

    state = set_mode(client, args.mode, args.wait)
    print(f"\nafter: {json.dumps(state, indent=2)}")

    record = {}
    if OUT.exists():
        record = json.loads(OUT.read_text(encoding="utf-8"))

    entry = {"index_state": state}
    if args.measure:
        print(f"\nMeasuring retrieval with {state['mode']} search...")
        entry["metrics"] = measure()
        for k, v in entry["metrics"].items():
            print(f"  {k:26} {v:.4f}")

    record[args.mode] = entry

    # If both arms are present, report the delta - the point of the exercise.
    if "exact" in record and "hnsw" in record:
        a = record["exact"].get("metrics")
        b = record["hnsw"].get("metrics")
        if a and b:
            print(f"\n{'metric':26} {'exact':>9} {'hnsw':>9} {'delta':>9}")
            print("-" * 56)
            deltas = {}
            for k in sorted(set(a) & set(b)):
                deltas[k] = round(b[k] - a[k], 4)
                print(f"{k:26} {a[k]:9.4f} {b[k]:9.4f} {deltas[k]:+9.4f}")
            record["delta_hnsw_minus_exact"] = deltas

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
