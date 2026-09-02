"""
Derive the abstention threshold from data instead of guessing it.

`RERANK_THRESHOLD` decides whether the system answers or declines, so it is the
single most consequential number in the configuration. It must be set from the
measured score distributions of in-corpus and out-of-corpus queries, and
re-derived whenever the corpus, the chunker or the reranker changes.

Reports, for every candidate threshold, the two error rates that actually
matter:

  * **false answer rate**  - out-of-corpus questions the system would answer
                             anyway (the failure the audit found: confident
                             answers to "how do I bake sourdough bread")
  * **false abstention rate** - answerable questions the system would refuse

    python -m tools.calibrate_threshold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import logfire

logfire.configure(service_name="threshold-calibration", send_to_logfire=False)

from app.config import settings
from app.services.retrieval.qdrant_service import search
from app.services.retrieval.ranking_service import score_chunks

DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset" / "golden_dataset.json"
OUT = Path(__file__).resolve().parent.parent / "evals" / "runs" / "threshold_calibration.json"

GRID = [
    0.0,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


def top_score(question: str) -> float:
    candidates = search(question, limit=settings.RETRIEVAL_CANDIDATES)
    if not candidates:
        return 0.0
    ordered = score_chunks(question, candidates)
    return float(ordered[0].rerank_score or 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--limit-per-class", type=int, default=60)
    args = ap.parse_args()

    settings.validate()
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    items = data["items"]

    in_corpus = [i for i in items if i["category"] == "answerable"][: args.limit_per_class]
    out_corpus = [i for i in items if i["category"] in ("unanswerable_in_domain", "off_domain")][
        : args.limit_per_class
    ]

    print(f"Scoring {len(in_corpus)} answerable and {len(out_corpus)} out-of-corpus questions...")

    pos = []
    for n, item in enumerate(in_corpus, 1):
        pos.append(top_score(item["question"]))
        if n % 20 == 0:
            print(f"  answerable {n}/{len(in_corpus)}")

    neg = []
    for n, item in enumerate(out_corpus, 1):
        neg.append(top_score(item["question"]))
        if n % 20 == 0:
            print(f"  out-of-corpus {n}/{len(out_corpus)}")

    pos_sorted, neg_sorted = sorted(pos), sorted(neg)

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, max(0, round(p * (len(values) - 1))))
        return values[idx]

    print("\nscore distribution")
    print(
        f"  answerable    min={pos_sorted[0]:.4f}  p05={pct(pos_sorted, 0.05):.4f}  "
        f"p50={pct(pos_sorted, 0.5):.4f}  max={pos_sorted[-1]:.4f}"
    )
    print(
        f"  out-of-corpus min={neg_sorted[0]:.4f}  p50={pct(neg_sorted, 0.5):.4f}  "
        f"p95={pct(neg_sorted, 0.95):.4f}  max={neg_sorted[-1]:.4f}"
    )

    print(f"\n{'threshold':>10} {'false_answer':>13} {'false_abstain':>14} {'youden_J':>9}")
    grid_rows = []
    best = None
    for t in GRID:
        false_answer = sum(1 for s in neg if s >= t) / len(neg) if neg else 0.0
        false_abstain = sum(1 for s in pos if s < t) / len(pos) if pos else 0.0
        # Youden's J: sensitivity + specificity - 1. Maximised at the threshold
        # that best separates the two classes.
        j = (1 - false_abstain) + (1 - false_answer) - 1
        grid_rows.append(
            {
                "threshold": t,
                "false_answer_rate": round(false_answer, 4),
                "false_abstention_rate": round(false_abstain, 4),
                "youden_j": round(j, 4),
            }
        )
        if best is None or j > best["youden_j"]:
            best = grid_rows[-1]
        print(f"{t:>10.3f} {false_answer:>13.3f} {false_abstain:>14.3f} {j:>9.3f}")

    print(
        f"\nbest by Youden's J: threshold={best['threshold']} "
        f"(false_answer={best['false_answer_rate']}, "
        f"false_abstain={best['false_abstention_rate']})"
    )
    print(f"currently configured RERANK_THRESHOLD={settings.RERANK_THRESHOLD}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "collection": settings.QDRANT_COLLECTION,
                "configured_threshold": settings.RERANK_THRESHOLD,
                "recommended_threshold": best["threshold"],
                "n_answerable": len(pos),
                "n_out_of_corpus": len(neg),
                "answerable_scores": [round(s, 5) for s in pos_sorted],
                "out_of_corpus_scores": [round(s, 5) for s in neg_sorted],
                "grid": grid_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
