"""
Add the hardest abstention cases to the golden dataset.

Threshold calibration showed the relevance threshold separates off-domain
questions from in-domain ones almost perfectly (off-domain tops out around
0.15; answerable questions start at 0.94). What it does NOT separate is
"in-domain and covered" from "in-domain but not covered" - a cross-encoder
scores topical relatedness, not whether the answer is actually present.

Concretely: "How do I install Kubernetes using kubeadm on Ubuntu?" scores 0.983
against a CronJob tutorial's "Before you begin" section, because that section
does talk about needing a cluster. Retrieval cannot decline it.

These are exactly the cases the dataset builder was DROPPING, because a
candidate that retrieval considers covered cannot be labelled
"unanswerable_in_domain" without making the abstention metric circular. Dropping
them lost the hardest evidence in the suite.

They are recorded here as a separate category with a different expected
behaviour: retrieval is allowed to return passages, but the GENERATOR must
decline rather than fabricate an answer from loosely-related context. That
measures the second line of defence, which nothing else in the suite tests.

    python -m evals.dataset.augment_hard_negatives
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import logfire

logfire.configure(service_name="eval-hard-negatives", send_to_logfire=False)

from app.config import settings
from app.services.retrieval.qdrant_service import search
from app.services.retrieval.ranking_service import select_relevant
from evals.dataset.build import UNANSWERABLE_CANDIDATES

DATASET = Path(__file__).resolve().parent / "golden_dataset.json"


def main() -> int:
    settings.validate()
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    items = data["items"]

    already = {i["question"] for i in items}
    hard: list[dict] = []

    print(
        "Finding in-domain questions retrieval considers covered but the corpus does not answer..."
    )
    for question in UNANSWERABLE_CANDIDATES:
        if question in already:
            continue  # already accepted as a clean abstention case
        kept, top = select_relevant(question, search(question, limit=settings.RETRIEVAL_CANDIDATES))
        if not kept:
            continue  # retrieval declines it; not a hard case
        hard.append(
            {
                "id": f"H{len(hard) + 1:03d}",
                "question": question,
                "reference": "",
                "category": "uncovered_retrieval_positive",
                "expected_behaviour": "decline_in_answer",
                "question_type": "uncovered_hard",
                "retrieval_top_score": round(top, 4),
                "retrieved_sources": sorted({c.source for c in kept}),
            }
        )
        print(f"  HARD (top={top:.3f}): {question[:70]}")

    data["items"] = items + hard
    counts = data["meta"]["counts"]
    counts["uncovered_retrieval_positive"] = len(hard)
    data["meta"]["total"] = len(data["items"])
    data["meta"]["notes"] = {
        "unanswerable_in_domain": (
            "Selected by dropping candidates that cleared the relevance threshold, "
            "so their scores are bounded by construction. Do NOT calibrate the "
            "threshold on this subset - use off_domain, which is unfiltered."
        ),
        "uncovered_retrieval_positive": (
            "In-domain questions the corpus does not answer but retrieval scores "
            "highly anyway. Retrieval cannot decline these; the generator must. "
            "Graded on whether the answer declines rather than fabricates."
        ),
    }

    DATASET.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAdded {len(hard)} hard negatives. Dataset now has {data['meta']['total']} items.")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
