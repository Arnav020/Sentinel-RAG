"""
Verify the golden dataset against the live corpus before anything is scored.

The old dataset's `relevant_contexts` were written from model memory: 10 of 19
did not appear verbatim anywhere in the corpus, and one had none of its
sentences present. Nothing checked, so nobody knew.

This runs as a gate in CI. It is cheap and deterministic, and it catches both
directions of drift: a dataset edited by hand, and a corpus re-ingested in a way
that moves or removes the passages the dataset points at.

    python -m tools.verify_dataset
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import logfire

logfire.configure(service_name="dataset-verify", send_to_logfire=False)

from app.config import settings
from app.services.retrieval.qdrant_service import get_client

DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset" / "golden_dataset.json"

REQUIRED_FIELDS = ("id", "question", "category", "expected_behaviour")
VALID_BEHAVIOURS = {
    "answer",
    "abstain",
    "refuse_or_abstain",
    "block",
    "answer_or_abstain",
    "converse",
    "decline_in_answer",
}
MIN_LEXICAL_GROUNDING = 0.55


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET))
    args = ap.parse_args()

    settings.validate()
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    items = data["items"]
    problems: list[str] = []

    # --- schema ------------------------------------------------------------
    ids = Counter(i.get("id") for i in items)
    for dupe, n in ids.items():
        if n > 1:
            problems.append(f"duplicate id {dupe!r} appears {n} times")
    for item in items:
        for field in REQUIRED_FIELDS:
            if not item.get(field):
                problems.append(f"{item.get('id', '?')}: missing {field}")
        behaviour = item.get("expected_behaviour")
        if behaviour and behaviour not in VALID_BEHAVIOURS:
            problems.append(f"{item['id']}: unknown expected_behaviour {behaviour!r}")

    questions = Counter(_norm(i["question"]) for i in items)
    for question, n in questions.items():
        if n > 1:
            problems.append(f"duplicate question appears {n} times: {question[:60]!r}")

    # --- ground truth actually exists in the live index ---------------------
    client = get_client()
    answerable = [i for i in items if i.get("category") == "answerable"]
    missing_chunks = 0

    for item in answerable:
        if not item.get("reference"):
            problems.append(f"{item['id']}: answerable item has no reference answer")
        grounding = item.get("lexical_grounding")
        if grounding is not None and grounding < MIN_LEXICAL_GROUNDING:
            problems.append(
                f"{item['id']}: lexical grounding {grounding} below {MIN_LEXICAL_GROUNDING} "
                "- the reference may contain claims absent from its source passage"
            )

    chunk_ids = [i["gold_chunk_id"] for i in answerable if i.get("gold_chunk_id")]
    for start in range(0, len(chunk_ids), 100):
        batch = chunk_ids[start : start + 100]
        found = client.retrieve(
            collection_name=settings.QDRANT_COLLECTION, ids=batch, with_payload=True
        )
        found_by_id = {str(p.id): p for p in found}
        for cid in batch:
            point = found_by_id.get(cid)
            if point is None:
                missing_chunks += 1
                problems.append(
                    f"gold chunk {cid} is not in collection "
                    f"'{settings.QDRANT_COLLECTION}' - the corpus moved under the dataset"
                )
                continue
            item = next(i for i in answerable if i.get("gold_chunk_id") == cid)
            indexed_body = _norm((point.payload or {}).get("body", ""))
            if _norm(item.get("gold_body", "")) != indexed_body:
                problems.append(
                    f"{item['id']}: gold_body no longer matches the indexed chunk "
                    "- re-chunking has changed this passage"
                )

    # --- report -------------------------------------------------------------
    counts = Counter(i.get("category") for i in items)
    print(f"dataset: {len(items)} items")
    for category, n in sorted(counts.items()):
        print(f"  {category:32} {n}")
    print(f"collection: {settings.QDRANT_COLLECTION}")
    print(f"gold chunks checked: {len(chunk_ids)}, missing: {missing_chunks}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems[:40]:
            print(f"  - {p}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1

    print("\nDataset is consistent with the live corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
