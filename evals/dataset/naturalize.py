"""
Rewrite generated questions in natural user voice.

The sharpest methodological objection to this eval set is circularity: every
answerable question was generated FROM the passage it is later measured against,
so it inherits that passage's vocabulary and the retriever is being asked to
match text against a paraphrase of itself. Scores come out optimistically high -
which is why the relevance threshold sits mid-band at 0.35 rather than tuned to
the observed 0.943 minimum.

This strips the vocabulary leakage without discarding the verified ground truth.
Each question is rewritten by a model that **never sees the source passage**, so
it cannot reintroduce documentation phrasing. The rewrite is then checked by a
second call to confirm it still asks for the same thing; anything that drifts is
rejected and the original is kept.

What this does NOT do is make the ground truth human-authored. The gold chunk,
the reference answer and the entailment verification are unchanged. It removes
one specific bias - shared vocabulary - and nothing more.

  question_original : the generated form, kept for audit
  question          : the naturalised form, used by every evaluation tier

    python -m evals.dataset.naturalize
    python -m evals.dataset.naturalize --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import logfire

logfire.configure(service_name="eval-naturalize", send_to_logfire=False)

from app.gateway import LLMError, complete

DATASET = Path(__file__).resolve().parent / "golden_dataset.json"

# The property that actually matters is that the rewriter is given ONLY the
# question, never the passage - that is what removes the vocabulary leakage.
# Model identity is secondary.
#
# The qwen models were the first choice (a different family from the model that
# generated the originals) but both were exhausted against the provider's
# 200k/day per-model cap, which refills at roughly 139 tokens per minute - about
# four and a half hours for the ~37k this pass needs. Substituting a model with
# budget is the right trade: the leakage being removed is vocabulary, and the
# reduction is measured below rather than assumed.
REWRITE_MODEL = "openai/gpt-oss-120b"
CHECK_MODEL = "openai/gpt-oss-20b"

PACE_SECONDS = 1.5

_REWRITE_PROMPT = """Rewrite the QUESTION the way a working platform engineer would
actually type it into a search box or a chat assistant.

Keep:
- the exact information being asked for
- any specific identifier, field name, flag, command or value it mentions

Change:
- the phrasing, so it does not read like documentation
- prefer how people really write: shorter, more direct, sometimes clipped
- drop formal scaffolding such as "What is the purpose of...", "Which of the
  following...", "In a Kubernetes Job spec, what does..."

Rules:
- Do NOT add detail that was not asked for.
- Do NOT make it vaguer - it must still have one specific answer.
- Do NOT answer it.
- Output the rewritten question alone, on one line, nothing else.

Examples:
  "What does the completions field do in a Job spec?"
      -> how does completions work in a Job

  "What is the purpose of the TTL-after-finished controller?"
      -> what does the ttl controller actually clean up

  "Which verbs are granted to the secret-reader ClusterRole?"
      -> what verbs does the secret-reader clusterrole get

QUESTION: {question}"""

_CHECK_PROMPT = """Two questions are given. Decide whether they ask for the SAME
specific information - so that one correct answer would answer both.

SAME     - same information need, only the wording differs
DIFFERENT - the rewrite asks for something else, is broader, is vaguer, or has
            lost a specific identifier the original named

Reply with exactly one word: SAME or DIFFERENT."""


def _clean(text: str) -> str:
    """Take the first usable line and strip conversational wrapping."""
    text = text.strip()
    # Reasoning models sometimes leak a <think> block; drop it.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    for line in text.splitlines():
        line = line.strip().strip("\"'`").strip()
        line = re.sub(r"^(rewritten question|question|answer)\s*:\s*", "", line, flags=re.I)
        if len(line) >= 10:
            return line
    return ""


def rewrite(question: str) -> str:
    try:
        result = complete(
            [{"role": "user", "content": _REWRITE_PROMPT.format(question=question)}],
            model=REWRITE_MODEL,
            temperature=0.7,  # some variety, or every rewrite lands on one template
            max_tokens=400,
            feature="eval-naturalize",
            bypass_cache=True,
        )
    except LLMError as e:
        print(f"    rewrite failed: {e}")
        return ""
    return _clean(result.content)


def same_intent(original: str, rewritten: str) -> bool:
    try:
        result = complete(
            [
                {"role": "system", "content": _CHECK_PROMPT},
                {"role": "user", "content": f"ORIGINAL: {original}\n\nREWRITE: {rewritten}"},
            ],
            model=CHECK_MODEL,
            temperature=0.0,
            max_tokens=600,
            feature="eval-naturalize-check",
            bypass_cache=True,
        )
    except LLMError as e:
        print(f"    intent check failed: {e}")
        return False
    return "DIFFERENT" not in result.content.upper()


def _content_words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_.-]{2,}", text.lower()))


def vocabulary_overlap(question: str, passage: str) -> float:
    """
    Fraction of the question's content words that also appear in the passage.

    This is the metric the rewrite is judged on: it is a direct proxy for the
    leakage being removed, and it is deterministic, so the reduction is measured
    rather than asserted.
    """
    q, p = _content_words(question), _content_words(passage)
    return len(q & p) / len(q) if q else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--limit", type=int, default=0, help="only the first N (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="do not write the dataset")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    random.seed(args.seed)
    path = Path(args.dataset)
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"]

    targets = [
        i for i in items if i.get("category") == "answerable" and "question_original" not in i
    ]
    if args.limit:
        targets = targets[: args.limit]

    if not targets:
        print("Nothing to do - every answerable item is already naturalised.")
        return 0

    print(f"Rewriting {len(targets)} question(s) with {REWRITE_MODEL}")
    print(f"Intent verified by {CHECK_MODEL}. The rewriter never sees the passage.\n")

    before, after = [], []
    rewritten_n = rejected_n = failed_n = 0

    for n, item in enumerate(targets, 1):
        original = item["question"]
        gold = item.get("gold_body", "")
        candidate = rewrite(original)
        time.sleep(PACE_SECONDS)

        if not candidate or candidate.lower() == original.lower():
            failed_n += 1
        elif not same_intent(original, candidate):
            rejected_n += 1
            time.sleep(PACE_SECONDS)
        else:
            time.sleep(PACE_SECONDS)
            before.append(vocabulary_overlap(original, gold))
            after.append(vocabulary_overlap(candidate, gold))
            item["question_original"] = original
            item["question"] = candidate
            item["naturalised_by"] = REWRITE_MODEL
            rewritten_n += 1

        if n % 10 == 0 or n == len(targets):
            print(
                f"  {n}/{len(targets)}  rewritten={rewritten_n} "
                f"rejected={rejected_n} failed={failed_n}"
            )

    print(f"\nrewritten={rewritten_n} rejected={rejected_n} failed={failed_n}")
    if before:
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        print("\nvocabulary overlap with the gold passage (lower = less leakage):")
        print(f"  before: {mean(before):.3f}")
        print(f"  after : {mean(after):.3f}")
        print(f"  change: {mean(after) - mean(before):+.3f}")

    print("\nExamples:")
    for item in [i for i in targets if "question_original" in i][:5]:
        print(f"  - {item['question_original']}")
        print(f"    -> {item['question']}")

    if args.dry_run:
        print("\n--dry-run: dataset not written.")
        return 0

    # Counts are recomputed over the whole dataset, not just this pass. The
    # script is idempotent and re-run to retry failures, so per-run counts would
    # under-report and the metadata is evidence like anything else.
    all_answerable = [i for i in items if i.get("category") == "answerable"]
    naturalised = [i for i in all_answerable if "question_original" in i]
    data["meta"]["naturalised"] = {
        "model": REWRITE_MODEL,
        "intent_checked_by": CHECK_MODEL,
        "naturalised": len(naturalised),
        "kept_original": len(all_answerable) - len(naturalised),
        "answerable_total": len(all_answerable),
        "rewritten_this_run": rewritten_n,
        "rejected_wrong_intent": rejected_n,
        "failed": failed_n,
        "note": (
            "Answerable questions were rewritten in natural user voice by a model "
            "that never saw the source passage, to remove the vocabulary leakage "
            "inherent in generating a question from the passage it is measured "
            "against. Intent was verified by a second model; rewrites that drifted "
            "were rejected and the original kept. The gold chunk, reference answer "
            "and entailment verification are unchanged. `question_original` "
            "preserves the pre-rewrite form."
        ),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
