"""
Answer-presence check: does the retrieved context actually answer the question?

This closes the gap the relevance threshold provably cannot. A cross-encoder
scores *topical relatedness*, not whether an answer is present, so an in-domain
question the corpus does not cover can still score highly:

    "What are the Kubernetes SIG groups?"  ->  "...expose groups of Pods..."   0.916
    "recommended way to run Kafka?"        ->  "...the recommended way to..."  0.744

Measured at candidate depth 60, that let 3 of 28 out-of-corpus questions past
retrieval (`retrieval_false_answer_rate` 0.107). The generator caught all three
via its decline instruction, so nothing wrong reached a user - but catching them
here is earlier, cheaper, and turns an abstention into a deliberate decision
rather than a lucky one.

One small-model call over the top few passages, not one per passage: the
question is whether the context *as a whole* supports an answer, which is the
same question the generator is about to be asked.

Fails OPEN. An outage here must not cause mass false abstentions, and the
generator's decline instruction is still downstream. That is the same posture as
the topical filter, and for the same reason: a second control exists.
"""

from __future__ import annotations

import logfire

from app.config import settings
from app.gateway import LLMError, complete
from app.services.retrieval.qdrant_service import RetrievedChunk

_PROMPT = """You decide whether a set of documentation passages contains the
information needed to answer a question.

Answer YES if the passages state the answer, even partially - a passage that
answers most of the question counts.

Answer NO if the passages are merely about a related topic, mention the same
words, or describe adjacent features, without actually containing the answer.

Be strict about the difference between "this is about the same area" and "this
answers the question". Sharing vocabulary is not answering.

Reply with exactly one word: YES or NO."""


def allocate_budget(bodies: list[str]) -> list[str]:
    """
    Fit passages into a shared character budget instead of capping each one.

    A flat per-passage cap is the wrong shape for real documentation. In one
    measured case the five passages ran 654, 1081, 252, 972 and 671 characters:
    a uniform 660 cap truncated the only two that carried the answer while the
    252-character passage left 408 of its allowance unused. Two answerable
    questions were refused that way - the answer sentence in one of them was
    sliced 43 characters in.

    Short passages therefore donate their unused share to long ones. The total
    stays at `ENTAILMENT_TOP_K * ENTAILMENT_MAX_CHARS`, so the token cost - the
    binding constraint on this tier - is exactly what it was, but the cut lands
    on the passage that can spare it rather than uniformly.
    """
    if not bodies:
        return []

    total = settings.ENTAILMENT_TOP_K * settings.ENTAILMENT_MAX_CHARS
    if sum(len(b) for b in bodies) <= total:
        return list(bodies)

    # Repeatedly give every still-oversized passage an equal share of what is
    # left, so that the passages under their share release the remainder.
    remaining, share = total, {i: len(b) for i, b in enumerate(bodies)}
    unresolved = set(share)
    while unresolved:
        allowance = remaining // len(unresolved)
        fits = {i for i in unresolved if share[i] <= allowance}
        if not fits:
            for i in unresolved:
                share[i] = allowance
            break
        for i in fits:
            remaining -= share[i]
        unresolved -= fits

    return [b[: share[i]] for i, b in enumerate(bodies)]


def _verdict_says_yes(raw: str) -> bool:
    """
    Read the model's verdict.

    Only an explicit NO rejects. Anything unparseable is treated as YES, which
    keeps this consistent with the module's fail-open posture: an ambiguous
    reply must not silently turn into a refusal the user cannot explain.
    """
    first = raw.strip().upper().lstrip("*_`\"' ").split(maxsplit=1)
    if not first:
        return True
    token = first[0].strip(".,:;!?*_`\"')")
    return token != "NO"


def context_answers_question(question: str, chunks: list[RetrievedChunk]) -> tuple[bool, bool]:
    """
    Does this context support an answer?

    Returns `(answers, checked)`. `checked` is False when the gate is disabled
    or the model was unreachable, so callers can tell "passed the check" apart
    from "was never checked" instead of silently treating them the same.
    """
    if not settings.ENTAILMENT_GATE_ENABLED or not chunks:
        return True, False

    top = chunks[: settings.ENTAILMENT_TOP_K]
    bodies = allocate_budget([c.body for c in top])
    passages = "\n\n".join(f"[{i}] {b}" for i, b in enumerate(bodies, 1))
    user = f"QUESTION: {question}\n\nPASSAGES:\n{passages}"

    try:
        result = complete(
            [{"role": "system", "content": _PROMPT}, {"role": "user", "content": user}],
            model=settings.ENTAILMENT_MODEL,
            temperature=0.0,
            max_tokens=settings.CLASSIFIER_MAX_TOKENS,
            feature="entailment",
        )
    except LLMError as e:
        logfire.error(f"Entailment gate unavailable, failing open: {e}")
        return True, False

    answers = _verdict_says_yes(result.content)
    if not answers:
        logfire.info(
            f"Entailment gate: retrieved context does not answer the question | {question[:70]!r}"
        )
    return answers, True
