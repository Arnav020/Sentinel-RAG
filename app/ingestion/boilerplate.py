"""
Boilerplate removal.

Measured motivation, not housekeeping. Increasing the retrieval candidate pool
from 20 to 100 raised chunk hit@5 from 0.859 to 0.976 - but it also broke
abstention on three previously-correct out-of-corpus questions, because a deeper
pool gives the cross-encoder more chances to promote a generic passage:

  "How do I configure Velero for backup and restore?"
      -> "You need to have a Kubernetes cluster, and the kubectl command-line
          tool must be configured..."          (prerequisite boilerplate, 3 docs)
  "What is the recommended way to run Kafka on Kubernetes?"
      -> "...This is the recommended way of managing Kubernetes applications"
  "What are the Kubernetes SIG groups?"
      -> "...expose groups of Pods over a network"

The first is pure boilerplate and removing it is free. The other two are real
content the cross-encoder mismatched on a surface word, and no ingestion filter
can fix those - they are what the generator's decline instruction is for.

This module removes what genuinely carries no answer:
  * passages repeated near-verbatim across documents (prerequisites, feature-gate
    notices) - they cannot be the unique answer to anything;
  * navigation sections ("What's next") that are lists of links;
  * raw HTML that leaked through extraction.
"""

from __future__ import annotations

import re

# Section headings whose content is navigation or administrivia rather than
# documentation. Matched on the LAST heading segment, case-insensitively.
NAVIGATION_HEADINGS = {
    "what's next",
    "whats next",
    "feedback",
    "next steps",
    "see also",
    "further reading",
}

# A chunk whose normalised body appears in at least this many distinct documents
# is boilerplate: it cannot be the unique answer to a question.
CROSS_DOC_DUPLICATE_THRESHOLD = 2

_HTML_TAG = re.compile(r"<[a-zA-Z/][^>]{0,200}>")
_WS = re.compile(r"\W+")


def normalise(text: str) -> str:
    """Aggressive normalisation used only for duplicate detection."""
    return _WS.sub(" ", text.lower()).strip()


def is_navigation(heading_path: str) -> bool:
    if not heading_path:
        return False
    leaf = heading_path.split(" > ")[-1].strip().lower().rstrip(":")
    return leaf in NAVIGATION_HEADINGS


def html_ratio(text: str) -> float:
    """Fraction of characters inside HTML tags - high means extraction leaked."""
    if not text:
        return 0.0
    tagged = sum(len(m.group(0)) for m in _HTML_TAG.finditer(text))
    return tagged / len(text)


def is_html_artifact(text: str) -> bool:
    return html_ratio(text) > 0.25


def strip_html(text: str) -> str:
    cleaned = _HTML_TAG.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def find_cross_document_duplicates(
    documents: dict[str, list[str]],
    threshold: int = CROSS_DOC_DUPLICATE_THRESHOLD,
) -> set[str]:
    """
    Normalised bodies appearing in `threshold` or more distinct documents.

    `documents` maps a document path to its chunk bodies.
    """
    seen: dict[str, set[str]] = {}
    for path, bodies in documents.items():
        for body in bodies:
            key = normalise(body)
            if len(key) < 40:
                continue  # too short to judge; the min-size filter handles these
            seen.setdefault(key, set()).add(path)
    return {key for key, paths in seen.items() if len(paths) >= threshold}


def classify(body: str, heading_path: str, duplicates: set[str]) -> str | None:
    """
    Reason this chunk should be dropped, or None to keep it.

    Returned as a string so the manifest can report *why* content was removed
    rather than silently shrinking the index.
    """
    if is_navigation(heading_path):
        return "navigation_section"
    if is_html_artifact(body):
        return "html_artifact"
    if normalise(body) in duplicates:
        return "cross_document_boilerplate"
    return None
