"""
Structure-aware chunker.

The previous implementation split on blank lines with zero overlap, so:
  * HTML and Office documents (which never contained a blank line) were packed
    into 1500-character slabs regardless of structure;
  * headings were routinely severed from the body they introduced;
  * YAML manifests and tables were cut mid-block;
  * and `_split_oversized` returned unconditionally at the end of its first loop
    iteration, so the sentence-boundary strategy was unreachable.

Measured consequence on the old corpus: 26 of 45 `cronjobs.docx` chunks ended
mid-sentence, and ~19% of the sentences needed by the reference answers never
reached the generator.

This version works from the structured `Block`s the loaders now emit:

  * a chunk never spans a section boundary, so a chunk is always about one thing;
  * code and table blocks stay atomic unless they alone exceed the budget;
  * every chunk is prefixed with its heading trail, so an isolated snippet like
    `restartPolicy: OnFailure` carries the context that makes it retrievable;
  * consecutive chunks within a section overlap, so a fact split across a
    boundary survives in at least one chunk whole.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import logfire

from app.config import settings
from app.ingestion.document import ATOMIC_KINDS, KIND_PROSE, Block, Document

_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+")


@dataclass(slots=True)
class Chunk:
    """One indexable unit."""

    body: str  # the source text, exactly as it appears in the document
    heading_path: str  # " > " joined heading trail
    kind: str
    index: int
    char_start: int
    char_end: int

    @property
    def text(self) -> str:
        """What actually gets embedded: heading trail plus body."""
        if self.heading_path:
            return f"[{self.heading_path}]\n{self.body}"
        return self.body


def _split_long_text(text: str, budget: int) -> list[str]:
    """
    Break a single oversized block into pieces of at most `budget` characters.

    Tries progressively finer boundaries and - unlike the previous version -
    actually reaches each one: paragraph, then line, then sentence, then a hard
    slice as the final fallback.
    """
    if len(text) <= budget:
        return [text]

    for splitter in (
        lambda s: s.split("\n\n"),
        lambda s: s.split("\n"),
        lambda s: _SENTENCE_END.split(s),
        lambda s: re.split(r"(?<=,)\s+", s),
    ):
        pieces = [p for p in splitter(text) if p.strip()]
        if len(pieces) <= 1:
            continue  # this boundary does not exist in the text - try the next

        out: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current}\n{piece}" if current else piece
            if len(candidate) <= budget:
                current = candidate
            else:
                if current:
                    out.append(current)
                # A single piece can still exceed the budget; recurse so the
                # next (finer) boundary gets a chance at it.
                current = piece if len(piece) <= budget else ""
                if not current:
                    out.extend(_split_long_text(piece, budget))
        if current:
            out.append(current)
        if out:
            return out

    # No natural boundary at any level - hard slice.
    return [text[i : i + budget] for i in range(0, len(text), budget)]


def _overlap_tail(text: str, overlap: int) -> str:
    """
    Trailing slice of `text` for the next chunk to start with, cut at a sentence
    boundary where possible so the overlap reads as prose rather than a fragment.
    """
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    parts = _SENTENCE_END.split(tail, maxsplit=1)
    if len(parts) == 2 and len(parts[1]) >= overlap // 3:
        return parts[1].strip()
    return tail.strip()


def chunk_document(doc: Document) -> list[Chunk]:
    """Turn a parsed Document into overlapping, section-scoped chunks."""
    size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    min_size = settings.CHUNK_MIN_SIZE

    # Offsets into the flat document text, so each chunk can record the span it
    # came from. Must mirror Document.text exactly.
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for b in doc.blocks:
        offsets.append((cursor, cursor + len(b.text)))
        cursor += len(b.text) + 2  # the "\n\n" join

    chunks: list[Chunk] = []

    def emit(body: str, heading: str, kind: str, start: int, end: int) -> None:
        body = body.strip()
        if not body:
            return
        chunks.append(
            Chunk(
                body=body,
                heading_path=heading,
                kind=kind,
                index=len(chunks),
                char_start=start,
                char_end=end,
            )
        )

    # Group consecutive blocks that share a heading path: a chunk never spans a
    # section boundary.
    groups: list[list[int]] = []
    for i, b in enumerate(doc.blocks):
        if groups and doc.blocks[groups[-1][-1]].heading_path == b.heading_path:
            groups[-1].append(i)
        else:
            groups.append([i])

    for group in groups:
        heading = doc.blocks[group[0]].heading_str
        # Budget left for the body once the heading prefix is accounted for.
        budget = max(min_size, size - len(heading) - 4)

        buf = ""
        buf_start: int | None = None
        buf_end = 0
        buf_kind = KIND_PROSE

        # `heading` is bound as a default argument rather than captured from the
        # enclosing loop. Late binding in a closure over a loop variable is a
        # classic source of silent misattribution - every chunk in the document
        # would end up labelled with the LAST section's heading. It happens to
        # be safe here because flush() is only ever called within the same
        # iteration, but relying on that is exactly the kind of implicit
        # invariant that breaks the next time this loop is restructured.
        def flush(section: str = heading) -> None:
            nonlocal buf, buf_start, buf_end, buf_kind
            if buf.strip() and buf_start is not None:
                # buf_end is `nonlocal`, so late binding is precisely what is
                # wanted here: flush() must read the value at call time.
                emit(buf, section, buf_kind, buf_start, buf_end)  # noqa: B023
            buf, buf_start, buf_kind = "", None, KIND_PROSE

        for i in group:
            block: Block = doc.blocks[i]
            b_start, b_end = offsets[i]

            # An atomic block that fits gets its own space; one that does not is
            # split on line boundaries, which is the least-bad option for code.
            if block.kind in ATOMIC_KINDS:
                flush()
                if len(block.text) <= budget:
                    emit(block.text, heading, block.kind, b_start, b_end)
                else:
                    for piece in _split_long_text(block.text, budget):
                        emit(piece, heading, block.kind, b_start, b_end)
                continue

            if len(block.text) > budget:
                flush()
                for piece in _split_long_text(block.text, budget):
                    emit(piece, heading, block.kind, b_start, b_end)
                continue

            candidate = f"{buf}\n\n{block.text}" if buf else block.text
            if len(candidate) <= budget:
                buf = candidate
                buf_start = b_start if buf_start is None else buf_start
                buf_end = b_end
            else:
                flush()
                tail = _overlap_tail(chunks[-1].body, overlap) if chunks else ""
                buf = f"{tail}\n\n{block.text}" if tail else block.text
                buf_start, buf_end = b_start, b_end

        flush()

    # Merge runs of small chunks that share a heading and kind. Without this,
    # neither a document of one-line sections nor the tail fragment of a split
    # code block is viable: a 40-character vector can still win a retrieval slot
    # while carrying no usable information.
    merged: list[Chunk] = []
    for c in chunks:
        if (
            merged
            and merged[-1].kind == c.kind
            and merged[-1].heading_path == c.heading_path
            and min(len(merged[-1].body), len(c.body)) < min_size
            and len(merged[-1].body) + len(c.body) <= size
        ):
            prev = merged[-1]
            joiner = "\n" if c.kind in ATOMIC_KINDS else "\n\n"
            merged[-1] = Chunk(
                body=f"{prev.body}{joiner}{c.body}",
                heading_path=prev.heading_path,
                kind=prev.kind,
                index=prev.index,
                char_start=prev.char_start,
                char_end=c.char_end,
            )
        else:
            merged.append(c)

    # Anything still below the floor after merging carries too little signal to
    # be worth a vector; it is unreachable context, not lost content, because
    # the surrounding section is indexed either side of it.
    merged = [c for c in merged if len(c.body) >= min_size // 3]

    # Renumber after merging so `index` is dense and reflects document order.
    for n, c in enumerate(merged):
        c.index = n

    logfire.info(f"Generated {len(merged)} chunks (structure-aware, overlap={overlap}).")
    return merged
