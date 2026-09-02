"""
The contract between loaders and the chunker.

Previously every loader returned one flat string, and the chunker split it on
blank lines. The HTML and Office loaders joined their output with single
newlines and so never produced a blank line at all, which meant four of six
documents were chunked as a single undifferentiated blob: headings were severed
from their bodies and code blocks were cut in half.

Loaders now emit an ordered list of `Block`s carrying the heading trail and the
kind of content. That is what lets the chunker keep a code sample intact, avoid
merging across a section boundary, and record `heading_path` metadata that makes
a citation resolvable to a section rather than only to a filename.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Block kinds. `code` and `table` are atomic: the chunker never splits them
# mid-block if it can avoid it, because half a YAML manifest is worse than none.
KIND_PROSE = "prose"
KIND_CODE = "code"
KIND_TABLE = "table"
KIND_LIST = "list"

ATOMIC_KINDS = frozenset({KIND_CODE, KIND_TABLE})


@dataclass(slots=True)
class Block:
    """One structurally coherent piece of a document."""

    text: str
    heading_path: tuple[str, ...] = ()
    kind: str = KIND_PROSE

    @property
    def heading_str(self) -> str:
        return " > ".join(self.heading_path)

    def __post_init__(self) -> None:
        self.text = self.text.strip()


@dataclass(slots=True)
class Document:
    """A parsed source document, ready for chunking."""

    filename: str
    blocks: list[Block] = field(default_factory=list)
    title: str = ""
    source_url: str = ""
    doc_topic: str = ""

    @property
    def text(self) -> str:
        """Flat rendering, used for diagnostics and content hashing."""
        return "\n\n".join(b.text for b in self.blocks if b.text)

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)

    def non_empty(self) -> Document:
        self.blocks = [b for b in self.blocks if b.text]
        return self
