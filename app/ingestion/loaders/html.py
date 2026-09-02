"""
HTML loader.

Walks the document in order, maintaining a heading stack, and emits one Block
per structural element. The previous implementation called `get_text()` on the
whole tree and joined lines with "\n", which destroyed every boundary the
chunker needed and merged code samples into surrounding prose.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from app.ingestion.document import (
    KIND_CODE,
    KIND_LIST,
    KIND_PROSE,
    KIND_TABLE,
    Block,
    Document,
)

_HEADINGS = {"h1": 0, "h2": 1, "h3": 2, "h4": 3, "h5": 4, "h6": 5}
_BLOCK_TAGS = set(_HEADINGS) | {"p", "pre", "table", "ul", "ol", "blockquote", "dl"}
_DROP_TAGS = ("script", "style", "meta", "noscript", "svg", "button")

_FRONT_MATTER = re.compile(r"<!--\s*(.*?)\s*-->", re.DOTALL)


def _parse_front_matter(raw: str) -> dict[str, str]:
    """Read the `key: value` comment block written by tools/fetch_corpus.py."""
    m = _FRONT_MATTER.search(raw[:2000])
    if not m:
        return {}
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip()
    return meta


def _clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table_to_text(tag: Tag) -> str:
    """Render a table row-wise so a chunk keeps the row/column association."""
    rows = []
    for tr in tag.find_all("tr"):
        cells = [_clean(td.get_text(" ")) for td in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _list_to_text(tag: Tag) -> str:
    items = []
    for li in tag.find_all("li", recursive=False):
        t = _clean(li.get_text(" "))
        if t:
            items.append(f"- {t}")
    return "\n".join(items)


def parse_html(file_path: str) -> Document:
    with open(file_path, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    meta = _parse_front_matter(raw)
    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(_DROP_TAGS):
        tag.decompose()

    blocks: list[Block] = []
    heading_stack: list[str] = []
    title = meta.get("title", "")

    for tag in soup.find_all(_BLOCK_TAGS):
        # `find_all` walks depth-first, so skip elements nested inside a block
        # we have already emitted whole (e.g. a <p> inside a <blockquote>).
        if tag.find_parent(["pre", "table", "ul", "ol", "blockquote", "dl"]):
            continue

        name = tag.name
        if name in _HEADINGS:
            text = _clean(tag.get_text(" "))
            if not text:
                continue
            level = _HEADINGS[name]
            del heading_stack[level:]
            heading_stack.append(text)
            if not title and level == 0:
                title = text
            continue

        if name == "pre":
            text = tag.get_text()
            kind = KIND_CODE
        elif name == "table":
            text = _table_to_text(tag)
            kind = KIND_TABLE
        elif name in ("ul", "ol"):
            text = _list_to_text(tag)
            kind = KIND_LIST
        else:
            text = _clean(tag.get_text(" "))
            kind = KIND_PROSE

        text = text.rstrip()
        if not text.strip():
            continue

        blocks.append(Block(text=text, heading_path=tuple(heading_stack), kind=kind))

    return Document(
        filename="",
        blocks=blocks,
        title=title,
        source_url=meta.get("source_url", ""),
        doc_topic=meta.get("doc_topic", ""),
    ).non_empty()
