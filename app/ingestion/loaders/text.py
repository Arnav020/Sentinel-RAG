"""
Plain-text loader.

Splits on blank lines and classifies uniformly-indented runs as code, so a
kubectl transcript or YAML sample survives chunking as one atomic block.
"""

from __future__ import annotations

import re

from app.ingestion.document import KIND_CODE, KIND_PROSE, Block, Document

# A run of lines that are indented or look like shell/YAML is treated as code.
_CODEISH = re.compile(r"^(\s{2,}|\t|\$ |# |[A-Za-z0-9_.-]+:\s|- )")
# A short line in Title Case or ALL CAPS with no terminal punctuation reads as a
# heading in hand-written .txt runbooks.
_HEADINGISH = re.compile(r"^[A-Z][A-Za-z0-9 ,/()'-]{2,70}$")


def _looks_like_code(para: str) -> bool:
    lines = [ln for ln in para.splitlines() if ln.strip()]
    if not lines:
        return False
    hits = sum(1 for ln in lines if _CODEISH.match(ln))
    return hits >= max(1, len(lines) // 2)


def _looks_like_heading(para: str) -> bool:
    if "\n" in para or len(para) > 70:
        return False
    stripped = para.strip().strip("=-_# ")
    if not stripped:
        return False
    return bool(_HEADINGISH.match(stripped)) and not stripped.endswith((".", ":", "?"))


def parse_text(file_path: str) -> Document:
    with open(file_path, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # Normalise line endings and collapse decorative rules used as separators.
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"^[-=_*]{4,}$", "", raw, flags=re.MULTILINE)

    blocks: list[Block] = []
    heading_stack: list[str] = []

    for para in re.split(r"\n\s*\n", raw):
        para = para.rstrip()
        if not para.strip():
            continue

        if _looks_like_heading(para):
            heading_stack = [para.strip().strip("=-_# ")]
            continue

        kind = KIND_CODE if _looks_like_code(para) else KIND_PROSE
        text = para if kind == KIND_CODE else re.sub(r"\s+", " ", para).strip()
        blocks.append(Block(text=text, heading_path=tuple(heading_stack), kind=kind))

    return Document(filename="", blocks=blocks).non_empty()
