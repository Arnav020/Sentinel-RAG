"""
PDF loader.

pypdf first, with pdfplumber retried only on pages that came back blank.

The previous version appended pdfplumber's recovered pages to the END of the
text list rather than at their original position, so any PDF with mid-document
image pages had its text silently reordered. Pages are now keyed by index and
reassembled in order.
"""

from __future__ import annotations

import re

import logfire
from pypdf import PdfReader

from app.ingestion.document import KIND_PROSE, Block, Document


def _paragraphs(page_text: str) -> list[str]:
    text = page_text.replace("\r\n", "\n").replace("\r", "\n")
    # De-hyphenate words broken across lines before paragraph splitting.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    parts = re.split(r"\n\s*\n", text)
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) > 1:
            out.append(p)
    return out


def parse_pdf(file_path: str) -> Document:
    reader = PdfReader(file_path)
    total = len(reader.pages)
    logfire.info(f"PDF has {total} pages.")

    pages: dict[int, str] = {}
    blank: list[int] = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logfire.warning(f"pypdf failed on page {i + 1}: {e}")
            text = ""
        if text.strip():
            pages[i] = text
        else:
            blank.append(i)

    if blank:
        logfire.info(f"pypdf returned blank on {len(blank)} page(s) - retrying with pdfplumber.")
        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                for i in blank:
                    if i >= len(pdf.pages):
                        continue
                    recovered = pdf.pages[i].extract_text() or ""
                    if recovered.strip():
                        pages[i] = recovered  # keyed by index: order preserved
        except Exception as e:
            logfire.warning(f"pdfplumber fallback failed: {e}")

    blocks: list[Block] = []
    for i in sorted(pages):
        heading = (f"page {i + 1}",)
        for para in _paragraphs(pages[i]):
            blocks.append(Block(text=para, heading_path=heading, kind=KIND_PROSE))

    if not blocks:
        logfire.warning(f"No text extracted from {file_path} - file may be fully image-based.")

    return Document(filename="", blocks=blocks).non_empty()
