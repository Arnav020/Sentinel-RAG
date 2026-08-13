import re
from typing import List
import logfire


def _split_oversized(paragraph: str, chunk_size: int) -> List[str]:
    """
    Break a single paragraph that exceeds chunk_size into smaller pieces.
    Falls through progressively finer boundaries: line breaks, then
    sentence boundaries, then hard character slicing as a last resort.
    Used when source text has no (or very sparse) "\\n\\n" breaks — e.g.
    PDF/HTML extraction that yields one giant blob for a whole document.
    """
    if len(paragraph) <= chunk_size:
        return [paragraph]

    for separator, joiner in (("\n", "\n"), (None, " ")):
        pieces_in = paragraph.split(separator) if separator else re.split(r"(?<=[.!?])\s+", paragraph)
        if len(pieces_in) <= 1:
            continue
        pieces_out = []
        current = ""
        for piece in pieces_in:
            if len(current) + len(piece) + len(joiner) <= chunk_size:
                current += piece + joiner
            else:
                if current.strip():
                    pieces_out.extend(_split_oversized(current.strip(), chunk_size))
                current = piece + joiner
        if current.strip():
            pieces_out.extend(_split_oversized(current.strip(), chunk_size))
        return pieces_out

    # No natural boundary found at all — hard slice.
    return [paragraph[i:i + chunk_size] for i in range(0, len(paragraph), chunk_size)]


def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Simple semantic-ish chunker that splits by paragraphs.
    Ensures chunks do not exceed the specified size.
    """
    joiner = "\n\n"

    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        paragraphs = text.split(joiner)
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) + len(joiner) <= chunk_size:
                current_chunk += p + joiner
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                if len(p) >= chunk_size:
                    chunks.extend(_split_oversized(p, chunk_size))
                    current_chunk = ""
                else:
                    current_chunk = p + joiner

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
