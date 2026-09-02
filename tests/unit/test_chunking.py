"""
Chunker tests.

These target the specific defects the audit measured, so a regression shows up
as a named failure rather than as a quiet drop in retrieval quality:

  * splitting on "\\n\\n" when the loaders never emit one (26 of 45 chunks in
    one real document ended mid-sentence);
  * zero overlap, so a fact spanning a boundary was in neither chunk whole;
  * code and table blocks cut in half;
  * `_split_long_text` returning after its first strategy, making the
    sentence-boundary fallback unreachable.
"""

from __future__ import annotations

import itertools

import pytest

from app.config import settings
from app.ingestion.chunking.splitter import _split_long_text, chunk_document
from app.ingestion.document import KIND_CODE, KIND_PROSE, KIND_TABLE, Block, Document
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text


def _doc(blocks: list[Block]) -> Document:
    return Document(filename="t.html", blocks=blocks)


class TestSplitLongText:
    def test_respects_budget(self):
        text = ". ".join(f"sentence number {i} with some padding" for i in range(80))
        pieces = _split_long_text(text, 200)
        assert pieces
        assert all(len(p) <= 200 for p in pieces)

    def test_sentence_strategy_is_reachable(self):
        """
        A single line with no newlines must still be split at sentence
        boundaries. The old implementation returned unconditionally after the
        newline strategy, so this fell through to a hard character slice.
        """
        text = " ".join(f"This is sentence {i}." for i in range(60))
        assert "\n" not in text
        pieces = _split_long_text(text, 180)
        assert len(pieces) > 1
        # A sentence-aware split leaves most pieces ending on a full stop.
        ending_well = sum(1 for p in pieces if p.rstrip().endswith("."))
        assert ending_well >= len(pieces) - 1

    def test_hard_slice_is_last_resort(self):
        text = "x" * 500
        pieces = _split_long_text(text, 100)
        assert len(pieces) == 5
        assert all(len(p) == 100 for p in pieces)

    def test_no_content_lost(self):
        text = "\n".join(f"line {i} content here" for i in range(50))
        pieces = _split_long_text(text, 120)
        joined = "".join(pieces).replace("\n", "").replace(" ", "")
        assert joined == text.replace("\n", "").replace(" ", "")


class TestStructureAwareChunking:
    def test_single_newline_source_is_still_chunked_by_structure(self, sample_html):
        """The regression test for the original defect: no blank lines anywhere."""
        raw = sample_html.read_text(encoding="utf-8")
        assert "\n\n" not in raw.split("-->")[1], "fixture must have no blank lines"

        doc = parse_html(str(sample_html))
        chunks = chunk_document(doc)

        assert len(chunks) >= 3, "structure must produce multiple chunks"
        # Every chunk belongs to exactly one section.
        assert all(c.heading_path for c in chunks)

    def test_chunks_carry_heading_path(self, sample_html):
        chunks = chunk_document(parse_html(str(sample_html)))
        headings = {c.heading_path for c in chunks}
        assert any("Restart policies" in h for h in headings)
        assert any("Parallelism" in h for h in headings)

    def test_heading_is_prefixed_into_embedded_text(self, sample_html):
        chunks = chunk_document(parse_html(str(sample_html)))
        code = [c for c in chunks if c.kind == KIND_CODE]
        assert code, "the fixture contains a code block"
        # An isolated YAML snippet is meaningless without its section title.
        assert code[0].text.startswith("[")
        assert "Restart policies" in code[0].text

    def test_code_block_stays_atomic(self, sample_html):
        chunks = chunk_document(parse_html(str(sample_html)))
        code = [c for c in chunks if c.kind == KIND_CODE]
        assert len(code) == 1
        assert "apiVersion: batch/v1" in code[0].body
        assert "restartPolicy: OnFailure" in code[0].body

    def test_table_preserved_row_wise(self, sample_html):
        chunks = chunk_document(parse_html(str(sample_html)))
        tables = [c for c in chunks if c.kind == KIND_TABLE]
        assert tables
        body = tables[0].body
        assert "completions | Successful Pods required" in body
        assert "parallelism | Maximum concurrent Pods" in body

    def test_never_merges_across_sections(self):
        blocks = [
            Block("Alpha section body text.", ("Doc", "Alpha"), KIND_PROSE),
            Block("Beta section body text.", ("Doc", "Beta"), KIND_PROSE),
        ]
        chunks = chunk_document(_doc(blocks))
        for c in chunks:
            assert not ("Alpha section" in c.body and "Beta section" in c.body)

    def test_overlap_exists_between_consecutive_chunks(self, monkeypatch):
        """
        Consecutive chunks within one section must share text, or a fact split
        across the boundary is lost from both.
        """
        monkeypatch.setattr(settings, "CHUNK_SIZE", 320)
        monkeypatch.setattr(settings, "CHUNK_OVERLAP", 90)
        blocks = [
            Block(
                f"Paragraph {i}. It contains enough words to matter for the budget "
                f"and it ends with a full stop.",
                ("Doc", "Section"),
                KIND_PROSE,
            )
            for i in range(8)
        ]
        chunks = chunk_document(_doc(blocks))
        assert len(chunks) >= 2

        overlapping = 0
        for prev, nxt in itertools.pairwise(chunks):
            tail_words = set(prev.body.split()[-12:])
            head_words = set(nxt.body.split()[:12])
            if tail_words & head_words:
                overlapping += 1
        assert overlapping >= 1, "no overlap found between any consecutive chunks"

    def test_mid_sentence_termination_is_rare(self):
        """
        The headline regression guard. On the old chunker this rate was 58% for a
        real document; anything above ~20% means structure handling has broken.
        """
        paragraphs = [
            Block(
                f"Section {i // 3} paragraph {i}: the controller reconciles state "
                f"until the desired count is reached.",
                ("Doc", f"Section {i // 3}"),
                KIND_PROSE,
            )
            for i in range(30)
        ]
        chunks = chunk_document(_doc(paragraphs))
        prose = [c for c in chunks if c.kind == KIND_PROSE]
        bad = [c for c in prose if c.body.rstrip()[-1:] not in '.!?:;)"`']
        assert len(bad) / len(prose) < 0.2

    def test_char_spans_are_sane(self, sample_html):
        doc = parse_html(str(sample_html))
        chunks = chunk_document(doc)
        total = len(doc.text)
        for c in chunks:
            assert 0 <= c.char_start <= c.char_end <= total + 2

    def test_indices_are_dense_and_ordered(self, sample_html):
        chunks = chunk_document(parse_html(str(sample_html)))
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_empty_document_yields_nothing(self):
        assert chunk_document(_doc([])) == []

    def test_tiny_fragments_are_merged_or_dropped(self):
        blocks = [Block(f"n{i}", ("Doc", "S"), KIND_PROSE) for i in range(10)]
        chunks = chunk_document(_doc(blocks))
        assert all(len(c.body) >= settings.CHUNK_MIN_SIZE // 3 for c in chunks)


class TestTextLoader:
    def test_detects_code_blocks(self, sample_text):
        doc = parse_text(str(sample_text))
        kinds = {b.kind for b in doc.blocks}
        assert KIND_CODE in kinds
        code = [b for b in doc.blocks if b.kind == KIND_CODE]
        assert any("kubectl apply" in b.text for b in code)

    def test_detects_headings(self, sample_text):
        doc = parse_text(str(sample_text))
        headings = {b.heading_str for b in doc.blocks}
        assert any("REDIS" in h.upper() for h in headings if h)


class TestHtmlLoader:
    def test_reads_front_matter(self, sample_html):
        doc = parse_html(str(sample_html))
        assert doc.title == "Test Doc"
        assert doc.source_url == "https://example.test/doc"
        assert doc.doc_topic == "workloads"

    def test_does_not_double_count_nested_blocks(self, sample_html):
        doc = parse_html(str(sample_html))
        list_blocks = [b for b in doc.blocks if b.kind == "list"]
        assert len(list_blocks) == 1
        # The <li> text must appear once, not once per nesting level.
        assert doc.text.count("Use OnFailure to retry in place") == 1

    @pytest.mark.parametrize("tag", ["script", "style", "noscript"])
    def test_strips_non_content_tags(self, tmp_path, tag):
        p = tmp_path / "x.html"
        p.write_text(
            f"<h1>T</h1><p>Real content here for the body.</p><{tag}>SECRET</{tag}>",
            encoding="utf-8",
        )
        doc = parse_html(str(p))
        assert "SECRET" not in doc.text
