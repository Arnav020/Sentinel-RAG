"""
End-to-end retrieval regression against a real Qdrant.

This is the tier that gates every pull request. It ingests a small fixture
corpus into a throwaway collection, then asserts the properties that actually
matter and that unit tests cannot reach:

  * ingestion reconciles - the index holds exactly the chunks it claims;
  * re-ingesting a shortened document removes the orphaned tail, which the old
    upsert-only processor left behind forever;
  * point ids are path-unique, so two same-named files cannot overwrite one
    another;
  * in-corpus questions retrieve the right document;
  * out-of-corpus questions abstain.

Runs against QDRANT_URL, which in CI is a `qdrant/qdrant` service container.
Skipped when no Qdrant is reachable, so the suite stays runnable offline.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path
from typing import ClassVar

import pytest

from app.config import settings

pytestmark = pytest.mark.integration

FIXTURE_CORPUS = Path(__file__).resolve().parent.parent / "fixtures" / "corpus"


def _qdrant_reachable() -> bool:
    try:
        from qdrant_client import QdrantClient

        QdrantClient(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=5
        ).get_collections()
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(not _qdrant_reachable(), reason="no reachable Qdrant instance")


@pytest.fixture(scope="module")
def seeded_collection():
    """Ingest the fixture corpus into a throwaway collection, then drop it."""
    from app.ingestion.processor import run
    from app.services.retrieval.qdrant_service import get_client

    name = f"test_{uuid.uuid4().hex[:12]}"
    previous = settings.QDRANT_COLLECTION
    settings.QDRANT_COLLECTION = name
    os.environ["QDRANT_COLLECTION"] = name

    exit_code = run(str(FIXTURE_CORPUS), wipe=True)
    assert exit_code == 0, "fixture ingestion must succeed cleanly"

    yield name

    with contextlib.suppress(Exception):
        get_client().delete_collection(name)
    settings.QDRANT_COLLECTION = previous
    os.environ["QDRANT_COLLECTION"] = previous


@requires_qdrant
class TestIngestionIntegrity:
    def test_manifest_reconciles(self, seeded_collection):
        import json

        manifest = json.loads(Path("processed_data/manifest.json").read_text(encoding="utf-8"))
        assert manifest["failed"] == 0
        assert manifest["reconciliation_problems"] == []
        assert manifest["total_chunks"] > 0

    def test_index_count_matches_manifest(self, seeded_collection):
        import json

        from app.services.retrieval.qdrant_service import get_client

        manifest = json.loads(Path("processed_data/manifest.json").read_text(encoding="utf-8"))
        actual = get_client().get_collection(seeded_collection).points_count
        assert actual == manifest["total_chunks"]

    def test_payload_carries_provenance(self, seeded_collection):
        from app.services.retrieval.qdrant_service import get_client

        points, _ = get_client().scroll(seeded_collection, limit=5, with_payload=True)
        for p in points:
            payload = p.payload or {}
            for field in (
                "text",
                "body",
                "source",
                "filename",
                "title",
                "source_url",
                "doc_topic",
                "heading_path",
                "kind",
                "chunk_index",
                "char_start",
                "char_end",
                "doc_sha256",
            ):
                assert field in payload, f"missing provenance field {field!r}"

    def test_reingest_removes_orphaned_chunks(self, seeded_collection, tmp_path):
        """
        The old processor only upserted, so shortening a document stranded its
        tail chunks in the index - still retrievable, and now stale.
        """
        from qdrant_client.http import models

        from app.ingestion.processor import index_document
        from app.services.retrieval.qdrant_service import get_client

        client = get_client()
        long_doc = tmp_path / "doc.html"
        body = "".join(
            f"<h2>Section {i}</h2><p>{'Sentence about controllers. ' * 30}</p>" for i in range(6)
        )
        long_doc.write_text(f"<h1>Doc</h1>{body}", encoding="utf-8")

        first = index_document(client, long_doc, "t/doc.html")
        assert first.status == "indexed" and first.chunks > 3

        long_doc.write_text(
            "<h1>Doc</h1><h2>Only</h2><p>"
            + ("The controller reconciles observed state toward desired state. " * 4)
            + "</p>",
            encoding="utf-8",
        )
        second = index_document(client, long_doc, "t/doc.html")
        assert second.status == "indexed"
        assert second.chunks < first.chunks

        remaining = client.count(
            collection_name=seeded_collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(key="source", match=models.MatchValue(value="t/doc.html"))
                ]
            ),
            exact=True,
        ).count
        assert remaining == second.chunks, "orphaned chunks were left in the index"

    def test_emptied_document_has_its_chunks_removed(self, seeded_collection, tmp_path):
        """
        A document reduced to nothing must have its old points deleted. Treating
        that as "skipped" left deleted content permanently retrievable - a worse
        outcome than never having indexed it.
        """
        from qdrant_client.http import models

        from app.ingestion.processor import index_document
        from app.services.retrieval.qdrant_service import get_client

        client = get_client()
        doc = tmp_path / "vanishing.html"
        doc.write_text(
            "<h1>Vanishing</h1><h2>Body</h2><p>"
            + ("Content that will shortly be deleted entirely. " * 8)
            + "</p>",
            encoding="utf-8",
        )
        first = index_document(client, doc, "t/vanishing.html")
        assert first.status == "indexed" and first.chunks >= 1

        doc.write_text("<h1></h1>", encoding="utf-8")
        second = index_document(client, doc, "t/vanishing.html")
        assert second.status == "emptied"

        remaining = client.count(
            collection_name=seeded_collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source", match=models.MatchValue(value="t/vanishing.html")
                    )
                ]
            ),
            exact=True,
        ).count
        assert remaining == 0, "emptied document left stale points in the index"

    def test_point_ids_are_path_unique(self):
        """
        Ids were keyed on (source_type, filename, index), so the same basename
        in two folders collided and silently overwrote.
        """
        from app.ingestion.processor import _point_id

        assert _point_id("a/doc.html", 0) != _point_id("b/doc.html", 0)
        assert _point_id("a/doc.html", 0) == _point_id("a/doc.html", 0)
        assert _point_id("a/doc.html", 0) != _point_id("a/doc.html", 1)

    def test_embedding_dim_comes_from_the_model(self):
        """
        A hardcoded 768 was compared against itself by the collection-dimension
        safety check, so a model change would corrupt the index while the check
        reported agreement.
        """
        from sentence_transformers import SentenceTransformer

        from app.services.retrieval.embedding import get_embedding_dim

        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        getter = (
            getattr(model, "get_embedding_dimension", None)
            or model.get_sentence_embedding_dimension
        )
        assert get_embedding_dim() == int(getter())


@requires_qdrant
class TestRetrievalQuality:
    IN_CORPUS: ClassVar[list[tuple[str, str]]] = [
        ("What does the kube-scheduler do?", "components"),
        ("How does the TTL controller clean up finished Jobs?", "ttlafterfinished"),
        ("What is the CronJob schedule syntax?", "cron-jobs"),
        ("How do I debug a pod stuck in Pending?", "debug-pods"),
    ]
    OUT_OF_CORPUS: ClassVar[list[str]] = [
        "How do I bake sourdough bread?",
        "What is the capital of France?",
        "Write a poem about the sea.",
        "xyzzy plugh frobnicate",
    ]

    def test_in_corpus_questions_retrieve_the_right_document(self, seeded_collection):
        from app.services.retrieval.qdrant_service import search
        from app.services.retrieval.ranking_service import select_relevant

        hits = 0
        for question, expected_slug in self.IN_CORPUS:
            kept, top = select_relevant(question, search(question, limit=20))
            assert kept, f"abstained on an answerable question: {question!r} (top={top:.3f})"
            if any(expected_slug in c.source for c in kept):
                hits += 1
        assert hits >= len(self.IN_CORPUS) - 1, (
            f"only {hits}/{len(self.IN_CORPUS)} routed correctly"
        )

    def test_out_of_corpus_questions_abstain(self, seeded_collection):
        """The flagship regression guard, end to end against real vectors."""
        from app.services.retrieval.qdrant_service import search
        from app.services.retrieval.ranking_service import select_relevant

        for question in self.OUT_OF_CORPUS:
            kept, top = select_relevant(question, search(question, limit=20))
            assert not kept, (
                f"answered an out-of-corpus question {question!r} with "
                f"{len(kept)} passages (top score {top:.4f})"
            )

    def test_reranker_is_healthy(self, seeded_collection):
        """
        A silently dead cross-encoder degrades to cosine scores, where any
        in-domain text scores ~0.7 and the abstention threshold stops working.
        """
        from app.services.retrieval.ranking_service import reranker_healthy

        assert reranker_healthy(), "cross-encoder failed - abstention is not reliable"
