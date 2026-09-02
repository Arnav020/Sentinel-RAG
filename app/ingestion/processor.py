"""
Ingestion: parse -> chunk -> embed -> index, with reconciliation.

The previous processor swallowed every per-file exception, logged it, and exited
0 regardless, so a half-failed run reported success and nothing in the system
could tell you whether the index matched DATA/. It also keyed point IDs on
(source_type, filename, chunk_index) and only ever upserted, which meant:

  * two same-named files in different folders silently overwrote each other, and
  * re-ingesting a document that produced fewer chunks stranded the tail in the
    index forever, still retrievable and now stale.

This version:
  * keys points on the repo-relative path, so identity is unambiguous;
  * deletes a document's existing points before writing the new set;
  * records a manifest and verifies the indexed count matches the expected count;
  * exits non-zero if any file failed or reconciliation disagrees.

    python -m app.ingestion.processor DATA/kubernetes --wipe
    python -m app.ingestion.processor DATA/kubernetes --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

if sys.platform == "win32":  # emoji in log messages vs the cp1252 console
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import logfire

from app.config import settings
from app.ingestion.boilerplate import classify as classify_boilerplate
from app.ingestion.boilerplate import find_cross_document_duplicates
from app.ingestion.chunking.splitter import Chunk, chunk_document
from app.ingestion.document import Document
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text
from app.services.retrieval.embedding import embed_texts, get_embedding_dim

logfire.configure(
    service_name="sentinel-rag-ingestion",
    send_to_logfire=bool(settings.LOGFIRE_ENABLED),
    token=settings.LOGFIRE_TOKEN,
)

from qdrant_client import QdrantClient
from qdrant_client.http import models

PROCESSED_DIR = Path("processed_data")
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"

# Namespace for deterministic point IDs. Keyed on the repo-relative POSIX path,
# so two files with the same basename in different folders never collide.
_ID_NAMESPACE = uuid.UUID("c9c8b1c4-0b8b-4b3a-9b3a-8f6f3f1b2b7a")

# Payload fields that must be indexed for filtered search to stay efficient as
# the corpus grows past the full-scan threshold.
_INDEXED_FIELDS = {
    "source": models.PayloadSchemaType.KEYWORD,
    "doc_topic": models.PayloadSchemaType.KEYWORD,
    "kind": models.PayloadSchemaType.KEYWORD,
}

SUPPORTED = {".html", ".htm", ".txt", ".md", ".pdf", ".docx", ".pptx"}


@dataclass
class FileResult:
    path: str
    status: str  # indexed | skipped | failed
    chunks: int = 0
    chars: int = 0
    doc_topic: str = ""
    title: str = ""
    sha256: str = ""
    error: str = ""
    dropped: int = 0
    drop_reasons: dict = field(default_factory=dict)


def _client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=max(120, settings.QDRANT_TIMEOUT),
    )


def _point_id(rel_path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{rel_path}#{chunk_index}"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_document(path: Path) -> Document:
    ext = path.suffix.lower()
    if ext in (".html", ".htm"):
        doc = parse_html(str(path))
    elif ext in (".txt", ".md"):
        doc = parse_text(str(path))
    elif ext == ".pdf":
        doc = parse_pdf(str(path))
    elif ext in (".docx", ".pptx"):
        from app.ingestion.loaders.office import parse_office

        doc = parse_office(str(path))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    doc.filename = path.name
    if not doc.title:
        doc.title = path.stem.replace("_", " ").replace("-", " ")
    return doc


def ensure_collection(client: QdrantClient, wipe: bool) -> None:
    name = settings.QDRANT_COLLECTION
    expected_dim = get_embedding_dim()

    if wipe and client.collection_exists(name):
        client.delete_collection(name)
        logfire.info(f"Collection '{name}' deleted.")

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=expected_dim, distance=models.Distance.COSINE),
        )
        logfire.info(f"Created collection '{name}' ({expected_dim}-dim, cosine).")
    else:
        existing = client.get_collection(name).config.params.vectors.size
        if existing != expected_dim:
            raise RuntimeError(
                f"Collection '{name}' is {existing}-dim but embedding model "
                f"'{settings.EMBEDDING_MODEL}' produces {expected_dim}-dim vectors. "
                "Re-run with --wipe; ingesting would corrupt search results."
            )

    # `field_name` rather than `field`: the dataclasses `field` import is in
    # scope here and shadowing it is a trap waiting for the next edit.
    for field_name, schema in _INDEXED_FIELDS.items():
        with contextlib.suppress(Exception):  # already exists is the normal case
            client.create_payload_index(
                collection_name=name, field_name=field_name, field_schema=schema
            )


def _delete_document(client: QdrantClient, rel_path: str) -> int:
    """
    Remove every point belonging to a document. Returns how many were removed.

    Upsert alone is not enough: re-ingesting a document that now produces fewer
    chunks would leave the tail behind, still retrievable and now stale.
    """
    selector = models.Filter(
        must=[models.FieldCondition(key="source", match=models.MatchValue(value=rel_path))]
    )
    existing = client.count(
        collection_name=settings.QDRANT_COLLECTION, count_filter=selector, exact=True
    ).count
    if existing:
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(filter=selector),
            wait=True,
        )
    return existing


def prepare_document(path: Path, rel_path: str) -> tuple[FileResult, Document | None, list[Chunk]]:
    """
    Parse and chunk one file without touching the index.

    Separated from writing so cross-document boilerplate can be detected over
    the whole corpus before anything is embedded - a passage repeated across
    documents cannot be judged from inside a single document.
    """
    result = FileResult(path=rel_path, status="failed")
    try:
        doc = load_document(path)
        chunks: list[Chunk] = chunk_document(doc) if doc.blocks else []
        result.chars = doc.char_count
        result.doc_topic = doc.doc_topic or path.parent.name
        result.title = doc.title
        result.sha256 = _sha256(path)
        result.chunks = len(chunks)
        result.status = "prepared"
        return result, doc, chunks
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logfire.error(f"Failed to parse {rel_path}: {result.error}")
        return result, None, []


def index_document(
    client: QdrantClient, path: Path, rel_path: str, dry_run: bool = False
) -> FileResult:
    """
    Parse, chunk, embed and index one file. Never raises: returns a result.

    Single-file convenience wrapper; `run()` uses the two-pass path so that
    cross-document boilerplate can be detected.
    """
    result, doc, chunks = prepare_document(path, rel_path)
    if doc is None:
        return result
    return write_document(client, rel_path, doc, chunks, result, set(), dry_run)


def write_document(
    client: QdrantClient,
    rel_path: str,
    doc: Document,
    chunks: list[Chunk],
    result: FileResult,
    duplicates: set[str],
    dry_run: bool = False,
) -> FileResult:
    """Filter boilerplate, embed, and replace this document's points."""
    try:
        kept: list[Chunk] = []
        reasons: dict[str, int] = {}
        for chunk in chunks:
            reason = classify_boilerplate(chunk.body, chunk.heading_path, duplicates)
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            kept.append(chunk)
        result.dropped = len(chunks) - len(kept)
        result.drop_reasons = reasons
        chunks = kept
        result.chunks = len(chunks)

        if not chunks:
            # The document parsed but yields nothing indexable - it was emptied,
            # or reduced below the minimum chunk size. Its previous chunks must
            # still be removed: skipping the delete leaves deleted content
            # permanently retrievable, which is worse than never indexing it.
            result.status = "emptied"
            result.error = "no extractable text" if not doc.blocks else "no chunks produced"
            if not dry_run:
                removed = _delete_document(client, rel_path)
                if removed:
                    logfire.warning(
                        f"{rel_path} now yields no chunks - removed its {removed} stale point(s)."
                    )
            return result

        if dry_run:
            result.status = "indexed"
            return result

        vectors = embed_texts([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(f"embedding count {len(vectors)} != chunk count {len(chunks)}")

        points = [
            models.PointStruct(
                id=_point_id(rel_path, c.index),
                vector=vec,
                payload={
                    "text": c.text,
                    "body": c.body,
                    "source": rel_path,
                    "filename": doc.filename or rel_path.rsplit("/", 1)[-1],
                    "title": doc.title,
                    "source_url": doc.source_url,
                    "doc_topic": result.doc_topic,
                    "heading_path": c.heading_path,
                    "kind": c.kind,
                    "chunk_index": c.index,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "doc_sha256": result.sha256,
                },
            )
            for c, vec in zip(chunks, vectors, strict=False)
        ]

        # Delete the document's existing points first: upsert alone leaves the
        # tail behind when a re-ingest produces fewer chunks than last time.
        _delete_document(client, rel_path)
        client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points, wait=True)

        result.status = "indexed"
        return result

    except Exception as e:
        result.status = "failed"
        result.error = f"{type(e).__name__}: {e}"
        logfire.error(f"Failed to index {rel_path}: {result.error}")
        return result


def discover(base: Path) -> list[Path]:
    if base.is_file():
        return [base]
    return sorted(
        p
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith(".")
    )


def reconcile(client: QdrantClient, results: list[FileResult]) -> list[str]:
    """Compare expected chunk counts against what Qdrant actually holds."""
    problems: list[str] = []
    for r in results:
        if r.status != "indexed":
            continue
        actual = client.count(
            collection_name=settings.QDRANT_COLLECTION,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=r.path))]
            ),
            exact=True,
        ).count
        if actual != r.chunks:
            problems.append(f"{r.path}: expected {r.chunks} points, index holds {actual}")
    return problems


def run(base_dir: str, wipe: bool = False, dry_run: bool = False) -> int:
    base = Path(base_dir)
    if not base.exists():
        print(f"error: path '{base_dir}' does not exist", file=sys.stderr)
        return 2

    files = discover(base)
    if not files:
        print(f"error: no supported files found under '{base_dir}'", file=sys.stderr)
        return 2

    client = _client()
    if not dry_run:
        ensure_collection(client, wipe)

    results: list[FileResult] = []
    with logfire.span("Ingestion", base=str(base), files=len(files)):
        # Pass 1 - parse and chunk everything. Nothing is embedded yet, because
        # boilerplate can only be recognised by looking ACROSS documents: a
        # passage repeated in several files cannot be the unique answer to
        # anything, and those generic passages are what a deep candidate pool
        # lets the cross-encoder mistakenly promote on out-of-corpus questions.
        prepared: list[tuple[FileResult, Document | None, list[Chunk], str]] = []
        for path in files:
            rel = path.relative_to(base.parent if base.is_file() else base).as_posix()
            rel = f"{base.name}/{rel}" if not base.is_file() else path.name
            result, doc, chunks = prepare_document(path, rel)
            prepared.append((result, doc, chunks, rel))
            if doc is None:
                print(f"  FAIL      0 chunks  {rel[:60]:60} {result.error[:36]}")

        duplicates = find_cross_document_duplicates(
            {rel: [c.body for c in chunks] for _, doc, chunks, rel in prepared if doc}
        )
        if duplicates:
            print(f"  boilerplate: {len(duplicates)} passage(s) repeated across documents")

        # Pass 2 - filter, embed, and replace each document's points.
        for result, doc, chunks, rel in prepared:
            if doc is None:
                results.append(result)
                continue
            r = write_document(client, rel, doc, chunks, result, duplicates, dry_run=dry_run)
            results.append(r)
            flag = {
                "indexed": "ok  ",
                "skipped": "skip",
                "prepared": "skip",
                "emptied": "wipe",
                "failed": "FAIL",
            }[r.status]
            note = f"-{r.dropped} boilerplate " if r.dropped else ""
            print(f"  {flag} {r.chunks:5d} chunks  {rel[:60]:60} {note}{r.error[:36]}")

    indexed = [r for r in results if r.status == "indexed"]
    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status in ("skipped", "emptied")]
    total_chunks = sum(r.chunks for r in indexed)

    problems: list[str] = []
    if not dry_run and indexed:
        problems = reconcile(client, indexed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_dir": str(base),
        "collection": settings.QDRANT_COLLECTION,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dim": get_embedding_dim(),
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "files": len(files),
        "indexed": len(indexed),
        "skipped": len(skipped),
        "failed": len(failed),
        "total_chunks": total_chunks,
        "dropped_boilerplate": sum(r.dropped for r in results),
        "drop_reasons": {
            reason: sum(r.drop_reasons.get(reason, 0) for r in results)
            for reason in sorted({k for r in results for k in r.drop_reasons})
        },
        "reconciliation_problems": problems,
        "documents": [asdict(r) for r in results],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"\nindexed={len(indexed)} skipped={len(skipped)} failed={len(failed)} "
        f"chunks={total_chunks} dropped={manifest['dropped_boilerplate']} "
        f"{manifest['drop_reasons']}  manifest={MANIFEST_PATH}"
    )
    for p in problems:
        print(f"  RECONCILE MISMATCH: {p}", file=sys.stderr)
    for r in failed:
        print(f"  FAILED: {r.path} -> {r.error}", file=sys.stderr)

    return 1 if (failed or problems) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    ap.add_argument("path", nargs="?", default="DATA/kubernetes")
    ap.add_argument("--wipe", action="store_true", help="drop and recreate the collection")
    ap.add_argument("--dry-run", action="store_true", help="parse and chunk only")
    args = ap.parse_args()

    settings.validate()
    return run(args.path, wipe=args.wipe, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
