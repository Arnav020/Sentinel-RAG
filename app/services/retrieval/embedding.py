"""
Local sentence-transformers embeddings.

Two corrections over the previous version:

1. The dimension is read from the loaded model, not from a hardcoded 768. The
   constant was also what the ingestion "safety check" compared the existing
   collection against, so changing the model name without editing the constant
   would create a wrong-sized collection while the check reported agreement.

2. Encoding is serialised with a lock. FastAPI runs sync endpoints in a
   threadpool, so concurrent requests previously called `encode()` on one shared
   model instance, which sentence-transformers does not guarantee is safe.
"""

from __future__ import annotations

import threading

import logfire
from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None
_dim: int | None = None
_load_lock = threading.Lock()
_encode_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model, _dim
    if _model is None:
        with _load_lock:
            if _model is None:  # re-check: another thread may have won the race
                logfire.info(f"Loading embedding model '{settings.EMBEDDING_MODEL}'.")
                model = SentenceTransformer(settings.EMBEDDING_MODEL)
                # sentence-transformers renamed this accessor; support both so
                # the dimension is always read from the model rather than
                # falling back to a constant that could disagree with it.
                getter = (
                    getattr(model, "get_embedding_dimension", None)
                    or model.get_sentence_embedding_dimension
                )
                _dim = int(getter())
                _model = model
                logfire.info(f"Embedding model ready ({_dim}-dim).")
    return _model


def get_embedding_dim() -> int:
    """Actual output dimension of the configured model."""
    _get_model()
    assert _dim is not None
    return _dim


def embed_query(query: str) -> list[float]:
    model = _get_model()
    with _encode_lock:
        vec = model.encode([query], show_progress_bar=False, normalize_embeddings=True)
    return vec[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed. Used by ingestion, where throughput matters more than latency."""
    model = _get_model()
    out: list[list[float]] = []
    batch = settings.EMBEDDING_BATCH_SIZE
    for i in range(0, len(texts), batch):
        window = texts[i : i + batch]
        with logfire.span("Embed batch", start=i, size=len(window)):
            with _encode_lock:
                vecs = model.encode(window, show_progress_bar=False, normalize_embeddings=True)
            out.extend(vecs.tolist())
    return out


def reset_for_tests() -> None:
    """Drop the cached model so a test can switch EMBEDDING_MODEL."""
    global _model, _dim
    _model, _dim = None, None
