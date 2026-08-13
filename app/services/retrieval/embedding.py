import logfire
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 50
EMBEDDING_DIM = 768  # all-mpnet-base-v2

_model: SentenceTransformer | None = None


def _init():
    """Load the local embedding model once per process. Called lazily on first use."""
    global _model
    if _model is None:
        logfire.info("Loading sentence-transformers embedding model (all-mpnet-base-v2, 768-dim).")
        _model = SentenceTransformer("all-mpnet-base-v2")


def get_embedding_dim() -> int:
    return EMBEDDING_DIM


def embed_query(query: str) -> list[float]:
    _init()
    return _model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", start=i, size=len(batch)):
            all_embeddings.extend(_model.encode(batch, show_progress_bar=False).tolist())
    return all_embeddings
