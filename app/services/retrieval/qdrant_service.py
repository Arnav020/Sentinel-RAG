import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query


class RetrievalError(Exception):
    """Raised when the vector DB search itself fails (outage, auth, bad
    collection) — distinct from a search that succeeds but finds nothing."""


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY
)

def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.

    Raises RetrievalError on failure — callers must not treat that the same
    as a clean zero-result search (see app/agents/nodes/retriever.py).
    """
    try:
        query_vector = embed_query(query)

        # Using query_points - the modern standard for Qdrant
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True # JSON
        )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score
            })

        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        raise RetrievalError(str(e)) from e
