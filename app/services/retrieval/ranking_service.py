import os
import time
import logfire
from flashrank import Ranker, RerankRequest

from app.config import settings

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily.
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.

    The cache directory comes from settings so a container can bake the model in
    at build time and never download on the request path. Falls back to
    FlashRank's own default only on a real filesystem error, and logs when it
    does — the previous version swallowed every exception, so the "specific
    cache directory" it advertised was in fact never used on Windows.
    """
    global _ranker
    if _ranker is None:
        cache_dir = settings.FLASHRANK_CACHE_DIR
        logfire.info(f"🧠 Initializing FlashRank Model (TinyBERT) from {cache_dir}...")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            _ranker = Ranker(cache_dir=cache_dir)
        except OSError as e:
            logfire.warning(
                f"FlashRank cache dir '{cache_dir}' unusable ({e}) — "
                "falling back to the library default (may re-download the model)."
            )
            _ranker = Ranker()
    return _ranker



def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    
    Why FlashRank? 
    Standard vector search (Cosine Similarity) is fast but mathematically "fuzzy."
    FlashRank uses a Cross-Encoder approach which is much more precise but usually slow.
    FlashRank solves this by using highly optimized, quantized ONNX models locally.
    """
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")

    try:
        ranker = _get_ranker()
        
        # FlashRank expects a list of dictionaries with 'id' and 'text'
        passages = [
            {"id": i, "text": doc}
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        
        # Results are returned sorted by highest semantic score first
        reranked_docs = []
        for res in results[:top_n]:
            reranked_docs.append(res['text'])

        duration = time.time() - start_time
        top_score = results[0]['score'] if results else 'N/A'
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}")
        
        return reranked_docs

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fallback to the original Qdrant order to ensure the user still gets an answer
        return documents[:top_n]
