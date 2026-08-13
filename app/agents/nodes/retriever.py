import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge, RetrievalError
from app.services.retrieval.ranking_service import rerank_documents

def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]


    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        try:
            raw_results = search_enterprise_knowledge(query, limit=15)
        except RetrievalError as e:
            # Distinct from "search succeeded, found nothing" — the KB/search
            # infra itself is broken, so say so instead of silently answering
            # as if the topic just isn't covered.
            logfire.error(f"Retrieval infra failure, not a zero-result search: {e}")
            return {
                "documents": [],
                "status": "Knowledge base search failed — retrieval service unavailable.",
                "plan": state["plan"] + ["Retrieval Error"]
            }
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        doc_contents = [doc['content'] for doc in raw_results]
        # Content -> source lookup so reranked chunks keep their citation
        # (rerank_documents only operates on plain text, not the full dicts).
        content_to_source = {doc['content']: doc['source'] for doc in raw_results}

        with logfire.span("⚖️ Semantic Reranking"):
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")

        formatted_docs = [
            f"SOURCE: {content_to_source.get(doc, 'Unknown')}\nCONTENT: {doc}"
            for doc in reranked_contents
        ]

    return {
        "documents": formatted_docs,
        "status": f"Found {len(formatted_docs)} relevant chunks." if formatted_docs else "No relevant context found.",
        "plan": state["plan"] + (["Context Retrieved"] if formatted_docs else ["No Context Found"])
    }
