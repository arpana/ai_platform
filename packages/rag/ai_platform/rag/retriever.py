from __future__ import annotations

from ai_platform.rag.chromadb_wrapper import ChromaDBWrapper


class RAGRetriever:
    """Retrieves document chunks from ChromaDB using environment-scoped collections."""

    def __init__(self, wrapper: ChromaDBWrapper) -> None:
        self._wrapper = wrapper

    def retrieve(self, query: str, environment: str, n_results: int = 3) -> list[str]:
        """Return up to n_results text chunks relevant to query from environment collection."""
        try:
            return self._wrapper.query(
                collection_name=environment,
                query_text=query,
                n_results=n_results,
            )
        except Exception:
            return []

    def index_document(
        self,
        text: str,
        doc_id: str,
        environment: str,
        metadata: dict | None = None,
    ) -> None:
        """Add a single document to the environment's collection."""
        metadatas = [metadata] if metadata is not None else None
        self._wrapper.add_documents(
            collection_name=environment,
            documents=[text],
            ids=[doc_id],
            metadatas=metadatas,
        )
