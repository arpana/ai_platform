from __future__ import annotations

import chromadb


class ChromaDBWrapper:
    """
    Thin wrapper around embedded ChromaDB.

    Supports ephemeral (in-memory) mode for tests and persistent
    (on-disk) mode for production usage.
    """

    def __init__(self, persist_dir: str = "./data/chroma", ephemeral: bool = False) -> None:
        """
        Initialize ChromaDB client.

        Args:
            persist_dir: Directory for persistent storage (ignored when ephemeral=True).
            ephemeral: If True, use an in-memory EphemeralClient (ideal for tests).
        """
        if ephemeral:
            self._client: chromadb.ClientAPI = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=persist_dir)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Return (or create) a ChromaDB collection by name."""
        return self._client.get_or_create_collection(name)

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """
        Add documents to the named collection.

        Args:
            collection_name: Target collection.
            documents: List of text strings to index.
            ids: Unique identifier for each document (must match length of documents).
            metadatas: Optional per-document metadata dicts.
        """
        collection = self.get_or_create_collection(collection_name)
        kwargs: dict = {"documents": documents, "ids": ids}
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        collection.add(**kwargs)

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 3,
    ) -> list[str]:
        """
        Query a collection and return matching document texts.

        Returns an empty list if the collection has no documents or does
        not exist yet.

        Args:
            collection_name: Target collection.
            query_text: Natural-language query string.
            n_results: Maximum number of results to return.

        Returns:
            Flat list of document strings (at most n_results items).
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            # ChromaDB raises if the collection is empty
            count = collection.count()
            if count == 0:
                return []

            # Cap n_results at available document count to avoid ChromaDB error
            effective_n = min(n_results, count)
            results = collection.query(query_texts=[query_text], n_results=effective_n)

            # results["documents"] is list[list[str]] — one inner list per query
            docs: list[list[str]] = results.get("documents") or [[]]
            return docs[0] if docs else []

        except Exception:
            return []
