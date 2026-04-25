import pytest
from ai_platform.rag.chromadb_wrapper import ChromaDBWrapper
from ai_platform.rag.retriever import RAGRetriever


@pytest.fixture
def retriever():
    wrapper = ChromaDBWrapper(ephemeral=True)
    return RAGRetriever(wrapper)


def test_retriever_retrieve_returns_strings(retriever):
    results = retriever.retrieve("any query", environment="banking")
    assert isinstance(results, list)
    for item in results:
        assert isinstance(item, str)


def test_retriever_empty_collection_returns_empty_list(retriever):
    results = retriever.retrieve("some query", environment="empty_env")
    assert results == []


def test_retriever_uses_environment_as_collection():
    wrapper = ChromaDBWrapper(ephemeral=True)
    wrapper.add_documents(
        collection_name="banking",
        documents=["banking doc about interest rates"],
        ids=["b1"],
    )
    wrapper.add_documents(
        collection_name="retail",
        documents=["retail doc about product returns"],
        ids=["r1"],
    )
    retriever = RAGRetriever(wrapper)

    banking_results = retriever.retrieve("interest rates", environment="banking")
    retail_results = retriever.retrieve("interest rates", environment="retail")

    assert any("banking" in r or "interest" in r for r in banking_results)
    assert not any("banking" in r for r in retail_results)


def test_retriever_index_and_retrieve(retriever):
    retriever.index_document(
        text="mortgage loan requires down payment",
        doc_id="mortgage1",
        environment="banking",
    )
    results = retriever.retrieve("mortgage down payment", environment="banking")
    assert len(results) > 0
    assert any("mortgage" in r for r in results)


def test_retriever_never_raises_on_missing_collection(retriever):
    try:
        results = retriever.retrieve("anything", environment="nonexistent_collection_xyz")
        assert results == []
    except Exception as exc:
        pytest.fail(f"retrieve() raised unexpectedly: {exc}")
