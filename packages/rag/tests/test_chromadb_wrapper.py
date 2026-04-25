import pytest
from ai_platform.rag.chromadb_wrapper import ChromaDBWrapper


@pytest.fixture
def wrapper():
    return ChromaDBWrapper(ephemeral=True)


def test_wrapper_creates_ephemeral_client():
    w = ChromaDBWrapper(ephemeral=True)
    assert w is not None


def test_get_or_create_collection(wrapper):
    col = wrapper.get_or_create_collection("test_collection")
    assert col is not None
    assert col.name == "test_collection"


def test_add_and_query_documents(wrapper):
    wrapper.add_documents(
        collection_name="banking",
        documents=[
            "savings account earns interest",
            "checking account for daily use",
            "credit card rewards",
        ],
        ids=["doc1", "doc2", "doc3"],
    )
    results = wrapper.query("banking", "interest on savings", n_results=2)
    assert isinstance(results, list)
    assert len(results) > 0
    assert any("savings" in r or "interest" in r for r in results)


def test_query_empty_collection_returns_empty_list(wrapper):
    results = wrapper.query("empty_collection", "any query", n_results=3)
    assert results == []


def test_query_returns_max_n_results(wrapper):
    wrapper.add_documents(
        collection_name="test_n",
        documents=["doc a", "doc b", "doc c", "doc d", "doc e"],
        ids=["a", "b", "c", "d", "e"],
    )
    results = wrapper.query("test_n", "doc", n_results=2)
    assert len(results) <= 2


def test_add_documents_with_metadata(wrapper):
    wrapper.add_documents(
        collection_name="meta_test",
        documents=["loan repayment schedule"],
        ids=["loan1"],
        metadatas=[{"category": "loans", "priority": "high"}],
    )
    results = wrapper.query("meta_test", "loan repayment", n_results=1)
    assert len(results) == 1
    assert "loan" in results[0]


def test_multiple_collections_isolated(wrapper):
    wrapper.add_documents(
        collection_name="banking",
        documents=["banking specific content about mortgages"],
        ids=["b1"],
    )
    wrapper.add_documents(
        collection_name="retail",
        documents=["retail specific content about product returns"],
        ids=["r1"],
    )

    banking_results = wrapper.query("banking", "mortgages", n_results=3)
    retail_results = wrapper.query("retail", "mortgages", n_results=3)

    assert any("mortgage" in r for r in banking_results)
    assert not any("mortgage" in r for r in retail_results)
