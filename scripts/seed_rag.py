"""
Seed ChromaDB with sample documents for banking and retail environments.
Run: python scripts/seed_rag.py

Requires the conda environment and packages to be installed first:
  ./scripts/run_local.sh install
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs" / "rag_sources"

BANKING_DOCS = [
    {
        "id": "banking_001",
        "content": "Home loan interest rates range from 6.0% to 7.5% APR depending on credit score and loan amount. "
        "Minimum credit score requirement is 680. Maximum loan-to-value ratio is 80%.",
        "metadata": {"source": "lending_guidelines", "environment": "banking"},
    },
    {
        "id": "banking_002",
        "content": "Personal loan eligibility requires minimum annual income of $40,000 and employment "
        "history of at least 2 years. Maximum personal loan amount is $50,000.",
        "metadata": {"source": "lending_guidelines", "environment": "banking"},
    },
    {
        "id": "banking_003",
        "content": "Auto loan terms available for 36, 48, and 60 months. Interest rates start at 4.5% APR "
        "for new vehicles and 5.5% APR for used vehicles. Maximum vehicle age for used auto loans is 7 years.",
        "metadata": {"source": "lending_guidelines", "environment": "banking"},
    },
]

RETAIL_DOCS = [
    {
        "id": "retail_001",
        "content": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. "
        "Same-day delivery is available in select metro areas for orders placed before 12 PM.",
        "metadata": {"source": "shipping_policy", "environment": "retail"},
    },
    {
        "id": "retail_002",
        "content": "Return policy allows returns within 30 days of delivery. Items must be in original packaging. "
        "Refunds are processed within 5-10 business days after receiving the return.",
        "metadata": {"source": "return_policy", "environment": "retail"},
    },
    {
        "id": "retail_003",
        "content": "Loyalty program members earn 1 point per dollar spent. Platinum members (5000+ points) "
        "receive free express shipping and early access to sales events.",
        "metadata": {"source": "loyalty_program", "environment": "retail"},
    },
]


def seed():
    try:
        import chromadb
    except ImportError:
        print("chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    client = chromadb.Client()

    for collection_name, docs in [("banking_docs", BANKING_DOCS), ("retail_docs", RETAIL_DOCS)]:
        collection = client.get_or_create_collection(name=collection_name)
        collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["content"] for d in docs],
            metadatas=[d["metadata"] for d in docs],
        )
        print(f"Seeded '{collection_name}' with {len(docs)} documents")

    print("Done.")


if __name__ == "__main__":
    seed()
