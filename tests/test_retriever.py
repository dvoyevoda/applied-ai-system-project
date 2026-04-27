from src.retriever import KnowledgeRetriever


def test_retriever_finds_refund_policy():
    retriever = KnowledgeRetriever("data/faq.json")
    docs = retriever.search("Can I cancel and get a refund tomorrow?", category="Refund Request")
    assert docs
    assert docs[0].item.id == "refund_deadline_policy"
