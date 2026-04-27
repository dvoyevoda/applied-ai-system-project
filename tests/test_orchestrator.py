from src.orchestrator import InboxTriageOrchestrator


def test_orchestrator_runs_with_local_fallback(tmp_path):
    orchestrator = InboxTriageOrchestrator(
        knowledge_path="data/faq.json",
        log_path=tmp_path / "logs.csv",
    )
    result = orchestrator.run("I registered for tonight but never got the Zoom link. It starts in 45 minutes.")

    assert result.classification.category == "Event Issue"
    assert result.classification.urgency == "High"
    assert result.retrieved_documents
    assert result.check.needs_human_review is True
