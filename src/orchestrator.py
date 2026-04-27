from __future__ import annotations

from pathlib import Path

from .checker import OutputChecker
from .classifier import MessageClassifier
from .generator import ResponseGenerator
from .logger import TriageLogger
from .models import TriageResult
from .retriever import KnowledgeRetriever


class InboxTriageOrchestrator:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-5.4-mini",
        knowledge_path: str | Path = "data/faq.json",
        log_path: str | Path = "logs/app_logs.csv",
    ):
        self.api_key = (api_key or "").strip()
        self.model = model.strip() or "gpt-5.4-mini"
        self.classifier = MessageClassifier(api_key=self.api_key, model=self.model)
        self.retriever = KnowledgeRetriever(knowledge_path=knowledge_path)
        self.generator = ResponseGenerator(api_key=self.api_key, model=self.model)
        self.checker = OutputChecker(api_key=self.api_key, model=self.model)
        self.logger = TriageLogger(log_path=log_path)

    def run(self, message: str, *, log: bool = True) -> TriageResult:
        clean_message = message.strip()
        if not clean_message:
            raise ValueError("Message cannot be empty.")

        classification = self.classifier.classify(clean_message)
        query = f"{clean_message} {classification.summary} {classification.category}"
        retrieved_documents = (
            self.retriever.search(query, category=classification.category, top_k=3)
            if classification.retrieval_needed
            else []
        )
        draft = self.generator.generate(
            message=clean_message,
            classification=classification,
            retrieved_documents=retrieved_documents,
        )
        check = self.checker.check(
            message=clean_message,
            classification=classification,
            retrieved_documents=retrieved_documents,
            draft=draft,
        )
        result = TriageResult(
            message=clean_message,
            classification=classification,
            retrieved_documents=retrieved_documents,
            draft=draft,
            check=check,
            model_used=self.model if self.api_key else "local-fallback",
        )
        if log:
            self.logger.log(result)
        return result
