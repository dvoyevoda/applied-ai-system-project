from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .models import TriageResult


class TriageLogger:
    HEADERS = [
        "timestamp",
        "message",
        "category",
        "urgency",
        "summary",
        "retrieved_ids",
        "retrieved_titles",
        "suggested_action",
        "draft_reply",
        "needs_human_review",
        "review_reason",
        "model_used",
    ]

    def __init__(self, log_path: str | Path = "logs/app_logs.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, result: TriageResult) -> None:
        row = self._flatten(result)
        file_exists = self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _flatten(self, result: TriageResult) -> dict[str, Any]:
        docs = result.retrieved_documents
        return {
            "timestamp": result.timestamp,
            "message": result.message,
            "category": result.classification.category,
            "urgency": result.classification.urgency,
            "summary": result.draft.summary,
            "retrieved_ids": ";".join(doc.item.id for doc in docs),
            "retrieved_titles": ";".join(doc.item.title for doc in docs),
            "suggested_action": result.draft.suggested_action,
            "draft_reply": result.draft.draft_reply,
            "needs_human_review": result.check.needs_human_review,
            "review_reason": result.check.review_reason,
            "model_used": result.model_used,
        }
