from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


CATEGORIES = [
    "Event Issue",
    "Schedule Question",
    "Refund Request",
    "Technical Problem",
    "General Information",
    "Urgent Complaint",
    "Ambiguous Request",
    "Spam",
    "Phishing Attempt",
    "Malicious Request",
]

URGENCY_LEVELS = ["Low", "Medium", "High"]


@dataclass(frozen=True)
class KnowledgeItem:
    id: str
    title: str
    category: str
    text: str
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeItem":
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            category=str(data["category"]),
            text=str(data["text"]),
            tags=[str(tag) for tag in data.get("tags", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationResult:
    category: str
    urgency: str
    summary: str
    retrieval_needed: bool = True
    needs_escalation: bool = False
    confidence: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedDocument:
    item: KnowledgeItem
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.item.id,
            "title": self.item.title,
            "category": self.item.category,
            "text": self.item.text,
            "tags": self.item.tags,
            "score": round(float(self.score), 4),
        }


@dataclass
class DraftResponse:
    summary: str
    suggested_action: str
    draft_reply: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckResult:
    needs_human_review: bool
    review_reason: str
    evidence_coverage: str
    professionalism: str
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriageResult:
    message: str
    classification: ClassificationResult
    retrieved_documents: list[RetrievedDocument]
    draft: DraftResponse
    check: CheckResult
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_used: str = "local-fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "classification": self.classification.to_dict(),
            "retrieved_documents": [doc.to_dict() for doc in self.retrieved_documents],
            "draft": self.draft.to_dict(),
            "check": self.check.to_dict(),
            "timestamp": self.timestamp,
            "model_used": self.model_used,
        }
