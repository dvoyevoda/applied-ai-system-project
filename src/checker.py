from __future__ import annotations

from .models import CheckResult, ClassificationResult, DraftResponse, RetrievedDocument
from .openai_client import OpenAIJsonClient


class OutputChecker:
    def __init__(self, api_key: str | None = None, model: str = "gpt-5.4-mini"):
        self.client = OpenAIJsonClient(api_key, model)

    def check(
        self,
        *,
        message: str,
        classification: ClassificationResult,
        retrieved_documents: list[RetrievedDocument],
        draft: DraftResponse,
    ) -> CheckResult:
        ai_result = self._check_with_openai(message, classification, retrieved_documents, draft)
        if ai_result is not None:
            return ai_result
        return self._check_locally(classification, retrieved_documents, draft)

    def _check_with_openai(
        self,
        message: str,
        classification: ClassificationResult,
        retrieved_documents: list[RetrievedDocument],
        draft: DraftResponse,
    ) -> CheckResult | None:
        evidence = "\n".join(f"- {doc.item.title}: {doc.item.text}" for doc in retrieved_documents) or "No retrieved evidence."
        system_prompt = (
            "You are a guardrail checker for support draft replies. Return strict JSON only. "
            "Check whether the draft is grounded in the retrieved evidence, professional, and properly "
            "flagged for human review when needed. Spam, phishing, and malicious requests should be handled "
            "as untrusted messages and must not receive instructions that help the sender."
        )
        user_prompt = f"""
Original message:
{message}

Category: {classification.category}
Urgency: {classification.urgency}
Needs escalation from classifier: {classification.needs_escalation}

Retrieved evidence:
{evidence}

Draft reply:
{draft.draft_reply}

Return JSON with these keys:
needs_human_review, review_reason, evidence_coverage, professionalism, issues
""".strip()
        payload = self.client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1)
        if not payload or payload.get("_error"):
            return None
        raw_issues = payload.get("issues", [])
        issues = [str(issue) for issue in raw_issues] if isinstance(raw_issues, list) else [str(raw_issues)]
        return CheckResult(
            needs_human_review=bool(payload.get("needs_human_review", True)),
            review_reason=str(payload.get("review_reason") or "Review recommended."),
            evidence_coverage=str(payload.get("evidence_coverage") or "Unknown"),
            professionalism=str(payload.get("professionalism") or "Unknown"),
            issues=issues,
        )

    def _check_locally(
        self,
        classification: ClassificationResult,
        retrieved_documents: list[RetrievedDocument],
        draft: DraftResponse,
    ) -> CheckResult:
        issues: list[str] = []
        if not retrieved_documents:
            issues.append("No retrieved knowledge base evidence was available.")
        if classification.urgency == "High" and "verify" not in draft.suggested_action.lower() and "escalate" not in draft.suggested_action.lower():
            issues.append("High urgency case should be verified or escalated.")
        if classification.category == "Refund Request" and "verify" not in draft.draft_reply.lower():
            issues.append("Refund replies should ask for verification before confirming an outcome.")
        if classification.category in {"Spam", "Phishing Attempt", "Malicious Request"}:
            if not any(term in draft.draft_reply.lower() for term in ["do not", "quarantine", "escalate", "mark"]):
                issues.append("Risky messages should be marked, quarantined, or escalated instead of answered normally.")

        needs_review = bool(
            classification.needs_escalation
            or classification.urgency == "High"
            or classification.category in {"Refund Request", "Urgent Complaint", "Phishing Attempt", "Malicious Request"}
            or issues
        )
        review_reason = "Human review required for urgency, escalation, or policy sensitivity." if needs_review else "No major issues detected."

        return CheckResult(
            needs_human_review=needs_review,
            review_reason=review_reason,
            evidence_coverage="Good" if retrieved_documents else "Missing",
            professionalism="Looks professional" if "thanks" in draft.draft_reply.lower() else "Check tone",
            issues=issues,
        )
