from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .models import Recommendation, RetrievedDocument, UserProfile


PROMPT_INJECTION_TERMS = [
    "ignore previous instructions",
    "reveal the system prompt",
    "system prompt",
    "developer message",
    "api key",
    "secret key",
    "bypass",
]

SENSITIVE_WELLBEING_TERMS = ["suicide", "self harm", "hurt myself", "end my life"]


def validate_query(query: str) -> Tuple[str, List[str]]:
    flags: List[str] = []
    sanitized = query.strip()

    if not sanitized:
        flags.append("empty_input")
        sanitized = "Recommend a balanced music discovery playlist."

    lowered = sanitized.lower()
    if len(sanitized) > 1200:
        flags.append("input_truncated")
        sanitized = sanitized[:1200]

    if any(term in lowered for term in PROMPT_INJECTION_TERMS):
        flags.append("prompt_injection_detected")

    if any(term in lowered for term in SENSITIVE_WELLBEING_TERMS):
        flags.append("wellbeing_sensitive")

    if len(sanitized.split()) < 4:
        flags.append("vague_preference")

    return sanitized, flags


def validate_profile(profile: UserProfile) -> UserProfile:
    profile.target_energy = min(1.0, max(0.0, profile.target_energy))
    profile.diversity = min(1.0, max(0.0, profile.diversity))
    profile.novelty = min(1.0, max(0.0, profile.novelty))
    if profile.desired_valence is not None:
        profile.desired_valence = min(1.0, max(0.0, profile.desired_valence))
    if profile.target_danceability is not None:
        profile.target_danceability = min(1.0, max(0.0, profile.target_danceability))
    return profile


def recommendation_confidence(
    score: float,
    top_score: float,
    parse_confidence: float,
    retrieval_coverage: float,
) -> float:
    score_strength = min(1.0, score / 7.0)
    relative_strength = 1.0 if top_score <= 0 else min(1.0, score / top_score)
    confidence = (
        0.36 * parse_confidence
        + 0.24 * retrieval_coverage
        + 0.24 * score_strength
        + 0.16 * relative_strength
    )
    return round(max(0.05, min(0.99, confidence)), 3)


def overall_confidence(
    recommendations: Sequence[Recommendation],
    parse_confidence: float,
    retrieval_coverage: float,
    flags: Sequence[str],
) -> float:
    if recommendations:
        rec_average = sum(rec.confidence for rec in recommendations[:3]) / min(3, len(recommendations))
    else:
        rec_average = 0.0
    confidence = 0.45 * rec_average + 0.35 * parse_confidence + 0.20 * retrieval_coverage
    if "prompt_injection_detected" in flags:
        confidence -= 0.08
    if "vague_preference" in flags:
        confidence -= 0.10
    if "empty_input" in flags:
        confidence -= 0.20
    return round(max(0.05, min(0.99, confidence)), 3)


def self_check(
    profile: UserProfile,
    documents: Sequence[RetrievedDocument],
    recommendations: Sequence[Recommendation],
    flags: Sequence[str],
    confidence: float,
) -> Dict[str, Any]:
    issues: List[str] = []
    needs_human_review = False

    if not recommendations:
        issues.append("No recommendations were produced.")
        needs_human_review = True

    if not any(doc.kind == "knowledge" for doc in documents):
        issues.append("No activity or safety guide was retrieved.")

    if confidence < 0.50:
        issues.append("Overall confidence is low, so the user should add more preferences.")
        needs_human_review = True

    if "prompt_injection_detected" in flags:
        issues.append("Prompt-injection language was treated as untrusted preference text.")
        needs_human_review = True

    if "vague_preference" in flags:
        issues.append("The request was vague; recommendations rely on default assumptions.")

    if "wellbeing_sensitive" in flags:
        issues.append("Sensitive wellbeing language detected; music recommendations should not be treated as help.")
        needs_human_review = True

    genres = {rec.song.genre for rec in recommendations}
    if len(recommendations) >= 4 and len(genres) == 1 and profile.diversity > 0.20:
        issues.append("Top results may be too narrow because all recommendations share one genre.")

    return {
        "passed": not needs_human_review and not issues,
        "needs_human_review": needs_human_review,
        "issues": issues,
        "profile_summary": (
            f"{profile.activity} profile: {profile.favorite_genre}, {profile.favorite_mood}, "
            f"energy {profile.target_energy:.2f}, acoustic={profile.likes_acoustic}"
        ),
    }
