from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "Song":
        return cls(
            id=int(row["id"]),
            title=str(row["title"]),
            artist=str(row["artist"]),
            genre=str(row["genre"]),
            mood=str(row["mood"]),
            energy=float(row["energy"]),
            tempo_bpm=float(row["tempo_bpm"]),
            valence=float(row["valence"]),
            danceability=float(row["danceability"]),
            acousticness=float(row["acousticness"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    activity: str = "general"
    desired_valence: Optional[float] = None
    target_danceability: Optional[float] = None
    diversity: float = 0.25
    novelty: float = 0.35
    raw_query: str = ""
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProfileParseResult:
    profile: UserProfile
    confidence: float
    assumptions: List[str]
    missing_fields: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "confidence": self.confidence,
            "assumptions": list(self.assumptions),
            "missing_fields": list(self.missing_fields),
        }


@dataclass
class RetrievedDocument:
    id: str
    title: str
    kind: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanStep:
    name: str
    status: str
    details: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class Recommendation:
    song: Song
    score: float
    confidence: float
    explanation: str
    evidence: List[str] = field(default_factory=list)
    guardrail_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "song": self.song.to_dict(),
            "score": self.score,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence": list(self.evidence),
            "guardrail_notes": list(self.guardrail_notes),
        }


@dataclass
class RecommendationResult:
    query: str
    sanitized_query: str
    profile: UserProfile
    parse_confidence: float
    llm_enabled: bool
    llm_used: bool
    llm_model: Optional[str]
    llm_error: Optional[str]
    retrieved_documents: List[RetrievedDocument]
    recommendations: List[Recommendation]
    overall_confidence: float
    guardrail_flags: List[str]
    plan_steps: List[PlanStep]
    self_check: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "sanitized_query": self.sanitized_query,
            "profile": self.profile.to_dict(),
            "parse_confidence": self.parse_confidence,
            "llm_enabled": self.llm_enabled,
            "llm_used": self.llm_used,
            "llm_model": self.llm_model,
            "llm_error": self.llm_error,
            "retrieved_documents": [doc.to_dict() for doc in self.retrieved_documents],
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "overall_confidence": self.overall_confidence,
            "guardrail_flags": list(self.guardrail_flags),
            "plan_steps": [step.to_dict() for step in self.plan_steps],
            "self_check": dict(self.self_check),
        }
