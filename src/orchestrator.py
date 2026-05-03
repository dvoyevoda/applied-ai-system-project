from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_client import LLMClientError, OpenAILLMClient
from .guardrails import (
    overall_confidence,
    recommendation_confidence,
    self_check,
    validate_profile,
    validate_query,
)
from .logger import RunLogger
from .models import PlanStep, Recommendation, RecommendationResult, Song
from .profile_parser import ProfileParser
from .recommender import diversify_ranked, load_songs, rank_songs, unique_labels
from .retriever import MusicRetriever


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


class MusicRecommendationAgent:
    def __init__(
        self,
        song_path: str | Path = ROOT / "data" / "songs.csv",
        knowledge_path: str | Path = ROOT / "data" / "music_knowledge.json",
        log_path: str | Path | None = ROOT / "logs" / "recommendation_runs.jsonl",
        use_llm: bool = False,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
        llm_client: Any | None = None,
    ):
        self.song_path = Path(song_path)
        self.knowledge_path = Path(knowledge_path)
        self.use_llm = use_llm
        self.llm_model = model if use_llm else None
        self.llm_error: str | None = None
        self.songs = load_songs(str(self.song_path))
        self.available_genres = unique_labels(self.songs, "genre")
        self.available_moods = unique_labels(self.songs, "mood")
        self.parser = ProfileParser(
            available_genres=self.available_genres,
            available_moods=self.available_moods,
        )
        self.retriever = MusicRetriever(self.songs, self.knowledge_path)
        self.logger = RunLogger(log_path)
        self.llm_client = llm_client
        if self.use_llm and self.llm_client is None:
            try:
                self.llm_client = OpenAILLMClient(api_key=api_key or "", model=model)
            except LLMClientError as exc:
                self.llm_error = str(exc)
                self.llm_client = None

    def run(
        self,
        query: str,
        k: int = 5,
        explicit_preferences: Optional[Dict[str, Any]] = None,
        should_log: bool = True,
    ) -> RecommendationResult:
        plan_steps: List[PlanStep] = []

        sanitized_query, flags = validate_query(query)
        plan_steps.append(
            PlanStep(
                name="Input guardrails",
                status="complete",
                details=f"Validated request and found {len(flags)} flag(s): {', '.join(flags) or 'none'}.",
            )
        )

        llm_used = False
        parse_result = self.parser.parse(sanitized_query, explicit_preferences=explicit_preferences)
        if self.use_llm and self.llm_client is not None:
            try:
                parse_result = self.llm_client.refine_profile(
                    query=sanitized_query,
                    local_parse=parse_result,
                    available_genres=self.available_genres,
                    available_moods=self.available_moods,
                )
                llm_used = True
                plan_steps.append(
                    PlanStep(
                        name="External LLM profile refinement",
                        status="complete",
                        details=f"Used {self.llm_model} to refine the user taste profile.",
                    )
                )
            except LLMClientError as exc:
                self.llm_error = str(exc)
                flags.append("llm_profile_fallback")
                plan_steps.append(
                    PlanStep(
                        name="External LLM profile refinement",
                        status="fallback",
                        details=f"LLM profile step failed, so the local parser was used: {exc}",
                    )
                )
        elif self.use_llm:
            flags.append("llm_unavailable")
            plan_steps.append(
                PlanStep(
                    name="External LLM profile refinement",
                    status="fallback",
                    details=self.llm_error or "LLM was requested but no usable client was available.",
                )
            )

        profile = validate_profile(parse_result.profile)
        flags.extend(f"missing_{field}" for field in parse_result.missing_fields)
        plan_steps.append(
            PlanStep(
                name="Profile parser",
                status="complete",
                details=(
                    f"Built a {profile.activity} profile with genre={profile.favorite_genre}, "
                    f"mood={profile.favorite_mood}, energy={profile.target_energy:.2f}."
                ),
            )
        )

        documents = self.retriever.retrieve(sanitized_query, profile, limit=8)
        context = self.retriever.build_context(documents)
        knowledge_count = sum(1 for doc in documents if doc.kind == "knowledge")
        song_context_count = sum(1 for doc in documents if doc.kind == "song")
        retrieval_coverage = min(1.0, 0.25 + 0.18 * knowledge_count + 0.08 * song_context_count)
        plan_steps.append(
            PlanStep(
                name="Retriever",
                status="complete",
                details=f"Retrieved {knowledge_count} guide document(s) and {song_context_count} song match(es).",
            )
        )

        ranked = rank_songs(profile, self.songs, context=context)
        pool_size = max(k * 3, k)
        diversified = diversify_ranked(ranked[:pool_size], k=k, diversity_strength=profile.diversity)
        plan_steps.append(
            PlanStep(
                name="Scoring and reranking",
                status="complete",
                details="Scored songs with the original Module 3 formula plus retrieved context and diversity reranking.",
            )
        )

        top_score = diversified[0][1] if diversified else 0.0
        evidence_titles = context.get("evidence_titles", [])[:4]
        recommendations: List[Recommendation] = []
        for song_data, score, reasons in diversified:
            rec_confidence = recommendation_confidence(
                score=score,
                top_score=top_score,
                parse_confidence=parse_result.confidence,
                retrieval_coverage=retrieval_coverage,
            )
            recommendations.append(
                Recommendation(
                    song=Song.from_dict(song_data),
                    score=round(score, 3),
                    confidence=rec_confidence,
                    explanation="; ".join(reasons),
                    evidence=evidence_titles,
                    guardrail_notes=list(flags),
                )
            )

        if self.use_llm and self.llm_client is not None and recommendations:
            try:
                llm_explanations = self.llm_client.narrate_recommendations(
                    query=sanitized_query,
                    profile=profile,
                    documents=documents,
                    recommendations=recommendations,
                )
                for rec, llm_explanation in zip(recommendations, llm_explanations):
                    rec.explanation = f"{llm_explanation} Deterministic evidence: {rec.explanation}"
                llm_used = True
                plan_steps.append(
                    PlanStep(
                        name="External LLM explanation generation",
                        status="complete",
                        details=f"Used {self.llm_model} to generate grounded recommendation explanations.",
                    )
                )
            except LLMClientError as exc:
                self.llm_error = str(exc)
                flags.append("llm_explanation_fallback")
                plan_steps.append(
                    PlanStep(
                        name="External LLM explanation generation",
                        status="fallback",
                        details=f"LLM explanation step failed, so deterministic explanations were kept: {exc}",
                    )
                )

        confidence = overall_confidence(
            recommendations=recommendations,
            parse_confidence=parse_result.confidence,
            retrieval_coverage=retrieval_coverage,
            flags=flags,
        )
        check = self_check(
            profile=profile,
            documents=documents,
            recommendations=recommendations,
            flags=flags,
            confidence=confidence,
        )
        plan_steps.append(
            PlanStep(
                name="Self-check",
                status="complete",
                details=(
                    f"Confidence={confidence:.2f}; "
                    f"human_review={'yes' if check['needs_human_review'] else 'no'}."
                ),
            )
        )

        result = RecommendationResult(
            query=query,
            sanitized_query=sanitized_query,
            profile=profile,
            parse_confidence=parse_result.confidence,
            llm_enabled=self.use_llm,
            llm_used=llm_used,
            llm_model=self.llm_model,
            llm_error=self.llm_error,
            retrieved_documents=documents,
            recommendations=recommendations,
            overall_confidence=confidence,
            guardrail_flags=sorted(set(flags)),
            plan_steps=plan_steps,
            self_check=check,
        )

        if should_log:
            self.logger.log(result)

        return result
