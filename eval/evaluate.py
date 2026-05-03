from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestrator import MusicRecommendationAgent


@dataclass
class CaseResult:
    case_id: str
    top_match: bool
    confidence_pass: bool
    flag_pass: bool
    retrieval_pass: bool
    overall_pass: bool
    top_song: str
    confidence: float
    flags: List[str]


def parse_options(value: str) -> set[str]:
    return {part.strip().lower() for part in value.split("|") if part.strip()}


def required_flags(value: str) -> set[str]:
    return parse_options(value) if value else set()


def evaluate_case(agent: MusicRecommendationAgent, row: dict) -> CaseResult:
    result = agent.run(row["query"], k=5, should_log=False)
    top = result.recommendations[0].song if result.recommendations else None
    expected_genres = parse_options(row["expected_top_genres"])
    expected_moods = parse_options(row["expected_top_moods"])
    expected_flags = required_flags(row.get("required_flags", ""))
    actual_flags = set(result.guardrail_flags)

    top_match = bool(
        top
        and (
            top.genre.lower() in expected_genres
            or top.mood.lower() in expected_moods
        )
    )
    confidence_pass = result.overall_confidence >= float(row["min_confidence"])
    flag_pass = expected_flags.issubset(actual_flags)
    retrieval_pass = any(doc.kind == "knowledge" for doc in result.retrieved_documents)
    overall_pass = top_match and confidence_pass and flag_pass and retrieval_pass

    return CaseResult(
        case_id=row["id"],
        top_match=top_match,
        confidence_pass=confidence_pass,
        flag_pass=flag_pass,
        retrieval_pass=retrieval_pass,
        overall_pass=overall_pass,
        top_song=f"{top.title} ({top.genre}, {top.mood})" if top else "none",
        confidence=result.overall_confidence,
        flags=sorted(actual_flags),
    )


def load_cases(path: Path) -> Iterable[dict]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        yield from csv.DictReader(csv_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the MoodMap recommender.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "evaluation_cases.csv",
        help="CSV file with evaluation inputs and expected behavior.",
    )
    args = parser.parse_args()

    agent = MusicRecommendationAgent(log_path=None)
    results = [evaluate_case(agent, row) for row in load_cases(args.cases)]

    total = len(results)
    passed = sum(result.overall_pass for result in results)
    top_matches = sum(result.top_match for result in results)
    confidence_passes = sum(result.confidence_pass for result in results)
    flag_passes = sum(result.flag_pass for result in results)
    retrieval_passes = sum(result.retrieval_pass for result in results)

    print("\nMoodMap Evaluation Summary")
    print(f"Cases passed: {passed}/{total}")
    print(f"Top genre/mood match: {top_matches}/{total}")
    print(f"Confidence threshold: {confidence_passes}/{total}")
    print(f"Required guardrail flags: {flag_passes}/{total}")
    print(f"Knowledge retrieval: {retrieval_passes}/{total}")
    print()

    for result in results:
        status = "PASS" if result.overall_pass else "FAIL"
        flags = ", ".join(result.flags) or "none"
        print(
            f"{status} {result.case_id}: top={result.top_song}; "
            f"confidence={result.confidence:.0%}; flags={flags}"
        )

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
