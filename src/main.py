from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from .orchestrator import MusicRecommendationAgent


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the MoodMap music recommender agent.")
    parser.add_argument(
        "query",
        nargs="?",
        default="I need calm focus music for coding with lofi and acoustic texture.",
        help="Natural-language music request.",
    )
    parser.add_argument("-k", "--count", type=int, default=5, help="Number of songs to recommend.")
    parser.add_argument("--use-llm", action="store_true", help="Use OpenAI for profile refinement and explanations.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"), help="OpenAI model name.")
    parser.add_argument("--json", action="store_true", help="Print the full structured result as JSON.")
    args = parser.parse_args()

    agent = MusicRecommendationAgent(
        use_llm=args.use_llm,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=args.model,
    )
    result = agent.run(args.query, k=args.count)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"\nRequest: {result.sanitized_query}")
    print(f"Profile: {result.self_check['profile_summary']}")
    print(f"Overall confidence: {result.overall_confidence:.0%}")
    print(f"LLM mode: {'used' if result.llm_used else 'not used'}")
    if result.llm_error:
        print(f"LLM fallback reason: {result.llm_error}")
    if result.guardrail_flags:
        print(f"Guardrail flags: {', '.join(result.guardrail_flags)}")
    print("\nTop recommendations:\n")
    for index, rec in enumerate(result.recommendations, start=1):
        song = rec.song
        print(f"{index}. {song.title} - {song.artist} ({song.genre}, {song.mood})")
        print(f"   Score: {rec.score:.2f} | Confidence: {rec.confidence:.0%}")
        print(f"   Because: {rec.explanation}")
    print()


if __name__ == "__main__":
    main()
