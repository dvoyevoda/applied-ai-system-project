from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestrator import InboxTriageOrchestrator  # noqa: E402


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate the AI Inbox Triage Assistant on a labeled CSV file.")
    parser.add_argument("--test-set", default=str(ROOT / "data" / "test_set.csv"), help="Path to labeled test CSV.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"), help="OpenAI model name.")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI instead of the local fallback.")
    parser.add_argument("--output", default="", help="Optional path to write per-example results as CSV.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY") if args.use_openai else None
    orchestrator = InboxTriageOrchestrator(
        api_key=api_key,
        model=args.model,
        knowledge_path=ROOT / "data" / "faq.json",
        log_path=ROOT / "logs" / "evaluation_logs.csv",
    )

    test_df = pd.read_csv(args.test_set)
    rows: list[dict[str, object]] = []
    for _, example in test_df.iterrows():
        result = orchestrator.run(str(example["message"]), log=False)
        retrieved_ids = [doc.item.id for doc in result.retrieved_documents]
        rows.append(
            {
                "message": example["message"],
                "true_category": example["true_category"],
                "pred_category": result.classification.category,
                "category_correct": result.classification.category == example["true_category"],
                "true_urgency": example["true_urgency"],
                "pred_urgency": result.classification.urgency,
                "urgency_correct": result.classification.urgency == example["true_urgency"],
                "expected_doc_id": example["expected_doc_id"],
                "retrieved_ids": ";".join(retrieved_ids),
                "retrieval_hit": example["expected_doc_id"] in retrieved_ids,
                "needs_human_review": result.check.needs_human_review,
            }
        )

    results = pd.DataFrame(rows)
    total = len(results)
    category_accuracy = results["category_correct"].mean() if total else 0.0
    urgency_accuracy = results["urgency_correct"].mean() if total else 0.0
    retrieval_hit_rate = results["retrieval_hit"].mean() if total else 0.0

    print("Evaluation results")
    print(f"Examples: {total}")
    print(f"Category accuracy: {category_accuracy:.1%}")
    print(f"Urgency accuracy: {urgency_accuracy:.1%}")
    print(f"Retrieval hit rate: {retrieval_hit_rate:.1%}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Wrote detailed results to {output_path}")


if __name__ == "__main__":
    main()
