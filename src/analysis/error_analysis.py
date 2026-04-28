from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.io import ensure_dir, read_jsonl


INTERESTING_PAIRS = [
    ("useful", "conflicting"),
    ("useful", "mixed"),
    ("reply_only", "useful"),
]


def load_variants(path: Path) -> dict[str, dict[str, Any]]:
    return {row["example_id"]: row for row in read_jsonl(path)}


def by_target(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["model"], row["target_id"])][row["condition"]] = row
    return grouped


def choose_cases(predictions: list[dict[str, Any]], variants: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for (model, target_id), condition_rows in sorted(by_target(predictions).items()):
        for base_condition, stress_condition in INTERESTING_PAIRS:
            base = condition_rows.get(base_condition)
            stress = condition_rows.get(stress_condition)
            if not base or not stress:
                continue
            gold = base["gold_label"]
            base_pred = base.get("metric_label") or base.get("predicted_label")
            stress_pred = stress.get("metric_label") or stress.get("predicted_label")
            if base_pred == stress_pred:
                continue
            if base_condition == "useful" and base_pred != gold:
                continue
            variant = variants.get(stress["example_id"], {})
            cases.append(
                {
                    "model": model,
                    "target_id": target_id,
                    "gold_label": gold,
                    "base_condition": base_condition,
                    "base_prediction": base_pred,
                    "stress_condition": stress_condition,
                    "stress_prediction": stress_pred,
                    "target_text": variant.get("target_text", ""),
                    "context_summary": " || ".join(
                        f"{item['role']}:{item['text']}" for item in variant.get("context_items", [])
                    ),
                    "construction_notes": ";".join(variant.get("construction_notes", [])),
                    "raw_base_output": base.get("raw_output", ""),
                    "raw_stress_output": stress.get("raw_output", ""),
                }
            )
            if len(cases) >= limit:
                return cases
    return cases


def write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "model",
        "target_id",
        "gold_label",
        "base_condition",
        "base_prediction",
        "stress_condition",
        "stress_prediction",
        "target_text",
        "context_summary",
        "construction_notes",
        "raw_base_output",
        "raw_stress_output",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(case)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paired prediction flips for manual analysis.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--out", default="results/tables/error_analysis_cases.csv")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    predictions = read_jsonl(args.predictions)
    variants = load_variants(Path(args.variants))
    cases = choose_cases(predictions, variants, args.limit)
    write_cases(Path(args.out), cases)
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()

