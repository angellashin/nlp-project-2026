from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

from src.common.io import ensure_dir, read_jsonl, write_jsonl


def run_majority(variants_path: Path, out_dir: Path, majority_label: Optional[str] = None) -> Path:
    variants = read_jsonl(variants_path)
    if majority_label is None:
        majority_label = Counter(row["label"] for row in variants).most_common(1)[0][0]

    output = out_dir / "predictions.jsonl"
    rows = []
    for row in variants:
        rows.append(
            {
                "example_id": row["example_id"],
                "split": row["split"],
                "platform": row.get("platform"),
                "event": row.get("event"),
                "model": "majority_baseline",
                "condition": row["condition"],
                "context_source": row.get("context_source"),
                "target_id": row["target_id"],
                "thread_id": row.get("thread_id"),
                "source_id": row.get("source_id"),
                "parent_id": row.get("parent_id"),
                "depth": row.get("depth"),
                "depth_bucket": row.get("depth_bucket"),
                "parent_available": row.get("parent_available"),
                "mixed_valid": row.get("mixed_valid"),
                "has_conflicting_reply": row.get("has_conflicting_reply"),
                "conflict_relation": row.get("conflict_relation"),
                "irrelevant_relation": row.get("irrelevant_relation"),
                "context_item_count": row.get("context_item_count"),
                "uses_gold_labels_for_stress_test": row.get("uses_gold_labels_for_stress_test"),
                "gold_label": row["label"],
                "predicted_label": majority_label,
                "metric_label": majority_label,
                "raw_output": majority_label,
                "invalid": False,
                "prompt_version": "baseline_no_prompt",
            }
        )
    write_jsonl(output, rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create majority-label baseline predictions.")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--out-dir", default="results/runs/majority_baseline")
    parser.add_argument("--majority-label")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    output = run_majority(Path(args.variants), out_dir, args.majority_label)
    print(output)


if __name__ == "__main__":
    main()
