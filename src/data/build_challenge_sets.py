from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from src.common.io import ensure_dir, read_jsonl, write_json, write_jsonl


CONDITIONS = ["reply_only", "useful", "irrelevant", "conflicting", "mixed"]
LABELS = ["support", "deny", "query", "comment"]
PRESETS = [
    "complete",
    "complete_balanced",
    "same_thread",
    "same_thread_balanced",
    "parent_available",
    "parent_available_balanced",
    "strict",
    "strict_balanced",
]


def bool_value(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def by_target(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["target_id"]][row["condition"]] = row
    return grouped


def has_required_conditions(condition_rows: dict[str, dict[str, Any]], conditions: list[str]) -> bool:
    return all(condition in condition_rows for condition in conditions)


def all_non_reply_same_thread(condition_rows: dict[str, dict[str, Any]], conditions: list[str]) -> bool:
    for condition in conditions:
        if condition == "reply_only":
            continue
        row = condition_rows.get(condition)
        if not row or row.get("context_source") != "same_thread":
            return False
    return True


def has_conflict_when_needed(condition_rows: dict[str, dict[str, Any]]) -> bool:
    for condition in ["conflicting", "mixed"]:
        row = condition_rows.get(condition)
        if not row or not bool_value(row.get("has_conflicting_reply")):
            return False
    return True


def parent_available(condition_rows: dict[str, dict[str, Any]]) -> bool:
    row = next(iter(condition_rows.values()))
    return bool_value(row.get("parent_available"))


def mixed_valid(condition_rows: dict[str, dict[str, Any]]) -> bool:
    row = condition_rows.get("mixed")
    return bool(row and bool_value(row.get("mixed_valid")))


def preset_predicate(preset: str, conditions: list[str]) -> Callable[[dict[str, dict[str, Any]]], bool]:
    base_preset = preset.removesuffix("_balanced")

    def complete(rows: dict[str, dict[str, Any]]) -> bool:
        return has_required_conditions(rows, conditions)

    def same_thread(rows: dict[str, dict[str, Any]]) -> bool:
        return complete(rows) and all_non_reply_same_thread(rows, conditions)

    def parent(rows: dict[str, dict[str, Any]]) -> bool:
        return complete(rows) and parent_available(rows)

    def strict(rows: dict[str, dict[str, Any]]) -> bool:
        return (
            complete(rows)
            and all_non_reply_same_thread(rows, conditions)
            and parent_available(rows)
            and mixed_valid(rows)
            and has_conflict_when_needed(rows)
        )

    predicates = {
        "complete": complete,
        "same_thread": same_thread,
        "parent_available": parent,
        "strict": strict,
    }
    if base_preset not in predicates:
        raise ValueError(f"Unknown preset: {preset}")
    return predicates[base_preset]


def flatten_condition_rows(
    grouped_rows: list[dict[str, dict[str, Any]]],
    conditions: list[str],
) -> list[dict[str, Any]]:
    output = []
    for condition_rows in grouped_rows:
        for condition in conditions:
            output.append(condition_rows[condition])
    return output


def balance_targets(
    grouped_rows: list[dict[str, dict[str, Any]]],
    max_per_label: Optional[int],
) -> list[dict[str, dict[str, Any]]]:
    by_label: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for condition_rows in grouped_rows:
        label = next(iter(condition_rows.values()))["label"]
        by_label[label].append(condition_rows)
    if max_per_label is None:
        positive_counts = [len(rows) for rows in by_label.values() if rows]
        max_per_label = min(positive_counts) if positive_counts else 0
    balanced = []
    for label in LABELS:
        balanced.extend(sorted(by_label.get(label, []), key=lambda rows: next(iter(rows.values()))["target_id"])[:max_per_label])
    return sorted(balanced, key=lambda rows: next(iter(rows.values()))["target_id"])


def summarize(rows: list[dict[str, Any]], preset: str) -> dict[str, Any]:
    target_ids = sorted({row["target_id"] for row in rows})
    summary: dict[str, Any] = {
        "preset": preset,
        "targets": len(target_ids),
        "variants": len(rows),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "label_counts": dict(Counter(row["label"] for row in rows if row["condition"] == "reply_only")),
        "context_source_counts": {},
        "depth_bucket_counts": dict(Counter(row.get("depth_bucket", "unknown") for row in rows if row["condition"] == "reply_only")),
        "platform_counts": dict(Counter(row.get("platform", "unknown") for row in rows if row["condition"] == "reply_only")),
    }
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        if condition_rows:
            summary["context_source_counts"][condition] = dict(
                Counter(row.get("context_source", "unknown") for row in condition_rows)
            )
    return summary


def build_challenge_sets(
    variants_path: Path,
    out_dir: Path,
    split: str,
    presets: list[str],
    conditions: list[str],
    max_per_label: Optional[int],
) -> dict[str, Any]:
    variants = read_jsonl(variants_path)
    grouped = by_target(variants)
    output_summary: dict[str, Any] = {
        "source_variants": str(variants_path),
        "split": split,
        "conditions": conditions,
        "presets": {},
    }
    ensure_dir(out_dir)
    for preset in presets:
        predicate = preset_predicate(preset, conditions)
        selected_targets = [
            condition_rows
            for _target_id, condition_rows in sorted(grouped.items())
            if predicate(condition_rows)
        ]
        if preset.endswith("_balanced"):
            selected_targets = balance_targets(selected_targets, max_per_label)
        rows = flatten_condition_rows(selected_targets, conditions)
        output_path = out_dir / f"{split}_{preset}_variants.jsonl"
        write_jsonl(output_path, rows)
        output_summary["presets"][preset] = {
            **summarize(rows, preset),
            "path": str(output_path),
        }
        print(f"{split}:{preset}: {len(selected_targets)} targets, {len(rows)} variants -> {output_path}")

    write_json(out_dir / f"{split}_challenge_summary.json", output_summary)
    return output_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validity-controlled RumourEval challenge subsets.")
    parser.add_argument("--variants", required=True, help="Input {split}_context_variants.jsonl")
    parser.add_argument("--out-dir", default="data/challenge")
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=PRESETS,
        default=[
            "complete",
            "complete_balanced",
            "same_thread",
            "parent_available",
            "strict",
            "strict_balanced",
        ],
    )
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--max-per-label", type=int, default=None)
    args = parser.parse_args()

    variants_path = Path(args.variants)
    split = args.split or variants_path.name.split("_", 1)[0]
    build_challenge_sets(
        variants_path,
        ensure_dir(args.out_dir),
        split,
        args.presets,
        args.conditions,
        args.max_per_label,
    )


if __name__ == "__main__":
    main()
