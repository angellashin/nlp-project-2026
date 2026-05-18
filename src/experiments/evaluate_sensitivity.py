from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from src.common.io import ensure_dir, read_jsonl, write_json
from src.experiments.evaluate import (
    CONTEXT_GAP_PAIRS,
    LABELS,
    is_correct,
    normalize_rows,
    safe_div,
    validity_subset_rows,
)


SCORE_KEYS = ["mean_logprob", "sum_logprob"]
PAIR_METRICS = [
    "gold_score_drop",
    "gold_prob_drop",
    "margin_drop",
    "entropy_increase",
    "correct_drop",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def score_key_name(score_key: str) -> str:
    if score_key not in SCORE_KEYS:
        raise ValueError(f"Unsupported score key: {score_key}")
    return score_key


def get_label_score(row: dict[str, Any], label: str, score_key: str) -> float:
    scores = row.get("label_scores")
    if not isinstance(scores, dict) or label not in scores:
        raise ValueError("Prediction rows must include label_scores from run_label_scoring.py")
    return float(scores[label][score_key])


def score_softmax(row: dict[str, Any], score_key: str) -> dict[str, float]:
    probabilities = row.get("label_probabilities")
    if isinstance(probabilities, dict) and all(label in probabilities for label in LABELS):
        return {label: float(probabilities[label]) for label in LABELS}
    values = {label: get_label_score(row, label, score_key) for label in LABELS}
    max_value = max(values.values())
    exp_values = {label: math.exp(value - max_value) for label, value in values.items()}
    total = sum(exp_values.values())
    return {label: value / total for label, value in exp_values.items()}


def entropy(probabilities: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in probabilities.values() if value > 0)


def attach_score_metrics(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        gold = row["gold_label"]
        scores = {label: get_label_score(row, label, score_key) for label in LABELS}
        probabilities = score_softmax(row, score_key)
        other_scores = [score for label, score in scores.items() if label != gold]
        copy = dict(row)
        copy["gold_score"] = scores[gold]
        copy["gold_prob"] = probabilities[gold]
        copy["score_margin"] = scores[gold] - max(other_scores)
        copy["score_entropy"] = entropy(probabilities)
        copy["correct"] = is_correct(row)
        output.append(copy)
    return output


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: list[float], rounds: int, seed: int) -> tuple[float, float]:
    if len(values) <= 1 or rounds <= 0:
        value = mean(values)
        return value, value
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(sample))
    means.sort()
    low_index = int(0.025 * (rounds - 1))
    high_index = int(0.975 * (rounds - 1))
    return means[low_index], means[high_index]


def summarize_scores(rows: list[dict[str, Any]], slice_keys: Optional[list[str]] = None) -> list[dict[str, Any]]:
    slice_keys = slice_keys or []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(key, "unknown") for key in [*slice_keys, "model", "condition"])
        grouped[key].append(row)

    output = []
    for key, group_rows in sorted(grouped.items()):
        prefix = dict(zip(slice_keys, key[: len(slice_keys)]))
        output.append(
            {
                **prefix,
                "model": key[-2],
                "condition": key[-1],
                "n": len(group_rows),
                "accuracy": round(safe_div(sum(1 for row in group_rows if row["correct"]), len(group_rows)), 6),
                "mean_gold_score": round(mean([row["gold_score"] for row in group_rows]), 6),
                "mean_gold_prob": round(mean([row["gold_prob"] for row in group_rows]), 6),
                "mean_margin": round(mean([row["score_margin"] for row in group_rows]), 6),
                "mean_entropy": round(mean([row["score_entropy"] for row in group_rows]), 6),
            }
        )
    return output


def paired_gap_rows(
    rows: list[dict[str, Any]],
    slice_keys: Optional[list[str]] = None,
    bootstrap_rounds: int = 1000,
    seed: int = 461,
) -> list[dict[str, Any]]:
    slice_keys = slice_keys or []
    by_target: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = tuple(row.get(key, "unknown") for key in [*slice_keys, "model", "target_id"])
        by_target[key][row["condition"]] = row

    raw_pairs: dict[tuple[Any, ...], list[dict[str, float]]] = defaultdict(list)
    for key, condition_rows in by_target.items():
        prefix = key[: len(slice_keys)]
        model = key[len(slice_keys)]
        for from_condition, to_condition in CONTEXT_GAP_PAIRS:
            source = condition_rows.get(from_condition)
            target = condition_rows.get(to_condition)
            if not source or not target:
                continue
            pair_key = (*prefix, model, from_condition, to_condition)
            raw_pairs[pair_key].append(
                {
                    "gold_score_drop": source["gold_score"] - target["gold_score"],
                    "gold_prob_drop": source["gold_prob"] - target["gold_prob"],
                    "margin_drop": source["score_margin"] - target["score_margin"],
                    "entropy_increase": target["score_entropy"] - source["score_entropy"],
                    "correct_drop": float(source["correct"]) - float(target["correct"]),
                }
            )

    output = []
    for key, pair_values in sorted(raw_pairs.items()):
        prefix = dict(zip(slice_keys, key[: len(slice_keys)]))
        model = key[len(slice_keys)]
        from_condition = key[len(slice_keys) + 1]
        to_condition = key[len(slice_keys) + 2]
        row = {
            **prefix,
            "model": model,
            "from_condition": from_condition,
            "to_condition": to_condition,
            "n": len(pair_values),
        }
        for metric in PAIR_METRICS:
            values = [item[metric] for item in pair_values]
            ci_low, ci_high = bootstrap_ci(values, bootstrap_rounds, seed)
            row[f"mean_{metric}"] = round(mean(values), 6)
            row[f"{metric}_ci95_low"] = round(ci_low, 6)
            row[f"{metric}_ci95_high"] = round(ci_high, 6)
        output.append(row)
    return output


def evaluate_sensitivity(
    predictions: Path,
    out_dir: Path,
    score_key: str,
    bootstrap_rounds: int,
    seed: int,
) -> dict[str, Any]:
    rows = attach_score_metrics(normalize_rows(read_jsonl(predictions)), score_key_name(score_key))
    ensure_dir(out_dir)
    summary_rows = summarize_scores(rows)
    validity_rows = attach_score_metrics(validity_subset_rows(rows), score_key_name(score_key))
    summary_by_validity = summarize_scores(validity_rows, ["validity_subset"])
    gap_rows = paired_gap_rows(rows, bootstrap_rounds=bootstrap_rounds, seed=seed)
    gap_by_validity = paired_gap_rows(validity_rows, ["validity_subset"], bootstrap_rounds, seed)
    gap_by_gold = paired_gap_rows(rows, ["gold_label"], bootstrap_rounds, seed)

    summary_fields = [
        "model",
        "condition",
        "n",
        "accuracy",
        "mean_gold_score",
        "mean_gold_prob",
        "mean_margin",
        "mean_entropy",
    ]
    validity_summary_fields = ["validity_subset", *summary_fields]
    gap_fields = [
        "model",
        "from_condition",
        "to_condition",
        "n",
        *[
            field
            for metric in PAIR_METRICS
            for field in [
                f"mean_{metric}",
                f"{metric}_ci95_low",
                f"{metric}_ci95_high",
            ]
        ],
    ]
    write_csv(out_dir / "score_summary.csv", summary_rows, summary_fields)
    write_csv(out_dir / "score_summary_by_validity_subset.csv", summary_by_validity, validity_summary_fields)
    write_csv(out_dir / "score_gaps.csv", gap_rows, gap_fields)
    write_csv(out_dir / "score_gaps_by_validity_subset.csv", gap_by_validity, ["validity_subset", *gap_fields])
    write_csv(out_dir / "score_gaps_by_gold_label.csv", gap_by_gold, ["gold_label", *gap_fields])
    payload = {
        "score_key": score_key,
        "bootstrap_rounds": bootstrap_rounds,
        "seed": seed,
        "summary": summary_rows,
        "summary_by_validity_subset": summary_by_validity,
        "gaps": gap_rows,
        "gaps_by_validity_subset": gap_by_validity,
        "gaps_by_gold_label": gap_by_gold,
    }
    write_json(out_dir / "sensitivity_metrics.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate label-score sensitivity under context perturbations.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-key", choices=SCORE_KEYS, default="mean_logprob")
    parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=461)
    args = parser.parse_args()

    payload = evaluate_sensitivity(
        Path(args.predictions),
        ensure_dir(args.out_dir),
        args.score_key,
        args.bootstrap_rounds,
        args.seed,
    )
    for row in payload["summary"]:
        print(row)


if __name__ == "__main__":
    main()
