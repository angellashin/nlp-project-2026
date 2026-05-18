from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from src.common.io import ensure_dir, read_jsonl, write_json


LABELS = ["support", "deny", "query", "comment"]
STANCE_LABELS = ["support", "deny", "query"]
USEFUL_CONDITION = "useful"
CONTEXT_GAP_PAIRS = [
    ("useful", "reply_only"),
    ("useful", "irrelevant"),
    ("useful", "conflicting"),
    ("useful", "mixed"),
    ("irrelevant", "conflicting"),
]
SLICE_KEYS = ["platform", "depth_bucket", "context_source", "parent_available"]
PAIR_SPECS = [
    ("reply_only_wrong_to_useful_correct", "reply_only", "useful"),
    ("useful_correct_to_conflicting_wrong", "useful", "conflicting"),
    ("useful_correct_to_mixed_wrong", "useful", "mixed"),
    ("useful_correct_to_irrelevant_wrong", "useful", "irrelevant"),
    ("conflicting_prediction_differs_from_useful", "useful", "conflicting"),
]


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def bool_key(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower()
    return "unknown"


def depth_bucket(row: dict[str, Any]) -> str:
    if row.get("depth_bucket"):
        return row["depth_bucket"]
    try:
        depth = int(row.get("depth"))
    except (TypeError, ValueError):
        return "unknown"
    return "depth_2plus" if depth >= 2 else "depth_1"


def metric_prediction(row: dict[str, Any]) -> str:
    return row.get("metric_label") or row.get("predicted_label") or "invalid"


def raw_prediction(row: dict[str, Any]) -> str:
    return row.get("predicted_label") or row.get("metric_label") or "invalid"


def is_correct(row: dict[str, Any]) -> bool:
    return row.get("gold_label") == metric_prediction(row)


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        copy = dict(row)
        copy["platform"] = copy.get("platform") or "unknown"
        copy["context_source"] = copy.get("context_source") or "unknown"
        copy["depth_bucket"] = depth_bucket(copy)
        copy["parent_available"] = bool_key(copy.get("parent_available"))
        copy["mixed_valid"] = bool_key(copy.get("mixed_valid"))
        normalized.append(copy)
    return normalized


def group_by_dimensions(
    rows: list[dict[str, Any]],
    dimensions: list[str],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(dimension) or "unknown" for dimension in dimensions)].append(row)
    return grouped


def confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows:
        gold = row["gold_label"]
        pred = metric_prediction(row)
        if gold in LABELS and pred in LABELS:
            matrix[gold][pred] += 1
    return matrix


def per_class_metrics(matrix: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }
    return metrics


def summarize_group(model: str, condition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = confusion(rows)
    class_metrics = per_class_metrics(matrix)
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in LABELS)
    invalid = sum(1 for row in rows if row.get("invalid") or row.get("predicted_label") == "invalid")
    return {
        "model": model,
        "condition": condition,
        "n": total,
        "accuracy": safe_div(correct, total),
        "macro_f1": sum(class_metrics[label]["f1"] for label in LABELS) / len(LABELS),
        "macro_f1_sdq": sum(class_metrics[label]["f1"] for label in STANCE_LABELS) / len(STANCE_LABELS),
        "invalid_rate": safe_div(invalid, len(rows)),
        "per_class": class_metrics,
        "confusion": matrix,
    }


def summary_to_rows(
    summary: dict[str, Any],
    extra_fields: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    extra_fields = extra_fields or {}
    summary_row = {
        **extra_fields,
        "model": summary["model"],
        "condition": summary["condition"],
        "n": summary["n"],
        "accuracy": round(summary["accuracy"], 6),
        "macro_f1": round(summary["macro_f1"], 6),
        "macro_f1_sdq": round(summary["macro_f1_sdq"], 6),
        "invalid_rate": round(summary["invalid_rate"], 6),
    }
    per_class_rows = [
        {
            **extra_fields,
            "model": summary["model"],
            "condition": summary["condition"],
            "label": label,
            "precision": round(metrics["precision"], 6),
            "recall": round(metrics["recall"], 6),
            "f1": round(metrics["f1"], 6),
            "support": int(metrics["support"]),
        }
        for label, metrics in summary["per_class"].items()
    ]
    return summary_row, per_class_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_confusion(path: Path, matrix: dict[str, dict[str, int]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + LABELS)
        for gold in LABELS:
            writer.writerow([gold] + [matrix[gold][pred] for pred in LABELS])


def summarize_rows(
    rows: list[dict[str, Any]],
    slice_keys: Optional[list[str]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slice_keys = slice_keys or []
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    dimensions = [*slice_keys, "model", "condition"]
    for key_values, group_rows in sorted(group_by_dimensions(rows, dimensions).items()):
        extra_fields = dict(zip(slice_keys, key_values[: len(slice_keys)]))
        model = key_values[-2]
        condition = key_values[-1]
        summary = summarize_group(model, condition, group_rows)
        summary_row, class_rows = summary_to_rows(summary, extra_fields)
        summary_rows.append(summary_row)
        per_class_rows.extend(class_rows)
    return summary_rows, per_class_rows


def context_gap_rows(summary_rows: list[dict[str, Any]], slice_keys: Optional[list[str]] = None) -> list[dict[str, Any]]:
    slice_keys = slice_keys or []
    by_key = {
        tuple(row[key] for key in [*slice_keys, "model", "condition"]): row for row in summary_rows
    }
    gap_rows: list[dict[str, Any]] = []
    slice_values = sorted({tuple(row[key] for key in slice_keys) for row in summary_rows}) or [()]
    for slice_value in slice_values:
        slice_prefix = dict(zip(slice_keys, slice_value))
        models = sorted(
            {
                row["model"]
                for row in summary_rows
                if tuple(row[key] for key in slice_keys) == slice_value
            }
        )
        for model in models:
            for from_condition, to_condition in CONTEXT_GAP_PAIRS:
                source = by_key.get((*slice_value, model, from_condition))
                target = by_key.get((*slice_value, model, to_condition))
                if not source or not target:
                    continue
                gap_rows.append(
                    {
                        **slice_prefix,
                        "model": model,
                        "from_condition": from_condition,
                        "to_condition": to_condition,
                        "macro_f1_drop": round(float(source["macro_f1"]) - float(target["macro_f1"]), 6),
                        "macro_f1_sdq_drop": round(
                            float(source.get("macro_f1_sdq", 0.0)) - float(target.get("macro_f1_sdq", 0.0)),
                            6,
                        ),
                        "accuracy_drop": round(float(source["accuracy"]) - float(target["accuracy"]), 6),
                    }
                )
    return gap_rows


def predicted_label_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for (model, condition), group_rows in sorted(group_by_dimensions(rows, ["model", "condition"]).items()):
        counts = Counter(raw_prediction(row) for row in group_rows)
        total = len(group_rows)
        for label in [*LABELS, "invalid"]:
            count = counts.get(label, 0)
            output_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "predicted_label": label,
                    "count": count,
                    "rate": round(safe_div(count, total), 6),
                    "n": total,
                }
            )
    return output_rows


def label_distribution_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for (model, condition), group_rows in sorted(group_by_dimensions(rows, ["model", "condition"]).items()):
        gold_counts = Counter(row.get("gold_label") for row in group_rows)
        prediction_counts = Counter(raw_prediction(row) for row in group_rows)
        total = len(group_rows)
        invalid_count = prediction_counts.get("invalid", 0)
        invalid_rate = safe_div(invalid_count, total)
        for label in LABELS:
            gold_count = gold_counts.get(label, 0)
            predicted_count = prediction_counts.get(label, 0)
            gold_rate = safe_div(gold_count, total)
            predicted_rate = safe_div(predicted_count, total)
            output_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "label": label,
                    "gold_count": gold_count,
                    "gold_rate": round(gold_rate, 6),
                    "predicted_count": predicted_count,
                    "predicted_rate": round(predicted_rate, 6),
                    "predicted_minus_gold_rate": round(predicted_rate - gold_rate, 6),
                    "invalid_count": invalid_count,
                    "invalid_rate": round(invalid_rate, 6),
                    "n": total,
                }
            )
    return output_rows


def target_ids_where(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> set[str]:
    return {row["target_id"] for row in rows if row.get("target_id") and predicate(row)}


def validity_subset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent_available_targets = target_ids_where(rows, lambda row: row.get("parent_available") == "true")
    mixed_valid_targets = target_ids_where(
        rows,
        lambda row: row["condition"] == "mixed" and row.get("mixed_valid") == "true",
    )
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("target_id"):
            by_target[row["target_id"]].append(row)
    no_fallback_targets = {
        target_id
        for target_id, target_rows in by_target.items()
        if all(
            row["condition"] == "reply_only" or row.get("context_source") == "same_thread"
            for row in target_rows
        )
    }
    subset_specs: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("all", lambda _row: True),
        (
            "same_thread_only",
            lambda row: row["condition"] == "reply_only" or row.get("context_source") == "same_thread",
        ),
        ("parent_available_only", lambda row: row.get("target_id") in parent_available_targets),
        ("mixed_valid_only", lambda row: row.get("target_id") in mixed_valid_targets),
        ("no_fallback_only", lambda row: row.get("target_id") in no_fallback_targets),
    ]
    output_rows: list[dict[str, Any]] = []
    for subset, predicate in subset_specs:
        for row in rows:
            if predicate(row):
                output = dict(row)
                output["validity_subset"] = subset
                output_rows.append(output)
    return output_rows


def paired_flip_outputs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        target_id = row.get("target_id")
        if not target_id:
            continue
        grouped[(row["model"], target_id)][row["condition"]] = row

    rate_counters: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"n": 0, "count": 0})
    case_rows: list[dict[str, Any]] = []
    for (model, target_id), condition_rows in sorted(grouped.items()):
        for event_name, base_condition, stress_condition in PAIR_SPECS:
            base = condition_rows.get(base_condition)
            stress = condition_rows.get(stress_condition)
            if not base or not stress:
                continue
            rate_counters[(model, event_name)]["n"] += 1
            base_pred = metric_prediction(base)
            stress_pred = metric_prediction(stress)
            base_correct = is_correct(base)
            stress_correct = is_correct(stress)
            if event_name == "reply_only_wrong_to_useful_correct":
                matched = (not base_correct) and stress_correct
            elif event_name == "conflicting_prediction_differs_from_useful":
                matched = base_pred != stress_pred
            else:
                matched = base_correct and not stress_correct
            if not matched:
                continue
            rate_counters[(model, event_name)]["count"] += 1
            case_rows.append(
                {
                    "model": model,
                    "target_id": target_id,
                    "event": event_name,
                    "gold_label": base.get("gold_label"),
                    "platform": base.get("platform"),
                    "depth": base.get("depth"),
                    "depth_bucket": base.get("depth_bucket"),
                    "parent_available": base.get("parent_available"),
                    "base_condition": base_condition,
                    "base_prediction": base_pred,
                    "base_correct": base_correct,
                    "stress_condition": stress_condition,
                    "stress_prediction": stress_pred,
                    "stress_correct": stress_correct,
                    "stress_context_source": stress.get("context_source"),
                    "stress_mixed_valid": stress.get("mixed_valid"),
                }
            )

    rate_rows = []
    for (model, event_name), counts in sorted(rate_counters.items()):
        rate_rows.append(
            {
                "model": model,
                "event": event_name,
                "count": counts["count"],
                "n": counts["n"],
                "rate": round(safe_div(counts["count"], counts["n"]), 6),
            }
        )
    return rate_rows, case_rows


def write_standard_outputs(
    out_dir: Path,
    rows: list[dict[str, Any]],
    prefix: str,
    slice_keys: list[str],
    payload: dict[str, Any],
) -> None:
    summary_rows, per_class_rows = summarize_rows(rows, slice_keys)
    gap_rows = context_gap_rows(summary_rows, slice_keys)

    summary_name = f"summary_by_{prefix}.csv" if prefix else "summary_metrics.csv"
    per_class_name = f"per_class_by_{prefix}.csv" if prefix else "per_class_metrics.csv"
    gap_name = f"context_gaps_by_{prefix}.csv" if prefix else "context_gaps.csv"
    field_prefix = slice_keys

    write_csv(
        out_dir / summary_name,
        summary_rows,
        [
            *field_prefix,
            "model",
            "condition",
            "n",
            "accuracy",
            "macro_f1",
            "macro_f1_sdq",
            "invalid_rate",
        ],
    )
    write_csv(
        out_dir / per_class_name,
        per_class_rows,
        [*field_prefix, "model", "condition", "label", "precision", "recall", "f1", "support"],
    )
    write_csv(
        out_dir / gap_name,
        gap_rows,
        [
            *field_prefix,
            "model",
            "from_condition",
            "to_condition",
            "macro_f1_drop",
            "macro_f1_sdq_drop",
            "accuracy_drop",
        ],
    )
    key = prefix or "overall"
    payload[key] = {"summary": summary_rows, "per_class": per_class_rows, "context_gaps": gap_rows}


def evaluate(predictions: Path, out_dir: Path) -> dict[str, Any]:
    rows = normalize_rows(read_jsonl(predictions))
    ensure_dir(out_dir)
    payload: dict[str, Any] = {}

    write_standard_outputs(out_dir, rows, "", [], payload)
    for slice_key in SLICE_KEYS:
        if any(row.get(slice_key) != "unknown" for row in rows):
            write_standard_outputs(out_dir, rows, slice_key, [slice_key], payload)

    validity_rows = validity_subset_rows(rows)
    write_standard_outputs(out_dir, validity_rows, "validity_subset", ["validity_subset"], payload)

    distribution_rows = predicted_label_distribution(rows)
    write_csv(
        out_dir / "predicted_label_distribution.csv",
        distribution_rows,
        ["model", "condition", "predicted_label", "count", "rate", "n"],
    )
    payload["predicted_label_distribution"] = distribution_rows

    label_bias_rows = label_distribution_comparison(rows)
    write_csv(
        out_dir / "label_distribution_comparison.csv",
        label_bias_rows,
        [
            "model",
            "condition",
            "label",
            "gold_count",
            "gold_rate",
            "predicted_count",
            "predicted_rate",
            "predicted_minus_gold_rate",
            "invalid_count",
            "invalid_rate",
            "n",
        ],
    )
    payload["label_distribution_comparison"] = label_bias_rows

    flip_rate_rows, flip_case_rows = paired_flip_outputs(rows)
    write_csv(out_dir / "paired_flip_rates.csv", flip_rate_rows, ["model", "event", "count", "n", "rate"])
    write_csv(
        out_dir / "paired_flip_cases.csv",
        flip_case_rows,
        [
            "model",
            "target_id",
            "event",
            "gold_label",
            "platform",
            "depth",
            "depth_bucket",
            "parent_available",
            "base_condition",
            "base_prediction",
            "base_correct",
            "stress_condition",
            "stress_prediction",
            "stress_correct",
            "stress_context_source",
            "stress_mixed_valid",
        ],
    )
    payload["paired_flip_rates"] = flip_rate_rows
    payload["paired_flip_cases"] = flip_case_rows

    for summary in payload["overall"]["summary"]:
        matrix = confusion(
            [
                row
                for row in rows
                if row.get("model") == summary["model"] and row.get("condition") == summary["condition"]
            ]
        )
        write_confusion(
            out_dir / f"confusion_{sanitize(summary['model'])}_{summary['condition']}.csv",
            matrix,
        )

    payload["summary"] = payload["overall"]["summary"]
    payload["per_class"] = payload["overall"]["per_class"]
    payload["context_gaps"] = payload["overall"]["context_gaps"]
    write_json(out_dir / "metrics.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RumourEval stance predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", default="results/tables")
    args = parser.parse_args()

    payload = evaluate(Path(args.predictions), ensure_dir(args.out_dir))
    for row in payload["summary"]:
        print(row)


if __name__ == "__main__":
    main()
