from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.io import ensure_dir, read_jsonl, write_json


LABELS = ["support", "deny", "query", "comment"]
USEFUL_CONDITION = "useful"


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def group_predictions(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model", "unknown"), row["condition"])].append(row)
    return grouped


def confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows:
        gold = row["gold_label"]
        pred = row.get("metric_label") or row.get("predicted_label")
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
        "invalid_rate": safe_div(invalid, len(rows)),
        "per_class": class_metrics,
        "confusion": matrix,
    }


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


def evaluate(predictions: Path, out_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(predictions)
    summaries = [
        summarize_group(model, condition, group_rows)
        for (model, condition), group_rows in sorted(group_predictions(rows).items())
    ]

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for summary in summaries:
        summary_rows.append(
            {
                "model": summary["model"],
                "condition": summary["condition"],
                "n": summary["n"],
                "accuracy": round(summary["accuracy"], 6),
                "macro_f1": round(summary["macro_f1"], 6),
                "invalid_rate": round(summary["invalid_rate"], 6),
            }
        )
        for label, metrics in summary["per_class"].items():
            per_class_rows.append(
                {
                    "model": summary["model"],
                    "condition": summary["condition"],
                    "label": label,
                    "precision": round(metrics["precision"], 6),
                    "recall": round(metrics["recall"], 6),
                    "f1": round(metrics["f1"], 6),
                    "support": int(metrics["support"]),
                }
            )
        write_confusion(
            out_dir / f"confusion_{sanitize(summary['model'])}_{summary['condition']}.csv",
            summary["confusion"],
        )

    by_model_condition = {
        (row["model"], row["condition"]): row for row in summary_rows
    }
    gap_rows: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in summary_rows}):
        useful = by_model_condition.get((model, USEFUL_CONDITION))
        if not useful:
            continue
        for condition in sorted({row["condition"] for row in summary_rows}):
            if condition == USEFUL_CONDITION:
                continue
            other = by_model_condition.get((model, condition))
            if not other:
                continue
            gap_rows.append(
                {
                    "model": model,
                    "from_condition": USEFUL_CONDITION,
                    "to_condition": condition,
                    "macro_f1_drop": round(float(useful["macro_f1"]) - float(other["macro_f1"]), 6),
                    "accuracy_drop": round(float(useful["accuracy"]) - float(other["accuracy"]), 6),
                }
            )

    write_csv(
        out_dir / "summary_metrics.csv",
        summary_rows,
        ["model", "condition", "n", "accuracy", "macro_f1", "invalid_rate"],
    )
    write_csv(
        out_dir / "per_class_metrics.csv",
        per_class_rows,
        ["model", "condition", "label", "precision", "recall", "f1", "support"],
    )
    write_csv(
        out_dir / "context_gaps.csv",
        gap_rows,
        ["model", "from_condition", "to_condition", "macro_f1_drop", "accuracy_drop"],
    )
    payload = {"summary": summary_rows, "per_class": per_class_rows, "context_gaps": gap_rows}
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

