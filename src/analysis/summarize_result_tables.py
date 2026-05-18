from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from src.common.io import ensure_dir


LABELS = ["support", "deny", "query", "comment"]
STANCE_LABELS = ["support", "deny", "query"]
CONDITIONS = ["reply_only", "useful", "irrelevant", "conflicting", "mixed"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_csv(tables_dir: Path, name: str) -> Optional[Path]:
    path = tables_dir / name
    if path.exists():
        return path
    stem = name.removesuffix(".csv")
    matches = sorted(tables_dir.glob(f"{stem}*.csv"))
    return matches[0] if matches else None


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value in (None, ""):
        return default
    return int(float(value))


def fmt(value: float) -> str:
    return f"{value:.3f}"


def fmt_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.3f}"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def index_by(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row.get(key, "") for key in keys): row for row in rows}


def per_class_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return index_by(rows, "condition", "label")


def macro_f1_sdq(condition: str, per_class: dict[tuple[str, str], dict[str, str]]) -> float:
    values = [as_float(per_class[(condition, label)], "f1") for label in STANCE_LABELS if (condition, label) in per_class]
    return sum(values) / len(values) if values else 0.0


def majority_baseline(label_rows: list[dict[str, str]]) -> dict[str, Any]:
    if not label_rows:
        return {}
    first_condition = label_rows[0]["condition"]
    gold_rows = [row for row in label_rows if row["condition"] == first_condition]
    total = as_int(gold_rows[0], "n") if gold_rows else 0
    gold_counts = {row["label"]: as_int(row, "gold_count") for row in gold_rows}
    label, count = max(gold_counts.items(), key=lambda item: item[1])
    precision = count / total if total else 0.0
    recall = 1.0 if total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "label": label,
        "accuracy": precision,
        "macro_f1": f1 / len(LABELS),
        "macro_f1_sdq": 0.0 if label == "comment" else f1 / len(STANCE_LABELS),
        "n": total,
    }


def notebook_run_evidence(notebook_path: Optional[Path]) -> list[str]:
    if notebook_path is None or not notebook_path.exists():
        return []
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    text = json.dumps(notebook, ensure_ascii=False)
    evidence = []
    prompt = re.search(r'PROMPT_VERSION\s*=\s*\\"([^\\"]+)\\"', text)
    if prompt:
        evidence.append(f"prompt version: `{prompt.group(1)}`")
    if "RUN_TEST_ONCE = True" in text or "Configured test run: True" in text:
        evidence.append("test mode was enabled")
    for run_id in ["test_qwen25_05b", "test_qwen25_15b"]:
        if run_id in text:
            evidence.append(f"`{run_id}` appears in notebook outputs")
    if "invalid=0" in text or "invalid_rate': 0.0" in text:
        evidence.append("invalid output rate was reported as 0")
    return evidence


def overall_result_table(summary_rows: list[dict[str, str]], per_class_rows: list[dict[str, str]]) -> str:
    per_class = per_class_index(per_class_rows)
    summary = index_by(summary_rows, "condition")
    rows = []
    for condition in CONDITIONS:
        if (condition,) not in summary:
            continue
        row = summary[(condition,)]
        rows.append(
            [
                condition,
                as_int(row, "n"),
                fmt(as_float(row, "accuracy")),
                fmt(as_float(row, "macro_f1")),
                fmt(as_float(row, "macro_f1_sdq", macro_f1_sdq(condition, per_class))),
                fmt(as_float(row, "invalid_rate")),
            ]
        )
    return md_table(["condition", "n", "accuracy", "macro-F1 all", "macro-F1 S/D/Q", "invalid"], rows)


def per_class_result_table(per_class_rows: list[dict[str, str]]) -> str:
    per_class = per_class_index(per_class_rows)
    rows = []
    for condition in CONDITIONS:
        if not any((condition, label) in per_class for label in LABELS):
            continue
        rows.append([condition] + [fmt(as_float(per_class[(condition, label)], "f1")) for label in LABELS])
    return md_table(["condition", *LABELS], rows)


def label_bias_table(label_rows: list[dict[str, str]]) -> str:
    rows = []
    for condition in CONDITIONS:
        condition_rows = {row["label"]: row for row in label_rows if row["condition"] == condition}
        if not condition_rows:
            continue
        comment = condition_rows["comment"]
        deny = condition_rows["deny"]
        query = condition_rows["query"]
        rows.append(
            [
                condition,
                fmt_pct(as_float(comment, "gold_rate")),
                fmt_pct(as_float(comment, "predicted_rate")),
                fmt_delta(as_float(comment, "predicted_minus_gold_rate")),
                fmt_pct(as_float(deny, "predicted_rate")),
                fmt_pct(as_float(query, "predicted_rate")),
            ]
        )
    return md_table(
        ["condition", "gold comment", "pred comment", "comment gap", "pred deny", "pred query"],
        rows,
    )


def slice_table(rows: list[dict[str, str]], slice_key: str) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row.get(slice_key, ""),
                row.get("condition", ""),
                as_int(row, "n"),
                fmt(as_float(row, "accuracy")),
                fmt(as_float(row, "macro_f1")),
            ]
        )
    return md_table([slice_key, "condition", "n", "accuracy", "macro-F1 all"], table_rows)


def paired_flip_summary(rows: list[dict[str, str]], n_targets: int) -> tuple[str, dict[str, Any]]:
    counts = Counter(row.get("event", "") for row in rows)
    gold_counts = Counter(row.get("gold_label", "") for row in rows)
    unique_targets = len({row.get("target_id", "") for row in rows if row.get("target_id")})
    table = md_table(
        ["event", "count", f"rate over {n_targets} targets"],
        [[event, count, fmt_pct(count / n_targets if n_targets else 0.0)] for event, count in sorted(counts.items())],
    )
    return table, {"event_counts": counts, "gold_counts": gold_counts, "unique_targets": unique_targets}


def render_analysis(tables_dir: Path, notebook_path: Optional[Path]) -> str:
    summary_rows = read_csv(tables_dir / "summary_metrics.csv")
    per_class_rows = read_csv(find_csv(tables_dir, "per_class_metrics.csv") or tables_dir / "per_class_metrics.csv")
    gap_rows = read_csv(tables_dir / "context_gaps.csv")
    label_rows = read_csv(tables_dir / "label_distribution_comparison.csv")
    depth_rows = read_csv(tables_dir / "summary_by_depth_bucket.csv")
    validity_rows = read_csv(tables_dir / "summary_by_validity_subset.csv")
    source_rows = read_csv(tables_dir / "summary_by_context_source.csv")
    flip_rows = read_csv(tables_dir / "paired_flip_cases.csv")
    model_names = sorted({row.get("model", "") for row in summary_rows if row.get("model")})
    model_text = ", ".join(f"`{model}`" for model in model_names) or "unknown"
    n_targets = as_int(summary_rows[0], "n") if summary_rows else 0
    per_class = per_class_index(per_class_rows)
    summary = index_by(summary_rows, "condition")
    majority = majority_baseline(label_rows)
    flip_table, flip_meta = paired_flip_summary(flip_rows, n_targets)
    evidence = notebook_run_evidence(notebook_path)

    useful = summary.get(("useful",), {})
    reply = summary.get(("reply_only",), {})
    conflicting = summary.get(("conflicting",), {})
    useful_sdq = macro_f1_sdq("useful", per_class)
    reply_sdq = macro_f1_sdq("reply_only", per_class)
    conflicting_sdq = macro_f1_sdq("conflicting", per_class)

    lines = [
        "# Test Result Analysis",
        "",
        "## Scope",
        "",
        f"- Result folder: `{tables_dir}`",
        f"- Detailed CSV model coverage in this folder: {model_text}",
        f"- Target count per condition: `{n_targets}`",
        "- These tables should be treated as final-test descriptive analysis. Do not tune prompts or context rules on these test numbers.",
    ]
    if evidence:
        lines.extend(["- Notebook evidence: " + "; ".join(evidence)])
    lines.extend(
        [
            "",
            "## Overall Metrics",
            "",
            overall_result_table(summary_rows, per_class_rows),
            "",
            "## Per-Class F1",
            "",
            per_class_result_table(per_class_rows),
            "",
            "## Label Distribution Bias",
            "",
            label_bias_table(label_rows),
            "",
            "## Context Source Slice",
            "",
            slice_table(source_rows, "context_source"),
            "",
            "## Depth Slice",
            "",
            slice_table(depth_rows, "depth_bucket"),
            "",
            "## Validity Subset Slice",
            "",
            slice_table(validity_rows, "validity_subset"),
            "",
            "## Paired Flip Summary",
            "",
            flip_table,
            "",
            f"- Unique targets appearing in flip cases: `{flip_meta['unique_targets']}`",
            f"- Flip-case gold labels: {dict(flip_meta['gold_counts'])}",
            "",
            "## Main Interpretation",
            "",
            (
                f"1. Useful context is better than reply-only, but the effect is modest: "
                f"macro-F1 all-labels {fmt(as_float(useful, 'macro_f1'))} vs "
                f"{fmt(as_float(reply, 'macro_f1'))}, and macro-F1 over support/deny/query "
                f"{fmt(useful_sdq)} vs {fmt(reply_sdq)}."
            ),
            (
                f"2. The original strong hypothesis that conflicting context is the most harmful is not supported by this aggregate test table. "
                f"Conflicting context is slightly above useful context on all-label macro-F1 "
                f"({fmt(as_float(conflicting, 'macro_f1'))} vs {fmt(as_float(useful, 'macro_f1'))}) "
                f"and nearly tied on support/deny/query macro-F1 ({fmt(conflicting_sdq)} vs {fmt(useful_sdq)})."
            ),
            (
                "3. The dominant failure mode is label calibration, not only context selection. "
                "Gold labels are mostly comment, but the model predicts deny/query for most examples and almost never predicts comment."
            ),
            (
                "4. Conversation structure matters. Depth-1 and depth-2plus behave differently; in depth-2plus, reply-only collapses and any context often acts as a scaffold, even when the context is not designed as useful."
            ),
            (
                "5. Fallback construction is a major validity issue. Same-event/cross-thread fallback rows behave very differently from same-thread rows, so paper claims should prioritize same-thread/no-fallback subsets."
            ),
            (
                "6. Paired flips show the model is locally sensitive to context, but many flips are wrong-to-wrong rather than useful-correct to stress-wrong. This is useful for error analysis, not enough by itself for a harm claim."
            ),
            "",
            "## Paper-Level Recommendation",
            "",
            "Reframe the project away from a simple ranking such as 'conflicting context hurts most.' A stronger and more defensible paper claim is:",
            "",
            "> In zero-shot small-LM RumourEval stance detection, context changes predictions, but the effect is mediated by label bias, reply depth, and fallback validity. Useful context helps compared with reply-only, while noisy context does not produce monotonic degradation under a strongly miscalibrated small LM.",
            "",
            "This keeps the context-error sensitivity question alive while honestly reporting that the first prompting setup is not a reliable standalone stance classifier.",
            "",
            "## Required Next Steps",
            "",
            "1. Copy the 0.5B test result tables into the same analysis folder so RQ3 can be evaluated on test, not only from notebook completion logs.",
            "2. Keep all future test analysis read-only. Any new prompt/model intervention should be developed on dev and reported as a separate exploratory follow-up, not used to rewrite the already-opened test result.",
            "3. Add macro-F1 over support/deny/query to future evaluator outputs alongside the existing all-label macro-F1.",
            "4. Add majority-label and simple lexical/classifier baselines in the report. The current 1.5B prompting result is below an always-comment baseline on all-label macro-F1, which must be acknowledged.",
            "5. For final claims, report aggregate plus same-thread/no-fallback plus depth slices. Do not make a strong mixed-condition claim without parent_available/mixed_valid filtering.",
        ]
    )
    if majority:
        lines.extend(
            [
                "",
                "## Majority Baseline Check",
                "",
                (
                    f"Always predicting `{majority['label']}` would get approximately "
                    f"accuracy {fmt(majority['accuracy'])} and all-label macro-F1 {fmt(majority['macro_f1'])} "
                    "under this label distribution. This is not a good stance model, but it is an essential sanity baseline because the test set is extremely imbalanced."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def render_onboarding(tables_dir: Path, notebook_path: Optional[Path]) -> str:
    summary_rows = read_csv(tables_dir / "summary_metrics.csv")
    per_class_rows = read_csv(find_csv(tables_dir, "per_class_metrics.csv") or tables_dir / "per_class_metrics.csv")
    label_rows = read_csv(tables_dir / "label_distribution_comparison.csv")
    per_class = per_class_index(per_class_rows)
    summary = index_by(summary_rows, "condition")
    evidence = notebook_run_evidence(notebook_path)
    n_targets = as_int(summary_rows[0], "n") if summary_rows else 0
    useful = summary.get(("useful",), {})
    reply = summary.get(("reply_only",), {})
    conflicting = summary.get(("conflicting",), {})
    mixed = summary.get(("mixed",), {})
    comment_pred_rates = {
        row["condition"]: as_float(row, "predicted_rate")
        for row in label_rows
        if row.get("label") == "comment"
    }
    lines = [
        "# Team Onboarding: RumourEval Test Results",
        "",
        "## 우리가 만든 것",
        "",
        "RumourEval 2019 Task A stance detection을 위한 controlled stress-test pipeline을 만들었다. 예측 단위는 reply 하나이고, 각 reply를 `reply_only`, `useful`, `irrelevant`, `conflicting`, `mixed` 5개 context condition으로 변환해 같은 target에 대해 context만 바뀌었을 때 small LM의 예측이 어떻게 달라지는지 본다.",
        "",
        "현재 MVP는 Qwen instruct model을 inference-only prompting으로 평가하는 형태다. 지금 이 폴더에 복사된 detailed test table은 1.5B 결과 중심이다. 노트북 로그상으로는 0.5B와 1.5B test run이 둘 다 완료됐지만, 최종 scale 비교를 하려면 `results/tables/test_qwen25_05b/`의 detailed CSV도 추가로 가져와야 한다.",
        "",
        "## 실제로 돌린 것",
        "",
        f"- Test targets: `{n_targets}` replies x 5 conditions.",
        "- Prompt version: copied notebook 기준 `qwen_mvp_v3`.",
        "- Copied summary table 기준 invalid label rate는 0.0.",
    ]
    if evidence:
        lines.append("- Notebook evidence: " + "; ".join(evidence))
    lines.extend(
        [
            "",
            "## 핵심 숫자",
            "",
            overall_result_table(summary_rows, per_class_rows),
            "",
            "## 결과 해석",
            "",
            (
                f"`useful` context는 `reply_only`보다 좋다. all-label macro-F1은 "
                f"{fmt(as_float(reply, 'macro_f1'))}에서 {fmt(as_float(useful, 'macro_f1'))}로 올라가고, "
                f"support/deny/query만 본 macro-F1도 {fmt(macro_f1_sdq('reply_only', per_class))}에서 "
                f"{fmt(macro_f1_sdq('useful', per_class))}로 올라간다."
            ),
            (
                f"하지만 aggregate table에서는 `conflicting`이 `useful`보다 나쁘지 않다: "
                f"{fmt(as_float(conflicting, 'macro_f1'))} vs {fmt(as_float(useful, 'macro_f1'))}. "
                "따라서 최종 paper에서 'conflicting이 가장 해롭다' 같은 단순한 ranking claim은 하면 안 된다."
            ),
            (
                f"`mixed`는 전체로 보면 `useful`보다 약간 낮다 "
                f"({fmt(as_float(mixed, 'macro_f1'))} vs {fmt(as_float(useful, 'macro_f1'))}). "
                "다만 mixed는 depth와 parent availability 문제가 크기 때문에 `parent_available` 또는 `mixed_valid` subset에서만 강하게 해석해야 한다."
            ),
            (
                "가장 큰 문제는 `comment` underprediction이다. Gold comment는 약 84.5%인데, predicted comment는 "
                + ", ".join(f"{cond}: {fmt_pct(rate)}" for cond, rate in sorted(comment_pred_rates.items()))
                + "에 불과하다."
            ),
            "",
            "## 보고서에서 어떻게 말해야 하는가",
            "",
            "말하면 안 되는 버전: 'conflicting context hurts small LMs the most.'",
            "",
            "더 좋은 버전: small LM은 context에 민감하지만, zero-shot prompting 결과는 강한 label bias의 영향을 받는다. Useful context는 reply-only보다 도움이 되지만, noisy context의 효과는 depth와 fallback validity에 따라 달라진다. 이 프로젝트의 기여는 SOTA classifier가 아니라, context-error sensitivity를 validity-aware하게 분석한 controlled study다.",
            "",
            "## 바로 해야 할 팀 작업",
            "",
            "1. Jupyter에서 `results/tables/test_qwen25_05b/`를 복사해 와서 0.5B vs 1.5B scale comparison을 완성한다.",
            "2. `paired_flip_cases.csv`에서 30-50개를 수동 검사한다. 특히 gold가 comment인데 query/deny로 흔들리는 케이스를 봐야 한다.",
            "3. 보고서에는 aggregate result만 놓지 말고 same-thread/no-fallback/depth slice를 같이 보여준다.",
            "4. `mixed` 결과는 전체 평균으로 강하게 주장하지 말고 `parent_available` 또는 `mixed_valid` subset 중심으로 해석한다.",
            "5. majority baseline과 가능하면 작은 supervised baseline을 추가해, 'small LM prompting 자체의 약점'과 'context sensitivity'를 분리한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize exported RumourEval result tables into report notes.")
    parser.add_argument("--tables-dir", required=True)
    parser.add_argument("--notebook", default=None)
    parser.add_argument("--analysis-out", default="docs/test_results_analysis.md")
    parser.add_argument("--onboarding-out", default="docs/team_onboarding_test_results.md")
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    notebook = Path(args.notebook) if args.notebook else None
    analysis_out = Path(args.analysis_out)
    onboarding_out = Path(args.onboarding_out)
    ensure_dir(analysis_out.parent)
    ensure_dir(onboarding_out.parent)
    analysis_out.write_text(render_analysis(tables_dir, notebook), encoding="utf-8")
    onboarding_out.write_text(render_onboarding(tables_dir, notebook), encoding="utf-8")
    print(f"wrote {analysis_out}")
    print(f"wrote {onboarding_out}")


if __name__ == "__main__":
    main()
