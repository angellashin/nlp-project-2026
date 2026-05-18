# MVP Implementation Guide

This guide runs the first-pass RumourEval context-error sensitivity pipeline.

## 1. Install Runtime Dependencies

Use the GPU/JupyterLab environment for model inference.

```bash
pip install -r requirements.txt
```

The data parser, context builder, majority baseline, evaluator, and unit tests use only the standard library.

## 2. Download And Parse Data

```bash
python -m src.data.download_rumoureval --out-dir data/raw/rumoureval2019
python -m src.data.parse_rumoureval --raw-dir data/raw/rumoureval2019 --out-dir data/processed
```

Expected reply-level target counts:

- train: 4890
- dev: 1447
- test: 1746

## 3. Build Context Variants

```bash
python -m src.data.build_context_variants \
  --processed-dir data/processed \
  --out-dir data/variants \
  --splits train dev test \
  --seed 461
```

Outputs include `{split}_context_variants.jsonl`, `{split}_prompts_qwen.jsonl`, and coverage JSON files.

The rendered prompt intentionally does not expose condition names such as `useful`, `conflicting`, or `mixed`. As of `qwen_mvp_v3`, it also hides construction roles such as `parent`, `irrelevant_reply`, and `conflicting_reply` behind neutral prompt labels like `Conversation Post 1`. The metadata still preserves the real construction role for analysis.

The prompt defines all four labels, including `comment`, and treats `comment` as the conservative fallback label when the target reply does not explicitly support, deny, or query. It also warns that context may be helpful, irrelevant, or misleading without telling the model which condition it is seeing.

For C2 irrelevant context, the builder first uses same-thread non-path `comment` replies. If none exist, it falls back to same-event `comment` replies from a different thread and records `same_event_fallback_comment` in coverage metadata.

For C3/C4 conflicting context, the builder requires an actual different-label `conflicting_reply`. It first uses same-thread candidates, then same-event candidates, then a marked same-platform cross-thread fallback only when needed.

Each variant carries analysis metadata including `platform`, `depth_bucket`, `parent_available`, `context_source`, `mixed_valid`, relation notes, and context item counts. Use these fields for platform/depth/fallback robustness checks before making context-error claims.

## 4. Run Sanity Baseline

```bash
python -m src.experiments.run_majority_baseline \
  --variants data/variants/dev_context_variants.jsonl \
  --out-dir results/runs/dev_majority_baseline

python -m src.experiments.evaluate \
  --predictions results/runs/dev_majority_baseline/predictions.jsonl \
  --out-dir results/tables/dev_majority_baseline
```

This should show why accuracy is misleading under class imbalance.

The evaluator also writes:

- `predicted_label_distribution.csv`
- `label_distribution_comparison.csv`
- `summary_by_platform.csv`
- `summary_by_depth_bucket.csv`
- `summary_by_context_source.csv`
- `summary_by_parent_available.csv`
- `summary_by_validity_subset.csv`
- `paired_flip_rates.csv`
- `paired_flip_cases.csv`

`summary_*` and `context_gaps*` include both all-label macro-F1 and `macro_f1_sdq`, the macro-F1 over the informative stance labels `support`, `deny`, and `query`. Report both because RumourEval is heavily skewed toward `comment`.

`no_fallback_only` is a target-level validity subset: all non-`reply_only` conditions for that target must use same-thread context. This keeps context-gap comparisons from mixing different target sets across conditions.

## 5. Run Qwen Smoke Test

```bash
python -m src.experiments.run_prompting \
  --variants data/variants/dev_context_variants.jsonl \
  --out-dir results/runs/dev_qwen05_smoke \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --limit 100 \
  --dtype float16 \
  --max-new-tokens 8 \
  --prompt-version qwen_mvp_v3
```

Then evaluate:

```bash
python -m src.experiments.evaluate \
  --predictions results/runs/dev_qwen05_smoke/predictions.jsonl \
  --out-dir results/tables/dev_qwen05_smoke
```

## 6. Full Dev Matrix

Run both models on dev before touching test:

```bash
python -m src.experiments.run_prompting \
  --variants data/variants/dev_context_variants.jsonl \
  --out-dir results/runs/dev_qwen25_05b \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-version qwen_mvp_v3

python -m src.experiments.run_prompting \
  --variants data/variants/dev_context_variants.jsonl \
  --out-dir results/runs/dev_qwen25_15b \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt-version qwen_mvp_v3
```

Evaluate each run separately, or concatenate prediction files before evaluation if comparing both in one table.

## 7. Error Analysis

```bash
python -m src.analysis.error_analysis \
  --predictions results/runs/dev_qwen25_05b/predictions.jsonl \
  --variants data/variants/dev_context_variants.jsonl \
  --out results/tables/dev_qwen25_05b_error_cases.csv \
  --limit 50
```

The extracted cases are designed for manual inspection of prediction flips between useful and noisy context.

## 8. Summarize Exported Test Tables

When result CSVs are copied back from Jupyter, generate readable analysis notes:

```bash
python -m src.analysis.summarize_result_tables \
  --tables-dir "Test Results" \
  --notebook "Test Results/rumoureval_mvp_runner (6).ipynb" \
  --analysis-out docs/test_results_analysis.md \
  --onboarding-out docs/team_onboarding_test_results.md
```

This report is read-only with respect to test predictions. Use it to interpret the result tables, not to tune prompt/context rules after seeing test performance.
