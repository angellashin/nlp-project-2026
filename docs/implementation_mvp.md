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

For C2 irrelevant context, the builder first uses same-thread non-path `comment` replies. If none exist, it falls back to same-event `comment` replies from a different thread and records `same_event_fallback_comment` in coverage metadata.

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

## 5. Run Qwen Smoke Test

```bash
python -m src.experiments.run_prompting \
  --variants data/variants/dev_context_variants.jsonl \
  --out-dir results/runs/dev_qwen05_smoke \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --limit 100 \
  --dtype float16 \
  --max-new-tokens 8
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
  --model Qwen/Qwen2.5-0.5B-Instruct

python -m src.experiments.run_prompting \
  --variants data/variants/dev_context_variants.jsonl \
  --out-dir results/runs/dev_qwen25_15b \
  --model Qwen/Qwen2.5-1.5B-Instruct
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
