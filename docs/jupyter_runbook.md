# Jupyter Runbook: RumourEval MVP

This runbook explains how teammates should run the first RumourEval context-error sensitivity experiment in JupyterLab without manually copying commands cell by cell.

## Open This File

Open:

```text
notebooks/rumoureval_mvp_runner.ipynb
```

In JupyterLab, choose **Run > Run All Cells** for the default smoke test.

## If Repo Root Detection Fails

If the first notebook cell raises:

```text
RuntimeError: Could not find repo root
```

it usually means the notebook was opened outside the cloned repo, or only the `.ipynb` file was uploaded to JupyterLab.

In a Jupyter terminal, run:

```bash
cd /home/ubuntu
git clone https://github.com/angellashin/nlp-project-2026.git
```

Then open this notebook from the cloned repo:

```text
/home/ubuntu/nlp-project-2026/notebooks/rumoureval_mvp_runner.ipynb
```

If the repo already exists somewhere else, edit the first code cell:

```python
MANUAL_PROJECT_ROOT = "/absolute/path/to/nlp-project-2026"
```

Then run all cells again.

## What The Default Run Does

The notebook default is intentionally conservative. It runs the complete data and evaluation setup, then performs a small Qwen smoke inference:

1. Finds the repo root automatically.
2. Checks Python and GPU visibility with `nvidia-smi`.
3. Installs missing dependencies from `requirements.txt` if needed.
4. Runs local unit tests.
5. Downloads RumourEval 2019 into `data/raw/rumoureval2019/`.
6. Parses reply-level target rows into `data/processed/`.
7. Builds C0-C4 context variants into `data/variants/`.
8. Runs the majority baseline on dev.
9. Runs `Qwen/Qwen2.5-0.5B-Instruct` on the first 100 dev variant rows.
10. Evaluates metrics and extracts error-analysis examples.

The smoke run corresponds to about 20 reply targets because each target has 5 context conditions.

## Default Settings

The main settings are in the notebook section **0. Run Settings**.

Important defaults:

```python
RUN_SMOKE_QWEN = True
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SMOKE_LIMIT = 100

RUN_FULL_DEV = False
RUN_TEST_ONCE = False

DTYPE = "float16"
MAX_NEW_TOKENS = 8
```

## After Smoke Test Passes

If the smoke run has a reasonable invalid label rate and runtime, change:

```python
RUN_FULL_DEV = True
```

Then run from **0. Run Settings** onward, or use **Run All Cells** again.

This runs full dev inference for:

```text
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
```

## Test Set Rule

Keep:

```python
RUN_TEST_ONCE = False
```

until prompt wording, context construction rules, and dev analysis are frozen.

Only after that, set:

```python
RUN_TEST_ONCE = True
```

and run the notebook again. Treat the test result as the final held-out evaluation.

## Important Output Paths

Parsed data:

```text
data/processed/reply_targets_{train,dev,test}.jsonl
data/processed/nodes_{train,dev,test}.jsonl
data/processed/dataset_summary.json
```

Context variants:

```text
data/variants/{split}_context_variants.jsonl
data/variants/{split}_coverage.json
data/variants/coverage_summary.json
```

Predictions:

```text
results/runs/{run_id}/predictions.jsonl
results/runs/{run_id}/run_metadata.json
```

Metrics:

```text
results/tables/{run_id}/summary_metrics.csv
results/tables/{run_id}/per_class_metrics.csv
results/tables/{run_id}/context_gaps.csv
results/tables/{run_id}/summary_by_platform.csv
results/tables/{run_id}/per_class_by_platform.csv
results/tables/{run_id}/context_gaps_by_platform.csv
results/tables/{run_id}/confusion_*.csv
```

Error analysis:

```text
results/tables/{run_id}_error_cases.csv
```

## Recommended Execution Order

For teammates:

1. Open `notebooks/rumoureval_mvp_runner.ipynb`.
2. Run all cells with the default settings.
3. Check `results/runs/dev_qwen25_05b_smoke/run_metadata.json`.
4. Check `results/tables/dev_qwen25_05b_smoke/summary_metrics.csv`.
5. If smoke is okay, set `RUN_FULL_DEV = True`.
6. Run again from **0. Run Settings** onward.
7. Share the generated `summary_metrics.csv`, `per_class_metrics.csv`, `context_gaps.csv`, platform-sliced metrics, and error cases with the team.

Do not commit `data/` or `results/` outputs unless the team explicitly decides to version a small derived artifact.
