# V2 Experiment Plan: Context Reliance vs Robustness

## Research Pivot

The first zero-shot generation test showed that Qwen2.5-1.5B is context-sensitive, but the clean context-error ranking is obscured by severe `comment` underprediction. V2 should therefore create a stronger contribution:

> Small LMs show a context-reliance vs context-robustness tradeoff: models or decoding protocols that make better use of useful context can become more vulnerable to misleading or mixed context.

This reframes the project from a simple condition ranking into a controlled robustness study.

## V2 Contributions

1. **Validity-controlled challenge sets**
   - Build target-level subsets where all C0-C4 variants exist and where fallback artifacts are separated.
   - Validity claim subset: `strict`, requiring same-thread context, parent availability, valid mixed context, and actual conflicting replies.
   - Class-balance analysis subset: `complete_balanced`, which keeps all C0-C4 rows and balances target labels before filtering by stricter validity constraints.

2. **Forced-choice label scoring**
   - Replace free-form label generation with direct scoring of `support`, `deny`, `query`, and `comment`.
   - This reduces invalid-format concerns and gives probability/margin measurements.

3. **Context training intervention**
   - Fine-tune Qwen2.5-1.5B with LoRA under two regimes:
     - `LoRA-C0`: trained on reply-only examples.
     - `LoRA-C1`: trained on useful-context examples.
   - Evaluate both on C0-C4. The key question is whether `LoRA-C1` gains clean-context performance while losing robustness under conflicting/mixed context.

4. **Sensitivity metrics beyond F1**
   - Report F1 and accuracy, but also:
     - gold-label probability drop,
     - gold-label log-probability drop,
     - margin drop,
     - entropy increase,
     - paired flip rate.

## Main Experimental Matrix

| setup | train data | inference | eval variants |
| --- | --- | --- | --- |
| majority baseline | none | constant `comment` | C0-C4 |
| zero-shot generation | none | label generation | C0-C4 |
| zero-shot scoring | none | forced-choice scoring | C0-C4 |
| LoRA-C0 scoring | reply_only train | forced-choice scoring | C0-C4 |
| LoRA-C1 scoring | useful train | forced-choice scoring | C0-C4 |

Use Qwen2.5-1.5B as the main V2 model. Add 0.5B only after the 1.5B V2 loop works.

## Observed Challenge Set Sizes

After regenerating variants with the current builder:

| split | complete | complete_balanced | same_thread | parent_available | strict | strict_balanced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 1447 | 292 | 399 | 639 | 303 | 8 |
| test | 1746 | 264 | 562 | 546 | 436 | 12 |

Interpretation:

- `strict` is large enough for validity-controlled aggregate analysis, but it is highly comment-heavy.
- `strict_balanced` is too small for main quantitative claims.
- `complete_balanced` is the practical subset for support/deny/query analysis.

## Commands

Run these from the repo root in the JupyterLab terminal.

### 1. Build Challenge Sets

```bash
python -m src.data.build_challenge_sets \
  --variants data/variants/dev_context_variants.jsonl \
  --out-dir data/challenge \
  --split dev

python -m src.data.build_challenge_sets \
  --variants data/variants/test_context_variants.jsonl \
  --out-dir data/challenge \
  --split test
```

Start all new analysis on dev. Treat test challenge sets as final-evaluation material only after dev decisions are frozen.

### 2. Zero-Shot Forced-Choice Scoring

```bash
python -m src.experiments.run_label_scoring \
  --variants data/challenge/dev_complete_variants.jsonl \
  --out-dir results/runs/dev_qwen25_15b_scoring_complete \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --selection mean \
  --prompt-version qwen_mvp_v3

python -m src.experiments.evaluate \
  --predictions results/runs/dev_qwen25_15b_scoring_complete/predictions.jsonl \
  --out-dir results/tables/dev_qwen25_15b_scoring_complete

python -m src.experiments.evaluate_sensitivity \
  --predictions results/runs/dev_qwen25_15b_scoring_complete/predictions.jsonl \
  --out-dir results/tables/dev_qwen25_15b_scoring_complete_sensitivity
```

Repeat on:

- `data/challenge/dev_strict_variants.jsonl` for validity-controlled claims,
- `data/challenge/dev_complete_balanced_variants.jsonl` for class-balanced support/deny/query analysis.

### 3. LoRA-C0 and LoRA-C1 Training

```bash
python -m src.experiments.run_lora_finetune \
  --train-variants data/variants/train_context_variants.jsonl \
  --out-dir checkpoints/qwen25_15b_lora_c0 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --condition reply_only \
  --dtype float16 \
  --gradient-checkpointing

python -m src.experiments.run_lora_finetune \
  --train-variants data/variants/train_context_variants.jsonl \
  --out-dir checkpoints/qwen25_15b_lora_c1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --condition useful \
  --dtype float16 \
  --gradient-checkpointing
```

On one NVIDIA L4, this should be realistic for Qwen2.5-1.5B with LoRA. If memory becomes tight, lower `--batch-size 1`, raise `--gradient-accumulation-steps`, or set `--max-length 768`.

### 4. Score LoRA Models

```bash
python -m src.experiments.run_label_scoring \
  --variants data/challenge/dev_complete_variants.jsonl \
  --out-dir results/runs/dev_qwen25_15b_lora_c0_scoring \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter checkpoints/qwen25_15b_lora_c0 \
  --dtype float16 \
  --selection mean \
  --prompt-version qwen_mvp_v3

python -m src.experiments.run_label_scoring \
  --variants data/challenge/dev_complete_variants.jsonl \
  --out-dir results/runs/dev_qwen25_15b_lora_c1_scoring \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter checkpoints/qwen25_15b_lora_c1 \
  --dtype float16 \
  --selection mean \
  --prompt-version qwen_mvp_v3
```

Evaluate both with `src.experiments.evaluate` and `src.experiments.evaluate_sensitivity`.

## Claim Criteria

Strong V2 claim requires:

1. `LoRA-C1` improves useful-context performance over zero-shot or `LoRA-C0`.
2. `LoRA-C1` shows larger useful-to-conflicting or useful-to-mixed drops in F1, gold probability, or margin.
3. The pattern appears on aggregate dev and does not vanish on `same_thread`, `no_fallback`, or `strict` subsets.
4. Minority-class claims are checked on `complete_balanced`; strict subsets are too comment-heavy to support strong support/deny/query claims alone.
5. The same frozen protocol is then evaluated on test once.

## If Time Is Tight

Minimum meaningful V2:

1. Build challenge sets.
2. Run zero-shot forced-choice scoring on dev complete and strict.
3. Train only Qwen2.5-1.5B `LoRA-C1`.
4. Compare zero-shot scoring vs LoRA-C1 scoring on C0-C4.

This still tests whether making the model more context-competent increases vulnerability to noisy context.
