# What Context Errors Hurt Small LMs Most?

Working subtitle: A Controlled Study on RumourEval Stance Detection

Alternative title: Context Error Sensitivity of Small Language Models in RumourEval

Source-of-truth status: initial project understanding and planning document, created before implementation.

Last updated: 2026-04-28

## 1. Refined Project Framing

This project studies how small language models behave when the conversational context supplied for RumourEval stance detection is useful, irrelevant, misleading, or mixed. The goal is not to show that context matters in general. The goal is to build a controlled stress-test for context selection errors and quantify which errors hurt small LMs most.

The core task is RumourEval 2019 Task A stance detection. The model predicts whether a target reply expresses `support`, `deny`, `query`, or `comment` toward the source rumour. Veracity prediction is out of scope except as background motivation.

The intended contribution is an empirical analysis, not a new large model or infrastructure-heavy retrieval system:

- define implementable context-error conditions;
- evaluate small LMs under matched inputs;
- measure performance drops from useful to bad context;
- analyze whether support/deny are more fragile than comment;
- compare robustness between roughly 0.5B and 1.5B models.

## 2. Motivation

RumourEval stance detection is a good setting for studying context sensitivity because replies are embedded in tree-structured conversations. A reply can be hard to interpret without the source post or parent reply, but extra context can also distract or mislead.

The project is motivated by recent work showing that:

- RumourEval provides tree-structured Twitter/Reddit conversations annotated for stance and veracity.
- Rumour stance classification is highly imbalanced, and support/deny are especially important minority classes.
- Long or unrefined social context can make even large LMs struggle.
- Evidence pollution can degrade detectors and calibration.
- Retrieval-augmented stance/fact-checking systems show the value and risk of context selection.

The gap this project targets is narrower: small LMs may be especially vulnerable to context errors, but the harm from different context-error types is not well characterized in a simple controlled setup.

## 3. Dataset Choice and Scope

Primary dataset:

- RumourEval 2019 Task A, stance detection only.
- Labels: `support`, `deny`, `query`, `comment`.
- Platforms in official data: Twitter and Reddit.
- Conversation structure: source post plus replies in a tree.

Out of scope:

- RumourEval Task B veracity prediction.
- External news/Wikipedia evidence for veracity.
- Large-scale retrieval-augmented generation.
- Training large models from scratch.

### Inspected Official Data Format

The official Figshare item `RumourEval 2019 data` contains:

- `rumoureval2019.tar.bz2`
- inside it:
  - `README`
  - `LICENSE`
  - `home_scorer_macro.py`
  - `final-eval-key.json`
  - `rumoureval-2019-training-data.zip`
  - `rumoureval-2019-test-data.zip`

The training/development zip contains:

- `train-key.json`
- `dev-key.json`
- platform/thread folders such as:
  - `twitter-english/<event>/<thread_id>/`
  - `reddit-training-data/<thread_id>/`
  - `reddit-dev-data/<thread_id>/`
- within each thread:
  - `structure.json`
  - `source-tweet/<source_id>.json`
  - `replies/<reply_id>.json`
  - Reddit threads may also include `raw.json`.

The test zip contains:

- `twitter-en-test-data/<event>/<thread_id>/`
- within each thread:
  - `structure.json`
  - `source-tweet/<source_id>.json`
  - `replies/<reply_id>.json`

`structure.json` is a nested JSON tree. The root key is the source post ID; nested keys are reply IDs. Example shape:

```json
{
  "source_id": {
    "reply_a": {
      "reply_b": []
    },
    "reply_c": []
  }
}
```

Twitter text is usually in the `text` field. Reddit source posts use fields such as `title` and `selftext`; Reddit replies use `body`.

### Important Dataset Ambiguity

The official Task A keys include labels for both source posts and replies. However, this project's research question is explicitly reply-level context sensitivity. Therefore the recommended main experimental unit is:

- one labeled reply as the target;
- source post, parent reply, ancestors, siblings, or other replies as optional context;
- source posts are used as context, not as target examples.

For transparency, the report should state that source-post targets are excluded from the main analysis. A small optional appendix/sanity result may include official all-node evaluation, but the main claims should use reply targets only.

Inspected label counts:

- train key: 5,217 labeled nodes total; 4,890 labeled replies after excluding source posts.
- dev key: 1,485 labeled nodes total; 1,447 labeled replies after excluding source posts.
- test key: 1,827 labeled nodes total; 1,746 labeled replies after excluding source posts.

Reply-only label counts from inspected data:

| Split | support | deny | query | comment | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train replies | 642 | 367 | 373 | 3508 | 4890 |
| dev replies | 73 | 79 | 114 | 1181 | 1447 |
| test replies | 104 | 100 | 66 | 1476 | 1746 |

This confirms severe class imbalance. Accuracy alone is not acceptable.

## 4. Task Definition

Input for each example:

- target reply text;
- optional source post text;
- optional conversational context selected by a controlled rule.

Output:

- one of `support`, `deny`, `query`, `comment`.

Prediction unit:

- a labeled reply node in a conversation tree.

Primary evaluation:

- reply-only target set.

## 5. Research Questions

RQ1. Which context error types hurt small LMs the most in RumourEval stance detection?

RQ2. Are all stance classes affected equally, or are informative minority classes such as support and deny especially vulnerable?

RQ3. Does scaling from a smaller model around 0.5B to a slightly larger small model around 1.5B reduce context-error sensitivity?

RQ4. Optional only if feasible: does thinking mode versus non-thinking mode change robustness under noisy or conflicting context?

## 6. Hypotheses

H1. Useful context improves stance detection over reply-only input.

H2. Topic-relevant but stance-irrelevant context harms performance.

H3. Conflicting or misleading context harms performance more than merely irrelevant context.

H4. The harm from bad context is larger for support/deny than for comment.

H5. A 1.5B-class small LM is more robust than a 0.5B-class model, but still vulnerable.

## 7. Controlled Context Condition Taxonomy

All conditions should be generated from the same underlying target examples. This makes each condition a paired intervention on the same target reply.

### C0. Reply Only

Input contains only the target reply.

Purpose:

- minimal baseline;
- measures how much stance can be inferred from the reply alone.

Implementation:

- `target_reply_text` only.

### C1. Useful Context

Context should plausibly help determine the target reply's stance.

Practical first-pass rule:

- include source post;
- include direct parent reply if the target is nested;
- optionally include ancestor chain from source to target, capped by token budget.

Recommended initial version:

- source post + parent reply + target reply.

Rationale:

- source defines the rumour claim;
- parent often determines what a nested reply is responding to;
- this is simple, deterministic, and reproducible.

### C2. Topic-Relevant But Stance-Irrelevant Context

Context is from the same rumour/thread but should not directly reveal the target stance.

Practical first-pass rule:

- sample sibling or same-thread replies that are not parent/ancestor/child of target;
- prefer replies labeled `comment`;
- avoid same author if available;
- avoid replies within the target's direct path.

Recommended initial version:

- source post + 1 to 3 same-thread `comment` replies outside the target path + target reply.
- If no same-thread non-path comment is available, use a same-event `comment` reply from another thread as a marked fallback.

Sanity check:

- manually inspect 50 sampled contexts;
- verify they are same-topic but usually not directly explanatory for the target.
- report same-thread versus same-event fallback coverage.

### C3. Conflicting / Misleading Context

Context carries stance signals likely to pull the model away from the target label.

Practical first-pass rule:

- choose same-thread replies with labels different from the target;
- prioritize labels from the opposite/informative set:
  - if target is `support`, prefer `deny` or `query`;
  - if target is `deny`, prefer `support` or `query`;
  - if target is `query`, prefer `support` or `deny`;
  - if target is `comment`, prefer support/deny/query distractors, but analyze comment separately because it is the majority class.

Recommended initial version:

- source post + one conflicting same-thread reply + target reply.

Fallback if same-thread conflicting examples are unavailable:

- use same event/topic conflicting reply from a different thread;
- mark as `same_event_conflict` rather than `same_thread_conflict`.

Sanity check:

- report how often conflict context comes from same thread vs same event;
- manually inspect a stratified sample.

### C4. Mixed Context

Context contains useful context plus distractors.

Practical first-pass rule:

- source post + parent reply + one irrelevant same-thread reply or one conflicting reply + target reply.

Recommended initial version:

- source + parent + conflicting same-thread reply + target.

Purpose:

- tests whether useful context remains helpful when polluted.

### C5. Lexical Distractor Context (Optional)

Context contains words that look stance-relevant but are not useful.

Practical first-pass rule:

- construct short neutral distractor snippets with words such as "true", "fake", "confirmed", "source?", "hoax", "really?";
- or sample unrelated replies containing stance cue words from other threads.

Risk:

- synthetic distractors may be less natural and harder to justify.

Recommendation:

- treat as optional appendix only after C0-C4 are complete.

## 8. Preprocessing Pipeline Plan

Robust pipeline stages:

1. Download and unpack official RumourEval 2019 data.
2. Load `train-key.json`, `dev-key.json`, and `final-eval-key.json`.
3. Traverse thread directories and parse each `structure.json` into:
   - `thread_id`;
   - `source_id`;
   - `node_id`;
   - `parent_id`;
   - `children`;
   - `depth`;
   - `platform`;
   - `event` if present.
4. Load source and reply JSON text:
   - Twitter: `text`;
   - Reddit source: `title` plus optional `selftext`;
   - Reddit reply: `body`.
5. Join labels by `node_id`.
6. Exclude source-post nodes from main target set.
7. Create one row per target reply with:
   - target text;
   - label;
   - source text;
   - parent text;
   - ancestor texts;
   - sibling candidates;
   - same-thread non-path candidates;
   - same-event candidates;
   - context condition metadata.
8. Generate context variants C0-C4 using deterministic rules and fixed random seeds.
9. Save processed examples in JSONL/Parquet with explicit context IDs, not only rendered prompts.
10. Save rendered model inputs separately so experiments are reproducible.

Critical checks:

- no target label leakage in prompt text;
- no using test labels to construct context conditions except for controlled test-time analysis after finalizing rules;
- paired examples preserved across all context conditions;
- class distribution reported for each split and condition;
- source targets excluded consistently.

## 9. Model Scope

Primary small-LM candidates:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`

Reason:

- both are instruction-tuned, small enough for one NVIDIA L4, and have long context support relative to the task needs.

Optional thinking/non-thinking comparison:

- use a Qwen3 model only if the core Qwen2.5 experiments finish early.
- Qwen2.5-Instruct itself is not the right model family for a built-in thinking/non-thinking switch.

Recommended main setup:

- inference-only prompting first for both Qwen2.5 models;
- then, if time permits, LoRA fine-tuning on the same context-formatted examples.

Realistic on one NVIDIA L4:

- batched inference for 0.5B and 1.5B;
- LoRA fine-tuning for 0.5B and 1.5B with short prompts;
- multiple controlled context conditions;
- repeated experiments with fixed seeds for sampling context.

Probably unrealistic or unnecessary:

- full fine-tuning many variants;
- 7B+ models as a main comparison;
- training from scratch;
- complex graph/RAG infrastructure;
- large synthetic data generation.

## 10. Baselines and Ablation Priorities

Main ablation:

- C0 reply only;
- C1 useful context;
- C2 topic-relevant but stance-irrelevant context;
- C3 conflicting context;
- C4 mixed context.

Core secondary ablation:

- 0.5B vs 1.5B model.

Optional ablations:

- thinking vs non-thinking with Qwen3, if time permits;
- context length or top-k context count;
- parent-only vs source-only vs source+parent useful context.

Non-generative sanity baselines:

- majority class baseline;
- simple TF-IDF/logistic regression or small encoder classifier if implementation time allows.

These sanity baselines are useful but should not become the main story.

## 11. Evaluation Plan

Primary metrics:

- macro-F1;
- per-class F1;
- support F1;
- deny F1;
- confusion matrix;
- context sensitivity gap.

Context sensitivity gap:

```text
gap(useful -> condition X) = macro_F1(C1 useful) - macro_F1(CX)
```

Also compute per-class gaps:

```text
gap_class(c, useful -> conflict) = F1_c(C1 useful) - F1_c(C3 conflict)
```

Recommended tables:

- Table 1: model x context condition, macro-F1 and accuracy.
- Table 2: model x context condition, per-class F1.
- Table 3: useful-to-bad context drops by model.
- Table 4: support/deny subset analysis.

Recommended plots:

- bar plot of macro-F1 by context condition;
- line plot of context condition by model size;
- heatmap/confusion matrix per key condition;
- robustness gap plot from useful context to irrelevant/conflicting/mixed.

Optional:

- confidence/calibration analysis using normalized label probabilities;
- expected calibration error if label probabilities can be extracted reliably;
- depth-based analysis: direct replies vs nested replies.

## 12. Error Analysis Plan

Manual inspection should focus on paired examples where prediction changes across context conditions.

Inspect:

- useful context correct, conflicting context wrong;
- reply-only wrong, useful context correct;
- support/deny examples flipped to comment;
- deny confused as query;
- query confused as comment;
- cases where irrelevant context helps unexpectedly;
- cases where model copies a distractor stance.

For each inspected case, record:

- target label;
- target reply;
- source post;
- selected context;
- predictions under C0-C4;
- suspected failure type.

Suggested failure categories:

- distractor stance override;
- source claim misunderstanding;
- parent-child relation ignored;
- sarcasm/irony;
- short ambiguous reply;
- question treated as comment;
- model defaults to majority class;
- context too long or poorly ordered.

## 13. Realistic Simplifications

If the project becomes too large, simplify in this order:

1. Keep only inference-only prompting, no fine-tuning.
2. Keep only Qwen2.5-0.5B and Qwen2.5-1.5B.
3. Keep C0-C4 only; drop lexical distractor and thinking mode.
4. Use source+parent as the only useful context rule.
5. Use one deterministic context sample per condition per target.
6. Evaluate on dev first; use test once after rules and prompts are finalized.

Minimum viable experiment:

- reply-only target dataset;
- Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct;
- C0 reply only, C1 source+parent useful, C3 conflicting context, C4 mixed context;
- macro-F1, per-class F1, support/deny analysis, confusion matrices;
- 30 to 50 manually analyzed failure cases.

## 14. Expected Risks and Mitigations

Risk: context condition labels are heuristic and imperfect.

Mitigation:

- frame them as controlled stress-test conditions, not gold human annotations;
- manually inspect samples;
- report construction statistics.

Risk: official Task A includes source-post labels, while project focuses on replies.

Mitigation:

- exclude source targets from main analysis;
- document this clearly;
- optionally report an all-node sanity check separately.

Risk: class imbalance makes results look better than they are.

Mitigation:

- use macro-F1 and per-class F1;
- emphasize support/deny;
- include confusion matrices.

Risk: small LMs produce invalid labels in prompting mode.

Mitigation:

- use constrained label parsing;
- retry once with a stricter prompt if invalid;
- record invalid rate;
- optionally score by label log probabilities instead of free-form generation.

Risk: context variants may change prompt length and create confounds.

Mitigation:

- cap number of context items;
- record token length;
- include a length analysis or matched-length subset if needed.

Risk: same-thread conflicting context may be unavailable for minority examples.

Mitigation:

- fallback to same-event context;
- keep metadata distinguishing same-thread and same-event distractors;
- report coverage.

Risk: LoRA fine-tuning multiplies experiment count.

Mitigation:

- make LoRA secondary;
- prioritize inference-only full context-condition matrix first.

## 15. Recommended Project Roadmap

Week 1:

- finalize project document and context taxonomy;
- implement data parser and produce reply-only target table;
- compute dataset statistics;
- create deterministic context variants;
- manually inspect sample variants.

Week 2:

- implement prompting/inference pipeline;
- run majority and simple baseline;
- run Qwen2.5-0.5B on dev for C0-C4;
- fix invalid label handling and prompt format.

Week 3:

- run Qwen2.5-1.5B on dev for C0-C4;
- tune prompt format on dev only;
- create first result tables and confusion matrices.

Week 4:

- run locked setup on test;
- run support/deny and depth analyses;
- perform manual error analysis.

Weeks 5-6, if available:

- optional LoRA fine-tuning;
- optional context length/top-k ablation;
- optional Qwen3 thinking/non-thinking ablation;
- write report figures and final narrative.

## 16. Recommended Repo Structure

```text
data/
  raw/
  processed/
  variants/
docs/
  project_proposal.md
  experiment_log.md
src/
  data/
    download_rumoureval.py
    parse_rumoureval.py
    build_context_variants.py
  experiments/
    run_prompting.py
    run_lora.py
    evaluate.py
  analysis/
    metrics.py
    plots.py
    error_analysis.py
configs/
  data.yaml
  models.yaml
  experiments/
results/
  runs/
  tables/
  figures/
notebooks/
  dataset_inspection.ipynb
  result_analysis.ipynb
```

Design principles:

- generated data files include condition metadata;
- every experiment has a config file;
- every result includes model, prompt version, split, seed, and context condition;
- reports use saved tables/figures, not copied notebook outputs.

## 17. Reference Anchors

- Gorrell et al. 2019, RumourEval 2019 Task 7: defines stance and veracity tasks, using tree-structured Twitter/Reddit conversations.
- Scarton et al. 2020, Measuring What Counts: warns that RumourEval stance classification is highly imbalanced and that support/deny need special attention.
- Zeng et al. 2025, Exploring LLMs for Effective Rumor Detection: motivates refined context because long structured social context can hinder LLMs.
- Wan et al. 2025, Evidence Pollution: motivates controlled polluted/misleading context and calibration analysis.
- Zhu et al. 2025, RATSD: motivates retrieval/context for truthfulness stance detection while this project remains a simpler controlled stress test.

## 18. Sharpened Final Scope

The strongest version of this project under the current constraints is:

- a controlled context-error sensitivity study on reply-level RumourEval stance detection;
- inference-first, with optional LoRA only after core results;
- Qwen2.5-0.5B vs Qwen2.5-1.5B as the main model-scale comparison;
- C0-C4 context conditions as the main contribution;
- macro-F1, per-class F1, support/deny fragility, and paired error analysis as the main evidence.

This keeps the project research-shaped, compute-realistic, and interpretable for a 3-person team working for 1-2 months on one NVIDIA L4 GPU.
