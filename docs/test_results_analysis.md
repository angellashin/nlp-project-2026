# Test Result Analysis

## Scope

- Result folder: `Test Results`
- Detailed CSV model coverage in this folder: `Qwen/Qwen2.5-1.5B-Instruct`
- Target count per condition: `1746`
- These tables should be treated as final-test descriptive analysis. Do not tune prompts or context rules on these test numbers.
- Notebook evidence: prompt version: `qwen_mvp_v3`; test mode was enabled; `test_qwen25_05b` appears in notebook outputs; `test_qwen25_15b` appears in notebook outputs; invalid output rate was reported as 0

## Overall Metrics

| condition | n | accuracy | macro-F1 all | macro-F1 S/D/Q | invalid |
| --- | --- | --- | --- | --- | --- |
| reply_only | 1746 | 0.110 | 0.131 | 0.158 | 0.000 |
| useful | 1746 | 0.134 | 0.150 | 0.169 | 0.000 |
| irrelevant | 1746 | 0.135 | 0.148 | 0.163 | 0.000 |
| conflicting | 1746 | 0.136 | 0.153 | 0.170 | 0.000 |
| mixed | 1746 | 0.133 | 0.147 | 0.163 | 0.000 |

## Per-Class F1

| condition | support | deny | query | comment |
| --- | --- | --- | --- | --- |
| reply_only | 0.146 | 0.150 | 0.177 | 0.049 |
| useful | 0.161 | 0.156 | 0.189 | 0.095 |
| irrelevant | 0.138 | 0.151 | 0.200 | 0.101 |
| conflicting | 0.178 | 0.151 | 0.179 | 0.104 |
| mixed | 0.144 | 0.148 | 0.195 | 0.100 |

## Label Distribution Bias

| condition | gold comment | pred comment | comment gap | pred deny | pred query |
| --- | --- | --- | --- | --- | --- |
| reply_only | 84.5% | 2.4% | -0.821 | 65.2% | 26.6% |
| useful | 84.5% | 4.6% | -0.800 | 63.9% | 24.7% |
| irrelevant | 84.5% | 4.9% | -0.797 | 64.9% | 25.4% |
| conflicting | 84.5% | 5.0% | -0.795 | 63.1% | 26.2% |
| mixed | 84.5% | 5.0% | -0.796 | 63.7% | 26.1% |

## Context Source Slice

| context_source | condition | n | accuracy | macro-F1 all |
| --- | --- | --- | --- | --- |
| cross_thread_fallback | conflicting | 27 | 0.000 | 0.000 |
| cross_thread_fallback | mixed | 28 | 0.000 | 0.000 |
| none | irrelevant | 179 | 0.212 | 0.224 |
| none | reply_only | 1746 | 0.110 | 0.131 |
| same_event_fallback | conflicting | 235 | 0.038 | 0.018 |
| same_event_fallback | irrelevant | 831 | 0.170 | 0.169 |
| same_event_fallback | mixed | 235 | 0.026 | 0.012 |
| same_thread | conflicting | 1484 | 0.154 | 0.169 |
| same_thread | irrelevant | 736 | 0.077 | 0.127 |
| same_thread | mixed | 1483 | 0.153 | 0.165 |
| same_thread | useful | 1746 | 0.134 | 0.150 |

## Depth Slice

| depth_bucket | condition | n | accuracy | macro-F1 all |
| --- | --- | --- | --- | --- |
| depth_1 | conflicting | 1200 | 0.164 | 0.177 |
| depth_1 | irrelevant | 1200 | 0.161 | 0.169 |
| depth_1 | mixed | 1200 | 0.163 | 0.174 |
| depth_1 | reply_only | 1200 | 0.147 | 0.169 |
| depth_1 | useful | 1200 | 0.167 | 0.181 |
| depth_2plus | conflicting | 546 | 0.073 | 0.124 |
| depth_2plus | irrelevant | 546 | 0.079 | 0.130 |
| depth_2plus | mixed | 546 | 0.068 | 0.092 |
| depth_2plus | reply_only | 546 | 0.029 | 0.024 |
| depth_2plus | useful | 546 | 0.062 | 0.077 |

## Validity Subset Slice

| validity_subset | condition | n | accuracy | macro-F1 all |
| --- | --- | --- | --- | --- |
| all | conflicting | 1746 | 0.136 | 0.153 |
| all | irrelevant | 1746 | 0.135 | 0.148 |
| all | mixed | 1746 | 0.133 | 0.147 |
| all | reply_only | 1746 | 0.110 | 0.131 |
| all | useful | 1746 | 0.134 | 0.150 |
| mixed_valid_only | conflicting | 546 | 0.073 | 0.124 |
| mixed_valid_only | irrelevant | 546 | 0.079 | 0.130 |
| mixed_valid_only | mixed | 546 | 0.068 | 0.092 |
| mixed_valid_only | reply_only | 546 | 0.029 | 0.024 |
| mixed_valid_only | useful | 546 | 0.062 | 0.077 |
| no_fallback_only | conflicting | 562 | 0.084 | 0.122 |
| no_fallback_only | irrelevant | 562 | 0.091 | 0.151 |
| no_fallback_only | mixed | 562 | 0.077 | 0.094 |
| no_fallback_only | reply_only | 562 | 0.048 | 0.071 |
| no_fallback_only | useful | 562 | 0.075 | 0.093 |
| parent_available_only | conflicting | 546 | 0.073 | 0.124 |
| parent_available_only | irrelevant | 546 | 0.079 | 0.130 |
| parent_available_only | mixed | 546 | 0.068 | 0.092 |
| parent_available_only | reply_only | 546 | 0.029 | 0.024 |
| parent_available_only | useful | 546 | 0.062 | 0.077 |
| same_thread_only | conflicting | 1484 | 0.154 | 0.169 |
| same_thread_only | irrelevant | 736 | 0.077 | 0.127 |
| same_thread_only | mixed | 1483 | 0.153 | 0.165 |
| same_thread_only | reply_only | 1746 | 0.110 | 0.131 |
| same_thread_only | useful | 1746 | 0.134 | 0.150 |

## Paired Flip Summary

| event | count | rate over 1746 targets |
| --- | --- | --- |
| conflicting_prediction_differs_from_useful | 286 | 16.4% |
| reply_only_wrong_to_useful_correct | 70 | 4.0% |
| useful_correct_to_conflicting_wrong | 29 | 1.7% |
| useful_correct_to_irrelevant_wrong | 35 | 2.0% |
| useful_correct_to_mixed_wrong | 29 | 1.7% |

- Unique targets appearing in flip cases: `345`
- Flip-case gold labels: {'comment': 365, 'support': 42, 'query': 21, 'deny': 21}

## Main Interpretation

1. Useful context is better than reply-only, but the effect is modest: macro-F1 all-labels 0.150 vs 0.131, and macro-F1 over support/deny/query 0.169 vs 0.158.
2. The original strong hypothesis that conflicting context is the most harmful is not supported by this aggregate test table. Conflicting context is slightly above useful context on all-label macro-F1 (0.153 vs 0.150) and nearly tied on support/deny/query macro-F1 (0.170 vs 0.169).
3. The dominant failure mode is label calibration, not only context selection. Gold labels are mostly comment, but the model predicts deny/query for most examples and almost never predicts comment.
4. Conversation structure matters. Depth-1 and depth-2plus behave differently; in depth-2plus, reply-only collapses and any context often acts as a scaffold, even when the context is not designed as useful.
5. Fallback construction is a major validity issue. Same-event/cross-thread fallback rows behave very differently from same-thread rows, so paper claims should prioritize same-thread/no-fallback subsets.
6. Paired flips show the model is locally sensitive to context, but many flips are wrong-to-wrong rather than useful-correct to stress-wrong. This is useful for error analysis, not enough by itself for a harm claim.

## Paper-Level Recommendation

Reframe the project away from a simple ranking such as 'conflicting context hurts most.' A stronger and more defensible paper claim is:

> In zero-shot small-LM RumourEval stance detection, context changes predictions, but the effect is mediated by label bias, reply depth, and fallback validity. Useful context helps compared with reply-only, while noisy context does not produce monotonic degradation under a strongly miscalibrated small LM.

This keeps the context-error sensitivity question alive while honestly reporting that the first prompting setup is not a reliable standalone stance classifier.

## Required Next Steps

1. Copy the 0.5B test result tables into the same analysis folder so RQ3 can be evaluated on test, not only from notebook completion logs.
2. Keep all future test analysis read-only. Any new prompt/model intervention should be developed on dev and reported as a separate exploratory follow-up, not used to rewrite the already-opened test result.
3. Add macro-F1 over support/deny/query to future evaluator outputs alongside the existing all-label macro-F1.
4. Add majority-label and simple lexical/classifier baselines in the report. The current 1.5B prompting result is below an always-comment baseline on all-label macro-F1, which must be acknowledged.
5. For final claims, report aggregate plus same-thread/no-fallback plus depth slices. Do not make a strong mixed-condition claim without parent_available/mixed_valid filtering.

## Majority Baseline Check

Always predicting `comment` would get approximately accuracy 0.845 and all-label macro-F1 0.229 under this label distribution. This is not a good stance model, but it is an essential sanity baseline because the test set is extremely imbalanced.
