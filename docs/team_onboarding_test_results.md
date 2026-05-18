# Team Onboarding: RumourEval Test Results

## 우리가 만든 것

RumourEval 2019 Task A stance detection을 위한 controlled stress-test pipeline을 만들었다. 예측 단위는 reply 하나이고, 각 reply를 `reply_only`, `useful`, `irrelevant`, `conflicting`, `mixed` 5개 context condition으로 변환해 같은 target에 대해 context만 바뀌었을 때 small LM의 예측이 어떻게 달라지는지 본다.

현재 MVP는 Qwen instruct model을 inference-only prompting으로 평가하는 형태다. 지금 이 폴더에 복사된 detailed test table은 1.5B 결과 중심이다. 노트북 로그상으로는 0.5B와 1.5B test run이 둘 다 완료됐지만, 최종 scale 비교를 하려면 `results/tables/test_qwen25_05b/`의 detailed CSV도 추가로 가져와야 한다.

## 실제로 돌린 것

- Test targets: `1746` replies x 5 conditions.
- Prompt version: copied notebook 기준 `qwen_mvp_v3`.
- Copied summary table 기준 invalid label rate는 0.0.
- Notebook evidence: prompt version: `qwen_mvp_v3`; test mode was enabled; `test_qwen25_05b` appears in notebook outputs; `test_qwen25_15b` appears in notebook outputs; invalid output rate was reported as 0

## 핵심 숫자

| condition | n | accuracy | macro-F1 all | macro-F1 S/D/Q | invalid |
| --- | --- | --- | --- | --- | --- |
| reply_only | 1746 | 0.110 | 0.131 | 0.158 | 0.000 |
| useful | 1746 | 0.134 | 0.150 | 0.169 | 0.000 |
| irrelevant | 1746 | 0.135 | 0.148 | 0.163 | 0.000 |
| conflicting | 1746 | 0.136 | 0.153 | 0.170 | 0.000 |
| mixed | 1746 | 0.133 | 0.147 | 0.163 | 0.000 |

## 결과 해석

`useful` context는 `reply_only`보다 좋다. all-label macro-F1은 0.131에서 0.150로 올라가고, support/deny/query만 본 macro-F1도 0.158에서 0.169로 올라간다.
하지만 aggregate table에서는 `conflicting`이 `useful`보다 나쁘지 않다: 0.153 vs 0.150. 따라서 최종 paper에서 'conflicting이 가장 해롭다' 같은 단순한 ranking claim은 하면 안 된다.
`mixed`는 전체로 보면 `useful`보다 약간 낮다 (0.147 vs 0.150). 다만 mixed는 depth와 parent availability 문제가 크기 때문에 `parent_available` 또는 `mixed_valid` subset에서만 강하게 해석해야 한다.
가장 큰 문제는 `comment` underprediction이다. Gold comment는 약 84.5%인데, predicted comment는 conflicting: 5.0%, irrelevant: 4.9%, mixed: 5.0%, reply_only: 2.4%, useful: 4.6%에 불과하다.

## 보고서에서 어떻게 말해야 하는가

말하면 안 되는 버전: 'conflicting context hurts small LMs the most.'

더 좋은 버전: small LM은 context에 민감하지만, zero-shot prompting 결과는 강한 label bias의 영향을 받는다. Useful context는 reply-only보다 도움이 되지만, noisy context의 효과는 depth와 fallback validity에 따라 달라진다. 이 프로젝트의 기여는 SOTA classifier가 아니라, context-error sensitivity를 validity-aware하게 분석한 controlled study다.

## 바로 해야 할 팀 작업

1. Jupyter에서 `results/tables/test_qwen25_05b/`를 복사해 와서 0.5B vs 1.5B scale comparison을 완성한다.
2. `paired_flip_cases.csv`에서 30-50개를 수동 검사한다. 특히 gold가 comment인데 query/deny로 흔들리는 케이스를 봐야 한다.
3. 보고서에는 aggregate result만 놓지 말고 same-thread/no-fallback/depth slice를 같이 보여준다.
4. `mixed` 결과는 전체 평균으로 강하게 주장하지 말고 `parent_available` 또는 `mixed_valid` subset 중심으로 해석한다.
5. majority baseline과 가능하면 작은 supervised baseline을 추가해, 'small LM prompting 자체의 약점'과 'context sensitivity'를 분리한다.
