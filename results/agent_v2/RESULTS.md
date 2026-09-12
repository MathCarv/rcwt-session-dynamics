# Local online-memory agent evaluation

Split: `test`. Paired episodes: 16.

Accuracy improvement against `summary`: **NOT DEMONSTRATED**.
Gate: lower endpoint of paired episode bootstrap 95% CI > 0 percentage points.

Frozen validation choice: `summary` (kind `summary`).
The summary baseline was retained: no memory-policy change was selected. The arm named `learned` is the unchanged baseline, not evidence of a learned improvement.

Offline evidence verification: **PASS**; verified steps: 512.
Verification scope: offline source/corpus binding, full scheduled cohorts, train-only proposal, validation selection, public actor/compressor inputs, raw-completion binding, cumulative chain, action replay and all summary counters.

| Policy | Episode-macro success | Entire episode success | Tokens / step | Calls | Decision p50 / p95 (s) | Step p50 / p95 (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned | 42.2% | 0.0% | 1808.2 | 240 | 4.496 / 5.327 | 6.982 / 9.488 |
| summary | 42.2% | 0.0% | 1808.2 | 240 | 4.481 / 5.325 | 7.019 / 9.024 |
| tail | 35.9% | 0.0% | 1142.1 | 128 | 4.526 / 5.356 | 4.548 / 5.488 |

Step latency includes memory compaction; decision latency is reported separately.

## Paired comparisons

- `learned` vs `summary`: +0.00 pp, 95% CI [+0.00, +0.00]; n=16 episodes; wins/ties/losses=0/16/0.
- `learned` vs `tail`: +6.25 pp, 95% CI [-2.34, +14.84]; n=16 episodes; wins/ties/losses=9/2/5.

Bootstrap unit: paired episode; the same sampled episode indices are used in both arms. Decisions are not treated as independent samples.

## By family

| Family | Policy | Episode-macro success | Entire episode success |
| --- | --- | ---: | ---: |
| evidence-chain | learned | 31.2% | 0.0% |
| evidence-chain | summary | 31.2% | 0.0% |
| evidence-chain | tail | 25.0% | 0.0% |
| insufficient-evidence | learned | 43.8% | 0.0% |
| insufficient-evidence | summary | 43.8% | 0.0% |
| insufficient-evidence | tail | 40.6% | 0.0% |
| topic-return | learned | 40.6% | 0.0% |
| topic-return | summary | 40.6% | 0.0% |
| topic-return | tail | 37.5% | 0.0% |
| updated-state | learned | 53.1% | 0.0% |
| updated-state | summary | 53.1% | 0.0% |
| updated-state | tail | 40.6% | 0.0% |

## Failure and resource accounting

- `learned`: failures invalid_action=4, unnecessary_deferral=26, unsafe_execution=12, wrong_case=2, wrong_decision=26, wrong_reason=4; valid actions 124/128; unsafe monetary-action attempts 12; unsafe amount actually booked 692950 fictional cents; prompt/completion tokens 199627/31819; memory truncations 5; episode wall time 902.139 s.
- `summary`: failures invalid_action=4, unnecessary_deferral=26, unsafe_execution=12, wrong_case=2, wrong_decision=26, wrong_reason=4; valid actions 124/128; unsafe monetary-action attempts 12; unsafe amount actually booked 692950 fictional cents; prompt/completion tokens 199627/31819; memory truncations 5; episode wall time 887.412 s.
- `tail`: failures invalid_action=8, unnecessary_deferral=20, unsafe_execution=22, wrong_case=2, wrong_decision=29, wrong_reason=1; valid actions 120/128; unsafe monetary-action attempts 22; unsafe amount actually booked 1154359 fictional cents; prompt/completion tokens 129707/16479; memory truncations 112; episode wall time 591.134 s.

Unsafe monetary-action attempts include rejected calls. Booked amounts are separate fictional ledger counters, not real transactions.

API charge: US$0 under the local-inference protocol. Total monetary cost: unknown, not zero.

## Training and policy-search overhead

This includes training episodes, the policy-proposal call, and validation of all candidates, including candidates not selected.

- Training/search: 241 model calls; 203559 prompt + 30341 completion tokens (233900 total); 802.488 s accounted wall time.
- Held-out evaluation, all arms: 608 model calls; 528961 prompt + 80117 completion tokens (609078 total); 2380.685 s accounted wall time.
- Combined training/search and evaluation: 849 model calls; 842978 tokens; 3183.172 s accounted wall time.

API charge remains US$0. Electricity, hardware depreciation, downloads/setup, and total monetary cost are not priced; these are not included in a claim of free execution.

## Limits

- Synthetic workflow tasks, one local quantized model, and no weight updates; not evidence of general recursive self-improvement.
- Confidence intervals cover episode sampling from this generator, not transfer to other domains, models, or stochastic inference runs.
- Latency depends on hardware, runtime, load, cache state, and execution order; it is not a universal model property.
- Zero API charge does not mean zero electricity, hardware, or total monetary cost; total monetary cost is unknown.
- Totals cover the supplied evaluation episodes; one-time training and policy-search costs must be reported separately.
- The accuracy gate does not certify latency, token-efficiency, safety, or cost improvements; family intervals are descriptive and not multiplicity-adjusted.

Normalized input SHA-256: `ae90a1dc0d4df754bb4836a46cfec0e6ca713df87e9335b712f1e1cd05d7521a`.
