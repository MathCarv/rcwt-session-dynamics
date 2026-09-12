# R1: one fixed recorded episode

Derived illustration, not a new run. The exporter first verifies all 512 recorded decisions and the exact R1 report bytes offline.

Selection: **manifest index 0**, `rcwta-test-78c3b2ca-0` (updated-state); focused **step 5 (index 4)**. Neither choice searches scores.

All accounts, observations, money and tool effects are fictional. Private expected actions and grades shown below were not actor inputs.

## All eight decisions, both policies

Each cell is `decision / amount in cents / reason`; the mark is the recorded exact-reference grade.

| Step | Request / case | Summary | Structured |
| --- | --- | --- | --- |
| 1 | payout / inv-test-78c3b2ca-0-0 | approve / 14018 cents / authorized_payout — correct | approve / 14018 cents / authorized_payout — correct |
| 2 | payout / inv-test-78c3b2ca-0-1 | hold / 0 cents / funds_unsettled — correct | hold / 0 cents / funds_unsettled — correct |
| 3 | refund / inv-test-78c3b2ca-0-0 | hold / 0 cents / account_restricted — correct | hold / 0 cents / account_restricted — correct |
| 4 | refund / inv-test-78c3b2ca-0-3 | hold / 0 cents / return_rejected — correct | hold / 0 cents / return_rejected — correct |
| 5 | payout / inv-test-78c3b2ca-0-1 | ask_info / 0 cents / missing_evidence — incorrect: unnecessary_deferral | approve / 2040 cents / authorized_payout — correct |
| 6 | refund / inv-test-78c3b2ca-0-0 | ask_info / 0 cents / missing_evidence — incorrect: unnecessary_deferral | refund / 14018 cents / eligible_refund — correct |
| 7 | refund / inv-test-78c3b2ca-0-1 | hold / 0 cents / account_restricted — correct | hold / 0 cents / account_restricted — correct |
| 8 | refund / inv-test-78c3b2ca-0-0 | ask_info / 0 cents / missing_evidence — incorrect: unnecessary_deferral | hold / 0 cents / already_completed — correct |

## Selected-pair totals

Summary: **5/8 correct**; structured: **8/8 correct**.

Expected actions follow each arm's own prior simulated ledger. Later reference actions can therefore differ between policies; this is not grading against one common outcome.

## Fixed step 5

[step5.json](step5.json) contains the exact public observation/request, retained memory, actor memory/context, recorded final action, tool receipt, and private score for both arms. Text fields are copied without rewriting. It omits raw generation plans and request envelopes; those remain in the full trace.

- **summary**: ask_info / 0 cents / missing_evidence; expected approve / 2040 cents / authorized_payout. Source: `traces.jsonl`, line 5; trace-chain SHA-256 `236384994e318fba35f146cea278544a3bf158e0ea76a7296855a6da93538af0`.
- **structured**: approve / 2040 cents / authorized_payout; expected approve / 2040 cents / authorized_payout. Source: `traces.jsonl`, line 13; trace-chain SHA-256 `90d80e495da8aab7c587dc2b2a859072cc65ae5fa8944f2ae7c5824f9f5d4661`.

The large [original trace](../agent_v4_replication/traces.jsonl) is the authority; if the hosting UI cannot render it, download it and use the recorded physical line numbers. The trace-chain hash is not the hash of the rendered JSON excerpt.

## Whole-cohort evidence, not an inference from this pair

Across all 32 paired episodes: **152/256 → 238/256 correct**, delta **+33.59 percentage points**, paired-episode bootstrap 95% CI **[27.34, 39.45]**.

Observed unsafe attempts: **11 → 4**. Aggregate guards are descriptive, not a production-safety certificate; family-level regressions remain visible in the complete report.

Recorded generation calls: **1,248**. Total model tokens: **1,165,452 → 972,074**; median full-step time: **10.46 → 7.72 seconds**. Local timings depend on hardware/load; API charge is US$0, while total monetary cost is unknown.

See the [complete report](../agent_v4_replication/RESULTS.md), [machine-readable analysis](../agent_v4_replication/analysis.json), [full pair summaries](../agent_v4_replication/episodes.jsonl), and [protocol](../agent_v4_replication/protocol.json). Earlier development and the interrupted confirmation are not pooled into R1.

This same-generator, one-model, one-inference-seed comparison evaluates engineered memory plus public-context organization under a shared actor. It does not establish intrinsic model improvement, recursive self-improvement, unseen-domain transfer, or production readiness.

## Reproduce this display without inference

```sh
python tools/export_r1_demo.py --run-dir results/agent_v4_replication --output-dir results/agent_v4_replication_demo --verify
```

Omit `--verify` only to create a previously absent sibling directory. Existing artifacts are never overwritten. Verification rebuilds expected display bytes in memory and checks both files exactly; it does not rewrite the archive or the display.

Input protocol SHA-256: `5fdb1fcb9b81bd4fc073d9429c90e199a44688dfbdf9b2bdfedc74f1c9092524`. Full trace-file SHA-256: `5d54c01d2911ee42f0f69c47c05ba053d5796912726ff8277d92da8e3ce6b8d9`. The JSON includes the complete archive hash snapshot and exporter hash for provenance.
