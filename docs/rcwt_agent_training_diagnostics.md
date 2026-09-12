# Aborted development run: training diagnostics

**Status: ABORTED DEVELOPMENT RUN.** These diagnostics describe the original development experiment, not the principal held-out evaluation. The coordinator stopped this run after 32 training decisions and 75 validation decisions; no held-out test decision was executed. Its evidence and source snapshot are retained, and its results must not be pooled with a subsequent, separately frozen experiment. This note does not inspect or analyze the validation decisions.

The generic rolling-summary baseline completed **6 of 32 decisions exactly correctly (18.75%)** in that development training cohort. The traces contain both loss of earlier evidence and failures to apply rules to evidence already present. These are different problems: the observed errors do not establish that changing memory alone will repair the agent. The original run used greedy, non-thinking inference and exhibited repetitive copying; development is evaluating the inference configuration before a new protocol is frozen. No improvement from that change is asserted here.

This note examines only the four original training episodes, their traces, the training failure counts and the resulting candidate policies. It does not use validation or test outcomes. Sources: [episode summaries](../results/agent_development/train/episodes.jsonl), [complete training traces](../results/agent_development/train/traces.jsonl), [training failure view](../results/agent_development/training-failures.json), and [candidate-policy record](../results/agent_development/candidate-policies.json). Step indices below are zero-based, matching the traces.

## What the labels count

| Recorded outcome | Decisions | Interpretation |
| --- | ---: | --- |
| `correct` | 6 | Exact case, decision, amount and reason |
| `unsafe_execution` | 18 | A monetary decision where the reference required deferral or a hold, including attempted duplicate operations |
| `invalid_action` | 4 | The strict action parser rejected the proposed arguments |
| `wrong_reason` | 2 | Decision and amount matched, but the reason did not |
| `wrong_amount` | 1 | The monetary decision matched, but the exact invoice amount did not |
| `unnecessary_deferral` | 1 | A valid operation was deferred |
| **Total** | **32** | **26 failures and 6 successes** |

These labels are mutually exclusive outcomes assigned by the scorer's precedence rules, not a causal taxonomy of memory defects. An `unsafe_execution` trace can also contain a wrong amount, but it is counted once under the earlier decision mismatch. The run recorded 28 valid action calls and 19 unsafe monetary attempts; the latter includes the wrong-amount attempt and excludes isolated wrong-reason failures. A rejected duplicate attempt is not an additional booking.

| Training family | Correct decisions | Failures |
| --- | ---: | --- |
| `updated-state` | 4/8 | 2 unsafe executions, 1 wrong amount, 1 wrong reason |
| `evidence-chain` | 0/8 | 6 unsafe executions, 1 unnecessary deferral, 1 wrong reason |
| `topic-return` | 2/8 | 2 unsafe executions, 4 invalid actions |
| `insufficient-evidence` | 0/8 | 8 unsafe executions |

The stored summary reached its cap and was truncated in **26 of 28 compaction opportunities**. This documents pressure on the 256-token memory budget; it does not by itself explain any particular error.

## Example 1: an earlier invoice amount disappears and another amount is reused

Episode `rcwta-train-135282f-0`, step **5**, targets `inv-train-135282f-0-1`. Its earlier invoice observation at step 0 contained `"total_cents": 110745`. The complete actor input at step 5 contains neither that amount nor a new invoice amount for the requested case. Its new observations restore the account to `active` and accept the return.

The retained memory instead begins with a previous action for a different invoice, `inv-train-135282f-0-2`, including `"amount_cents": 18555`. The actor outputs `"decision": "refund", "amount_cents": 18555` for the current case. The reference requires a refund of **110745 cents**. The simulator accepts the proposed fictional booking and the scorer reports `wrong_amount`. [Initial invoice](../results/agent_development/train/traces.jsonl#L1); [failing decision](../results/agent_development/train/traces.jsonl#L6).

The value **18555** originally belonged to a third invoice, `inv-train-135282f-0-3`, correctly refunded at step 3. It was already copied into an incorrect payout attempt at step 4; that attempt was rejected as `operation_already_booked`, yet its proposed amount remains prominent in the next memory. [Steps 3 and 4](../results/agent_development/train/traces.jsonl#L4).

**Observed:** the necessary historical amount is absent, another case's amount remains, and the subsequent action reuses that amount. **Inference:** loss of per-case evidence and retention of prior proposals are plausible contributors. This trace alone does not prove a causal improvement from a different compressor: no counterfactual actor call with restored evidence is evaluated here. Given its actual incomplete input, the actor also fails to abstain from inventing the missing amount.

## Example 2: the blocking fact is in the current observations

Episode `rcwta-train-135282f-0`, step **1**, receives all relevant records for a new invoice, including a payment response with `"status": "pending"`. Its amount is **113840 cents**. The supplied business rules require `hold`, zero cents, and `funds_unsettled`. The actor instead outputs `"decision": "approve", "amount_cents": 113840` with reason `authorized_payout`. [Trace](../results/agent_development/train/traces.jsonl#L2).

**Observed:** the blocking payment status is directly present in the current, uncompacted observations, but the actor approves. **Interpretation:** loss of that status from earlier memory cannot explain this decision. Rule application, evidence use or interference from other context are possible causes; this diagnostic does not distinguish among them. The action was accepted by the intentionally permissive fictional booking tool, demonstrating that tool acceptance is not a correctness verdict.

## Example 3: approval without required evidence, before any memory compaction

Episode `rcwta-train-135282f-1`, step **0**, starts with `"memory_before": ""`. It provides an invoice of **56788 cents** and an active, matching account, but no payment status. A support message also says `"no payment status update was supplied"`. The reference requires `ask_info` with `missing_evidence`. The actor issues an approval for **56788 cents**. [Trace](../results/agent_development/train/traces.jsonl#L9).

**Observed:** an unsafe approval occurs on the first decision, before an earlier summary could lose anything. **Interpretation:** this is an evidence-sufficiency failure under the supplied rules, not a demonstrated compaction failure. Memory-policy changes cannot recover a payment observation that was never provided.

## What was proposed from training

The training failure counts selected the existing **`evidence_dependencies`** template as one candidate. Its instruction asks the compressor to maintain exact per-case identifiers, amounts and dependencies, explicitly retain missing/revoked/unknown prerequisites, and distinguish decisions from evidence. In particular: “Do not infer missing proof from a prior action or a successful tool receipt.” This targets the distinction exposed by the first example. It is a heuristic template choice based on training counts, not a claim that validation selected it.

The local model also produced a separate **`repair_proposed`** instruction from the aggregated training taxonomy. It emphasizes dependency tracking, latest updates, uncertainty and confirmation of evidence before proceeding: “In cases of insufficient evidence, require explicit confirmation from the most recent observations before proceeding.” The recorded proposal used 395 prompt tokens and 181 completion tokens and ended with `finish_reason: stop`. [Proposal and full candidate instructions](../results/agent_development/candidate-policies.json).

The proposal received counts by family and failure class, not the examples analyzed in this note or an answer key. Some proposed language asks the compressor to validate or reject actions, although the compressor only writes retained memory. Its effect on the acting model therefore remains a hypothesis. Neither candidate changes the model weights, the actor's rules or the executable tool's validation. These candidates belong to the aborted development run; this record does not select the final experiment's policy. No gain is claimed from this training diagnostic.

## Limits

This is a descriptive inspection of **4 development training episodes and 32 decisions**, with one episode per family, one local model and one memory budget. Episodes are sequential: an early incorrect fictional booking changes later idempotency expectations, so later failures are not independent observations. The examples were selected after inspecting training failures and are not an exhaustive causal classification. They justify examining evidence retention and decision-rule compliance separately; they do not establish generalization, a memory-only cause, a successful repair or recursive self-improvement. The aborted cohort is development evidence only and must remain separate from the new principal experiment and its held-out results.
