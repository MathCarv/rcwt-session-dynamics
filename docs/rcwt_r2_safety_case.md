# R2: what the execution controls establish

R2 is a local, fictional financial-workflow experiment. It has no payment
integration, real customer data, production permissions or network interface
in its execution kernel. Its evidence can support bounded engineering claims;
it cannot certify production safety. The prospective experimental criteria
are in [the R2 protocol](rcwt_r2_protocol.md).

## Separate learning from permission

The model proposes an action. An independent executor consumes the current
host-provided observation delta and its own trusted record/booking store,
then either allows that exact proposal or rejects it with zero new monetary
effect. Rejection does not manufacture a correct replacement action. A private
evaluator grades the raw proposal outside the executor, using the actual
booking state before that step.

The learner may propose six integer retention weights. Those weights can
change only whole-component eviction under the existing memory token limit.
They cannot edit facts, tool schemas, financial rules, permissions, evaluator
references, test data or this executor. Its trusted history is explicitly
outside the actor's counted rolling memory; it is identical infrastructure
for every experimental arm, not a hidden retrieval advantage for the learner.

## Bounded claims and evidence

- **Authorized local effects only.** Monetary effects require the requested
  case and operation, a positive exact BRL invoice amount, an explicitly linked
  active account with matching owner, cleared payment, and an accepted return
  for refunds. An operation already in the actual booking ledger is denied.
- **No permission from model assertions.** Actor text, the actor's evidence
  check, retained memory and free-form messages are not authorization inputs.
  Only allowlisted structured records received through the assumed trusted
  host channel update the executor's record store.
- **Stale fields cannot survive an incomplete source update.** A source row
  replaces its previous row as a whole. Missing fields become unavailable,
  rather than inheriting an earlier favorable value.
- **Malformed trusted updates fail closed.** Invalid source envelopes,
  conflicting/repeated event identifiers and out-of-order/repeated requests
  raise before any state mutation. The orchestrator must abort the campaign,
  not catch the error and proceed with old state.
- **Denied proposals have no new financial effect.** A denied proposal is
  recorded unchanged, has no submitted replacement, and books zero cents.
  Independent quality and safety metrics prevent a guard's intervention from
  being counted as improved model reasoning.

The synthetic suite checks 5,760 combinations of account, owner, payment,
return, amount, operation and prior-booking conditions against both the
authorization predicate and the actual in-memory execution effects. Additional
tests cover malformed updates, duplicates, message spoofing, state isolation,
preservation of raw proposals and prohibited I/O. These are finite tests of
specified properties, not an exhaustive proof over arbitrary programs or data.

The zero-weight retention tests require exact agreement with the prior fixed
writer, including the sequence of tokenizer calls. Restricted-schema and
feedback tests reject extra fields, executable policy text, invalid numbers,
non-TRAIN labels and identity-bearing additions to the critic payload.
The orchestrator must additionally prove that supplied feedback came from
the sealed, complete TRAIN traces; a string saying `train` is not provenance.

## Assumptions and nonclaims

The host must genuinely control the trusted channel. A JSON field saying
`source: tool` is not authentication. This module does not authenticate an
external bank, validate a remote signature, implement tenant isolation or
establish whether a real backend record is truthful and current.

Execution is sequential and in-memory, for one isolated episode. Duplicate
protection here is not durable, distributed idempotency. Process crashes,
concurrent requests, database transactions, remote settlement failures and
rollback after real financial effects require additional implementation and
evidence. The finite message-spoof tests do not establish general resistance
to prompt injection, data exfiltration or compromised tools.

Zero observed unsafe effects in R2 means zero in the measured fictional
cohort under these assumptions. It does not mean zero risk in a deployment.
Likewise a valid changed retention policy proves that a bounded adaptation
step ran, not that it learned something useful. Held-out quality and feedback
control comparisons must independently establish that outcome.

## Route to an authorized production pilot

First bind the kernel to a separately authorized staging backend with explicit
tenant identities, authenticated/versioned records, least-privilege tool
credentials, durable transactional idempotency and an append-only audit store.
Keep credentials and customer data out of public experiment artifacts.

Next run a read-only shadow pilot on approved, minimized data. No proposed
action is executed. Register task boundaries and thresholds in advance, and
measure raw correctness, unsafe attempts, proposed rejections, false financial
blocks, p50/p95 latency, full adaptation overhead and drift. Preserve negative
outcomes and maintain a fixed-policy comparator.

Only a separate approval may enable a narrowly limited canary with human
approval for consequential actions, hard per-action and total exposure limits,
a tested kill switch, rollback to a pinned policy and incident procedures.
Do not let the learner change those limits or approve its own release. New
policies stay quarantined until independent validation passes. The R2 local
experiment does not grant that production approval.
