# V3 post-hoc evidence-check diagnosis

OFFLINE RECORDED DIAGNOSIS / NO MODEL CALLS / NO FEEDBACK

This describes reference-check and action agreement after a completed run. It is not an extra success endpoint, a repaired trajectory, or evidence that memory caused an error.

Only the final completion labelled `action:policy` is executed and graded. A `draft:policy` completion, when present, is shown solely as an unexecuted draft; there is no fallback to it.

Episodes: 4; split: `train`; arms: summary and structured.

| Policy | Decisions | Exact successes | Matching checks | Divergent checks | Invalid checks | Failures despite matching check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 32 | 17 | 11 | 21 | 0 | 2 |
| structured | 32 | 20 | 19 | 11 | 2 | 5 |

A matching check requires all six fields to match the reference, including actual prior bookings. Invalid envelopes are not treated as evidence of memory failure.

## Field mismatches

| Policy | Amount | Account | Ownership | Payment | Return | Already booked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 19 | 14 | 16 | 16 | 9 | 2 |
| structured | 5 | 1 | 4 | 3 | 2 | 1 |

## Action agreement with the actor's own check

- `summary`: 28 consistent and 4 inconsistent among 32 valid envelopes; 0 not evaluable.
- `structured`: 22 consistent and 8 inconsistent among 30 valid envelopes; 2 not evaluable.

## Paired task outcomes (descriptive)

Across 32 task pairs: structured correct / summary incorrect = 6; summary correct / structured incorrect = 3; both correct = 14; both incorrect = 9.

These counts are not independent samples. Prior bookings can differ by arm. The primary result remains the separate paired-episode analysis.

## Deterministic examples

First occurrence in public-manifest order, then step, then summary/structured: matching-check failure, divergent-check failure, invalid envelope, and each direction of paired outcome disagreement when present.

### Example 1: failed_matching_reference_check

[rcwta-train-78c3b2c1-0/summary/1](<../traces.jsonl#L10>) — `failed_matching_reference_check`.

- Trace line/hash: 10 / `752ef0d2b9ef99b59da133b7656c9f16d427486947aed0b04812af18276922d1`.
- Original failure category: `unsafe_execution`; field mismatches: none.
- Actual action: `{"arguments": {"amount_cents": 65489, "case_id": "inv-train-78c3b2c1-0-1", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Executed completion purpose: `action:summary`.

Stored memory before this action (persistent writer state):

```text
Case inv-train-78c3b2c1-0-0: $1408.28 (BRL) approved via record_decision; account acct-train-78c3b2c1-0-0 active; invoice cleared; support queue review pending address correction with shipping team.
```

Actual memory view supplied to the actor:

```text
Case inv-train-78c3b2c1-0-0: $1408.28 (BRL) approved via record_decision; account acct-train-78c3b2c1-0-0 active; invoice cleared; support queue review pending address correction with shipping team.
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `67`.

Draft model output (NOT EXECUTED):

```text
{
"evidence_check": {
"invoice_amount_cents": 65489,
"account_status": "active",
"ownership_match": "yes",
"payment_status": "pending",
"return_status": "accepted",
"operation_already_booked": "no_record"
},
"tool": "record_decision",
"arguments": {
"case_id": "inv-train-78c3b2c1-0-1",
"decision": "approve",
"amount_cents": 65489,
"reason_code": "authorized_payout"
}
}
```

Draft finish reason: `stop`; proposed action, not executed: `"{\"arguments\":{\"amount_cents\":65489,\"case_id\":\"inv-train-78c3b2c1-0-1\",\"decision\":\"approve\",\"reason_code\":\"authorized_payout\"},\"tool\":\"record_decision\"}"`.

### Example 2: failed_divergent_check

[rcwta-train-78c3b2c1-0/summary/4](<../traces.jsonl#L13>) — `failed_divergent_check`.

- Trace line/hash: 13 / `8b75c9ea93d80935cab89c7f6b4569d3351326f4565fe8e5ef33bb42d40b0901`.
- Original failure category: `wrong_decision`; field mismatches: invoice_amount_cents: claimed=null, reference=65489; account_status: claimed="unknown", reference="active"; ownership_match: claimed="unknown", reference="yes"; return_status: claimed="unknown", reference="accepted"; operation_already_booked: claimed="no_record", reference="yes".
- Actual action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "already_completed"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Executed completion purpose: `action:summary`.

Stored memory before this action (persistent writer state):

```text
{"case_id":"inv-train-78c3b2c1-0-3","amount_cents":48916,"decision":"refund","reason_code":"eligible_refund","account_id":"acct-train-78c3b2c1-0-3","verification_status":"active","currency":"BRL","status":"cleared","notes":["Warehouse sent opening-hours update; document contains no return inspection"]}
```

Actual memory view supplied to the actor:

```text
{"case_id":"inv-train-78c3b2c1-0-3","amount_cents":48916,"decision":"refund","reason_code":"eligible_refund","account_id":"acct-train-78c3b2c1-0-3","verification_status":"active","currency":"BRL","status":"cleared","notes":["Warehouse sent opening-hours update; document contains no return inspection"]}
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `91`.

Draft model output (NOT EXECUTED):

```text
{
  "evidence_check": {
    "invoice_amount_cents": null,
    "account_status": "unknown",
    "ownership_match": "unknown",
    "payment_status": "cleared",
    "return_status": "unknown",
    "operation_already_booked": "no_record"
  },
  "tool": "record_decision",
  "arguments": {
    "case_id": "inv-train-78c3b2c1-0-1",
    "decision": "ask_info",
    "amount_cents": 0,
    "reason_code": "missing_evidence"
  }
}
```

Draft finish reason: `stop`; proposed action, not executed: `"{\"arguments\":{\"amount_cents\":0,\"case_id\":\"inv-train-78c3b2c1-0-1\",\"decision\":\"ask_info\",\"reason_code\":\"missing_evidence\"},\"tool\":\"record_decision\"}"`.

### Example 3: invalid_envelope

[rcwta-train-78c3b2c1-1/structured/2](<../traces.jsonl#L19>) — `invalid_envelope`.

- Trace line/hash: 19 / `72be650e206c1b06bf6eaf8152964adde6da873add97e8d3e135fb1f50c80b68`.
- Original failure category: `invalid_action`; field mismatches: none.
- Actual action: `"INVALID_ACTOR_OUTPUT"`.
- Reference action: `{"arguments": {"amount_cents": 27953, "case_id": "inv-train-78c3b2c1-1-3", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `null`.
- Executed completion purpose: `action:structured`.

Stored memory before this action (persistent writer state):

```text
{"replace_in_ids":{"~":"-train-78c3b2c1-1-"},"records":[{"invoice":"inv~2","cents":8398,"currency":"BRL","account":"acct~2","merchant":"merchant~2","order":"ord~2","account_status":"active","holder":"merchant~2","ownership_match":"yes","payment":"cleared","booked_payout":8398},{"invoice":"inv~3","cents":27953,"currency":"BRL","account":"acct~3","merchant":"merchant~3","order":"ord~3","ownership_match":"unknown"}]}
```

Actual memory view supplied to the actor:

```text
{"case_id":"inv-train-78c3b2c1-1-3","operation":"payout","retained_evidence":{"invoice_amount_cents":27953,"return_status":"unknown","operation_already_booked":"no_record"},"retained_ids":{"account_id":"acct-train-78c3b2c1-1-3","merchant_id":"merchant-train-78c3b2c1-1-3","order_id":"ord-train-78c3b2c1-1-3"},"invalidated_fields":["account_status","ownership_match","payment_status","holder_merchant_id"]}
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `137`.

Draft model output (NOT EXECUTED):

```text
{
  "evidence_check": {
    "invoice_amount_cents": 27953,
    "account_status": "active",
    "ownership_match": "yes",
    "payment_status": "cleared",
    "return_status": "unknown",
    "operation_already_booked": "no_record"
  },
  "tool": "record_decision",
  "arguments": {
    "case_id": "inv-train-78c3b2c1-1-3",
    "decision": "approve",
    "amount_cents": 27953,
    "reason_code": "authorized_payout"
  }
}
```

Draft finish reason: `stop`; proposed action, not executed: `"{\"arguments\":{\"amount_cents\":27953,\"case_id\":\"inv-train-78c3b2c1-1-3\",\"decision\":\"approve\",\"reason_code\":\"authorized_payout\"},\"tool\":\"record_decision\"}"`.

### Example 4: paired_structured_correct_summary_incorrect

[rcwta-train-78c3b2c1-0/summary/1](<../traces.jsonl#L10>) — `failed_matching_reference_check`.

- Trace line/hash: 10 / `752ef0d2b9ef99b59da133b7656c9f16d427486947aed0b04812af18276922d1`.
- Original failure category: `unsafe_execution`; field mismatches: none.
- Actual action: `{"arguments": {"amount_cents": 65489, "case_id": "inv-train-78c3b2c1-0-1", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Executed completion purpose: `action:summary`.

Stored memory before this action (persistent writer state):

```text
Case inv-train-78c3b2c1-0-0: $1408.28 (BRL) approved via record_decision; account acct-train-78c3b2c1-0-0 active; invoice cleared; support queue review pending address correction with shipping team.
```

Actual memory view supplied to the actor:

```text
Case inv-train-78c3b2c1-0-0: $1408.28 (BRL) approved via record_decision; account acct-train-78c3b2c1-0-0 active; invoice cleared; support queue review pending address correction with shipping team.
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `67`.

Draft model output (NOT EXECUTED):

```text
{
"evidence_check": {
"invoice_amount_cents": 65489,
"account_status": "active",
"ownership_match": "yes",
"payment_status": "pending",
"return_status": "accepted",
"operation_already_booked": "no_record"
},
"tool": "record_decision",
"arguments": {
"case_id": "inv-train-78c3b2c1-0-1",
"decision": "approve",
"amount_cents": 65489,
"reason_code": "authorized_payout"
}
}
```

Draft finish reason: `stop`; proposed action, not executed: `"{\"arguments\":{\"amount_cents\":65489,\"case_id\":\"inv-train-78c3b2c1-0-1\",\"decision\":\"approve\",\"reason_code\":\"authorized_payout\"},\"tool\":\"record_decision\"}"`.

[rcwta-train-78c3b2c1-0/structured/1](<../traces.jsonl#L2>) — `correct_matching_reference_check`.

- Trace line/hash: 2 / `1106ca1879852b5d915558aba6bbcf0766a6b00bc465e69819535c155622a411`.
- Original failure category: `correct`; field mismatches: none.
- Actual action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Executed completion purpose: `action:structured`.

Stored memory before this action (persistent writer state):

```text
{"replace_in_ids":{"~":"-train-78c3b2c1-0-0"},"records":[{"invoice":"inv~","cents":140828,"currency":"BRL","account":"acct~","merchant":"merchant~","order":"ord~","account_status":"active","holder":"merchant~","ownership_match":"yes","payment":"cleared","booked_payout":140828}]}
```

Actual memory view supplied to the actor:

```text
(empty)
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `0`.

Draft model output (NOT EXECUTED):

```text
{
  "evidence_check": {
    "invoice_amount_cents": 65489,
    "account_status": "active",
    "ownership_match": "yes",
    "payment_status": "pending","return_status": "accepted",
    "operation_already_booked": "no_record"
  },
  "tool": "record_decision",
  "arguments": {
    "case_id": "inv-train-78c3b2c1-0-1",
    "decision": "hold",
    "amount_cents": 65489,
    "reason_code": "funds_unsettled"
  }
}
```

Draft finish reason: `stop`; proposed action, not executed: `"INVALID_ACTOR_OUTPUT"`.

### Example 5: paired_summary_correct_structured_incorrect

[rcwta-train-78c3b2c1-1/summary/2](<../traces.jsonl#L27>) — `correct_matching_reference_check`.

- Trace line/hash: 27 / `2078f67aeb3a720e882f0f8d72bd70d90a6fee57749577651f27ff0210028e69`.
- Original failure category: `correct`; field mismatches: none.
- Actual action: `{"arguments": {"amount_cents": 27953, "case_id": "inv-train-78c3b2c1-1-3", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 27953, "case_id": "inv-train-78c3b2c1-1-3", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 27953, "case_id": "inv-train-78c3b2c1-1-3", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Executed completion purpose: `action:summary`.

Stored memory before this action (persistent writer state):

```text
"ask_info decision recorded for case inv-train-78c3b2c1-1-3 (missing evidence); invoice inv-train-78c3b2c1-1-2 cleared; total_cents 27953 on inv-train-78c3b2c1-1-3; customer service requested receipt copy without payment status update."
```

Actual memory view supplied to the actor:

```text
"ask_info decision recorded for case inv-train-78c3b2c1-1-3 (missing evidence); invoice inv-train-78c3b2c1-1-2 cleared; total_cents 27953 on inv-train-78c3b2c1-1-3; customer service requested receipt copy without payment status update."
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `83`.

Draft model output (NOT EXECUTED):

```text
{
"evidence_check": {
"invoice_amount_cents": 27953,
"account_status": "active",
"ownership_match": "yes",
"payment_status": "cleared",
"return_status": "unknown",
"operation_already_booked": "no_record"
},
"tool": "record_decision",
"arguments": {
"case_id": "inv-train-78c3b2c1-1-3",
"decision": "approve",
"amount_cents": 27953,
"reason_code": "authorized_payout"
}
}
```

Draft finish reason: `stop`; proposed action, not executed: `"{\"arguments\":{\"amount_cents\":27953,\"case_id\":\"inv-train-78c3b2c1-1-3\",\"decision\":\"approve\",\"reason_code\":\"authorized_payout\"},\"tool\":\"record_decision\"}"`.

[rcwta-train-78c3b2c1-1/structured/2](<../traces.jsonl#L19>) — `invalid_envelope`.

- Trace line/hash: 19 / `72be650e206c1b06bf6eaf8152964adde6da873add97e8d3e135fb1f50c80b68`.
- Original failure category: `invalid_action`; field mismatches: none.
- Actual action: `"INVALID_ACTOR_OUTPUT"`.
- Reference action: `{"arguments": {"amount_cents": 27953, "case_id": "inv-train-78c3b2c1-1-3", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `null`.
- Executed completion purpose: `action:structured`.

Stored memory before this action (persistent writer state):

```text
{"replace_in_ids":{"~":"-train-78c3b2c1-1-"},"records":[{"invoice":"inv~2","cents":8398,"currency":"BRL","account":"acct~2","merchant":"merchant~2","order":"ord~2","account_status":"active","holder":"merchant~2","ownership_match":"yes","payment":"cleared","booked_payout":8398},{"invoice":"inv~3","cents":27953,"currency":"BRL","account":"acct~3","merchant":"merchant~3","order":"ord~3","ownership_match":"unknown"}]}
```

Actual memory view supplied to the actor:

```text
{"case_id":"inv-train-78c3b2c1-1-3","operation":"payout","retained_evidence":{"invoice_amount_cents":27953,"return_status":"unknown","operation_already_booked":"no_record"},"retained_ids":{"account_id":"acct-train-78c3b2c1-1-3","merchant_id":"merchant-train-78c3b2c1-1-3","order_id":"ord-train-78c3b2c1-1-3"},"invalidated_fields":["account_status","ownership_match","payment_status","holder_merchant_id"]}
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `137`.

Draft model output (NOT EXECUTED):

```text
{
  "evidence_check": {
    "invoice_amount_cents": 27953,
    "account_status": "active",
    "ownership_match": "yes",
    "payment_status": "cleared",
    "return_status": "unknown",
    "operation_already_booked": "no_record"
  },
  "tool": "record_decision",
  "arguments": {
    "case_id": "inv-train-78c3b2c1-1-3",
    "decision": "approve",
    "amount_cents": 27953,
    "reason_code": "authorized_payout"
  }
}
```

Draft finish reason: `stop`; proposed action, not executed: `"{\"arguments\":{\"amount_cents\":27953,\"case_id\":\"inv-train-78c3b2c1-1-3\",\"decision\":\"approve\",\"reason_code\":\"authorized_payout\"},\"tool\":\"record_decision\"}"`.

## Provenance and limits

Original-run offline verification: `PASS`. This is recorded-artifact consistency verification, not independent inference attestation.

Inspector SHA-256: `2e9516823d85542d3c5ff80a1d32f584e1674173b540fb0d5790eaae53f3b327`. Input-manifest SHA-256: `37078742ab0926017e8ede056c978f6bba03589456ae50b1a1c6eabc84ca4146`. Per-file, frozen-source and reused-helper hashes are in diagnosis.json.

- Post-hoc descriptive diagnosis of verified recordings, not a new confirmatory endpoint, new inference, or independent attestation of physical inference.
- Checks are actor self-reports. A divergent field does not identify compression loss, extraction error, wrong linkage, stale-state resolution, or unsupported inference as the cause.
- Reference fields represent evidence available across the episode and actual pre-action bookings, not only stored memory or the query-conditioned view supplied to the actor. A fact never supplied is not a memory-compression failure.
- All six check fields are compared, including fields irrelevant to an operation or overridden by higher-priority rules. A check mismatch need not cause an action error.
- Failure despite a matching reference check describes inconsistent rule application in the recorded answer, not hidden reasoning or a neural mechanism.
- The action implied by the claimed check is computed only for offline comparison. It is never executed, substituted, fed back, or used to change the original grade.
- Paired task counts are not independent statistical samples. The two arms may have different prior actual bookings; their contrast is not a controlled causal attribution to memory.
- Examples follow a frozen descriptive selection rule: first occurrence in public-manifest order, then step, then summary/structured. Both disagreement directions are shown when present.
- The structured intervention combines engineered storage and query-conditioned retrieval, including deterministic selection, joins and stale-field invalidation. It is not compaction alone and not autonomous policy learning or recursive self-improvement. This diagnosis does not isolate which component caused an outcome or establish production safety or cross-domain transfer.
- When present, a draft is an unexecuted model output. Only the completion labelled action:policy is executed and graded; the same mandatory two-pass actor is used in both arms. Draft quality is not counted as final performance, and the diagnosis does not isolate the effect of self-review.
