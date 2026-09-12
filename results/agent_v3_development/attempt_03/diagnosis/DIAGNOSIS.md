# V3 post-hoc evidence-check diagnosis

OFFLINE RECORDED DIAGNOSIS / NO MODEL CALLS / NO FEEDBACK

This describes reference-check and action agreement after a completed run. It is not an extra success endpoint, a repaired trajectory, or evidence that memory caused an error.

Episodes: 4; split: `train`; arms: summary and structured.

| Policy | Decisions | Exact successes | Matching checks | Divergent checks | Invalid checks | Failures despite matching check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 32 | 14 | 11 | 21 | 0 | 2 |
| structured | 32 | 19 | 22 | 9 | 1 | 6 |

A matching check requires all six fields to match the reference, including actual prior bookings. Invalid envelopes are not treated as evidence of memory failure.

## Field mismatches

| Policy | Amount | Account | Ownership | Payment | Return | Already booked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 16 | 13 | 17 | 14 | 8 | 3 |
| structured | 1 | 3 | 4 | 3 | 2 | 1 |

## Action agreement with the actor's own check

- `summary`: 27 consistent and 5 inconsistent among 32 valid envelopes; 0 not evaluable.
- `structured`: 23 consistent and 8 inconsistent among 31 valid envelopes; 1 not evaluable.

## Paired task outcomes (descriptive)

Across 32 task pairs: structured correct / summary incorrect = 8; summary correct / structured incorrect = 3; both correct = 11; both incorrect = 10.

These counts are not independent samples. Prior bookings can differ by arm. The primary result remains the separate paired-episode analysis.

## Deterministic examples

First occurrence in public-manifest order, then step, then summary/structured: matching-check failure, divergent-check failure, invalid envelope, and each direction of paired outcome disagreement when present.

### Example 1: failed_matching_reference_check

[rcwta-train-78c3b2c1-0/summary/1](<../traces.jsonl#L10>) — `failed_matching_reference_check`.

- Trace line/hash: 10 / `7715c608eee94b299b00d4c1bbbeba0347fbe58ab1684a89dfe907d8b58a9392`.
- Original failure category: `unsafe_execution`; field mismatches: none.
- Actual action: `{"arguments": {"amount_cents": 65489, "case_id": "inv-train-78c3b2c1-0-1", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.

Stored memory before this action (persistent writer state):

```text
Case inv-train-78c3b2c1-0-0: $1408.28 (BRL) approved via record_decision; account acct-train-78c3b2c1-0-0 active; invoice cleared; support queue review pending address correction with shipping team.
```

Actual memory view supplied to the actor:

```text
Case inv-train-78c3b2c1-0-0: $1408.28 (BRL) approved via record_decision; account acct-train-78c3b2c1-0-0 active; invoice cleared; support queue review pending address correction with shipping team.
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `67`.

### Example 2: failed_divergent_check

[rcwta-train-78c3b2c1-0/summary/4](<../traces.jsonl#L13>) — `failed_divergent_check`.

- Trace line/hash: 13 / `883d59a71daae20f17656fac0ced35a22b7e0f04d10c1248de247a16a2b39f9b`.
- Original failure category: `wrong_decision`; field mismatches: invoice_amount_cents: claimed=null, reference=65489; account_status: claimed="unknown", reference="active"; ownership_match: claimed="unknown", reference="yes"; return_status: claimed="unknown", reference="accepted"; operation_already_booked: claimed="no_record", reference="yes".
- Actual action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "already_completed"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.

Stored memory before this action (persistent writer state):

```text
{"case_id":"inv-train-78c3b2c1-0-3","amount_cents":48916,"decision":"refund","reason_code":"eligible_refund","account_id":"acct-train-78c3b2c1-0-3","verification_status":"active","currency":"BRL","status":"cleared","notes":["Warehouse sent opening-hours update; document contains no return inspection"]}
```

Actual memory view supplied to the actor:

```text
{"case_id":"inv-train-78c3b2c1-0-3","amount_cents":48916,"decision":"refund","reason_code":"eligible_refund","account_id":"acct-train-78c3b2c1-0-3","verification_status":"active","currency":"BRL","status":"cleared","notes":["Warehouse sent opening-hours update; document contains no return inspection"]}
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `91`.

### Example 3: invalid_envelope

[rcwta-train-78c3b2c1-0/structured/1](<../traces.jsonl#L2>) — `invalid_envelope`.

- Trace line/hash: 2 / `6041080059b1698c06e4402d584c56fa8362fcb2234c4e1f7018c636ac009531`.
- Original failure category: `invalid_action`; field mismatches: none.
- Actual action: `"INVALID_ACTOR_OUTPUT"`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `null`.

Stored memory before this action (persistent writer state):

```text
{"replace_in_ids":{"~":"-train-78c3b2c1-0-0"},"records":[{"invoice":"inv~","cents":140828,"currency":"BRL","account":"acct~","merchant":"merchant~","order":"ord~","account_status":"active","holder":"merchant~","ownership_match":"yes","payment":"cleared","booked_payout":140828}]}
```

Actual memory view supplied to the actor:

```text
(empty)
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `0`.

### Example 4: paired_structured_correct_summary_incorrect

[rcwta-train-78c3b2c1-0/summary/4](<../traces.jsonl#L13>) — `failed_divergent_check`.

- Trace line/hash: 13 / `883d59a71daae20f17656fac0ced35a22b7e0f04d10c1248de247a16a2b39f9b`.
- Original failure category: `wrong_decision`; field mismatches: invoice_amount_cents: claimed=null, reference=65489; account_status: claimed="unknown", reference="active"; ownership_match: claimed="unknown", reference="yes"; return_status: claimed="unknown", reference="accepted"; operation_already_booked: claimed="no_record", reference="yes".
- Actual action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "already_completed"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.

Stored memory before this action (persistent writer state):

```text
{"case_id":"inv-train-78c3b2c1-0-3","amount_cents":48916,"decision":"refund","reason_code":"eligible_refund","account_id":"acct-train-78c3b2c1-0-3","verification_status":"active","currency":"BRL","status":"cleared","notes":["Warehouse sent opening-hours update; document contains no return inspection"]}
```

Actual memory view supplied to the actor:

```text
{"case_id":"inv-train-78c3b2c1-0-3","amount_cents":48916,"decision":"refund","reason_code":"eligible_refund","account_id":"acct-train-78c3b2c1-0-3","verification_status":"active","currency":"BRL","status":"cleared","notes":["Warehouse sent opening-hours update; document contains no return inspection"]}
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `91`.

[rcwta-train-78c3b2c1-0/structured/4](<../traces.jsonl#L5>) — `correct_matching_reference_check`.

- Trace line/hash: 5 / `0842f5d7de3716a1badc8e4ad2e7a67d4bee053e9ef48121a0a0c6c4998a5a12`.
- Original failure category: `correct`; field mismatches: none.
- Actual action: `{"arguments": {"amount_cents": 65489, "case_id": "inv-train-78c3b2c1-0-1", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 65489, "case_id": "inv-train-78c3b2c1-0-1", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 65489, "case_id": "inv-train-78c3b2c1-0-1", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.

Stored memory before this action (persistent writer state):

```text
{"replace_in_ids":{"~":"-train-78c3b2c1-0-"},"records":[{"invoice":"inv~1","cents":65489,"currency":"BRL","account":"acct~1","merchant":"merchant~1","order":"ord~1","account_status":"active","holder":"merchant~1","ownership_match":"yes","return":"accepted","payment":"pending"},{"invoice":"inv~0","cents":140828,"currency":"BRL","account":"acct~0","merchant":"merchant~0","order":"ord~0","account_status":"revoked","holder":"merchant~0","ownership_match":"yes","payment":"cleared","booked_payout":140828},{"invoice":"inv~3","cents":48916,"currency":"BRL","account":"acct~3","merchant":"merchant~3","order":"ord~3","account_status":"active","holder":"merchant~3","ownership_match":"yes","return":"accepted","payment":"cleared","booked_refund":48916}]}
```

Actual memory view supplied to the actor:

```text
{"case_id":"inv-train-78c3b2c1-0-1","operation":"payout","retained_evidence":{"invoice_amount_cents":65489,"account_status":"active","ownership_match":"yes","return_status":"accepted","operation_already_booked":"no_record"},"retained_ids":{"account_id":"acct-train-78c3b2c1-0-1","merchant_id":"merchant-train-78c3b2c1-0-1","holder_merchant_id":"merchant-train-78c3b2c1-0-1","order_id":"ord-train-78c3b2c1-0-1"},"invalidated_fields":["payment_status"]}
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `158`.

### Example 5: paired_summary_correct_structured_incorrect

[rcwta-train-78c3b2c1-2/summary/1](<../traces.jsonl#L34>) — `correct_matching_reference_check`.

- Trace line/hash: 34 / `ba70a3dcba0319d93e67629bf044a561f380a44367d2f299cec101c0fff174eb`.
- Original failure category: `correct`; field mismatches: none.
- Actual action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-2-2", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-2-2", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-2-2", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.

Stored memory before this action (persistent writer state):

```text
Invoice inv-train-78c3b2c1-2-0 ($124.49) payout approved for merchant-train-78c3b2c1-2-0 (acct-train-78c3b2c1-2-0); order ord-train-78c3b2c1-2-0 accepted; invoice status cleared; customer service requested receipt copy but no payment status update provided.
```

Actual memory view supplied to the actor:

```text
Invoice inv-train-78c3b2c1-2-0 ($124.49) payout approved for merchant-train-78c3b2c1-2-0 (acct-train-78c3b2c1-2-0); order ord-train-78c3b2c1-2-0 accepted; invoice status cleared; customer service requested receipt copy but no payment status update provided.
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `97`.

[rcwta-train-78c3b2c1-2/structured/1](<../traces.jsonl#L42>) — `failed_matching_reference_check`.

- Trace line/hash: 42 / `fe6e9875a9ec174043252eb7c7358c2357ff8d61858c8193a16cd8e03b87dd0b`.
- Original failure category: `unsafe_execution`; field mismatches: none.
- Actual action: `{"arguments": {"amount_cents": 127333, "case_id": "inv-train-78c3b2c1-2-2", "decision": "refund", "reason_code": "eligible_refund"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-2-2", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-2-2", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.

Stored memory before this action (persistent writer state):

```text
{"replace_in_ids":{"~":"-train-78c3b2c1-2-0"},"records":[{"invoice":"inv~","cents":124486,"currency":"BRL","account":"acct~","merchant":"merchant~","order":"ord~","account_status":"active","holder":"merchant~","ownership_match":"yes","return":"accepted","payment":"cleared","booked_payout":124486}]}
```

Actual memory view supplied to the actor:

```text
(empty)
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `0`.

## Provenance and limits

Original-run offline verification: `PASS`. This is recorded-artifact consistency verification, not independent inference attestation.

Inspector SHA-256: `8353b10de8c15495ec792b3dbf8f25b55dd9e9632f47a8f6e8a0a7e27c4f0964`. Input-manifest SHA-256: `79a215cca60e7aedcdecd31e656b8269e05dfb8d7cfee9797c5fe5509a3c5752`. Per-file, frozen-source and reused-helper hashes are in diagnosis.json.

- Post-hoc descriptive diagnosis of verified recordings, not a new confirmatory endpoint, new inference, or independent attestation of physical inference.
- Checks are actor self-reports. A divergent field does not identify compression loss, extraction error, wrong linkage, stale-state resolution, or unsupported inference as the cause.
- Reference fields represent evidence available across the episode and actual pre-action bookings, not only stored memory or the query-conditioned view supplied to the actor. A fact never supplied is not a memory-compression failure.
- All six check fields are compared, including fields irrelevant to an operation or overridden by higher-priority rules. A check mismatch need not cause an action error.
- Failure despite a matching reference check describes inconsistent rule application in the recorded answer, not hidden reasoning or a neural mechanism.
- The action implied by the claimed check is computed only for offline comparison. It is never executed, substituted, fed back, or used to change the original grade.
- Paired task counts are not independent statistical samples. The two arms may have different prior actual bookings; their contrast is not a controlled causal attribution to memory.
- Examples follow a frozen descriptive selection rule: first occurrence in public-manifest order, then step, then summary/structured. Both disagreement directions are shown when present.
- The structured intervention combines engineered storage and query-conditioned retrieval, including deterministic selection, joins and stale-field invalidation. It is not compaction alone and not autonomous policy learning or recursive self-improvement. This diagnosis does not isolate which component caused an outcome or establish production safety or cross-domain transfer.
