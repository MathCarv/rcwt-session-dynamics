# V3 post-hoc evidence-check diagnosis

OFFLINE RECORDED DIAGNOSIS / NO MODEL CALLS / NO FEEDBACK

This describes reference-check and action agreement after a completed run. It is not an extra success endpoint, a repaired trajectory, or evidence that memory caused an error.

Only the final completion labelled `action:policy` is executed and graded. A `draft:policy` completion, when present, is shown solely as an unexecuted draft; there is no fallback to it.

Episodes: 4; split: `train`; arms: summary and structured.

| Policy | Decisions | Exact successes | Matching checks | Divergent checks | Invalid checks | Failures despite matching check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 32 | 16 | 8 | 23 | 1 | 0 |
| structured | 32 | 23 | 22 | 9 | 1 | 1 |

A matching check requires all six fields to match the reference, including actual prior bookings. Invalid envelopes are not treated as evidence of memory failure.

## Field mismatches

| Policy | Amount | Account | Ownership | Payment | Return | Already booked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 16 | 12 | 16 | 14 | 10 | 1 |
| structured | 2 | 3 | 2 | 6 | 2 | 0 |

## Action agreement with the actor's own check

- `summary`: 28 consistent and 3 inconsistent among 31 valid envelopes; 1 not evaluable.
- `structured`: 27 consistent and 4 inconsistent among 31 valid envelopes; 1 not evaluable.

## Paired task outcomes (descriptive)

Across 32 task pairs: structured correct / summary incorrect = 9; summary correct / structured incorrect = 2; both correct = 14; both incorrect = 7.

These counts are not independent samples. Prior bookings can differ by arm. The primary result remains the separate paired-episode analysis.

## Deterministic examples

First occurrence in public-manifest order, then step, then summary/structured: matching-check failure, divergent-check failure, invalid envelope, and each direction of paired outcome disagreement when present.

### Example 1: failed_matching_reference_check

[rcwta-train-78c3b2c1-2/structured/3](<../traces.jsonl#L44>) — `failed_matching_reference_check`.

- Trace line/hash: 44 / `4add8148e45252b01fa080622f1b8076e4a43df1e5720f50c1bf3fbee098ab12`.
- Original failure category: `wrong_decision`; field mismatches: none.
- Actual action: `{"arguments": {"amount_cents": 124486, "case_id": "inv-train-78c3b2c1-2-0", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 124486, "case_id": "inv-train-78c3b2c1-2-0", "decision": "refund", "reason_code": "eligible_refund"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 124486, "case_id": "inv-train-78c3b2c1-2-0", "decision": "refund", "reason_code": "eligible_refund"}, "tool": "record_decision"}`.
- Executed completion purpose: `action:structured`.

Stored memory before this action (persistent writer state):

```text
{"replace_in_ids":{"~":"-train-78c3b2c1-2-"},"records":[{"invoice":"inv~0","cents":124486,"currency":"BRL","account":"acct~0","merchant":"merchant~0","order":"ord~0","account_status":"active","holder":"merchant~0","ownership_match":"yes","return":"accepted","payment":"cleared","booked_payout":124486},{"invoice":"inv~2","cents":127333,"currency":"BRL","account":"acct~2","merchant":"merchant~2","order":"ord~2","account_status":"active","holder":"merchant~2","ownership_match":"yes","return":"accepted","payment":"pending"},{"invoice":"inv~1","cents":31908,"currency":"BRL","account":"acct~1","merchant":"merchant~1","order":"ord~1","account_status":"blocked","holder":"merchant~1","ownership_match":"yes","payment":"cleared"}]}
```

Actual memory view supplied to the actor:

```text
{"case_id":"inv-train-78c3b2c1-2-0","operation":"refund","retained_evidence":{"invoice_amount_cents":124486,"account_status":"active","ownership_match":"yes","payment_status":"cleared","return_status":"accepted","operation_already_booked":"no_record"},"retained_ids":{"account_id":"acct-train-78c3b2c1-2-0","merchant_id":"merchant-train-78c3b2c1-2-0","holder_merchant_id":"merchant-train-78c3b2c1-2-0","order_id":"ord-train-78c3b2c1-2-0"},"invalidated_fields":[]}
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `162`.

Free-text plan (NOT EXECUTED; NOT PARSED AS A TOOL CALL):

```text
Request: refund for invoice inv-train-78c3b2c1-2-0 (req-train-78c3b2c1-2-3).
Known Facts: Invoice amount 124486 cents; account active, ownership matches merchant; payment cleared; return accepted. No prior booking found.
Current Updates: New observation shows invoice inv-train-78c3b2c1-2-2 status pending and a note about future statement changes. These do not match the target invoice identifier for this refund request.
Absent Evidence: The new observation provides no data on inv-train-78c3b2c1-2-0, so payment_status cannot be inferred from it despite being cleared in memory. However, memory is retained and valid.
Applicable Rule: Priority (6) applies because all required evidence for a refund (account status, ownership, payment cleared, return accepted, amount known) is present in retained memory and no higher-priority block exists.
Decision: approve with reason authorized_payout. Amount: 124486 cents.
```

First-pass format: `unconstrained_text_plan`; finish reason: `stop`.

The raw plan supplies no executable action or extracted evidence check. Its finish reason does not classify the final action as invalid; the original final completion is evaluated independently.

### Example 2: failed_divergent_check

[rcwta-train-78c3b2c1-0/structured/1](<../traces.jsonl#L2>) — `failed_divergent_check`.

- Trace line/hash: 2 / `62c7b50dc1da0c3a31e2b745b2b3064856c6cfa0c831e8adbcdb7840a8ab21dc`.
- Original failure category: `wrong_decision`; field mismatches: payment_status: claimed="unknown", reference="pending".
- Actual action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
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

Free-text plan (NOT EXECUTED; NOT PARSED AS A TOOL CALL):

```text
Case: inv-train-78c3b2c1-0-1 (Payout).
Known facts: Account acct-train-78c3b2c1-0-1 is active and holder matches merchant. Invoice total_cents is 65489, status pending. Order ord-train-78c3b2c1-0-1 has accepted return inspection.
Unknown facts: Payment status is not explicitly recorded as cleared in the provided observations; a message notes customer service requested a receipt but no update was supplied.
Priority check: Rule (6) applies for payouts, but Rule (3) regarding missing payment status takes precedence if the payment status cannot be confirmed as cleared. Since the observation only shows "pending" and lacks explicit confirmation of funds being settled or cleared, I must verify this before approving.
Decision: ask_info/missing_evidence with amount_cents=0 to confirm payment clearance. Reason: Payment status unknown.
```

First-pass format: `unconstrained_text_plan`; finish reason: `stop`.

The raw plan supplies no executable action or extracted evidence check. Its finish reason does not classify the final action as invalid; the original final completion is evaluated independently.

### Example 3: invalid_envelope

[rcwta-train-78c3b2c1-2/summary/6](<../traces.jsonl#L39>) — `invalid_envelope`.

- Trace line/hash: 39 / `c722288cccf460a928fad6893badee6fc01048a3d6dab58a6a721c72f3eda60e`.
- Original failure category: `invalid_action`; field mismatches: none.
- Actual action: `"INVALID_ACTOR_OUTPUT"`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-2-3", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `null`.
- Executed completion purpose: `action:summary`.

Stored memory before this action (persistent writer state):

```text
{"case_id":"inv-train-78c3b2c1-2-1","decision":"ask_info","reason_code":"missing_evidence","notes":[{"content":"Weekly reconciliation note: merchant statement layout will change next month.","event_id":"obs-train-78c3b2c1-2-19"}]}
```

Actual memory view supplied to the actor:

```text
{"case_id":"inv-train-78c3b2c1-2-1","decision":"ask_info","reason_code":"missing_evidence","notes":[{"content":"Weekly reconciliation note: merchant statement layout will change next month.","event_id":"obs-train-78c3b2c1-2-19"}]}
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `71`.

Free-text plan (NOT EXECUTED; NOT PARSED AS A TOOL CALL):

```text
Requested case: inv-train-78c3b2c1-2-3. Operation: payout.

Known facts: Invoice inv-train-78c3b2c1-2-3 exists with total_cents=86451 (obs-train-78c3b2c1-2-20). Account acct-train-78c3b2c1-2-3 is associated with this invoice. Merchant ID matches account holder context implied by IDs. No evidence of revoked/blocked status, ownership mismatch, or uncleared funds in current observations or memory for this specific case ID.

Unknown facts: Payment status (cleared/unsettled), return status (not applicable for payout).

Priority check:
1. Revoked/blocked account: Not observed.
2. Ownership mismatch: IDs align; not observed otherwise.
3. Missing evidence/payment status: No indication of missing required fields for a standard payout other than potential payment status, but rule 4 covers funds.
4. Funds unsettled: No observation of payment rejection or hold on funds.
5. Return rejected: N/A for payout.
6. Payout logic: Apply approve/authorized_payout.

Decision: Approve the payout with authorized_payout reason code. Amount must be exact invoice total 86451 cents.
```

First-pass format: `unconstrained_text_plan`; finish reason: `stop`.

The raw plan supplies no executable action or extracted evidence check. Its finish reason does not classify the final action as invalid; the original final completion is evaluated independently.

### Example 4: paired_structured_correct_summary_incorrect

[rcwta-train-78c3b2c1-0/summary/5](<../traces.jsonl#L14>) — `failed_divergent_check`.

- Trace line/hash: 14 / `961587b52261c52a2cae3432f8ec04524ae3b5b1f7a3902c379dec2bb9b57478`.
- Original failure category: `unnecessary_deferral`; field mismatches: invoice_amount_cents: claimed=null, reference=140828; payment_status: claimed="unknown", reference="cleared".
- Actual action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-0", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 140828, "case_id": "inv-train-78c3b2c1-0-0", "decision": "refund", "reason_code": "eligible_refund"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-0", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Executed completion purpose: `action:summary`.

Stored memory before this action (persistent writer state):

```text
{"case_id":"inv-train-78c3b2c1-0-1","decision":"ask_info","reason_code":"missing_evidence","invoice_id":"inv-train-78c3b2c1-0-1","status":"cleared","amount_cents":0,"account_id":"acct-train-78c3b2c1-0-1","merchant_id":"merchant-train-78c3b2c1-0-1","order_id":"ord-train-78c3b2c1-0-1","request_id":"req-train-78c3b2c1-0-4","observations":[{"content":{"invoice_id":"inv-train-78c3b2c1-0-1","status":"cleared"},"event_id":"obs-train-78c3b2c1-0-17","source":"tool","tool":"read_payment"},{"content":"Customer service asked for a receipt copy; no payment status update was supplied.","event_id":"obs-train-78c3b2c1-0-18","source":"message"}]}
```

Actual memory view supplied to the actor:

```text
{"case_id":"inv-train-78c3b2c1-0-1","decision":"ask_info","reason_code":"missing_evidence","invoice_id":"inv-train-78c3b2c1-0-1","status":"cleared","amount_cents":0,"account_id":"acct-train-78c3b2c1-0-1","merchant_id":"merchant-train-78c3b2c1-0-1","order_id":"ord-train-78c3b2c1-0-1","request_id":"req-train-78c3b2c1-0-4","observations":[{"content":{"invoice_id":"inv-train-78c3b2c1-0-1","status":"cleared"},"event_id":"obs-train-78c3b2c1-0-17","source":"tool","tool":"read_payment"},{"content":"Customer service asked for a receipt copy; no payment status update was supplied.","event_id":"obs-train-78c3b2c1-0-18","source":"message"}]}
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `246`.

Free-text plan (NOT EXECUTED; NOT PARSED AS A TOOL CALL):

```text
**Case:** inv-train-78c3b2c1-0-0
**Operation:** refund

**Known Facts:**
*   **Account:** acct-train-78c3b2c1-0-0 (active, holder matches merchant).
*   **Return:** ord-train-78c3b2c1-0-0 (inspection accepted; delivery address correction noted but not rejected).
*   **Invoice Amount/Status:** Unknown. No payment or invoice details observed for inv-train-78c3b2c1-0-0 in current observations or memory.

**Absent Evidence:**
*   Invoice amount (cents).
*   Payment status (cleared/unsettled/rejected).
*   Invoice return status.

**Rule Application:**
Priority Rule (3) applies: "missing/unknown invoice amount, account status, ownership, or payment status... -> ask_info/missing_evidence". Although the return inspection is accepted, the lack of any invoice data prevents applying higher-priority rules like (6) or (4). The retained memory for case inv-train-78c3b2c1-0-1 does not apply to this current request.

**Decision Plan:**
Ask for missing evidence regarding the invoice amount and payment status before proceeding.

*   **Action:** ask_info
*   **Reason Code:** missing_evidence
*   **Amount:** 0
```

First-pass format: `unconstrained_text_plan`; finish reason: `stop`.

The raw plan supplies no executable action or extracted evidence check. Its finish reason does not classify the final action as invalid; the original final completion is evaluated independently.

[rcwta-train-78c3b2c1-0/structured/5](<../traces.jsonl#L6>) — `correct_matching_reference_check`.

- Trace line/hash: 6 / `0e1472ad5865b171d69c25bfc699091ebad098ee2ceb46366b3c3e3a3dabd0ab`.
- Original failure category: `correct`; field mismatches: none.
- Actual action: `{"arguments": {"amount_cents": 140828, "case_id": "inv-train-78c3b2c1-0-0", "decision": "refund", "reason_code": "eligible_refund"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 140828, "case_id": "inv-train-78c3b2c1-0-0", "decision": "refund", "reason_code": "eligible_refund"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 140828, "case_id": "inv-train-78c3b2c1-0-0", "decision": "refund", "reason_code": "eligible_refund"}, "tool": "record_decision"}`.
- Executed completion purpose: `action:structured`.

Stored memory before this action (persistent writer state):

```text
{"replace_in_ids":{"~":"-train-78c3b2c1-0-"},"records":[{"invoice":"inv~0","cents":140828,"currency":"BRL","account":"acct~0","merchant":"merchant~0","order":"ord~0","account_status":"revoked","holder":"merchant~0","ownership_match":"yes","payment":"cleared","booked_payout":140828},{"invoice":"inv~3","cents":48916,"currency":"BRL","account":"acct~3","merchant":"merchant~3","order":"ord~3","account_status":"active","holder":"merchant~3","ownership_match":"yes","return":"accepted","payment":"cleared","booked_refund":48916},{"invoice":"inv~1","cents":65489,"currency":"BRL","account":"acct~1","merchant":"merchant~1","order":"ord~1","account_status":"active","holder":"merchant~1","ownership_match":"yes","return":"accepted","payment":"cleared"}]}
```

Actual memory view supplied to the actor:

```text
{"case_id":"inv-train-78c3b2c1-0-0","operation":"refund","retained_evidence":{"invoice_amount_cents":140828,"payment_status":"cleared","operation_already_booked":"no_record"},"retained_ids":{"account_id":"acct-train-78c3b2c1-0-0","merchant_id":"merchant-train-78c3b2c1-0-0","order_id":"ord-train-78c3b2c1-0-0"},"invalidated_fields":["account_status","ownership_match","return_status","holder_merchant_id"]}
```

Memory-view source: `recorded_actor_view`; bound to recorded request: `True`; recorded actor-view tokens: `138`.

Free-text plan (NOT EXECUTED; NOT PARSED AS A TOOL CALL):

```text
Case: inv-train-78c3b2c1-0-0; Operation: refund.
Known facts: Account acct-train-78c3b2c1-0-0 is active and matches merchant (retained memory). Invoice amount is 140,828 cents with cleared payment status (retained memory). Return order ord-train-78c3b2c1-0-0 shows inspection_status "accepted" (current observation). No existing booking for this refund operation.
Absent evidence: Current observations do not explicitly report the return's rejection or acceptance status beyond the inspection, nor do they re-verify the account/merchant match definitively in the new step data. However, retained memory indicates these were previously validated and invalidated as fields needing update only if conflicting. Priority (3) requires missing evidence for refund eligibility (return status). Since current observations lack explicit "return accepted" confirmation and rely on retained memory which may be stale regarding return validity, strict adherence to priority (3) mandates asking for the return status to confirm eligibility before approving.
Decision: ask_info/missing_evidence with amount_cents=0. Reason: Missing explicit return acceptance status in current observations despite cleared payment.
```

First-pass format: `unconstrained_text_plan`; finish reason: `stop`.

The raw plan supplies no executable action or extracted evidence check. Its finish reason does not classify the final action as invalid; the original final completion is evaluated independently.

### Example 5: paired_summary_correct_structured_incorrect

[rcwta-train-78c3b2c1-0/summary/1](<../traces.jsonl#L10>) — `correct_divergent_check`.

- Trace line/hash: 10 / `356c2e4cdcdcbb2afcfac3c411200629947af88eedd641d9244b10a144eaafd6`.
- Original failure category: `correct`; field mismatches: return_status: claimed="unknown", reference="accepted".
- Actual action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
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

Free-text plan (NOT EXECUTED; NOT PARSED AS A TOOL CALL):

```text
Requested case: inv-train-78c3b2c1-0-1. Operation: payout.

**Known Facts:**
*   **Account (acct-train-78c3b2c1-0-1):** Active, holder matches invoice merchant (merchant-train-78c3b2c1-0-1).
*   **Invoice (inv-train-78c3b2c1-0-1):** Status pending, total_cents 65489.
*   **Payment (inv-train-78c3b2c1-0-1):** No evidence of cleared funds; status remains pending.
*   **Return:** Not applicable for payout.

**Absent Evidence:**
*   Payment status is not "cleared". The message indicates a missing receipt copy, but the payment record itself does not confirm clearance.

**Rule Application:**
Priority rule (4) applies: "payment not cleared -> hold/funds_unsettled". Although the account and ownership are valid, the lack of evidence confirming the payment has been cleared prevents authorization. Rule (6) is skipped because the prerequisite condition (cleared payment) is unmet.

**Decision Plan:**
Hold the payout request with reason `funds_unsettled`. Do not approve or refund.
```

First-pass format: `unconstrained_text_plan`; finish reason: `stop`.

The raw plan supplies no executable action or extracted evidence check. Its finish reason does not classify the final action as invalid; the original final completion is evaluated independently.

[rcwta-train-78c3b2c1-0/structured/1](<../traces.jsonl#L2>) — `failed_divergent_check`.

- Trace line/hash: 2 / `62c7b50dc1da0c3a31e2b745b2b3064856c6cfa0c831e8adbcdb7840a8ab21dc`.
- Original failure category: `wrong_decision`; field mismatches: payment_status: claimed="unknown", reference="pending".
- Actual action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Reference action: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "hold", "reason_code": "funds_unsettled"}, "tool": "record_decision"}`.
- Action implied only by the claimed check: `{"arguments": {"amount_cents": 0, "case_id": "inv-train-78c3b2c1-0-1", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
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

Free-text plan (NOT EXECUTED; NOT PARSED AS A TOOL CALL):

```text
Case: inv-train-78c3b2c1-0-1 (Payout).
Known facts: Account acct-train-78c3b2c1-0-1 is active and holder matches merchant. Invoice total_cents is 65489, status pending. Order ord-train-78c3b2c1-0-1 has accepted return inspection.
Unknown facts: Payment status is not explicitly recorded as cleared in the provided observations; a message notes customer service requested a receipt but no update was supplied.
Priority check: Rule (6) applies for payouts, but Rule (3) regarding missing payment status takes precedence if the payment status cannot be confirmed as cleared. Since the observation only shows "pending" and lacks explicit confirmation of funds being settled or cleared, I must verify this before approving.
Decision: ask_info/missing_evidence with amount_cents=0 to confirm payment clearance. Reason: Payment status unknown.
```

First-pass format: `unconstrained_text_plan`; finish reason: `stop`.

The raw plan supplies no executable action or extracted evidence check. Its finish reason does not classify the final action as invalid; the original final completion is evaluated independently.

## Provenance and limits

Original-run offline verification: `PASS`. This is recorded-artifact consistency verification, not independent inference attestation.

Inspector SHA-256: `e1b98d5a6c97dc21df767150331656102b44688919c9f59043fa7ad4e0c52a78`. Input-manifest SHA-256: `84dc96ced4d831f39c652e790f3d22ecf84ec55caf69e9efea9a4677b9097df4`. Per-file, frozen-source and reused-helper hashes are in diagnosis.json.

- Post-hoc descriptive diagnosis of verified recordings, not a new confirmatory endpoint, new inference, or independent attestation of physical inference.
- Checks are actor self-reports. A divergent field does not identify compression loss, extraction error, wrong linkage, stale-state resolution, or unsupported inference as the cause.
- Reference fields represent evidence available across the episode and actual pre-action bookings, not only stored memory or the query-conditioned view supplied to the actor. A fact never supplied is not a memory-compression failure.
- All six check fields are compared, including fields irrelevant to an operation or overridden by higher-priority rules. A check mismatch need not cause an action error.
- Failure despite a matching reference check describes inconsistent rule application in the recorded answer, not hidden reasoning or a neural mechanism.
- The action implied by the claimed check is computed only for offline comparison. It is never executed, substituted, fed back, or used to change the original grade.
- Paired task counts are not independent statistical samples. The two arms may have different prior actual bookings; their contrast is not a controlled causal attribution to memory.
- Examples follow a frozen descriptive selection rule: first occurrence in public-manifest order, then step, then summary/structured. Both disagreement directions are shown when present.
- The structured intervention combines engineered storage and query-conditioned retrieval, including deterministic selection, joins and stale-field invalidation. It is not compaction alone and not autonomous policy learning or recursive self-improvement. This diagnosis does not isolate which component caused an outcome or establish production safety or cross-domain transfer.
- When present, a draft is an unexecuted model output. A draft request without response_format is a schema-free text plan: it is not parsed as an action or evidence check, even if its text resembles JSON. Legacy schema-constrained JSON drafts are displayed separately. Only the completion labelled action:policy is executed and graded; the same mandatory two-pass actor is used in both arms. Plan or draft quality is not counted as final performance, and the diagnosis does not isolate the effect of planning or self-review.
