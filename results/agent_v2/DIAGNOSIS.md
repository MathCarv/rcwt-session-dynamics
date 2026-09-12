# Post-hoc evidence-check diagnosis

This artifact describes recorded evidence/action agreement. It is not an additional success gate, a corrected agent run, or proof that memory caused an error.

Episodes: 16; split: test; feedback to agent: none.

| Policy | Decisions | Exact successes | Failed with matching check | Failed with divergent check | Invalid envelope | Action consistent with own check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tail | 128 | 46 | 14 | 60 | 8 | 72/120 valid |
| summary | 128 | 54 | 9 | 61 | 4 | 87/124 valid |
| learned | 128 | 54 | 9 | 61 | 4 | 87/124 valid |

Matching reference check means all six fields match the reference snapshot and actual pre-action booking state. Divergent check is a descriptive mismatch, not a causal memory-failure label. Invalid envelopes are excluded from the own-check consistency denominator.

## Field disagreements

| Policy | Amount | Account | Ownership | Payment | Return | Already booked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tail | 66 | 34 | 70 | 52 | 37 | 17 |
| summary | 61 | 49 | 66 | 65 | 39 | 9 |
| learned | 61 | 49 | 66 | 65 | 39 | 9 |

## Paired task outcomes

Across 128 scheduled summary/learned task pairs: learned correct / summary incorrect = 0; summary correct / learned incorrect = 0; both correct = 54; both incorrect = 74.

These are descriptive counts, not independent samples. Prior actual bookings may differ between arms. The principal result remains the separately computed paired episode analysis.

## Deterministically selected examples

First recorded matching-check failure, first recorded divergent-check failure; then first episode-ID/step-sorted learned-correct contrast with opposite-direction contrast if available.

### Example 1: failed_matching_reference_check

[rcwta-test-135282f-1 / tail / step 4](<./test/traces.jsonl#L37>) — `failed_matching_reference_check`.

- Field deltas: none.
- Recorded action: `{"arguments": {"amount_cents": 0, "case_id": "inv-test-135282f-1-0", "decision": "hold", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Reference action using actual prior bookings: `{"arguments": {"amount_cents": 0, "case_id": "inv-test-135282f-1-0", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Action implied by the actor's check alone: `{"arguments": {"amount_cents": 0, "case_id": "inv-test-135282f-1-0", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Consistent with own check: `False`. Prior actual booked keys: `[["inv-test-135282f-1-3", "payout"]]`.

Retained public memory before this action:

```text
completed_request": {"case_id": "inv-test-135282f-1-1", "instruction": "Record the refund decision for invoice inv-test-135282f-1-1 using the latest known evidence.", "operation": "refund", "request_id": "req-test-135282f-1-3"}, "observations": [{"content": {"inspection_status": "rejected", "order_id": "ord-test-135282f-1-1"}, "event_id": "obs-test-135282f-1-10", "source": "tool", "tool": "read_return"}, {"content": "Support queue reviewed. Delivery address correction remains with the shipping team.", "event_id": "obs-test-135282f-1-11", "source": "message"}], "tool_result": {"accepted": true, "amount_booked_cents": 0, "case_id": "inv-test-135282f-1-1", "decision": "hold", "simulated": true, "step_index": 3, "tool": "record_decision"}}
```

Current public observations and the complete check are retained in diagnosis.json and the linked trace. No missing historical fact is automatically classified as a compression defect.

### Example 2: failed_divergent_check

[rcwta-test-135282f-0 / summary / step 4](<./test/traces.jsonl#L5>) — `failed_divergent_check`.

- Field deltas: invoice_amount_cents: claimed=null, reference=16048; account_status: claimed="unknown", reference="active"; ownership_match: claimed="unknown", reference="yes"; return_status: claimed="unknown", reference="accepted".
- Recorded action: `{"arguments": {"amount_cents": 0, "case_id": "inv-test-135282f-0-2", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Reference action using actual prior bookings: `{"arguments": {"amount_cents": 16048, "case_id": "inv-test-135282f-0-2", "decision": "approve", "reason_code": "authorized_payout"}, "tool": "record_decision"}`.
- Action implied by the actor's check alone: `{"arguments": {"amount_cents": 0, "case_id": "inv-test-135282f-0-2", "decision": "ask_info", "reason_code": "missing_evidence"}, "tool": "record_decision"}`.
- Consistent with own check: `True`. Prior actual booked keys: `[["inv-test-135282f-0-1", "payout"]]`.

Retained public memory before this action:

```text
{"case_id":"inv-test-135282f-0-0","amount_cents":99207,"currency":"BRL","status":"hold","decision":"hold","reason_code":"return_rejected","account_id":"acct-test-135282f-0-0","merchant_id":"merchant-test-135282f-0-0","verification_status":"active","order_id":"ord-test-135282f-0-0","invoice_status":"cleared"}
```

Current public observations and the complete check are retained in diagnosis.json and the linked trace. No missing historical fact is automatically classified as a compression defect.

## Provenance and limits

Offline evidence verification: `PASS`. This is consistency verification of recorded artifacts, not independent inference attestation.

Inspector SHA-256: `b178ef9408c3b942117de45ef8969e99eac99f524112729fad0eaa9f11ca2756`. Input-manifest SHA-256: `f985ecfeebdaa801a84b96fc73976ac80347a5fecac66a3902c5472e994a9c29`. Per-file hashes and frozen-source hashes are in diagnosis.json.

- This is post-hoc diagnosis of verified recorded artifacts, not a new confirmatory endpoint or an independent attestation of physical inference.
- A self-reported evidence check may be wrong. Divergence does not distinguish compression loss, extraction, linkage, stale-state resolution, or unsupported inference.
- Reference facts describe all public evidence seen so far, which may exceed the retained memory. A missing fact never supplied is not a compression failure.
- All six fields are compared, including facts that may be irrelevant to this operation or overridden by higher-priority rules. A divergent check need not cause the action error.
- A matching check with a failed action demonstrates inconsistency with the reference rule application, not a hidden-reasoning or neural-mechanism diagnosis.
- The action implied by the claimed check is computed for offline comparison only. It is never executed, fed back, substituted into the trace, or used to rewrite the original score.
- Paired policy decisions share a scheduled task but may have different prior bookings. A learned-correct/summary-incorrect example is not by itself a controlled causal memory experiment.
- Examples are selected deterministically after evaluation; they are illustrations, not unbiased frequency estimates. Both directions of paired disagreement are counted.
