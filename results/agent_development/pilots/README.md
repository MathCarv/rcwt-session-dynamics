# Metered development pilots

These four receipts preserve exploratory configuration work **outside the final benchmark**. They use synthetic training cases or literal JSON-formatting fixtures, not test/held-out cases. They are not independent validation data, a memory-policy comparison, or evidence of production readiness.

## Known metered lower bound

| Receipt | Calls | Input tokens | Output tokens | Summed call wall time |
| --- | ---: | ---: | ---: | ---: |
| [Action/schema diagnostics](action-schema-diagnostic-receipt.json) | 10 | 5,404 | 473 | 11.5239 s |
| [Qwen3 thinking pilot](thinking-pilot-receipt.json) | 6 | 5,443 | 2,984 | 56.4406 s |
| [Qwen3.5 direct-action pilot](qwen35-pilot-receipt.json) | 6 | 5,521 | 371 | 12.5344 s |
| [Qwen3.5 evidence-before-action pilot](evidence-action-pilot-receipt.json) | 6 | 5,827 | 836 | 27.9124 s |
| **Included subtotal** | **28** | **22,195** | **4,664** | **108.4113 s** |

The subtotal is 26,859 tokens. Output counts include server-accounted hidden reasoning in the thinking pilot; final visible text alone cannot reproduce that count. Wall time is the sum of per-call measurements, not total elapsed development time. [index.json](index.json) contains unrounded values, source/published SHA256 hashes, byte sizes and exact metering field paths.

All inference was local through `http://127.0.0.1:18085`; API-provider charges were **US$ 0**. This does not imply zero total cost: electricity, hardware depreciation and human time were not measured. Runtime setup, downloads, warm-up and any other unreceipted probes have unknown cost/duration.

Four additional non-greedy diagnostics made by the campaign coordinator are **not metered in this package**. The aborted greedy development campaign has separate accounting and is not included here. Do not mistake this subtotal for the full development cost, or pool these calls or scores into final benchmark results.

## What the pilots show

- Action/schema diagnostics: three literal fixtures returned the requested `ask_info` object, while several realistic training prompts still produced unsupported approvals. The JSON grammar did not universally force the `approve` enum. This is a configuration diagnosis, not a task-success benchmark.
- Qwen3 with a 512-token reasoning budget: 4/6 training probes passed, including 3/4 initial steps. The pending-payment probe failed.
- Qwen3.5 direct-action, non-thinking: 3/6 passed, including 2/4 initial steps. Missing-evidence and pending-payment failures remained.
- Qwen3.5 evidence-before-action, non-thinking: 5/6 passed, including all four initial steps. The pending-payment probe still produced an unsafe approval.

These are small, adaptively inspected development probes with changed model/prompt configurations. Their fractions are not a causal estimate of any single change and do not demonstrate a memory gain or generalization. The last interface was chosen for subsequent evaluation; that evaluation must report its own frozen protocol and results.

The four initial probes use current public observations with empty retained memory. The pending-payment probe uses the public current step. The revoked-account probe also supplies preceding public observations to preserve invoice/account identity; it is not an empty-memory test. Expected actions stored under `evaluation_only_not_in_request` or `score` are simulator grading evidence, separate from the captured model requests. No hidden oracle labels or future observations were supplied to the model.

## Privacy, provenance and integrity

Only the absolute runtime-directory prefix was replaced with `<LOCAL_RUNTIME>`: 5 metadata values in the action/schema receipt, 9 in the thinking receipt, 6 in the Qwen3.5 receipt and none in the evidence-before-action receipt. The exact JSON paths are listed in the index. No credentials were found in the inspected source receipts. No fields were dropped, and all records (requests, generated final responses, scores, token usage and timing) were checked for semantic equality with the originals. Formatting/line endings may differ; the original hashes identify the untouched local inputs.

Raw hidden reasoning was not present in the inspected originals and is not published. Existing reasoning hashes and character counts remain; the public `evidence_check` object is the requested final evidence record, not a hidden reasoning transcript.

Qwen3 used the Qwen-published GGUF. Qwen3.5 used an **Unsloth third-party quantization** of the Qwen base model. Its receipt pins the quantized revision/hash and reports the base revision observed during setup; it does **not** attest the exact source revision used for conversion. Runtime/model provenance is retained where embedded in the original receipt; the evidence-before-action receipt has no independent embedded installation/server snapshot.

Financial actions are confined to the local fictional simulator. There is no execution against CloudWalk or another external financial system. Packaging these receipts made no inference calls and modified no original receipts. File hashes prove artifact identity/integrity, not independent attestation of model execution.
