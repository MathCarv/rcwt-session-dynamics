# Online agent evaluation — local-only protocol

This is a separate experiment from the deterministic `session_v1` artifact.
Its unit is an eight-step episode in a fictional financial workflow. No real
payment service is reachable, and no paid inference API is used.

## Question and hypotheses

Can a failure-informed rolling-memory instruction improve an actual local
model's exact tool decisions on held-out instances, compared with a generic
rolling summary using the same model and memory cap?

Primary endpoint: the paired difference in per-episode exact-action success
between the validation-selected policy and `summary`. The success gate is a
strictly positive lower endpoint of a 5,000-sample, seed-fixed, 95% paired
episode bootstrap interval, and an actual change from the baseline instruction.
Comparison with `tail` is secondary. This is an accuracy gate, not a claim of
better safety, latency, efficiency, or general recursive self-improvement.

## Fixed design

- Model/runtime: Qwen3.5-4B, using the
  [Unsloth-published Q4_K_M GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
  and llama.cpp. This is a third-party quantization of the
  [Qwen model](https://huggingface.co/Qwen/Qwen3.5-4B), not an official Qwen GGUF.
  Model and runtime revisions and SHA-256 values are pinned in the
  [runtime receipt](rcwt_agent_runtime.json). Local loopback only, one slot,
  prompt cache disabled. All actor, compressor and policy-proposal calls use
  non-thinking mode and the same seeded sampler: temperature 0.7, top_p 0.8,
  top_k 20, min_p 0, presence penalty 1.5 and repetition penalty 1.0
  (`repeat_penalty` in the llama.cpp request). These are the model card's
  [general-task non-thinking settings](https://huggingface.co/Qwen/Qwen3.5-4B#best-practices).
  `chat_template_kwargs.enable_thinking=false` and a runtime reasoning budget
  of zero are explicit; a `/no_think` text suffix is not the mode switch.
- Actor output: at most **512 generated tokens** for a single evidence-and-action
  envelope. This is a generation limit, not an increase in retained memory.
- Memory: at most 256 tokens measured with the running model's tokenizer.
  New observations and public business rules are outside the storage cap and
  identical across arms. All inference tokens, including these inputs and the
  memory compressor's calls, are charged to the measured workload. The compressor
  may generate up to 320 tokens (budget + 64), but retained text is re-tokenized
  and explicitly truncated to at most 256. The policy-proposal limit is 384
  generated tokens and its cost is separate from evaluation-only totals.
- Four families: updated state, cross-source evidence chains, return to an
  earlier topic, and insufficient evidence. Seed-dependent variations include
  amounts, identities, source arrival order and decision outcomes.
- Train: 4 episodes (32 decisions), generic-summary baseline only.
- Validation: 4 disjoint episodes, 3 policies (96 decisions).
- Test: 16 disjoint episodes, 3 arms (384 decisions). If validation retains
  the baseline, `learned` is a repeated summary arm, not a third distinct policy.
- Split namespaces and 128-bit seed partitions prevent shared case identifiers.
  The same family distribution is used across splits. Test novelty means new
  instances and combinations, **not unseen families or production distribution**.
- Monetary actions affect only an in-memory simulator. Its ledger persists
  across steps; later idempotency scoring reflects what the agent actually did,
  including earlier mistakes, rather than an imaginary perfect trajectory.
- The fixture supplies the observation schedule. The actor chooses the
  `record_decision` action; it does not plan which external sources to query.
  `ask_info` records a deferral, not an autonomous information-gathering call.
- An exact success requires the correct case, decision, integer amount and
  reason. Invalid envelopes, invalid actions and length-truncated model outputs
  are failed tool decisions. No oracle repair, retry, hidden retrieval, or mock
  fallback is allowed. Actor architecture, evidence-check schema, extraction,
  sampling and generation limits are identical across all memory arms.

The prepared run's `protocol.json`, runtime receipt and frozen source hashes are
the executable configuration record. They must agree before starting a new
held-out run; a document edit is not permission to change a frozen experiment.

## Public evidence-and-action envelope

The actor is instructed to produce a JSON object with `evidence_check` first,
followed by the original `tool` and `arguments` fields at the same level.
The check is a short, explicit inventory of the evidence the actor believes it
has for the requested case, not an oracle-provided state or a hidden-reasoning
transcript. It uses only the current public observations and retained memory.

| `evidence_check` field | Allowed representation | What it refers to |
| --- | --- | --- |
| `invoice_amount_cents` | Nonnegative integer or `null` | Exact invoice amount, or unknown |
| `account_status` | `unknown`, `active`, `blocked`, `revoked` | Latest status for the linked account |
| `ownership_match` | `unknown`, `yes`, `no` | Whether account holder matches invoice merchant |
| `payment_status` | `unknown`, `pending`, `cleared` | Latest status of the linked payment |
| `return_status` | `unknown`, `accepted`, `rejected` | Latest linked return status, when required |
| `operation_already_booked` | `no_record`, `yes` | Evidence of an accepted booking for this invoice and operation |

`unknown`, `null`, and `no_record` preserve missing evidence; they do not certify
that an external fact is false. A prior proposed action is not proof that it was
booked, and a booking receipt is not proof that it was authorized. Payout and
refund maintain separate idempotency keys under the public rules.

The strict extractor requires that top-level field order, rejects duplicate
keys and non-finite JSON numbers, and validates shape and field types/enums,
then serializes **only the model's unchanged `tool` and `arguments`** as a
canonical JSON action. It neither executes `evidence_check` nor derives, fixes,
or overrides a decision from it. If the actor recognizes a pending payment but
still approves, that action is executed by the fictional tool and scored as
proposed; the host does not rescue it. Canonical serialization changes JSON
formatting, not argument values. Invalid or truncated envelopes produce a
non-executable failure sentinel, not a guessed or corrected action.

The trace retains the raw model output, extracted action and `evidence_check`.
For an invalid envelope, the extracted check is `null`; the raw output remains
available for inspection.
Offline verification binds extraction to that raw output. The check's presence
and schema validity do **not** establish that its contents are true or that a
failure was caused by memory. Primary scoring remains the executed tool action.
Neither the actor nor the rolling compressor receives a private correctness
verdict or a corrected check during execution.

## Cumulative memory and learning boundary

Each acting call sees public rules, previously retained memory, and only the
current step's observations and task. After executing the JSON tool call, the
memory compressor sees that same prior memory plus current observations,
completed request, extracted actual action and public tool receipt. The actor's
evidence inventory is recorded for diagnosis, not used as an oracle-corrected
memory update. The compressor does not receive
the private score, expected action, simulator's private state or future tasks.
Discarded history never reappears. The final step does not pay for unused memory.

Policies:

1. `tail`: keep the newest tokens of prior memory plus the public interaction.
2. `summary`: a generic LLM-generated rolling summary, an active baseline rather
   than only a deliberately weak truncation baseline.
3. `learned`: an instruction selected on validation, frozen before test. Search
   compares the unchanged summary against (a) one template selected by training
   failure categories and (b) a fresh local-model instruction proposed using
   only aggregated training failures. These are prompt/policy changes, not
   model-weight updates. All three candidates and the proposal call are saved.

Selection is deterministic: highest validation episode-macro success, then
fewest unsafe monetary attempts, then fewest completion tokens, then name.
If the generic summary wins, it is retained and **no policy improvement is
claimed**, even if its duplicate test arm varies numerically.

The test freeze records source, protocol, corpus, selected policy and schedule
hashes before any test inference. An existing test directory cannot be rerun or
overwritten by the runner. A later exploratory change requires a distinct run,
and must not silently replace the confirmatory result.

Model choice, sampler and actor-envelope changes made during training-only
development are shared engineering changes, not the learned memory treatment.
Only the frozen memory-policy comparison within one model/actor configuration
can support the stated memory-gain endpoint.

## Cost, latency and uncertainty

Report real prompt and generated token counts from llama.cpp, model-call counts,
decision latency, end-to-end step latency including memory compaction, episode
wall time, and one-time training/search overhead. API charge is US$0. Electricity,
hardware depreciation and total monetary cost are unknown, not zero.

Episode execution order is a seed-shuffled initial permutation cyclically
rotated across episodes. This balances policy position without using outcomes.
Latency remains sensitive to concurrent desktop load, thermals and hardware.

Bootstrap the 16 paired episodes, not 384 decisions as independent samples.
Family results are descriptive; intervals do not imply transfer to other
families, models, raw customer conversations or stochastic inference runs.
One sampled trajectory per episode/arm does not estimate across-seed inference
variance. With only four validation episodes, policy selection is particularly
noisy; retaining the summary baseline is a valid outcome.

## Running and auditing

Start the pinned llama.cpp server on `127.0.0.1:18085`, alias
`rcwt-local-qwen35-4b`, with the launch settings in the receipt. Model storage,
context/cache buffers and GPU working memory are distinct resource requirements;
downloads are not inference API calls.
No credentials or remote model code are required.

```bash
PYTHONPATH=src python src/rcwt_agent_run.py --stage all \
  --model rcwt-local-qwen35-4b \
  --runtime-receipt docs/rcwt_agent_runtime.json \
  --output-dir .runs/agent_v2
```

PowerShell:

```powershell
$env:PYTHONPATH='src'
python src/rcwt_agent_run.py --stage all --model rcwt-local-qwen35-4b --runtime-receipt docs/rcwt_agent_runtime.json --output-dir .runs/agent_v2
```

The stages can also be run separately with `--stage train`, `validation`, `test`
and `report`, always using the same output directory. `train` requires a new
directory. `verify` and `report` are offline: they replay recorded actions and
check source/corpus/selection/trace bindings without rerunning inference.
The actor's public inputs, evidence envelope and extracted action are checked
against the recorded completion; the compressor's inputs are checked against
the retained memory and actual public interaction. These checks detect artifact
inconsistency; they do not independently re-run inference or prove that a
self-reported evidence field was read correctly by the model.
Reported token counts are sealed observations, not an offline re-tokenization
or cryptographic attestation of the physical model that ran.

All fixtures and traces are fictional. Unit tests use fake clients only to test
contracts; those calls never enter reported real-model results.

## Development disclosure

An initial Qwen3-4B greedy-decoding run was stopped during validation, before any
test call or test freeze. It obtained 6/32 training decisions, with repetitive
compactor outputs and failures even before compaction. The original sources,
traces and [abort receipt](../results/agent_development/ABORTED.json) are preserved
separately. That archive records 32 training and 75 validation decisions, not a
completed confirmatory experiment. Its planned test corpus is not evidence of
test execution. Recorded development cost is a lower bound: the receipt states
that an interrupted call and subsequent sampler diagnostics are not fully
metered into its totals.

Subsequent training-only checks informed selection of Qwen3.5-4B and the explicit
evidence envelope. A small development check is a feasibility diagnostic, not a
held-out accuracy estimate. At this protocol revision no held-out decision has
been executed. The new campaign repeats training and validation with one shared
model, actor and sampler before freezing the test arm. Earlier model/configuration
results remain separate; their apparent improvement is not called a memory-policy
gain and is not pooled into the new endpoint.

## Diagnostic interpretation, not an extra success gate

Failure categories describe the scorer's first action mismatch. They are not a
causal taxonomy: `wrong_amount` can follow loss of an earlier amount, a wrong
link between cases, or incorrect use of an amount already present. The visible
evidence inventory permits a more specific trace inspection without changing
the acting agent:

- Compare the relevant historical public observation with `memory_before` and
  current observations: was the required fact ever supplied, and is it still
  available to the actor? A fact never supplied cannot have been lost by the
  compressor.
- Compare that available evidence with `evidence_check`: a wrong check despite
  sufficient input is consistent with extraction, linkage or update-resolution
  error; it is not demonstrated memory loss.
- Compare a correct evidence inventory with the extracted action under the
  public rules: a disagreement isolates an observable rule-application or
  action-selection mismatch, without automatically repairing it.
- Track prior actual bookings: one early mistake changes later idempotency
  expectations. Downstream steps are a trajectory, not independent failures.

These are descriptive diagnostics. Counterfactual replay with restored evidence,
or an additional full-history control, would be a separate exploratory experiment
with its own cost and disclosure, not a post-hoc modification of this test. The
current primary analysis does not automatically assign these causal labels.
