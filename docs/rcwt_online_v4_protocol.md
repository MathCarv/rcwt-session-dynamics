# RCWT-S v4: fixed local evidence-compilation experiment

Status at registration: no v4 model inference or real development/test corpus
has been run or inspected. This new series follows the user's explicit request
to continue after the negative v3 closure. All five v3 attempts, their costs,
and their failed safety screens remain historical evidence. This does not
relabel those attempts as a successful experiment.

## Hypothesis and intervention

Failure inspection on the completed v3 training traces found current payment
facts omitted from the old retained-only card and confusion across invoice
identities. It also found errors despite fully correct evidence. The hypothesis
is that compiling retained and current public facts into one case-specific card
can improve a local agent's decisions. Improvement is not assumed.

The candidate keeps the frozen v3 deterministic structured writer. Before each
decision, its new reader overlays current structured public tool records onto a
copy decoded from the retained text. The last same-source/same-ID record
replaces the whole row, including missing fields, before literal invoice joins
are resolved. The card includes explicit unknowns, links and six factual
fields. It selects no action or reason, reads no private oracle or future step,
and does not infer absent records. Only retained actual receipts establish a
previous booking. The card and stored text each fit 256 actual model tokens;
only stored text persists. The writer never receives the transient card or
free-text plan as additional persistent state.

Both arms receive the same raw current step and public business rules. The
summary arm keeps the frozen LLM summary reducer. The candidate additionally
normalizes current observations, so actor inputs can differ even at step zero.
This comparison concerns the combined engineered memory-and-context system,
not memory compression alone, autonomous learning, or intrinsic model gains.

Both arms use the same v3.3 free-text-plan then final-JSON workflow, with 512
tokens per pass. A common new output schema binds only the public literal case
ID and operation-compatible decision/reason vocabulary. It never reads facts,
memory, self-checks or reference actions, and does not constrain amounts from
evidence. Semantically unsafe actions remain possible. Only the final model
answer executes. There is no semantic veto, action repair, selective retry,
fallback or oracle feedback. This shared schema differs from the v3 baseline;
v4 must be compared with its contemporaneous summary arm, not an old run.

## Fixed cohorts and selection

- Development: 8 paired episodes, two per family, 8 steps per arm, split
  `train`, dataset seed `2026091205`: 128 executed decisions and 312 generation
  calls (128 structured, 184 summary).
- Confirmation: one cohort of 32 new paired episodes, eight per family, 8
  steps per arm, split `test`, dataset seed `2026091206`: 512 decisions and
  1,248 generation calls. The 32 paired episodes are the statistical units.
- Inference seed: `20260911`. Schedule seed: `2026091208`, with arm order
  balanced within each family. Analysis bootstrap seed: `2026091207`.
- At most two development candidate versions in this series. A second version
  is allowed only after diagnosing the first completed development screen.
  Preserve both outcomes and source snapshots. Do not select or revise using
  confirmation outcomes. The first candidate passing development is frozen
  for the single confirmation; do not search further for a preferred score.

Development requires strictly more exact actions than summary, no increase in
unsafe monetary attempts and no increase in unsafe fictional cents actually
booked. It is selection evidence, not independent proof. Confirmation is not
generated or inspected until a complete verified development run passes.
Fresh instances share the four synthetic task families and generator; they
are not unseen domains, real customer data or CloudWalk transactions.

## Predeclared confirmation endpoints

The primary endpoint is the mean paired episode difference in exact-action
accuracy, structured minus summary. Case ID, decision, exact amount and reason
must all match the private reference under that arm's actual simulated ledger.
Require both a gain of at least 10 percentage points and a strictly positive
lower endpoint of a two-sided 95% paired-episode percentile bootstrap interval,
using 10,000 resamples. Do not resample correlated steps as independent units.

Separately require no increase in aggregate unsafe attempts and unsafe
fictional cents actually booked. These are descriptive guards, not a
statistical noninferiority test, zero-risk certificate or production clearance.
Report failed guards even if accuracy improves. Per-family results and
case-specific regressions are descriptive, not extra confirmatory hypotheses.

Report exact accuracy, entire-episode success, invalid actions, failure types,
prompt/completion tokens, generation calls, compaction, memory truncations,
decision and full-step p50/p95 latency, and total episode wall time. Both actor
passes count. Tokenizer requests and deterministic context work count in step
latency. Development costs are separate from confirmation costs.

## Runtime, integrity and stopping

Use the existing pinned local llama.cpp runtime and Qwen3.5-4B Unsloth Q4_K_M
weights, as recorded in `docs/rcwt_agent_runtime.json`. This is a third-party
quantization, not an official Qwen GGUF. No paid model API or weight update is
permitted. API charge is US$0; electricity, hardware depreciation and total
monetary cost are unknown, not zero.

Freeze all 15 source hashes, protocol, corpus, schedule and runtime before
inference. The six v2 and eleven v3 source hashes must remain unchanged. Bind
the negative v3 closure and development index. Before confirmation, copy and
bind the complete verified development run. Record raw requests, completions,
tokenization, action effects, grades and hash-chained traces. Strict offline
replay must consume every event and reconstruct all inputs and counters.
Recorded token IDs and timings are server claims, not hardware attestation.

No early success stopping, cohort resizing, replacement episodes, repeated
confirmation seeds, edits during a run, or hidden unfavorable outcomes. An
exclusive start marker prevents silent restarts. Preserve partial/aborted runs
and disclose possibly unmetered in-flight calls. If the fixed gates fail, report
that gain was not demonstrated; do not change the criterion after seeing it.

```powershell
python src/rcwt_online_v4.py --mode development --output-dir .runs/v4_development_01
python src/rcwt_online_v4.py --mode confirmatory --dataset-seed 2026091206 --development-run .runs/v4_development_01 --output-dir .runs/v4_confirmation
```

Offline verification uses `--stage verify`; it makes no inference calls.
These seeds reproduce this experiment, not another independent confirmation.
No publication or email is part of this run. Any favorable final claim must be
limited to measured gain on new instances of this synthetic local benchmark.
