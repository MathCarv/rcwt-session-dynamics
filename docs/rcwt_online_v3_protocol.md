# V3: confirm an engineered memory improvement against the active baseline

Status at registration: **no v3 held-out inference has run**. The v2 result is
negative and remains intact. Its failure diagnosis now informs development;
its test data cannot be reused as fresh confirmation.

Final disposition: **all five development attempts failed the screen**. The
last recorded structured 23/32 versus summary 16/32, but two unsafe attempts
versus one and 86451 versus 27953 fictional cents unsafely booked. Development
is closed for this series; no v3 held-out confirmation was run. See the
[closure](../results/agent_v3_development/RESULTS.md) and
[complete attempt ledger](../results/agent_v3_development/DEVELOPMENT.md).
The design below records the experiment that was prepared, not a claim that
its confirmatory stage occurred.

## Intervention and constants

The revision-3.3 memory intervention changes memory writing **and reading**,
not just the rolling-memory reducer. Both arms now use the **same two-pass
actor**: a short text plan followed by a fixed final-JSON model pass.
The six inherited v2 source files, final schema, extraction, simulator,
business rules, real local model and sampler remain byte-identical. The new
first-pass wrapper removes exactly the original requirement to return JSON
and appends a plain-text planning instruction; all other base text is retained.
Each actor pass has a
512-token generation cap. Both arms carry at most **256
actual model tokens** into the next decision. There are no paid API calls.

- `summary`: the original LLM rolling-summary compressor inside that same
  two-pass actor, including all its calls, tokens and latency.
- `structured`: an engineered deterministic adapter for the public structured
  tool schemas. It retains observed source records, explicit identifier links
  and accepted monetary receipts. At read time it selects retained facts for
  the requested invoice and omits old facts superseded by matching current
  observations. It does not compute a recommended decision, call the grader,
  import current observation values into the read view, infer a missing fact,
  or use the actor's check as evidence.

The structured policy serializes **all** persistent state, including any
identifier compression metadata and lexical replacement legend, into the capped text.
There is no hidden per-case store. On overflow, it evicts whole components by
the frozen rule. It cannot recover previously evicted history. Both writers
run after an action, at the same point in the loop; neither sees future tasks
or private reference facts. The structured writer ignores facts present only
in free text: this is a domain-specific adapter, not general semantic memory.

The structured reader runs before the next action. It decodes only the stored
text, selects by literal requested case ID, and emits a named fact card also
capped at **256 actual model tokens**. Empty stored memory returns an empty
view, preserving identical first-step actor inputs. The writer always receives
the original stored text, not the selected card. All tokenizer calls and read
work are recorded and included in step latency.

Matching current observations are used only to invalidate stale fields, never
to fill the card with newly supplied values. An updated invoice also invalidates
its old account/ownership/return links; account, payment and return updates
invalidate the corresponding old fields by explicit IDs. The shared actor
must extract current values from the unchanged `current_step` payload. Bookings
remain actual retained receipts indexed by invoice and operation.

This reader transfers selection and public-record joins from the LLM to code.
A resulting gain is attributable to the **combined memory system**, not an
isolated proof that serialization alone improved the model or that a learned
policy autonomously emerged. Card overflow fails closed; JSON is never cut.

Agent decisions still come from Qwen3.5-4B, Unsloth Q4_K_M, on the pinned local
llama.cpp runtime. No model weights are updated. The first pass has no output
schema and requests a short plan: case/operation, known and unknown facts,
first applicable rule and proposed decision. Code does not parse the plan or
derive an action from it. The same fixed review message
is appended after the original final-JSON prompt and raw assistant plan, in both
arms. It asks the model to recheck the original evidence and rule precedence;
it receives no oracle, grader feedback, tool result or computed correct action.
The final pass always runs, including for empty or truncated plans. Only the final
model answer is executed, with the unchanged structural validator; no semantic
veto or replacement action is introduced. The first plan remains in the raw
trace but is not an extra executed task or persistent memory. It is never
passed to the memory writer. Both calls are charged to decision
tokens and latency. The summary arm additionally pays its original compaction
calls. This is failure-informed engineering, **not autonomous recursive
self-improvement**.

This is a new controlled comparison conditional on a shared two-pass actor,
not a claim that the single-pass v2 workflow stayed unchanged. Any final gain
is measured against the two-pass summary arm, not against an earlier result.
An ablation proving that self-review itself helps is not included.

## Development and held-out separation

The development screen uses four episodes (one per family), eight decisions
each, in both arms: dataset seed `2026091201`, split `train`. The candidate must
have more correct decisions and no more observed unsafe attempts or unsafe
fictional cents booked before confirmation is allowed. A development screen
is not held-out proof. If development requires another code revision, preserve
its failed attempt and source snapshot; do not relabel it as confirmation.

The one fixed confirmatory run uses **32 new episodes**, eight per family and
eight sequential decisions each, in exactly two arms: **512 decisions, 32
paired statistical units**. Dataset seed `2026091202`, split `test`, is distinct
from the previously observed v2 seed and development. The new cases use the
same generator and task families, not unseen domains or production traffic.
No tail arm is needed: the primary comparator is the stronger active summary.
With two actor passes per decision, the fixed cohort schedules 1,024 actor
generations plus 224 summary-compaction generations: **1,248 model calls**.
The four-episode development cohort schedules 156 calls. Tokenization requests
are recorded separately and are not counted as generation calls.

Before any confirmatory call, freeze source hashes, public and private corpus
hashes, the full schedule, the runtime receipt, analysis configuration and the
verified development package. The six inherited source hashes must equal v2's
frozen hashes. New modules do not match v2's source glob and cannot invalidate
its historical replay.

Inference seed stays `20260911` in both arms. Schedule seed is `2026091204`:
the initial arm order is randomized per family and alternates on subsequent
instances of that family. Every family has four episodes in each order. There
is one inference trajectory per episode/arm, not a multi-inference-seed study.

## Predeclared endpoints

The primary endpoint is the mean paired episode difference in exact-action
accuracy: `structured - summary`. Identity, decision, exact amount and reason
must all match the private reference under that arm's actual simulated ledger.

The accuracy criterion requires **both**:

1. mean difference of at least **+10 percentage points**;
2. lower endpoint of the two-sided **95% paired episode bootstrap interval
   strictly above zero**.

Use 10,000 percentile resamples, analysis seed `2026091203`. Resample paired
episodes, never individual correlated decisions. Family-level comparisons are
descriptive and not additional confirmatory hypotheses.

Separately require no increase in aggregate unsafe monetary attempts **and**
unsafe fictional cents actually booked. These are descriptive safeguards, not
a statistical non-inferiority test or safety certification. Report any failed
guard even if accuracy improves; lower totals can conceal case-specific harm.

Record prompt/completion tokens, number of calls, decision-only and full-step
latency p50/p95, entire-episode success, invalid actions, failure categories,
memory eviction/truncation and fictional ledger effects. Per-step cost includes
compaction and tokenization; no model calls are hidden by the deterministic
policy. API charge is US$0, but electricity, hardware and total monetary cost
are unknown. Report development costs separately from held-out results.

## Stopping, integrity and disclosure

No early success stopping, test resizing, optional additional test seeds,
replacement of unfavorable episodes, or candidate edits during confirmation.
Report the entire scheduled cohort, including regressions and invalid outputs.
If the gate fails, that run does not prove the requested gain.

The first completed development attempt, `v3_development_01`, failed its
screen: both policies made 14/32 correct decisions; structured made five unsafe
attempts versus four for summary. Its source snapshot and traces are retained.
Post-hoc training inspection found the required facts recoverable in all 32
structured inputs, but 12 self-reported amounts matched another retained
invoice. This motivated removing numeric identifier indirection and grouping
textual invoice identities with their facts. It is a development diagnosis,
not a causal neural explanation or held-out improvement.

The second attempt, `v3_development_02`, used named per-invoice records with
public ownership joins. It also failed the screen: structured 15/32 versus
summary 14/32, but ten unsafe attempts versus four. The model still mixed
amounts between retained invoices and sometimes preferred stale memory over a
current account update. This motivated the additional query-conditioned reader
and stale-field invalidation in revision 3.1. Both completed failed attempts
remain disclosed, and no held-out data was used to design this revision.

The third attempt, `v3_development_03` (revision 3.1), also failed: structured
19/32 versus summary 14/32, but five unsafe attempts versus four and 163684
versus 101840 fictional cents unsafely booked. Structured self-reports matched
all six reference fields in 22/32 decisions, versus 11/32 for summary. However,
all five structured unsafe decisions contradicted their own self-reported
evidence; four checks were entirely correct. Two unsafe attempts were rejected
duplicates and three were booked. These are descriptive training diagnostics,
not evidence of held-out gain or a hidden reasoning mechanism.

This motivated revision 3.2's symmetric JSON-draft/JSON-review actor, not
another change in the memory writer or reader. The fourth complete development
attempt also failed: structured 20/32 versus summary 17/32, four versus two
unsafe attempts and 135731 versus 73887 fictional cents unsafely booked. The
review repeated unsafe decisions despite correctly reported evidence.

Revision 3.3 replaces the JSON draft with a short free-text plan in both arms,
keeping two 512-token passes and the same final review and business rules.
The fifth full development attempt is the **last development revision in this
series**. If it fails the unchanged screen, stop this series with a negative
result; do not spend the confirmatory cohort or introduce a deterministic
decision replacement to manufacture a passing result. If it passes, run the
one originally fixed confirmation unchanged. All previous attempts and costs
remain disclosed. No held-out corpus or inference was used to design these
revisions. Repeated development can overfit those four known episodes;
passing development is not independent evidence.

Write an exclusive start marker before any inference. A run that aborts even
before the first complete step cannot silently restart in the same directory.
Preserve partial events and disclose potentially unmetered in-flight calls.

Record raw model requests/completions and the exact tokenization operations.
Offline verification re-executes the reducer and simulator with a strict replay
client, checks every model input and output binding, consumes all operations in
order, and reconstructs every grade and aggregate counter. Token IDs and wall
times remain recorded server claims, not independent hardware attestation.

Reproduction after the pinned local runtime is ready:

```powershell
python src/rcwt_online_v3.py --mode development --output-dir .runs/my_v3_development
python src/rcwt_online_v3.py --mode confirmatory --dataset-seed 2026091202 --development-run .runs/my_v3_development --output-dir .runs/my_v3_confirmation
```

The second command refuses a failed development screen. Both commands require
new nonexistent directories. Offline replay uses `--stage verify` or `--stage
report` with the recorded output directory and makes no inference calls.
These fixed seeds reproduce this experiment; rerunning them is not another
independent unseen test or an additional confirmation hypothesis.

When a later development revision changes the current source, replay the
earlier **trusted project** archive using its saved sources, without changing
its evidence:

```powershell
python tools/verify_online_v3_archive.py --run-dir .runs/v3_development_01
```

This helper executes the project's archived Python exclusively in verification
mode. It is not intended for untrusted or downloaded source bundles.

Any final claim must say **gain on new instances of this synthetic local-agent
benchmark**. It cannot claim production readiness, general model improvement,
autonomous research capability, or a demonstrated CloudWalk customer outcome.
