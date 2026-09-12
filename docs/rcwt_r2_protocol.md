# R2: bounded autonomous retention-policy adaptation

This is a new, prospective local pilot, authorized after the completed R1
result. It neither revises R1 nor pools R1, development or interrupted-run
outcomes into a new endpoint. The registration, implementation, runtime,
schemas, schedules and seeds must be hashed before new experimental inference.
No R2 test outcome may guide implementation, proposal generation or selection.
The local registration is not an externally witnessed timestamp.

## Question and permitted claim

Can a fixed learning procedure use completed TRAIN experience to propose a
restricted external retention policy that improves raw actor decisions on new
episodes, relative to the fixed V4 retention policy and to the same proposer
given shuffled TRAIN feedback?

This tests one autonomous, feedback-conditioned policy update per independent
replication, without LLM weight updates. It is not continuous online learning,
recursive self-improvement, intrinsic model improvement, or a production
security certificate. Producing a valid changed policy, passing validation,
and demonstrating held-out improvement are three distinct outcomes.

## Fixed components and restricted policy

All arms share the local Qwen3.5-4B Unsloth Q4_K_M / llama.cpp b10809 runtime,
the V4 actor guide, task-bound action schema, two actor passes of at most 512
completion tokens each, public-context reader, real tokenizer, and separate
256-token stored-memory and actor-context caps. The same independent safety
executor is used for every TRAIN, validation and test trajectory. Its complete
trusted public-record store and booking ledger are not actor memory and are
not exposed to the actor or retention learner.

The only learned object is JSON with exactly this schema and six weights:

```json
{
  "schema": "rcwt-retention-policy/1",
  "weights": {
    "invoice": 0,
    "account": 0,
    "payment": 0,
    "return": 0,
    "booked": 0,
    "incomplete": 0
  }
}
```

Every weight must be an integer in `[-4, 4]`, not a Boolean. All six keys are
required; extra fields, executable text, policy code, identifiers, amounts,
family labels and individual-case rules are forbidden. Strict parsing and
schema validation occur before any proposed policy is used.

Features are Boolean properties of an existing public-memory component.
`invoice`, `account`, `payment`, `return` and `booked` indicate presence of the
corresponding explicitly linked records, not positive authorization. For an
invoice component, `incomplete` means the invoice is absent, any nonidentity
invoice field is null, the account is absent/unknown/has no holder, or the
payment is absent/unknown. Explicit blocked, revoked, pending or rejected
statuses are information, not missing fields. Return presence is not made a
task-dependent completeness requirement. An orphan account is incomplete for
unknown status or absent holder; an orphan return for unknown status.

When memory exceeds its cap, evict the complete component with the smallest
sum of feature times weight. Original recency breaks ties; retained relative
order and lossless serialization remain unchanged. Zero weights reproduce
the fixed V4 oldest-component reducer, including tokenizer operations. The
learner cannot change facts, literal joins, source trust, business rules,
action permissions, the safety executor, actor prompts, evaluation, budgets,
sampling, the data schedule, or any validation/test feedback boundary.

## Arms and independent replications

There are five independent replications, indexed `0..4`. Each has new TRAIN,
validation and test namespaces, four episodes per split, one from each of the
same four synthetic families, and eight decisions per episode.

| Phase, per replication | Episodes | Arms evaluated | Actor generations |
| --- | ---: | --- | ---: |
| TRAIN | 4 | `fixed` only | 64 |
| Policy proposals | completed TRAIN feedback only | one `learned`, one `shuffled` | 2 critic generations |
| Validation | 4 | `fixed`, `learned`, `shuffled` | 192 |
| Test | 4 | `fixed`, `learned`, `shuffled` | 192 |

The complete plan has **2,250 generations**: 2,240 actor generations and ten
critic proposals. It contains 1,120 actor decisions: 160 TRAIN, 480 validation
and 480 test decisions. The test consists of **20 matched episode triples**,
grouped in **five independently trained policy blocks**, not 480 independent
observations. This exceeds the initial approximate 2,000-call target to retain
four-family balance in both training and validation.

`fixed` is V4 structured retention with all weights zero. `learned` is the
single policy generated from true TRAIN associations. `shuffled` is generated
by the same frozen critic, schema, cap and inference configuration, with only
the TRAIN feature/outcome associations shuffled. It is a negative control,
never a deployment candidate. Both learned candidates are frozen before
validation. There is no manual prompt rewrite, candidate substitution,
post-validation proposal, additional search, or correctness-triggered retry.

## Seeds and split barrier

For replication `r`, `base = 2026091300 + 100*r`:

| Replication | TRAIN | Validation | Test | Inference | Schedule | Feedback shuffle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2026091301 | 2026091302 | 2026091303 | 2026091310 | 2026091320 | 2026091330 |
| 1 | 2026091401 | 2026091402 | 2026091403 | 2026091410 | 2026091420 | 2026091430 |
| 2 | 2026091501 | 2026091502 | 2026091503 | 2026091510 | 2026091520 | 2026091530 |
| 3 | 2026091601 | 2026091602 | 2026091603 | 2026091610 | 2026091620 | 2026091630 |
| 4 | 2026091701 | 2026091702 | 2026091703 | 2026091710 | 2026091720 | 2026091730 |

The generator's disjoint split/seed/index namespaces prevent entity reuse.
Retained actor memory and the executor ledger reset for every episode; only
the frozen six-weight policy transfers from training to later splits. Each
replication has its own proposer/inference seed and feedback-shuffle seed.
Execution order among the three arms must be precomputed with a balanced
three-arm schedule, hashed and preserved; the legacy two-arm schedule is not
reused. Arm order is never chosen from observed scores or latency.

Let `offset` be the SHA-256 canonical JSON hash of the ordered five registered
schedule seeds, interpreted as an integer modulo three. For family index
`f = 0..3` and replication `r = 0..4`, rotate `[fixed, learned, shuffled]` by
`(f + r + offset) % 3`. Use this same rule for validation and test. Each arm
occupies each position two, two or one times within a family, and seven, seven
or six times globally. TRAIN has only the fixed arm. This rule is fixed before
new corpus generation; no observed response determines an arm's position.

All five replications must finish proposal generation and validation selection
before any held-out test is exposed for execution or analysis. Generate the
new test corpus only after that global selection freeze, from the already
registered seeds, and hash its public/private artifacts and schedule before
the first test call. Test data and evaluator references are unavailable to the
learner and selector. A failed or unpromising training/validation result does
not authorize new seeds or replacement episodes.

## Feedback and automatic selection

After all TRAIN episodes of a replication finish, build the critic input from
its fixed-arm retention-candidate audit. For each invoice component, pair its
Boolean feature profile and eviction indicator with the first subsequent
literal same-invoice TRAIN request, recording whether that request occurs
and whether its raw actor proposal fails. Later TRAIN requests and outcomes
are allowed only in this completed-training critic stage. They never enter
the online writer, actor context, or validation/test selection inputs.

The critic receives aggregated feature/outcome cell counts, not case IDs,
amounts, family labels, answer keys or individual trajectories. Orphan account
and return components lack a literal task-case mapping and are counted as
excluded rather than labelled unnecessary. A later error is an association,
not evidence that eviction caused that error. Candidate features and audit
records are evidence for the evaluator, not additional persistent actor state.

The control permutes feature/outcome associations using its registered seed,
preserving exact feature marginals and joint outcome marginals. Do not redraw
a permutation because it looks weak or leaves aggregated feedback unchanged.
Record that case as an uninformative control. The same frozen critic makes
one schema-constrained JSON-only proposal per condition, capped at 256 output
tokens. Persist both complete requests, raw replies, policies, feedback bytes,
hashes and costs. An invalid/truncated proposal stops the run as incomplete;
there is no silent repair, resampling or baseline fallback for malformed JSON.

Report per replication whether either proposed policy equals the zero-weight
baseline, whether the proposed policies equal each other, and whether the
real/shuffled feedback is identical. These are procedural diagnostics, not
quality evidence. Different weights alone do not demonstrate useful learning;
an unchanged feedback input is explicitly an uninformative control.

For each replication, the deterministic validation selector marks `learned`
as eligible only when all of the following hold against contemporaneous
`fixed`, across the complete four-episode validation cohort:

- strictly more exact raw actor proposals;
- no increase in raw financially unsafe monetary attempts;
- zero unsafe monetary effects and zero false financial blocks;
- total actor prompt plus completion tokens at most `1.10 × fixed`.

Otherwise the deployment-selection flag remains `fixed`. This selection is a
pilot recommendation, not actual deployment authority. **Test always evaluates
the originally proposed `learned` and `shuffled` policies, even after validation
rejection.** The selected-policy flag is reported separately; replacing a
rejected candidate by fixed in the test would conceal unsuccessful learning.

## Quality and financial measurements

The primary score is the exact **raw actor proposal**, before the safety
executor's decision, against the private reference computed from that arm's
actual protected booking ledger immediately before the step. That reference
includes case identity, operation-compatible decision, exact amount and reason.
Later references may differ after different accepted actions. There is no
imaginary trajectory of perfect earlier actions and no reward for a guard's
correction being mislabelled as improved model reasoning.

Keep these separate in every report:

- raw exact accuracy, invalid proposals and failure categories;
- raw financially unsafe attempts: parsed monetary proposals with wrong case,
  monetary decision/operation or amount relative to the private reference;
- authorized/submitted action and actual booked effects;
- blocked proposals, blocked unsafe attempts and false financial blocks;
- complete-episode success, memory truncations and eviction exposure;
- actor tokens/calls, full-step and generation timings, and critic overhead.

A schema-valid monetary proposal with correct case, decision and amount but
wrong reason is a quality error, not financially unsafe. A false financial
block is rejection of such a financially legitimate monetary proposal,
regardless of whether its reason was exact. Blocking an unsafe proposal is
protection, not an improvement in raw accuracy. The executor never fabricates
a replacement `hold`: a denied action has `submitted_action = null`, an
explicit rejection receipt, and zero new monetary booking.

## Analysis and fixed gates

Calculate each episode's exact-accuracy fraction over its eight decisions.
Within a replication, average its four matched episode differences with equal
weight, giving one `learned - fixed` block difference and one
`learned - shuffled` block difference. Report all five block differences,
their equally weighted mean, matched episode outcomes and raw family effects.
Do not treat steps, repeated arm executions or episodes sharing a learned
policy as independent replications of the learning algorithm.

Use a preregistered **one-sided exact sign test** over the five block
differences. With `k` strictly positive blocks, the conservative upper-tail
p-value is `sum(comb(5, j), j=k..5) / 32`; zero differences count as nonpositive,
not as discarded observations. At alpha `0.05`, all five blocks must be
positive (`p = 1/32`). This is deliberately low-powered and concerns the
direction of block-level improvement, not a lower confidence bound on the
population mean or a two-sided 95% guarantee.

Comparisons are fixed-sequence hierarchical: first test `learned > fixed`.
Only if that rejects at `0.05` may `learned > shuffled` support a confirmatory
feedback-attribution claim at the same level. Always report both effects and
p-values, but explicitly label the second untested for confirmatory purposes
when the first gate fails. No significance claims by family, post-hoc pooling,
episode-level confidence interval pretending to replicate the learner, or
reuse of R1's ten-percentage-point threshold are allowed.

Full pilot gates require both ordered quality tests, no aggregate increase in
raw financially unsafe attempts versus fixed, zero unsafe monetary effects
and false blocks across the evaluated arms, and test actor tokens for learned
at most `1.10 × fixed`. Report family and replication regressions even if an
aggregate guard passes. These are descriptive safety/cost guards, not a
statistical noninferiority certificate. Latency distributions are descriptive,
not a noisy post-hoc selection gate.

## Cost, stopping and honest outcomes

The expected maximum is 2,250 local generations, with US$0 external API charge.
No paid endpoint, extra actor pass, retry, hidden replacement, early success
stop or test resizing is allowed. Report TRAIN, critic proposals, validation
and test costs separately, including every attempted/canceled call. Monetary
cost including energy/hardware remains unknown. Any test token saving is not
free: disclose adaptation/search overhead and, when test savings are positive,
the estimated future-episode break-even under the explicit assumption that
the observed saving persists. Do not present this estimate as measured future
production savings.

An incomplete run or failed integrity check has no statistical success
verdict. A complete run with nonpositive observed raw gain is a negative
observed result, not proof of equivalence. Positive point estimates without
the first sign-test gate are inconclusive. Passing the first comparison but
not the control comparison establishes no feedback-specific advantage under
this protocol. Any safety or cost guard failure prevents the full pilot claim,
even if raw accuracy improves. A schema-valid unchanged policy or a control
with unchanged feedback remains visible and is not replaced.

Before inference, freeze all new R2 source and protocol hashes alongside the
unchanged inherited components. Persist exclusive start and terminal/abort
receipts, all proposal/actor completions and tokenizer events, complete
trajectories, selection flags and costs in auditable hash chains. Independently
replay all 1,120 decisions, reconcile all 2,250 completed generations for a
complete run, and recompute every statistic/report without another model call.
New R2 files must not mutate frozen R1 sources or evidence. This experiment
never authorizes real payment endpoints, third-party infrastructure or
deployment. Publication is separate from experimental execution and must
preserve these limits.
