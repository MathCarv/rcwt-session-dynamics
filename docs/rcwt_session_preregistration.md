# RCWT-S preregistration: critical-state survival under session compaction

Status: confirmatory protocol locked before the first full-corpus run

Date: 2026-09-11

Protocol version: `session-v1`

An engineering pilot with one case per family was used to validate serialization,
hashing, and scorer plumbing. No 64-case confirmatory artifact had been generated
or scored when this protocol was frozen. A pre-run adversarial review strengthened
source binding, chronology fixtures, and correlated-data intervals without changing
H1, H2, H5, the non-inferiority margin, or the tested budgets.

## Research question

With the task and checkpoint query held intact, and under the same maximum
coordination-token budget, does a versioned state view preserve current decision-critical
facts better than retaining only the tail of a growing agent transcript?

RCWT-S is a deterministic, offline session-memory primitive. It isolates memory
selection and compaction from model quality. It does **not** measure the net
benefit of multi-agent coordination, agent intelligence, tool reliability, or
end-to-end task performance.

## Locked experimental design

The corpus contains 64 public synthetic sessions, each with 64 events and four
checkpoints at turns 8, 16, 32, and 64. Cases are balanced across four families:

- `update-heavy`: critical facts are revised or revoked during the session;
- `dependency-heavy`: a later critical fact depends on earlier evidence;
- `chronology-sensitive`: correct interpretation requires preserving order;
- `insufficient-evidence`: the trace never contains enough evidence for a
  decision and the context must say so rather than inventing a fact.

Each case is evaluated at hard caps of 256, 512, and 1024 `cl100k_base`
construction tokens. The same case, checkpoint query, and cap are used for every
treatment. Treatments are not padded to identical realized token counts; realized
use and utilization are reported.

### Treatments

1. `tail` (baseline): retain the newest complete transcript events that fit the
   budget.
2. `state_latest`: resolve the latest non-revoked value for each relevant
   `entity.field`, include its transitive dependencies, preserve causal order,
   and fill remaining capacity by recency.
3. `state_first` (ablation): use the same state-selection machinery but retain
   the first value for each `entity.field`, intentionally ignoring later
   supersession.

The builder receives only public cases. Oracle event IDs, stale-conflict IDs,
expected dispositions, and scoring metadata are stored separately and are not
arguments to the context builder.

## Outcomes

The primary binary outcome is `decision_ready`:

```text
all required current events are present
AND no stale conflicting event is present
AND every required causal edge is preserved
```

For an `insufficient` checkpoint, `decision_ready` additionally requires an
explicit `MISSING` marker and forbids fabrication of a current fact.

Secondary outcomes are:

- required-fact recall;
- stale-exposure rate;
- conservative stale-conflict exposure (reported as contradiction in v1);
- realized coordination tokens;
- turns to first failure;
- `B95`, the smallest tested budget reaching at least 95% decision readiness;
- paired differences `state_latest - tail` and
  `state_latest - state_first`.

All rates are reported overall and by family, checkpoint turn, and budget.

## Pre-specified hypotheses and falsification checks

- **H1:** `state_latest` has higher overall decision readiness than `tail` in
  long (`turn >= 32`) update-heavy and dependency-heavy checkpoints.
- **H2:** `state_latest` outperforms `state_first` on update-heavy checkpoints.
- **H3:** `state_latest` and `state_first` are byte-identical on append-only
  fixtures that contain no supersession.
- **H4:** permuting distractors without changing facts or dependencies does not
  change the critical event set selected by `state_latest`.
- **H5:** no treatment fabricates sufficient evidence in the
  insufficient-evidence family.
- **Control:** chronology-sensitive results are reported independently. A gain
  on update-heavy cases does not justify a general claim if chronology-sensitive
  readiness regresses by more than 5 percentage points relative to `tail`.

The main claim is supported only if H1 and H2 have positive paired 95%
bootstrap intervals, all validity checks pass, H5 has zero violations, and the
chronology control stays within the stated non-inferiority margin. Any other
outcome is reported as mixed or null.

## Analysis plan

- The independent resampling unit is the case/seed, not a checkpoint or context
  record.
- Paired differences and treatment rates use 10,000 deterministic percentile
  bootstrap resamples clustered by case/seed.
- Naive 95% Wilson intervals may be retained as a descriptive diagnostic, labeled
  explicitly as non-clustered; they are not used for confirmation or the main plot.
- `B95` is computed only from tested budgets and is `null` when no budget
  reaches the threshold.
- No case, family, checkpoint, or budget may be removed after inspecting the
  result unless a pre-specified validity check fails. Any invalidation is
  reported with the original manifest hash and a new protocol version.

## Run invalidation criteria

A run is invalid if any of the following occurs:

- public case, oracle, configuration, or sealed context hash mismatch;
- a context exceeds its declared token budget;
- source case/checkpoint/budget differs across treatments;
- a context header, query, event payload, event order, family, or seed differs from
  its source public case;
- the builder imports or reads oracle data;
- two runs with the same seed differ byte-for-byte;
- API credentials, network availability, or provider SDK state changes output;
- a causal edge is scored from event IDs not present in the rendered context;
- a persisted score cannot be independently rederived from contexts and oracle.

## Locked full-run commands

```bash
PYTHONPATH=src python src/rcwt_session_generate.py \
  --seed 20260911 \
  --cases 64 \
  --turns 64 \
  --output-dir results/session_v1

PYTHONPATH=src python src/rcwt_session_run.py \
  --cases results/session_v1/public_cases.jsonl \
  --manifest results/session_v1/manifest.json \
  --treatments tail,state_latest,state_first \
  --checkpoints 8,16,32,64 \
  --budgets 256,512,1024 \
  --output results/session_v1/contexts.jsonl

PYTHONPATH=src python src/rcwt_session_score.py \
  --contexts results/session_v1/contexts.jsonl \
  --cases results/session_v1/public_cases.jsonl \
  --oracle results/session_v1/oracle_cases.jsonl \
  --manifest results/session_v1/manifest.json \
  --output results/session_v1/aggregates.json
```

The source commit, preregistration digest, configuration, and generated artifact
digests must be recorded in `results/session_v1/manifest.json` before results
are interpreted. The runner seals the context digest into that manifest before
the scorer starts.

`tiktoken==0.12.0` and the encoding identity are pinned. A cold first load may
populate tiktoken's local `cl100k_base` vocabulary cache over HTTPS; after that
bootstrap, generation and scoring make no provider/model calls and are required
to remain byte-identical with network access blocked.

## Pre-run review amendments

Before the confirmatory run, a separate adversarial review found four validity
risks and the protocol was amended in place:

- chronology fixtures now carry explicit dependency edges across grant/revoke
  transitions;
- required historical dependencies cannot simultaneously be labeled stale
  conflicts;
- scoring is bound to the actual public corpus, not only to a claimed case hash;
- contexts are sealed into the manifest and correlated rate intervals use the
  case/seed cluster.

These corrections were completed before the confirmatory corpus was interpreted.
