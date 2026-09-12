# V4 replication R1: fresh confirmation after an interrupted run

Registration time: 2026-09-12T01:39:00Z (local written registration; not an
externally witnessed timestamp). The user explicitly authorized one new local
confirmation after being told that the first stopped at 173/512 decisions.
No R1 corpus has been generated, inspected or run at registration. This is an
explicit new experiment, not a resumed or silently replaced first attempt.

## Immutable candidate and retained history

Use exactly the fifteen current v4 source files selected in development
attempt 02, whose protocol SHA-256 is
`750768b5615a15c6a0c0542bf7464045116fa8587dd3f7b61f894dba124de73b`.
The shared actor, guide, planning/review prompts, schema, memory writer,
public-context reader, summary baseline, simulator, extractor, analysis and
sampling stay byte-identical. No new development, learning, output repair,
semantic veto, retries or model changes are allowed. A separate orchestration
module may register the new cohort, verify its evidence and report it; it
must reuse the frozen step executor and statistical functions unchanged.

Preserve both v4 development attempts, all v2/v3 evidence, and the incomplete
confirmation archived at `results/agent_v4_interrupted`. Its protocol hash is
`b09076db9d9311e7699d072a223bbd62d95b112fbb7973bbab87c950af101654`;
its trace hash is
`beb01cde1aa808cc2204344bd9adef27fc0ac7ac10f0da8434c14501b2607676`.
It contains 421 completed calls and one additional canceled server task with
no final persisted response. Its partial outcomes are not pooled, used for
tuning or promoted to an independent result. Its known costs and the missing
in-flight accounting remain disclosed alongside R1 costs.

## Fixed fresh cohort and endpoints

- One cohort only: **32 paired episodes**, eight per each of the same four
  synthetic families, eight steps per arm: **512 decisions**.
- Dataset split `test`, seed **2026091210**, chosen now without preview.
  Do not use the interrupted seed 2026091206 or development seed 2026091205.
- Inference seed **20260911**, balanced within-family schedule seed
  **2026091208**, bootstrap seed **2026091207**; no alternate seeds or reruns.
- Same pinned Qwen3.5-4B Unsloth Q4_K_M and llama.cpp b10809 loopback runtime
  in `docs/rcwt_agent_runtime.json`. No paid API. The restart changes process
  identity, not runtime/model configuration. Server start/stop and hashes
  must be recorded; health/model-inventory probes generate no completions.
- Same two 512-token actor passes and 256-token persistent/context caps.
  Expected model generations: **1248** (736 summary, 512 structured).

The primary unit is a paired episode, not an independent step. Require mean
structured-minus-summary exact-action accuracy **>=10 percentage points** and
the lower endpoint of a two-sided 95% paired-episode percentile bootstrap
interval **>0**, using **10000** resamples. Reuse the unchanged v4 analysis.
This does not require or prove that the population gain exceeds 10 points.

Both descriptive guards must also pass: no increase in unsafe monetary
attempts and no increase in unsafe fictional cents actually booked. They are
not statistical noninferiority tests or production safety certification.
Report failures even if accuracy improves. References depend on each arm's
actual prior simulated ledger; this compares policy trajectories under their
own states, not identical future labels after divergent actions.

Report complete-cohort accuracy, full-episode success, invalid actions,
failure types, both-pass/compaction tokens and calls, truncations, decision
and full-step p50/p95 latencies and total recorded episode wall time. Separate
this cohort's costs from earlier development and the incomplete confirmation.
Local API charge is US$0; electricity, hardware and total monetary cost remain
unknown. Timing is descriptive for this local hardware/load, not universal.

## Freeze, integrity and stopping

Before first inference, bind this registration, all fifteen immutable source
hashes, the new orchestration source, runtime, generated corpus, schedule,
selected complete development and interrupted-run evidence. Start into a
nonexistent output directory. The first possible inference must follow an
exclusive start marker. Persist all completions, tokenization, actions and
effects in a hash chain, plus terminal completion or available partial/abort
records. No hidden reattempts, early success stop, replacement episodes,
sample resizing or new candidate revision based on this cohort.

After completion, replay every event with the frozen executor, recompute
grades/counters, and independently verify exact statistical report bytes.
Reconcile the new server's generation count with this cohort, and disclose
previous logs separately. If interrupted again, preserve it as incomplete;
if any fixed endpoint fails, report that the combined gain was not established.
Do not change the criterion or silently run another confirmation.

R1 is new-parameter-instance evidence from the same generator, one model and
one inference seed. It does not test unseen domains, CloudWalk customer data,
production traffic, autonomous policy learning, weight updates or recursive
self-improvement. The source freeze and local telemetry establish auditable
consistency, not independent physical attestation or proof that no unlisted
runs ever occurred. No publication, Git push or email is authorized here.
