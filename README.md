# RCWT-S Online: Memory and Public Context for a Stateful Agent

[![Verify](https://github.com/MathCarv/rcwt-session-dynamics/actions/workflows/verify.yml/badge.svg?branch=main)](https://github.com/MathCarv/rcwt-session-dynamics/actions/workflows/verify.yml)

An independent extension of [CloudWalk's RCWT artifact](https://github.com/cloudwalk/rcwt-agent-coordination)
by [Matheus Carvalho](https://github.com/MathCarv). The original authors' attribution
and [MIT license](LICENSE) are preserved.

An agent can retain a readable summary while losing the facts needed for its
next action. This repository tests that problem with a local LLM making
successive decisions over a fictional financial ledger.

**The completed R1 confirmation improved exact actions from 152/256 (59.38%) to
238/256 (92.97%): +33.59 percentage points, with a paired-episode bootstrap
95% interval of [+27.34, +39.45] pp.** The gain belongs to an engineered memory
and public-context system under a common frozen actor. It is not autonomous
learning or an intrinsic improvement of the model.

Start with the [short case study](docs/rcwt_r1_case_study.md),
[complete results](results/agent_v4_replication/RESULTS.md), or
[small first-episode demo](results/agent_v4_replication_demo/README.md).

## The agent, the failure, and the change

A real Qwen3.5-4B model receives public observations about invoices, accounts,
payments and returns. It produces a text plan and then a JSON action. Only
the final action executes in the simulator; its consequences affect later
decisions. All identities, tools and money are synthetic.

The baseline uses an active LLM rolling summary. In the first recorded episode,
the fifth request returns to an earlier invoice. The summary contains another
invoice, and the agent asks for information that had already been observed.
The structured arm retains the earlier record, joins it to the new payment
update, and approves the correct fictional amount. This is an inspectable
example, not proof that memory caused every error.

The candidate combines a deterministic structured-memory writer with a reader
that organizes retained and current public facts by their identifiers. Both
arms use the same frozen actor, public rules, output schema and local model:
two passes capped at 512 tokens each, with 256-token memory/context caps.
No private reference selects the action, no semantic repair overrides it, and
no correctness-triggered retry is allowed. Record joining and derived-field
calculation move work into deterministic code, so this tests the combined
system rather than compaction alone.

## A fresh, fixed confirmation

R1 used **32 paired episodes**, eight from each of four synthetic families,
with eight decisions per arm: **512 decisions and 1,248 model generations**.
The runtime was Qwen3.5-4B, Unsloth Q4_K_M, on llama.cpp b10809.

The [prospective registration](docs/rcwt_v4_replication_protocol.md) froze the
selected development candidate before generating the new cohort. The accuracy
criterion required a mean gain of at least 10 pp and a strictly positive lower
endpoint of a two-sided 95% interval. The interval uses 10,000 bootstrap
resamples of **paired episodes**, not 512 independent decisions.

| Complete R1 cohort | Rolling summary | Structured memory + public context |
| --- | ---: | ---: |
| Exact actions | 152/256 (59.38%) | 238/256 (92.97%) |
| Entirely correct episodes | 2/32 | 19/32 |
| Unsafe fictional action attempts | 11 | 4 |
| Unsafe fictional cents actually booked | 743,614 | 297,683 |
| Prompt + completion tokens | 1,165,452 | 972,074 |
| Model calls, including compaction | 736 | 512 |
| Step latency: median / p95 | 10.460 / 14.936 s | 7.723 / 11.532 s |

The candidate made **86 additional correct decisions**, used **16.59% fewer
tokens**, and reduced median step latency from **10.46 to 7.72 seconds**.
Tokens and step times include both actor passes and memory compaction; step
time also includes tokenization and context processing. These are descriptive
measurements on the local hardware and load. API charges were US$0;
electricity, hardware and total monetary cost remain unknown.

Both **aggregate descriptive** safety guards passed, but four unsafe attempts
remain. In `insufficient-evidence`, unsafe attempts increased from 1 to 2 and
unsafe booked cents from 142,330 to 160,695. In `updated-state`, attempts
increased from 0 to 1, with no unsafe money booked. Memory truncations increased
from 8 to 10. This is not a production-safety certificate.

## Inspect the saved evidence

With Python 3.12, run from the repository root:

```bash
python -m pip install -r requirements.txt
python tools/verify_v4_replication_report.py --run-dir results/agent_v4_replication
python tools/replay_v4_replication.py --run-dir results/agent_v4_replication
```

The verifier replays every recorded decision with the frozen code, checks the
source/corpus/schedule bindings, and compares the recomputed analysis and report
byte for byte. The replay always displays the **first manifest episode**, both
arms and all eight steps, including failures. Neither command starts a model
or requires an API key.

For a browser-friendly view, the [small demo](results/agent_v4_replication_demo/README.md)
shows all eight decisions in both arms. Its [focused step 5 JSON](results/agent_v4_replication_demo/step5.json)
contains the public inputs, memory/context, actions and grades, with original
trace hashes and line numbers. Selection is fixed to manifest index 0 and step
index 4, never a search for a favorable score. Optionally verify its exact bytes:

```bash
python tools/export_r1_demo.py --verify
```

The [analysis](results/agent_v4_replication/analysis.json),
[replay receipt](results/agent_v4_replication/verification.json) and
[registered protocol](docs/rcwt_v4_replication_protocol.md) support the public
report checks. An integrity PASS is separate from the accuracy and safety
verdicts. The original R1 closure also passed **583 Python regression tests**;
tests of the implementation are not additional model experiments.

The complete [derived runtime log](results/agent_v4_replication_public_runtime/server.redacted.txt)
is also publicly verifiable. Exactly **one local directory prefix in model-load
metadata** is redacted; every task line, timestamp and recorded timing remains
unchanged. The unchanged auditor reconciles **1,248 server starts, final timing
pairs and recorded calls** against the public traces; see the
[public accounting report](results/agent_v4_replication_public_accounting/CALL-ACCOUNTING.md).
Recheck it without model calls or the private runtime bundle:

```bash
python tools/audit_v4_replication_calls.py --run-dir results/agent_v4_replication --server-log results/agent_v4_replication_public_runtime/server.redacted.txt --output-dir results/agent_v4_replication_public_accounting --verify
```

The [derived-log manifest](results/agent_v4_replication_public_runtime/manifest.json)
binds the public log to hashes of the original private log, custody receipt and
accounting. Public verification checks the derived log's consistency with the
recorded calls; it does **not** independently reproduce private custody or prove
the redaction against unavailable original bytes. Neither check is independent
physical attestation of inference or proof that no unlisted runs occurred.

## Limits

The confirmation tests new parameter instances from the **same four synthetic
families**, one quantized model and one inference seed. It does not establish
transfer to unseen domains, other models or seeds, CloudWalk customer data,
or production traffic. Each arm is graded against its own actual prior
simulated ledger, so later references can differ after different actions.

There are no model-weight updates, autonomously learned policies or recursive
self-improvement. The confidence interval concerns the registered comparison;
the criterion does not prove a population-wide minimum gain of 10 pp. Local
hashes and recorded telemetry establish consistency, not independent
attestation that no unlisted runs occurred.

## Preserved experiments and attribution

Earlier negative and incomplete outcomes remain part of the evidence:

| Experiment | What it measures or concluded | Evidence |
| --- | --- | --- |
| Original RCWT and intact-task ablation | Coordination-token overhead in single-call tasks | [Upstream artifact](https://github.com/cloudwalk/rcwt-agent-coordination), [local aggregates](results/rcwt_controlled_aggregates.json), [intact-task results](results/intact_ablation/rcwt_intact_ablation_aggregates.json) |
| Static RCWT-S `session_v1` | Retained-evidence scoring, **not LLM action accuracy** or cumulative online memory | [Registration](docs/rcwt_session_preregistration.md), [results](results/session_v1/RESULTS.md) |
| Online v2 | No learned-memory improvement; selection retained the summary baseline | [Results](results/agent_v2/RESULTS.md), [diagnosis](results/agent_v2/DIAGNOSIS.md), [demo](docs/rcwt_agent_demo.md) |
| Online v3 | Five failed development screens; no held-out confirmation | [Closure](results/agent_v3_development/RESULTS.md), [all attempts](results/agent_v3_development/DEVELOPMENT.md), [demo](docs/rcwt_online_v3_demo.md) |
| Online v4 development | First revision failed safety guards; the second passed development | [Both attempts](results/agent_v4_development/RESULTS.md), [registered revision](docs/rcwt_online_v4_revision_02.md) |
| Original v4 confirmation | Interrupted at 173/512 decisions; remains incomplete | [Interruption record](results/agent_v4_interrupted/INTERRUPTION.md) |

R1 was a separately authorized fresh cohort after that interruption was
disclosed. It did not resume, pool or replace the 173-decision prefix, and no
candidate was retuned for R1. The prefix retains 421 completed calls and one
additional canceled task without final persisted response/usage. Historical
costs are disclosed separately from the R1 comparison.

The [Makefile](Makefile) retains the legacy reproduction and verification
entrypoints. `make verify-session` rebuilds the static session experiment;
`make verify` rebuilds and checks the legacy deterministic artifacts and runs
the Python tests. `make replication-verify` checks the complete R1 report.
Those experiments answer different questions and their scores must not be pooled.

The original RCWT is by **Brenda Carolina Belmiro Lelis and Rodrigo
Cabral-Carvalho**. This extension preserves their copyright notice and MIT
license; it does not represent an endorsement by the original authors.
