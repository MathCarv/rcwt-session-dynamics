# R1 case study: keeping the facts needed for the next action

A short memory can be readable and still omit the record an agent needs next.
R1 tested whether an engineered combination of structured memory and public
context improved actual decisions over an active LLM rolling-summary baseline.
The completed comparison produced **238/256 correct actions versus 152/256**,
with fewer aggregate unsafe attempts, tokens and recorded step time. It also
left consequential errors and safety regressions.

This is an independent extension of [CloudWalk's RCWT artifact](https://github.com/cloudwalk/rcwt-agent-coordination).
The original attribution and [MIT license](../LICENSE) are preserved.

## The agent and its task

A local Qwen3.5-4B model, using the Unsloth Q4_K_M quantization, processes
fictional invoices, accounts, payments and returns across eight decisions.
Tool observations supply the facts; the model generates a text plan followed
by a JSON action. Only the final action changes the simulated ledger. Its
earlier actions therefore affect what should happen later.

Both arms use the same frozen actor, public rules and task-bound schema, with
512 tokens per actor pass and 256-token memory/context caps. The baseline
updates a rolling text summary through model calls. The candidate uses a
deterministic writer and a reader that joins retained records to current
public observations. The reader exposes relevant facts; it does not consult
the private grading reference or choose, repair or retry the model's action.

## A failure in the first recorded episode

The replay fixes selection to manifest episode zero,
`rcwta-test-78c3b2ca-0`, from the `updated-state` family. It is not an episode
selected for having the largest gain. The summary arm gets five of its eight
actions right; the structured arm gets eight.

At step 2, both arms correctly hold invoice `inv-test-78c3b2ca-0-1` because
payment is pending. Its amount is 2,040 fictional cents. At step 5, a new
`read_payment` observation says that payment has cleared, and the request
returns to that invoice.

| At step 5 | Rolling summary | Structured memory + context |
| --- | --- | --- |
| Retained context | Describes another invoice, ending in `-0-3` | Retains the requested invoice's amount, account and ownership links |
| Current payment update | The raw observation is supplied | The raw observation is supplied and joined to the retained record |
| Final action | `ask_info`, zero cents, `missing_evidence` | `approve`, 2,040 cents, `authorized_payout` |
| Recorded grade | Unnecessary deferral | Correct action |

The baseline's final evidence check marks the amount and account status as
unknown. The structured context contains the requested amount, active account,
matching ownership and cleared payment. These observations explain what each
actor received and did; they do not isolate memory as the cause of every
failure. The change also reorganizes current evidence and moves record joining
into deterministic code.

The [small demo](../results/agent_v4_replication_demo/README.md) shows all eight
decisions in both arms. Its [focused step 5 JSON](../results/agent_v4_replication_demo/step5.json)
copies the public inputs, memory/context, final actions, tool receipts and
private grades. Provenance is retained: the summary record is `traces.jsonl`
line **5**, and the structured record is line **13**, with both original
trace-chain hashes in the JSON. The full CLI replay additionally includes raw
plans, final responses and all costs. Later references follow each arm's own ledger: at step 8 the
structured arm should hold an already completed refund, while the summary arm
still has an eligible refund to perform. Identical later labels are not assumed.

## What the complete new cohort established

After preserving an earlier confirmation interrupted at 173/512 decisions,
R1 registered one fresh cohort without changing the selected candidate:
32 paired episodes, eight per synthetic family, and 512 decisions. The
[registration](rcwt_v4_replication_protocol.md) preceded cohort generation.

Exact-action accuracy rose from **59.38% to 92.97%**, a **+33.59 pp** difference.
The paired-episode bootstrap 95% interval is **[+27.34, +39.45] pp**, using
10,000 resamples. The statistical unit is the paired episode. Entirely correct
episodes increased from 2/32 to 19/32.

The candidate used **16.59% fewer tokens**; median recorded step time fell
from **10.46 to 7.72 seconds**. The whole cohort consumed 2,137,526 prompt and
completion tokens across 1,248 generations. These local measurements include both actor passes and
compaction; step time includes tokenization and context work. API charges were
US$0, while electricity, hardware and total monetary cost remain unknown.

The accuracy criterion and both aggregate descriptive safety guards passed.
Unsafe attempts fell from 11 to 4 and unsafe booked fictional cents from
743,614 to 297,683. But `insufficient-evidence` worsened from 1 to 2 attempts
and from 142,330 to 160,695 unsafe cents; `updated-state` worsened from 0 to 1
attempt, without unsafe money booked. Memory truncations increased from 8 to
10. The system is not certified for production safety.

## Inspect the evidence and its limits

From the repository root, these commands verify the recorded cohort and show
the fixed first episode without making new model calls:

```bash
python tools/verify_v4_replication_report.py --run-dir results/agent_v4_replication
python tools/replay_v4_replication.py --run-dir results/agent_v4_replication
python tools/export_r1_demo.py --verify
```

The [complete report](../results/agent_v4_replication/RESULTS.md) and
[analysis](../results/agent_v4_replication/analysis.json) contain the full
metrics. Public report verification checks the frozen source and evidence
bindings, replays the 512 decisions, and recomputes exact statistical report
bytes. The small demo is also compared byte for byte after full report verification;
it is a derived display with fixed selection, not a new experiment.

The [complete derived runtime log](../results/agent_v4_replication_public_runtime/server.redacted.txt)
and [public accounting](../results/agent_v4_replication_public_accounting/CALL-ACCOUNTING.md)
allow the unchanged auditor to reconcile all **1,248 generations** from public
artifacts. Exactly one local directory prefix in model-load metadata is
redacted; all task lines, timestamps and recorded timings remain unchanged.
This additional check requires no private runtime bundle:

```bash
python tools/verify_r1_public_accounting.py
```

The verifier checks the exact released receipt hashes and all recomputed values;
only JSON object-key order is canonicalized for Linux/Windows portability.

The [manifest](../results/agent_v4_replication_public_runtime/manifest.json) binds
hashes of the original private log, custody and accounting to the derived log.
Those hashes do not independently reproduce private custody or prove the
redaction without access to the originals. Public accounting verifies recorded
consistency, not physical attestation of inference or absence of unlisted runs.

This is evidence for the combined engineered memory-and-context system on new
instances from the same generator, one model and one inference seed. It does
not demonstrate transfer to unseen domains, customer traffic, intrinsic model
improvement, autonomously learned policies, weight updates or recursive
self-improvement. The [earlier incomplete run](../results/agent_v4_interrupted/INTERRUPTION.md)
and the negative development history remain separate; neither quality scores
nor costs were silently substituted for R1.
