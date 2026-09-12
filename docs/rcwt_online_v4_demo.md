# Replaying the recorded v4 experiment

This is an offline replay of saved evidence, not live inference. It does not
contact a model or a financial system. All identities, accounts, tools and
amounts are fictional. A replay or integrity PASS is not proof of improvement.

Current state: the registered confirmation was interrupted after 173 of 512
decisions. Its [partial archive](../results/agent_v4_interrupted/INTERRUPTION.md)
is not a complete demonstration or proof of held-out improvement. No new run
was substituted. The development replay below remains available.

Run these commands from the repository root. The tools require a completed,
frozen archive that passes offline verification before displaying its traces.
They do not select the best-scoring example or repair an unsuccessful action.

## First development episode

```powershell
python tools/replay_online_v4.py --run-dir results/agent_v4_development/attempt_02 --episode-index 0
```

Episode index zero is the first episode in the frozen public manifest. Both
policies are shown for every request, including incorrect or invalid actions.
This directory is **development / TRAIN**. Replaying it does not turn it into
held-out evidence. Earlier revisions remain in the development index; their
reused training episodes must not be pooled as independent quality evidence.

The Makefile equivalent, where GNU Make and a POSIX shell are available, is:

```sh
make online-v4-demo ONLINE_V4_RESULTS=results/agent_v4_development/attempt_02 AGENT_EPISODE=0
```

## First confirmation episode, once the complete archive is available

Only use the following command when the complete confirmation archive is
available in the checkout at `results/agent_v4`. An absent or incomplete directory is an
error, not permission to generate another run.

```powershell
python tools/replay_online_v4.py --run-dir results/agent_v4 --episode-index 0
```

This is a recording from the declared **confirmatory / TEST** cohort, not a
fresh replication. One displayed episode cannot establish cohort-level gain;
interpret it alongside the complete paired analysis, uncertainty interval,
safety safeguards, costs and limitations in that archive's `RESULTS.md`.

## What to inspect

1. Compare the exact requested invoice ID with the raw public observations.
2. Read **stored memory before** separately from **exact context sent to the
   actor**. The structured context joins retained and current public facts by
   literal IDs. It is transient, not another persistent ledger. Persistent
   memory and actor context each have their own counted token cap.
3. Inspect the raw, unexecuted text plan and the final model response. Both
   policies use the same actor procedure, public task-bound schema and, in
   the final development revision, the same ordered public-rule guide. The
   schema restricts public request identity and operation; it does not decide
   eligibility or calculate the amount to authorize.
4. Compare the model's self-reported evidence check, its actual final action,
   the fictional tool receipt and the private evaluation. A correct evidence
   check does not guarantee a correct decision. A booking receipt records an
   effect, not its correctness. The plan was never executed as a fallback.
5. Inspect all recorded generation calls, including planning, final response
   and summary compression, plus retained memory after the action. Per-step
   time includes deterministic context work and recorded tokenizer operations;
   it is not whole-run calendar time or startup overhead.

Private references and grades are displayed afterward for the reader. They
were not provided to the actor, reviewer, context builder or memory writer.
Each policy's reference uses its own actual simulated ledger: once actions
diverge, the reference state need not remain identical across the pair.

## Verify without regenerating evidence

The development index and both archived diagnoses are mandatory checks:

```powershell
python tools/report_online_v4_development.py --runs results/agent_v4_development/attempt_01 results/agent_v4_development/attempt_02 --output-dir results/agent_v4_development --verify
python tools/verify_online_v4_diagnosis_archive.py --run-dir results/agent_v4_development/attempt_01 --diagnosis-dir results/agent_v4_development/attempt_01/diagnosis_verified
python tools/verify_online_v4_diagnosis_archive.py --run-dir results/agent_v4_development/attempt_02 --diagnosis-dir results/agent_v4_development/attempt_02/diagnosis
```

The historical-diagnosis wrapper reconstructs a temporary workspace using
the run's frozen sources and the diagnosis's three pinned tool snapshots.
It compares recomputed facts and exact Markdown without changing originals.
Never use these helpers with untrusted downloaded Python archives; their
hashes check consistency, not authorship, and they are not security sandboxes.

For an available complete confirmation, the strict checks are:

```powershell
python tools/verify_online_v4_report.py --run-dir results/agent_v4
python tools/inspect_online_v4.py --run-dir results/agent_v4 --output-dir results/agent_v4/diagnosis --verify
```

`make online-v4-verify` verifies development and, only if the confirmation
path exists, runs those strict confirmation checks. An existing but partial,
corrupt or incompatible archive fails; only an entirely absent confirmation
path is skipped with an explicit message. `make online-v4-verify-confirmation`
requires confirmation even when the path is absent. CI does not attempt to
audit the unpublished local `.runs/` inventory.

No command above starts a server or calls a model. Local API charges are zero,
but electricity, hardware and total monetary cost are unmeasured. Token counts
and durations are recorded runtime claims, not independent hardware
attestation. Latency depends on hardware. This synthetic, single-model,
quantized benchmark does not establish production safety, cross-domain
generalization, weight learning or recursive self-improvement.

The interrupted prefix has its own strict offline check, which can return
`PARTIAL_VERIFIED` but never a complete confirmation or a gain verdict:

```powershell
python tools/verify_online_v4_partial.py --run-dir results/agent_v4_interrupted
```

`make online-v4-verify` also checks this prefix when its archive exists. Missing
decisions and a possibly unmetered in-flight call remain explicitly disclosed.

See the [series protocol](rcwt_online_v4_protocol.md) and the
[frozen final-development revision](rcwt_online_v4_revision_02.md) for the
registered design and interpretation limits.
