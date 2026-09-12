# V4 confirmation: interrupted, no confirmed gain

**Status: INCOMPLETE.** The single registered confirmation did not finish.
173 of 512 scheduled decisions were durably recorded; 339 are missing. There
are 21 complete episode/arm summaries out of 64, plus five recorded steps of
the next trajectory. No confirmatory accuracy estimate, confidence interval
or favorable safety verdict is reported from this incomplete prefix.

The [two development attempts](../agent_v4_development/DEVELOPMENT.md) remain
separate and verified. The selected second candidate achieved 60/64 exact
actions versus 38/64, with 3 versus 4 unsafe attempts. This is development
selection evidence, not independent proof of gain on new tasks.

## Observed interruption

- Confirmation started at `2026-09-12T00:55:29.685194Z` under the frozen
  [protocol](protocol.json), with unchanged candidate code and no further tuning.
- The final persisted trace is episode `rcwta-test-78c3b2c6-a`, policy
  `summary`, zero-based step 4. Its file's last-write time is
  `2026-09-12T01:18:07.6759998Z`.
- The local server log's last-write time is
  `2026-09-12T01:18:11.6989938Z` (22:18:11 in Sao Paulo on September 11).
  Its final task, `197522`, has an explicit cancellation and slot release,
  without a final completion timing record.
- Read-only checks at approximately `2026-09-12T01:24:48Z` through
  `01:25:13Z` found no owned PID 22728, no matching RCWT runner/server process,
  and no listener on port 18085.
- There is no runner `completion.json`, `aborted.json` or `partial-step.json`.
  No cleanup receipt for PID 22728 was found in the scoped runtime receipt
  directories. This does not establish who stopped it, why it stopped, or
  whether a shutdown was graceful. No cause is inferred from the interruption.

The coordinator did not restart, replace or resume this run. It did not
fabricate a runner abort receipt or a completion manifest. The existing
complete-run verifier and campaign auditor correctly reject it as incomplete.
Their gates were not relaxed to accept a favorable partial result.

A separate read-only prefix verifier subsequently reconstructed all **173
recorded steps** and **21 complete trajectory summaries**, including exact
actor inputs, saved completions, tokenization, memory transitions, simulated
effects, grades and counters. It returned `PARTIAL_VERIFIED`, with
`confirmation_status=INCOMPLETE` and `gain=NOT_EVALUATED`. This does not repair
or fill the 339 missing decisions and makes no new inference call.

```powershell
python tools/verify_online_v4_partial.py --run-dir results/agent_v4_interrupted
```

Final software checks: **511 regression tests passed** in 114.580 seconds,
including 12 new partial-prefix tests. Both development archives and their
historical diagnoses passed `make online-v4-verify`; that command reported the
interrupted prefix separately and explicitly reported no complete confirmation.
Historical v2 replay (512 decisions) and the five-attempt v3 development index
also passed unchanged. Compilation and `git diff --check` passed. These are
software/evidence-integrity checks, not a substitute for model-performance
confirmation.

## Preservation and accounting

The original `.runs/v4_confirmation` directory remains intact. Its 49 files
(16782247 bytes), including frozen code and selected development evidence,
were copied byte-identically to this archive without overwriting any file.
Runtime receipts and the original server log were copied separately to
`runtime/`; that directory contains local machine paths and is not a sanitized
public export. No GitHub publication or external message was performed.

The 173 complete steps contain **421 saved generation calls**, **646649 input
tokens** and **76320 output tokens** (722969 total). All passes, including
summary compaction, are counted. These are lower bounds on interrupted-run
work because the canceled request has no completed response in the traces.

Across the two development runs and this prefix, **1045 completed calls** and
**1679254 saved tokens** are accounted for. The same server log records **1046
parent-task starts**: 1045 completed requests and one final canceled request.
This does not turn partial confirmation into complete evidence, and it does
not attest that no unlisted work occurred elsewhere.

API-provider charge was US$0 under the local-only protocol. Electricity,
hardware, setup and the total monetary cost are unknown, not zero. The canceled
request's complete usage and latency cannot be reconstructed from a missing
response; they are not entered as zero. See [ACCOUNTING.md](ACCOUNTING.md).

## Evidence identifiers

| Artifact | SHA-256 |
| --- | --- |
| protocol.json | `b09076db9d9311e7699d072a223bbd62d95b112fbb7973bbab87c950af101654` |
| traces.jsonl | `beb01cde1aa808cc2204344bd9adef27fc0ac7ac10f0da8434c14501b2607676` |
| episodes.jsonl | `3a450546223c188c2cd139dcc418e0a91ddc4e39b7858100888c3a67a0183d4f` |
| runtime/server.stderr.log | `cd8f2cd409a502936db005d3d31360f0c81ce3ffd88740f28d20b95e5fbb5785` |

## What would require a new decision

The [registered v4 series](../../docs/rcwt_online_v4_protocol.md) allows one
confirmation, no silent restart, replacement cohort or outcome-dependent
retry. This series therefore cannot establish its final claim. Any recovery
or independent replication needs an explicit prospective protocol that retains
and discloses this partial run, rather than relabeling it as an unseen first
attempt. The selected system must remain unchanged if the question is whether
its development advantage generalizes.

Even a future complete positive result would concern an engineered memory and
public-evidence system, one local quantized model and new instances of four
synthetic families. It would not establish autonomous learning, recursive
self-improvement, CloudWalk customer outcomes or production safety.
