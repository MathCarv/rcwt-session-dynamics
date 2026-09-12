# Replaying the paired local-agent experiment

This demonstration replays saved evidence. It does **not** run inference or
contact a financial system. All identities, accounts and money are fictional.
Quality claims belong to the complete cohort report, not to one example.

The released recording is the **last failed development attempt**, not a
held-out confirmation. Its entire paired result and safety regression are
reported in the [closure](../results/agent_v3_development/RESULTS.md).

```powershell
python tools/replay_online_v3.py --run-dir results/agent_v3_development/attempt_05 --episode-index 0 --show-memory
```

The default is episode zero in the frozen manifest, never an automatically
selected best case. Both memory policies are shown for all eight requests.
The command refuses an incomplete or non-replaying artifact bundle.

For each request, inspect:

1. The original public observations and exact requested invoice ID.
2. Persistent memory and the actual memory view passed to the model. The
   structured reader selects from retained history; it cannot use the private
   reference or import current observation values into that view.
3. The unexecuted short text plan, then the final model JSON response. Both
   policies receive the same fixed review instruction. Only the final action
   reaches the simulator; there is no fallback to a more favorable draft.
4. The actual fictional receipt, unchanged private grade, and retained memory
   after execution. A receipt confirms a booking, not correctness.
5. All generation calls, tokens and full-step wall time, including both actor
   passes and any summary-compaction call.

The private grade is displayed **afterward for the reader**. It was never
provided to the actor, reviewer, writer or retrieval policy.

Offline checks:

```powershell
python src/rcwt_online_v3.py --stage verify --output-dir results/agent_v3_development/attempt_05
python tools/inspect_online_v3.py --run-dir results/agent_v3_development/attempt_05 --output-dir results/agent_v3_development/attempt_05/diagnosis --verify
```

For a development recording, replace the directory with that complete
development run. It remains training evidence, not held-out confirmation.
An older development revision is verified with its own archived source:

```powershell
python tools/verify_online_v3_archive.py --run-dir results/agent_v3_development/attempt_01
```

The archive helper executes trusted project code in verification mode. Do not
use it on untrusted downloaded Python bundles. Model tokens and elapsed times
are recorded local-runtime claims, not independent hardware attestation.

See the [protocol](rcwt_online_v3_protocol.md) for the fixed cohort, statistical
unit, accuracy criterion, safety safeguards and limits on interpretation.
