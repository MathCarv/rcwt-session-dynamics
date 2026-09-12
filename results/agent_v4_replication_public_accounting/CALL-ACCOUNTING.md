# R1 server-call accounting

PASS — recorded accounting only. Gain: NOT_EVALUATED.

Complete offline replay: 512 decisions and 64 trajectory summaries.
Server starts / complete final timing pairs / matched trace calls: 1248 / 1248 / 1248.
Canceled or extra tasks: 0. No inference or endpoint request was made by this audit.

Prompt tokens: 1906468. Completion tokens: 231058. Total: 2137526.
Recorded client inference duration: 4758.089287 s.
Server prompt evaluation: 1104.095570 s; generation evaluation: 3633.302690 s.
Per-field timing rounding tolerance: 0.01 ms; maximum observed delta: 0.005000 ms.
Provider API cost: USD 0. Total monetary cost: unknown (electricity and hardware are not measured).

Server-log SHA-256: `64b74addd5505f9f3d0e3a0543fa35fa8055c81504f76fd72ac0aa3d139dc34e`.
Protocol SHA-256: `5fdb1fcb9b81bd4fc073d9429c90e199a44688dfbdf9b2bdfedc74f1c9092524`.
Audit-helper SHA-256: `1e99ec11151944d285abda1b15f9411636a17f6e626ec265cb81b20274f281ac`.

All 1248 ordered task-to-trace mappings and the input/source hashes are preserved in call-accounting.json.

PASS means complete recorded call accounting only; quality or improvement was not evaluated.
The supplied complete log and traces are local claims, not independent hardware or global run attestation.
Process start/stop, log-copy completeness and absence of later appends require separate runtime custody receipts.
Client inference duration includes request overhead; server evaluation time is reported separately.
Provider API charges are zero; electricity, hardware and total monetary cost are unknown.
This audit neither pools nor replaces the preserved interrupted confirmation.
