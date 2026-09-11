# RCWT-S confirmatory results

Protocol: `session-v1`

Source commit: `6153fbe9d154483346abbecde52b9172b2fbde07`

Confirmatory run: 64 sessions, 4,096 events, 256 checkpoints, 2,304 contexts

## Result

The preregistered main claim is **supported inside this synthetic benchmark**.
Under the same maximum construction-token caps, versioned latest-state context
was decision-ready in 83.3% of cells, compared with 54.2% for transcript tail.
The paired improvement was **+29.2 percentage points** with a 95% case-cluster
bootstrap interval of **+24.6 to +33.6 points**.

| Treatment | Decision-ready | 95% case-cluster bootstrap CI | Stale exposure | Mean tokens |
|---|---:|---:|---:|---:|
| `state_latest` | 83.3% (640/768) | 79.0–87.6% | 0.0% | 541.1 |
| `tail` | 54.2% (416/768) | 47.0–61.3% | 2.1% | 525.8 |
| `state_first` | 37.5% (288/768) | 28.5–46.9% | 37.5% | 537.7 |

The proposed treatment did not win by receiving a smaller context. Its realized
context was 15.3 tokens larger than tail on average (paired 95% CI 12.6–18.1)
because both structured-state arms filled residual capacity with recent,
dependency-closed events.

## Budget response

| Maximum tokens | `state_latest` | `tail` | `state_first` |
|---:|---:|---:|---:|
| 256 | 62.5% | 43.8% | 37.5% |
| 512 | 87.5% | 53.1% | 37.5% |
| 1,024 | **100.0%** | 65.6% | 37.5% |

`state_latest` reached the preregistered B95 threshold at 1,024 tokens. Neither
tail nor the first-state ablation reached 95% at a tested budget.

## Preregistered checks

- **H1 passed.** In long (turn 32/64) update-heavy and dependency-heavy cells,
  `state_latest - tail` was +41.7 points (95% paired cluster bootstrap CI
  +38.5 to +44.3; 32 independent sessions).
- **H2 passed.** On update-heavy cells, `state_latest - state_first` was +75.0
  points (95% CI +75.0 to +75.0; 16 sessions). The degenerate interval reflects
  identical deterministic case-level effects, not unlimited external precision.
- **H3 passed in regression tests.** Latest and first state render byte-identical
  output on the append-only fixture with no supersession.
- **H4 passed in regression tests.** Permuting distractors does not change the
  decision-critical set selected by latest state.
- **H5 passed.** All 576 insufficient-evidence contexts retained an explicit
  `MISSING` disposition with zero fabricated-sufficiency violations after source
  validation.
- **Chronology control passed.** Latest state improved rather than regressed:
  +37.5 points versus tail (95% CI +35.4 to +39.6).

## Integrity checks

The scorer returned `validity.status=PASS` after rederiving and checking all
2,304 context hashes, token counts, rendered event lists, public source bindings,
oracle keys, case hashes, and the complete treatment matrix. The manifest was
sealed before scoring.

| Artifact | SHA-256 |
|---|---|
| `public_cases.jsonl` | `d44cdfb9bf11043b97c0ed3e023545e48ef69a08456b1f090b618cd77571bd5a` |
| `oracle_cases.jsonl` | `9a5dcdb08e049b60f7ab1a4a222d0505b02509088a6260299f6dae23de087d09` |
| `contexts.jsonl` | `07514a0fa27bd863de5517de7468e09b81f9ee3aace0825ed8b9c761ad0e4b28` |
| `manifest.json` | `587cc15307575fd9f9e9c2b4d9fdc0d6c36a890e04635626f82f3eff2006a6af` |
| `aggregates.json` | `6637e468a6185c021012d04c93bea229b5d8fab4691520bd9bbba6c876733429` |
| `decision_readiness.svg` | `50a6bfaf07c9ee058eedb64a652438014cbf2f37c60fdb1a1849487e2098cb21` |

## Boundaries

This result isolates memory selection over structured events whose entities,
fields, supersession links, and dependencies are already known. It does not test
semantic extraction from raw conversations, LLM intelligence, production tool
reliability, net multi-agent benefit, CloudWalk traffic, or customer outcomes.
The four synthetic families are regression fixtures, not a representative
sample of all agent sessions. “Contradiction” in v1 is a conservative stale-
exposure proxy. A cold first load of `cl100k_base` may populate tiktoken's local
vocabulary cache over HTTPS; the benchmark itself makes no model/provider calls.

The full preregistration and pre-run amendments are in
[`docs/rcwt_session_preregistration.md`](../../docs/rcwt_session_preregistration.md).
