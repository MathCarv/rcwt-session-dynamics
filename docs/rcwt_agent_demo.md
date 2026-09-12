# Three-minute recorded-agent demonstration

This demo replays one already recorded episode. It does not start a model, call a provider, re-time inference, sleep between steps or generate replacement results. The terminal banner is **RECORDED REPLAY / NO MODEL CALLS**. All invoice amounts, tool effects and ledger entries are fictional.

The replay requires a held-out freeze and a successful full offline evidence verification before displaying any episode. Its default is episode index **0** in the frozen public test manifest, not the episode with the best outcome. Keep that index when comparing policies. The replay also displays failures.

The included run did **not** demonstrate learned memory improvement. Validation retained the original summary policy, so the `learned` and `summary` arms have identical decisions. The [aggregate report](../results/agent_v2/RESULTS.md) records 42.2% accuracy for these arms versus 35.9% for tail, with an inconclusive paired interval versus tail. Present the demo as an inspectable local-agent experiment, not a successful self-improvement claim.

## 0:00-0:30: verify the saved experiment

From the repository root, with its Python dependencies installed:

```powershell
python src/rcwt_agent_run.py --stage verify --output-dir results/agent_v2
```

Explain what this checks: the saved sources and corpora, training-only proposal, frozen validation selection, complete scheduled cohorts, cumulative memory, raw model outputs, tool execution and episode summaries. The token counts and elapsed times remain recorded claims; this command does not independently remeasure model tokens or runtime. A missing/incomplete run or changed experiment sources must fail verification.

## 0:30-2:00: replay all eight recorded decisions

```powershell
python tools/replay_agent_episode.py --run-dir results/agent_v2 --episode-index 0 --policy summary
```

For each step, point to the newly supplied observations, the actor's self-reported evidence check, its actual extracted tool call and the resulting fictional tool receipt. An accepted booking is not proof that the decision was correct. The section marked **PRIVATE EVALUATION** shows the recorded reference grade, which was never given to the acting model or rolling-memory compressor. The ledger totals show the consequences of actions that were actually booked in the simulator.

The displayed decision latency belongs to the saved action call. Step latency also includes memory work. Prompt and completion token totals include the model calls recorded for that step. They are replayed numbers, not a claim about how quickly this terminal demo executes.

## 2:00-3:00: inspect cumulative memory or the paired arm

Use the same episode index for the validation-selected policy arm:

```powershell
python tools/replay_agent_episode.py --run-dir results/agent_v2 --episode-index 0 --policy learned
```

The header prints the frozen selected policy. The arm name `learned` does not guarantee that validation selected a changed policy or that an improvement was established. Refer to the aggregate held-out report for that conclusion; a single episode cannot demonstrate a general gain.

To inspect exactly what survived between decisions:

```powershell
python tools/replay_agent_episode.py --run-dir results/agent_v2 --episode-index 0 --policy summary --show-memory
```

The tool prints the full retained memory before and after every step. It cannot retrieve history that was absent from the recorded input. `--policy tail` is also available for the same fixed episode. API charge was zero under the local inference protocol; electricity, hardware use and total monetary cost remain unknown.

## Verify the failure diagnosis and regenerate the report

The [diagnosis](../results/agent_v2/DIAGNOSIS.md) includes deterministic examples and aggregate comparisons of the actor's evidence check with the private reference. It does not establish causal attribution to memory. Verify its exact saved contents without changing them:

```powershell
python tools/inspect_agent_traces.py --run-dir results/agent_v2 --output-dir results/agent_v2 --verify
```

To reconstruct the aggregate report and analysis from the recorded evidence, without inference:

```powershell
python src/rcwt_agent_run.py --stage report --output-dir results/agent_v2
```

## A new real run is a separate operation

The commands above work without a running model server once verified results are available. To perform new inference, first start the pinned local `llama.cpp` server using the model, executable hashes and configuration in [the runtime receipt](rcwt_agent_runtime.json), at `http://127.0.0.1:18085`. The receipt documents the required text-only, non-thinking setup; model weights are not included in the replay.

Then use a **new, nonexistent output directory** and a separately recorded seed:

```powershell
python src/rcwt_agent_run.py --stage all --output-dir .runs/agent_new_run --runtime-receipt docs/rcwt_agent_runtime.json --endpoint http://127.0.0.1:18085 --model rcwt-local-qwen35-4b --seed 20260912 --train-count 4 --validation-count 4 --test-count 16 --memory-budget 256
```

This command executes the real local agent, proposes and selects a memory policy, and evaluates its frozen held-out cohort. It is not part of the three-minute recorded demo, may take substantially longer, and does not promise a positive result. Do not overwrite an existing run or pool development attempts with the held-out report.

The replay tests use explicitly fake transports and temporary fixtures to verify these display and integrity contracts. Their passing results are software tests, not model-performance evidence.
