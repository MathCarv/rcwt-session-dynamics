# R1: newly authorized confirmation after an interrupted v4 cohort

This is the one explicitly authorized fresh 32-pair cohort, registered before generating dataset seed 2026091210. It reuses the selected development attempt 02 and all fifteen v4 sources unchanged. The earlier confirmation was interrupted at 173/512 decisions and remains INCOMPLETE; it is not resumed, silently replaced, pooled into this cohort, or used to tune this candidate.

Registration SHA-256: `0afb3d5b4ce22f27c06a91a963914be409ec61d51ea57f192a9469a0f1282f12`. Selected development protocol: `750768b5615a15c6a0c0542bf7464045116fa8587dd3f7b61f894dba124de73b`. Preserved interrupted protocol: `b09076db9d9311e7699d072a223bbd62d95b112fbb7973bbab87c950af101654`; traces: `beb01cde1aa808cc2204344bd9adef27fc0ac7ac10f0da8434c14501b2607676`.

Separate earlier-interruption costs: 421 persisted completed calls; 646649 prompt tokens and 76320 completion tokens; 1353.636752 s recorded inference. One additional canceled server task has no final persisted response or final usage. These prior totals are incomplete and are excluded from every R1 quality, cost and timing statistic below. Both previous development attempts and all earlier evidence remain retained; their costs are separately documented in the preserved interruption accounting.

R1 tests new parameter instances from the same four synthetic families under one model and inference seed. It does not establish transfer to unseen domains, CloudWalk customer data, production traffic, autonomous learning or recursive self-improvement. Source hashes and local telemetry establish auditable consistency, not independent physical attestation or proof that no unlisted runs occurred.

The mean-gain threshold is at least 10 percentage points and the paired-episode 95% bootstrap lower endpoint must exceed zero; this does not prove the population gain exceeds 10 points. Both observed monetary safety guards must also pass. References follow each arm's own prior fictional ledger. Integrity PASS alone is not a gain result. Local API charge is US$0; electricity, hardware and total monetary cost remain unknown.

New server generation counts and process start/stop evidence are reconciled separately from the preserved earlier server logs. No publication is authorized.
# RCWT-S Online v4: paired memory-and-public-context confirmation

Result: **PASS - accuracy criterion and descriptive safety guards met**.

This compares engineered deterministic memory storage plus unified retained-and-current public-evidence reading with the unchanged LLM summary memory policy, conditional on the same frozen two-pass actor in both arms. The structured writer is unchanged from v3; the new reader resolves current observations together with retained source records. Both arms also use the same public-task-bound output schema. Each actor generates a short free-text plan and then a final JSON action; only the final action is executed. The structured writer and reader make no LLM memory calls. This is not autonomous policy learning or a production-deployment certificate.

Both arms use the same planning and final prompts, review instruction, local model and 512-token cap per pass. The planning wrapper retains the v3.3 workflow and the public rules. A shared public-task-bound output schema adds the literal case_id and operation-compatible decision/reason vocabulary. It reads only task identity and operation, not observations, memory or self-reported evidence; amounts and evidence-check fields remain unconstrained by those facts. No post-generation semantic veto, deterministic action repair, oracle feedback, fallback to the plan or correctness-triggered retry is applied. Semantically unsafe decisions remain possible. This is not the unchanged end-to-end v3 actor workflow and does not isolate the effect of planning, the schema or self-review.

The intervention includes storage, public-record joins, derived-field calculations and unified reading of retained and current facts, not compaction alone or only memory retention. Deterministic code resolves same-source/same-ID updates and recomputes affected links before selecting the requested case. It does not choose the action, consult an oracle, infer absent facts or retrieve discarded history. Any gain describes the combined memory and public-context system under the common actor; it is not proof of an intrinsic LLM improvement.

Stored state and the actor's unified context each have a 256-token cap. The actor receives the context rather than both stored text and context; the raw current-step input stays unchanged. The reader imports current observation values, so even an empty stored state can yield a nonempty context and first-step inputs can differ between arms. The final pass additionally sees its own raw text plan. That plan is never parsed into a tool call and never passed to the memory writer; the writer still receives the stored state.

Recorded model: `rcwt-local-qwen35-4b`. Split: `test`. Paired episodes: 32 (protocol test count: 32); 8 decisions per episode per arm.
The final development revision adds the same constant ordered-public-rule walkthrough to both system prompts. It reiterates first-applicable-rule priority, retention semantics and coherent decision/reason/amount generation. Code does not evaluate predicates, select actions or repair outputs. The writer and unified reader remain unchanged from the first v4 development attempt. This is a contemporaneous comparison under the shared guide, not an ablation of that guide or a comparison against an older baseline score.
Offline evidence verification: **PASS**. Verified steps: 512.
Verification scope: Registered R1 sources, runtime configuration, history, corpus and schedule; exact offline replay of every actor input, raw completion, tokenization, action, ledger effect, grade, counter and recorded timing.

## Primary comparison

`structured` minus `summary`: **+33.59 pp**, two-sided 95% CI **[+27.34, +39.45] pp**; paired episode wins/ties/losses: 30/1/1.
Accuracy gate: paired episode mean delta >= 10 percentage points AND lower endpoint of two-sided paired bootstrap 95% CI > 0. Numerically met: yes.
Bootstrap: 10,000 percentile resamples, seed 2026091207; unit: paired episode. The 512 decisions are not independent statistical units.

| Policy | Exact actions | Episode-macro success | Fully successful episodes | Tokens / step | Model calls | Decision p50 / p95 (s) | Step p50 / p95 (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| summary | 152/256 | 59.38% | 2/32 | 4552.55 | 736 | 8.443 / 12.521 | 10.460 / 14.936 |
| structured | 238/256 | 92.97% | 19/32 | 3797.16 | 512 | 7.684 / 11.464 | 7.723 / 11.532 |

Decision latency sums draft and final-review generation time. Step latency also includes memory compaction, unified public-evidence reading, tokenization and other measured per-step work. Episode wall time includes all steps. No timing uncertainty or universal speed claim is inferred from these descriptive measurements.

## Descriptive safety guards

| Observed counter | Summary | Structured | Difference | No worse observed |
| --- | ---: | ---: | ---: | --- |
| unsafe_actions | 11 | 4 | -7 | yes |
| unsafe_booked_cents | 743614 | 297683 | -445931 | yes |

Both descriptive guards met: **yes**. Unsafe monetary attempts include rejected calls; unsafe booked cents count only accepted fictional ledger effects. These aggregate observations are not a statistical noninferiority test, real transactions, or a production safety guarantee.

## Failure and resource accounting

- `summary`: unnecessary_deferral=67, unsafe_execution=11, wrong_decision=22, wrong_reason=4; valid actions 256/256; prompt/completion tokens 1033346/132106 (1165452 total); 8 memory truncations; accounted episode wall time 2709.096 s.
- `structured`: unnecessary_deferral=7, unsafe_execution=4, wrong_decision=4, wrong_reason=3; valid actions 256/256; prompt/completion tokens 873122/98952 (972074 total); 10 memory truncations; accounted episode wall time 2071.715 s.

Both test arms: 1248 model calls; 2137526 tokens; 4780.811 s accounted episode wall time. Tokens, calls, and wall time include both actor passes, memory compaction and unified public-evidence reading; earlier development, setup, and unmeasured inter-episode overhead are excluded.

API charge: **US$0** under the local-only protocol. Total monetary cost: **unknown, not zero**; electricity and hardware depreciation are not priced.

## By family (descriptive only)

| Family | Paired episodes | Summary success | Structured success | Delta (pp) |
| --- | ---: | ---: | ---: | ---: |
| evidence-chain | 8 | 50.00% | 96.88% | +46.88 |
| insufficient-evidence | 8 | 79.69% | 96.88% | +17.19 |
| topic-return | 8 | 48.44% | 81.25% | +32.81 |
| updated-state | 8 | 59.38% | 96.88% | +37.50 |

## Limits

- The last development revision appends a constant ordered walkthrough of the existing public rules to both actors. The model evaluates every condition and chooses its own action; the guide is not an executable guard or repair. This confirmation compares contemporaneous arms under that common guide, not the earlier development actor, and does not isolate the guide's effect.
- The structured candidate combines the unchanged engineered deterministic v3 memory writer with a new unified reader of retained and current public evidence. It is not an autonomously learned policy, model-weight update, or evidence of recursive self-improvement.
- This compares combined storage and public-evidence organization against the unchanged summary memory policy, conditional on the same frozen two-pass actor and public-task-bound output schema in both arms. The shared schema changes generation relative to v3.3; this is not the unchanged end-to-end v2 or v3 actor workflow and does not isolate a planning, schema, or self-review effect.
- The intervention is not compaction alone or a memory-only change. Selection, public-record joins, derived-field calculations and resolution of current updates move work from the LLM to deterministic code. Any observed gain belongs to the combined memory and public-context system under this shared actor, not to a demonstrated intrinsic improvement of the LLM.
- Both arms always generate a short plain-text plan without an action schema, then one final JSON action, each with a 512-token output cap. Code never parses the plan into an action or rule choice. Only the final action is executed. No post-generation semantic veto, deterministic action repair, oracle feedback, draft fallback, or selective retry based on correctness is applied.
- The public-task-bound final schema is identical across policies for a given task. It constrains only the literal case_id and operation-compatible decision/reason enums. It does not condition on evidence, compute amounts, choose the first applicable rule, repair model output, or certify safety. The unchanged extractor still processes the final model answer.
- The reader unifies serialized retained facts with current public tool observations, resolving records by source and literal identifiers before selecting the requested case. Unlike v3, it imports current observation values and recomputes affected joins. It does not recover discarded history, infer unobserved statuses, choose a decision, or use the private reference. The independent verifier must bind its input, derived context and actual actor request.
- Stored memory and the unified actor context are separately capped at 256 actual model tokens; the actor receives the context, not both texts concatenated, alongside the unchanged raw current step. Current observations can produce a nonempty context even when stored memory is empty, so first-step inputs can differ between arms. The final pass receives the original actor prompt and its own raw plan. The plan is transient and never enters the memory writer, which consumes the stored state rather than its selected context.
- New held-out instances come from the same four synthetic task families and generator. This is not transfer to unseen families, production traffic, or other domains.
- The confidence interval resamples paired episodes, not individual decisions; it covers generator episode sampling, not variation across inference seeds, models, or hardware.
- Family comparisons are descriptive and not multiplicity-adjusted; only structured versus summary over the complete frozen cohort is the primary comparison.
- Observed safety counters are descriptive guards, not statistical noninferiority evidence or production safety certification. Lower aggregate counts can conceal case-specific regressions.
- Tokens and step/episode latency include both actor passes, memory compaction and unified public-evidence reading; local tokenization and read processing are included in step time. Decision latency sums draft and final-review generation time. Local latency also depends on hardware, load, cache state, and execution order.
- API charge is US$0 under the local-only protocol; electricity, hardware depreciation, and total monetary cost are unknown, not zero.
- Evaluation totals exclude earlier development and setup. V2 and v3 evidence informs development of this new v4 series and is not pooled into its confirmation; both historical negative results remain unchanged.
- Summary validation and statistical calculation alone do not verify actual inference, corpus novelty, pre-test freezing, or raw trace integrity; those require the independent evidence verifier.
- A passing accuracy/safety gate does not require or certify lower latency, fewer tokens, or production deployability.

Production deployability: **not_evaluated**.
Normalized input SHA-256: `c343b8bd6533a180a93879ca0d0a6f99ba98ef5d91e88bc29d1cfe4fb1e1bd92`.
