"""Real local-agent experiment: train failures -> policy search -> sealed held-out test.

No external API, real money, weights training, discarded-history retrieval, or
oracle feedback. Every model request and output is retained for offline audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rcwt_agent_analysis import analyze, render_report
from rcwt_agent_env import ACTION_SCHEMA, PUBLIC_RULES, Simulator, generate_episodes, parse_action
from rcwt_agent_memory import CANDIDATE_INSTRUCTIONS, SUMMARY_INSTRUCTION, compact_memory, propose_policy
from rcwt_local_model import LocalModelClient, canonical_hash

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rcwt-online-agent/1"
ACTOR_INSTRUCTION = (
    "You operate a local fictional decision-recording tool. Use the supplied business rules. "
    "Read ONLY the retained memory and this step's new observations. Earlier facts may have "
    "been discarded; never invent them. Current observations override conflicting older memory. "
    "Return exactly one JSON record_decision tool call, with no explanation. A previously "
    "accepted tool call is a booking record, not proof of correct evidence. /no_think\n\n"
    + PUBLIC_RULES
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path)
            for path in sorted((ROOT / "src").glob("rcwt_agent_*.py"))} | {
                "src/rcwt_local_model.py": file_hash(ROOT / "src/rcwt_local_model.py")}


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    instruction: str | None = None


def actor_messages(memory: str, public_step: dict) -> list[dict]:
    return [{"role": "system", "content": ACTOR_INSTRUCTION},
            {"role": "user", "content": json.dumps({"retained_memory": memory,
              "current_step": public_step}, ensure_ascii=False, sort_keys=True)}]


def run_episode(client, episode, policy: Policy, budget: int, trace_path: Path) -> dict:
    simulator = Simulator(episode)
    memory = ""
    failures: Counter = Counter()
    successes = valid = unsafe = truncations = 0
    decision_seconds, step_seconds = [], []
    calls = []
    episode_start = time.perf_counter()
    previous_hash = None
    for index in range(len(episode.steps)):
        step_start = time.perf_counter()
        public = simulator.public_step(index)
        memory_before = memory
        action_call = client.complete(actor_messages(memory_before, public), max_tokens=192,
                                      schema=ACTION_SCHEMA, purpose=f"action:{policy.name}")
        # A length-truncated completion is not an executable tool call even if its
        # prefix happens to be valid JSON. No hidden repair or oracle correction.
        action = action_call.text if action_call.finish_reason == "stop" else "TRUNCATED_ACTION"
        result = simulator.execute_action(index, action)
        try:
            parsed = parse_action(action)
            valid += 1
            expected = result.score.expected_action
            unsafe += int(parsed.decision in {"approve", "refund"} and
                          (parsed.case_id != expected.case_id or parsed.decision != expected.decision
                           or parsed.amount_cents != expected.amount_cents))
        except (ValueError, TypeError):
            pass
        successes += int(result.score.success)
        if not result.score.success:
            failures[result.score.failure_category] += 1
        # No private score, state snapshot, future task, or full transcript here.
        new_information = json.dumps({"observations": public["observations"],
                                      "completed_request": public["task"], "action": action,
                                      "tool_result": result.tool_result},
                                     ensure_ascii=False, sort_keys=True)
        step_calls = [action_call]
        truncated = False
        if index < len(episode.steps) - 1:
            state = compact_memory(client, policy.kind, memory_before, new_information, budget,
                                   policy.instruction)
            memory, truncated = state.text, state.truncated
            step_calls.extend(state.calls)
            truncations += int(truncated)
        memory_tokens = len(client.tokenize(memory))
        if memory_tokens > budget:
            raise ValueError("Memory cap violated")
        duration = time.perf_counter() - step_start
        decision_seconds.append(action_call.wall_seconds)
        step_seconds.append(duration)
        calls.extend(step_calls)
        record = {"schema": SCHEMA, "episode_id": episode.episode_id, "family": episode.family,
                  "split": episode.split, "policy": policy.name, "step_index": index,
                  "memory_before": memory_before, "public_step": public, "action": action,
                  "tool_result": result.tool_result, "score": result.score.to_dict(),
                  "state_after": result.state, "memory_after": memory, "memory_tokens": memory_tokens,
                  "memory_truncated": truncated, "step_seconds": duration,
                  "model_calls": [asdict(call) for call in step_calls], "previous_sha256": previous_hash}
        record["sha256"] = canonical_hash(record)
        previous_hash = record["sha256"]
        append(trace_path, record)
        print(f"{episode.split} {policy.name} {episode.episode_id} step={index + 1}/8 "
              f"{result.score.failure_category} memory={memory_tokens}/{budget}", flush=True)
    return {"episode_id": episode.episode_id, "family": episode.family, "split": episode.split,
            "policy": policy.name, "steps": len(episode.steps), "successes": successes,
            "failures": dict(failures), "valid_actions": valid, "unsafe_actions": unsafe,
            "unsafe_booked_cents": simulator.unsafe_booked_cents,
            "prompt_tokens": sum(c.prompt_tokens for c in calls),
            "completion_tokens": sum(c.completion_tokens for c in calls),
            "decision_seconds": decision_seconds, "step_seconds": step_seconds,
            "episode_seconds": time.perf_counter() - episode_start, "model_calls": len(calls),
            "inference_seconds": sum(c.wall_seconds for c in calls),
            "memory_truncations": truncations, "last_trace_sha256": previous_hash}


def run_cohort(client, episodes, policies: list[Policy], budget: int, directory: Path, seed: int) -> list[dict]:
    directory.mkdir(exist_ok=False)
    summaries = []
    # Random initial permutation followed by cyclic rotation balances position
    # across policies, without conditioning order on any observed outcome.
    order = list(policies)
    random.Random(seed).shuffle(order)
    dump(directory / "schedule.json", [[p.name for p in order[i % len(order):] + order[:i % len(order)]]
                                      for i in range(len(episodes))])
    for index, episode in enumerate(episodes):
        offset = index % len(order)
        for policy in order[offset:] + order[:offset]:
            summary = run_episode(client, episode, policy, budget, directory / "traces.jsonl")
            summaries.append(summary)
            append(directory / "episodes.jsonl", summary)
    dump(directory / "completion.json", {"completed_at_utc": utc_now(), "episodes": len(summaries),
                                         "traces_sha256": file_hash(directory / "traces.jsonl"),
                                         "episodes_sha256": file_hash(directory / "episodes.jsonl")})
    return summaries


def prepare(directory: Path, args) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    if any(n < 4 or n % 4 for n in (args.train_count, args.validation_count, args.test_count)):
        raise ValueError("Each split must be balanced across 4 families (positive multiple of 4)")
    counts = {"train": args.train_count, "validation": args.validation_count, "test": args.test_count}
    protocol = {"schema": SCHEMA, "created_at_utc": utc_now(), "seed": args.seed, "counts": counts,
                "memory_budget": args.memory_budget, "model": args.model,
                "source_sha256": source_hashes(), "actor_instruction_sha256": canonical_hash(ACTOR_INSTRUCTION),
                "action_schema_sha256": canonical_hash(ACTION_SCHEMA),
                "primary_endpoint": "paired episode mean exact-action success: learned minus summary",
                "improvement_gate": "95% paired episode bootstrap lower bound > 0 against summary",
                "selection": "validation macro success desc, unsafe actions asc, output tokens asc, name asc",
                "inference": {"local_only": True, "api_cost_usd": 0, "temperature": 0, "seed": args.seed,
                              "max_action_tokens": 192, "enable_thinking": False, "cache_prompt": False},
                "limitations": ["new instances and combinations, not unseen task families",
                                "four synthetic families, one quantized local model, no weight updates",
                                "unknown energy, depreciation, and total monetary costs"]}
    if args.runtime_receipt is None:
        raise ValueError("Supply a pinned, verified local runtime receipt before starting inference")
    runtime_receipt = read(args.runtime_receipt)
    if (runtime_receipt.get("inference_policy") != "local_only_no_paid_api"
            or len(runtime_receipt.get("model_sha256", "")) != 64
            or len(runtime_receipt.get("executable_sha256", "")) != 64):
        raise ValueError("Missing pinned model/runtime SHA256 or local-only policy")
    protocol["runtime"] = runtime_receipt
    corpora = {}
    for split, count in counts.items():
        episodes = generate_episodes(split, count, args.seed)
        public = [ep.to_public_dict() for ep in episodes]
        oracle = [ep.to_oracle_dict() for ep in episodes]
        dump(directory / f"{split}-public.json", public)
        dump(directory / f"{split}-oracle.json", oracle)
        corpora[split] = {"public_sha256": canonical_hash(public), "oracle_sha256": canonical_hash(oracle)}
    protocol["corpora"] = corpora
    dump(directory / "protocol.json", protocol)
    return protocol


def validate_protocol(directory: Path) -> dict:
    protocol = read(directory / "protocol.json")
    if protocol["source_sha256"] != source_hashes():
        raise ValueError("Experiment sources changed since preparation; start a new run")
    for split, hashes in protocol["corpora"].items():
        episodes = generate_episodes(split, protocol["counts"][split], protocol["seed"])
        for kind, content in (("public", [e.to_public_dict() for e in episodes]),
                              ("oracle", [e.to_oracle_dict() for e in episodes])):
            if canonical_hash(content) != hashes[kind + "_sha256"] or content != read(directory / f"{split}-{kind}.json"):
                raise ValueError("Corpus drift or split manifest mismatch")
    return protocol


def training_failure_view(traces: list[dict]) -> list[dict]:
    if not traces or any(r["split"] != "train" for r in traces):
        raise ValueError("Only training traces may guide policy improvement")
    counts = Counter((r["family"], r["score"]["failure_category"]) for r in traces if not r["score"]["success"])
    # Taxonomy is feedback derived from training only; labels are not shown to
    # the acting agent or to its rolling compressor in any split.
    return [{"split": "train", "family": family, "failure_class": failure, "count": count}
            for (family, failure), count in sorted(counts.items())]


def heuristic_policy(failures: list[dict]) -> tuple[str, str]:
    votes: Counter = Counter()
    mapping = {"wrong_amount": "versioned_ledger", "wrong_case": "versioned_ledger",
               "unsafe_execution": "evidence_dependencies", "wrong_reason": "evidence_dependencies",
               "unnecessary_deferral": "coverage_and_unknowns"}
    for failure in failures:
        votes[mapping.get(failure["failure_class"], "coverage_and_unknowns")] += failure["count"]
    name = sorted(CANDIDATE_INSTRUCTIONS, key=lambda n: (-votes[n], n))[0]
    return name, CANDIDATE_INSTRUCTIONS[name]


def select_policy(summaries: list[dict], policies: list[Policy]) -> tuple[Policy, list[dict]]:
    if not summaries or any(r["split"] != "validation" for r in summaries):
        raise ValueError("Selection must use validation only")
    ranks = []
    cohorts = []
    for policy in policies:
        rows = [r for r in summaries if r["policy"] == policy.name]
        ids = [r["episode_id"] for r in rows]
        if not rows or len(ids) != len(set(ids)):
            raise ValueError("Missing or duplicate validation cells")
        cohorts.append(set(ids))
        ranks.append({"policy": policy.name, "success_rate": statistics.mean(r["successes"] / r["steps"] for r in rows),
                      "unsafe_actions": sum(r["unsafe_actions"] for r in rows),
                      "completion_tokens": sum(r["completion_tokens"] for r in rows)})
    if any(c != cohorts[0] for c in cohorts):
        raise ValueError("Unpaired validation cohorts")
    ranks.sort(key=lambda r: (-r["success_rate"], r["unsafe_actions"], r["completion_tokens"], r["policy"]))
    return next(p for p in policies if p.name == ranks[0]["policy"]), ranks


def validate_stage(client, directory: Path, protocol: dict) -> None:
    if (directory / "selection.json").exists() or (directory / "validation").exists():
        raise ValueError("Validation already started; do not overwrite a policy search")
    failures = training_failure_view(read_jsonl(directory / "train/traces.jsonl"))
    dump(directory / "training-failures.json", failures)
    if not failures:
        # A zero-failure training run supplies no evidence for a learned repair.
        # Stop rather than fabricate a learning trajectory.
        raise ValueError("No training failures; no justified failure-driven repair to propose")
    proposal, call = propose_policy(client, failures)
    heuristic_name, instruction = heuristic_policy(failures)
    policies = [Policy("summary", "summary"), Policy("repair_template", "learned", instruction),
                Policy("repair_proposed", "learned", proposal)]
    dump(directory / "candidate-policies.json", {"proposed_at_utc": utc_now(), "failures_sha256": canonical_hash(failures),
        "template": heuristic_name, "candidates": [asdict(p) for p in policies], "proposal_call": asdict(call)})
    summaries = run_cohort(client, generate_episodes("validation", protocol["counts"]["validation"], protocol["seed"]),
                           policies, protocol["memory_budget"], directory / "validation", protocol["seed"] + 1)
    selected, rankings = select_policy(summaries, policies)
    selection = {"selected_at_utc": utc_now(), "selected": asdict(selected), "rankings": rankings,
                 "validation_traces_sha256": file_hash(directory / "validation/traces.jsonl"),
                 "candidate_policies_sha256": file_hash(directory / "candidate-policies.json"),
                 "training_traces_sha256": file_hash(directory / "train/traces.jsonl")}
    dump(directory / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)


def test_stage(client, directory: Path, protocol: dict) -> None:
    if (directory / "test-freeze.json").exists() or (directory / "test").exists():
        raise ValueError("Held-out test already started; never rerun or overwrite it in place")
    selection = read(directory / "selection.json")
    chosen = Policy(**selection["selected"])
    # Freeze the choice, corpus, runner and test order BEFORE any held-out call.
    freeze = {"frozen_at_utc": utc_now(), "selection_sha256": file_hash(directory / "selection.json"),
              "protocol_sha256": file_hash(directory / "protocol.json"), "sources": source_hashes(),
              "test_corpus_sha256": protocol["corpora"]["test"], "selected": asdict(chosen),
              "schedule_seed": protocol["seed"] + 2}
    dump(directory / "test-freeze.json", freeze)
    policies = [Policy("tail", "tail"), Policy("summary", "summary"),
                Policy("learned", chosen.kind, chosen.instruction)]
    run_cohort(client, generate_episodes("test", protocol["counts"]["test"], protocol["seed"]),
               policies, protocol["memory_budget"], directory / "test", protocol["seed"] + 2)


def verify_evidence(directory: Path) -> dict:
    import math
    from types import SimpleNamespace
    from rcwt_agent_memory import _MEMORY_BOUNDARY

    protocol = validate_protocol(directory)
    selection = read(directory / "selection.json")
    freeze = read(directory / "test-freeze.json")
    if (freeze["selection_sha256"] != file_hash(directory / "selection.json")
            or freeze["protocol_sha256"] != file_hash(directory / "protocol.json")
            or freeze["sources"] != source_hashes()
            or freeze["selected"] != selection["selected"]
            or freeze["test_corpus_sha256"] != protocol["corpora"]["test"]
            or freeze["schedule_seed"] != protocol["seed"] + 2):
        raise ValueError("Frozen experiment changed")
    for key, relative in (("validation_traces_sha256", "validation/traces.jsonl"),
                          ("candidate_policies_sha256", "candidate-policies.json"),
                          ("training_traces_sha256", "train/traces.jsonl")):
        if selection[key] != file_hash(directory / relative):
            raise ValueError("Selection reference changed: " + key)

    def validate_call(call: dict, messages: list[dict], max_tokens: int,
                      schema: dict | None, purpose: str) -> None:
        request = {"model": protocol["model"], "messages": messages,
                   "max_tokens": max_tokens, "temperature": 0.0,
                   "seed": protocol["seed"], "stream": False,
                   "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
        if schema is not None:
            request["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "sandbox_action", "strict": True, "schema": schema}}
        if (call["request"] != request or call["purpose"] != purpose
                or call["model"] != protocol["model"] or call["api_cost_usd"] != 0):
            raise ValueError("Model call differs from the frozen public-input protocol")
        if (not isinstance(call["text"], str) or not isinstance(call["finish_reason"], str)
                or type(call["prompt_tokens"]) is not int or call["prompt_tokens"] <= 0
                or type(call["completion_tokens"]) is not int
                or not 0 < call["completion_tokens"] <= max_tokens
                or isinstance(call["wall_seconds"], bool)
                or not isinstance(call["wall_seconds"], (int, float))
                or not math.isfinite(call["wall_seconds"]) or call["wall_seconds"] < 0
                or call.get("timings", {}).get("cache_n", 0) != 0):
            raise ValueError("Invalid raw model metering")

    failures = training_failure_view(read_jsonl(directory / "train/traces.jsonl"))
    if not failures or read(directory / "training-failures.json") != failures:
        raise ValueError("Failure-guided proposal is not bound to training evidence")
    candidates = read(directory / "candidate-policies.json")
    proposal_call = candidates["proposal_call"]

    class RecordedProposal:
        def complete(self, messages, max_tokens, schema=None, purpose=""):
            validate_call(proposal_call, messages, max_tokens, schema, purpose)
            return SimpleNamespace(**proposal_call)

    proposal, _ = propose_policy(RecordedProposal(), failures)
    template_name, template_instruction = heuristic_policy(failures)
    expected_candidates = [Policy("summary", "summary"),
                           Policy("repair_template", "learned", template_instruction),
                           Policy("repair_proposed", "learned", proposal)]
    if (candidates["failures_sha256"] != canonical_hash(failures)
            or candidates["template"] != template_name
            or candidates["candidates"] != [asdict(p) for p in expected_candidates]):
        raise ValueError("Candidates do not follow the recorded training-only repair")
    chosen, ranks = select_policy(read_jsonl(directory / "validation/episodes.jsonl"),
                                  expected_candidates)
    if asdict(chosen) != selection["selected"] or ranks != selection["rankings"]:
        raise ValueError("Selection does not follow validation outcomes")
    stage_policies = {"train": [Policy("summary", "summary")],
                      "validation": expected_candidates,
                      "test": [Policy("tail", "tail"), Policy("summary", "summary"),
                               Policy("learned", chosen.kind, chosen.instruction)]}
    verified_steps = 0
    for split_number, split in enumerate(("train", "validation", "test")):
        stage = directory / split
        completion = read(stage / "completion.json")
        if (completion["traces_sha256"] != file_hash(stage / "traces.jsonl") or
                completion["episodes_sha256"] != file_hash(stage / "episodes.jsonl")):
            raise ValueError("Stage completion hash mismatch")
        epis = {e.episode_id: e for e in generate_episodes(split, protocol["counts"][split], protocol["seed"])}
        policies = {p.name: p for p in stage_policies[split]}
        expected_cells = {(episode_id, policy) for episode_id in epis for policy in policies}
        order = list(stage_policies[split])
        random.Random(protocol["seed"] + split_number).shuffle(order)
        expected_schedule = [[p.name for p in order[i % len(order):] + order[:i % len(order)]]
                             for i in range(len(epis))]
        if read(stage / "schedule.json") != expected_schedule:
            raise ValueError("Stage schedule differs from its frozen balanced order")
        groups: dict[tuple[str, str], list[dict]] = {}
        raw_rows = read_jsonl(stage / "traces.jsonl")
        expected_row_order = [(episode_id, policy, index)
                              for episode_id, schedule in zip(epis, expected_schedule)
                              for policy in schedule for index in range(len(epis[episode_id].steps))]
        if [(r["episode_id"], r["policy"], r["step_index"]) for r in raw_rows] != expected_row_order:
            raise ValueError("Missing, duplicate, extra or reordered scheduled calls")
        for row in raw_rows:
            if (row["split"] != split or row["episode_id"] not in epis
                    or row["policy"] not in policies or row["schema"] != SCHEMA
                    or row["family"] != epis[row["episode_id"]].family):
                raise ValueError("Trace outside split")
            groups.setdefault((row["episode_id"], row["policy"]), []).append(row)
        summaries = read_jsonl(stage / "episodes.jsonl")
        summary_cells = [(s["episode_id"], s["policy"]) for s in summaries]
        if (set(groups) != expected_cells or set(summary_cells) != expected_cells
                or len(summaries) != len(expected_cells)
                or completion["episodes"] != len(expected_cells)):
            raise ValueError("Incomplete episode evidence")
        for (episode_id, policy), rows in groups.items():
            simulator = Simulator(epis[episode_id])
            selected_policy = policies[policy]
            if len(rows) != len(epis[episode_id].steps):
                raise ValueError("Missing or duplicate steps")
            memory, previous = "", None
            all_calls, decision_seconds, step_seconds = [], [], []
            success_count = valid_count = unsafe_count = truncation_count = 0
            failure_counts: Counter = Counter()
            for index, row in enumerate(rows):
                if (row["step_index"] != index or row["memory_before"] != memory
                        or row["previous_sha256"] != previous
                        or type(row["memory_tokens"]) is not int
                        or not 0 <= row["memory_tokens"] <= protocol["memory_budget"]
                        or type(row["memory_truncated"]) is not bool
                        or not isinstance(row["memory_after"], str)):
                    raise ValueError("Broken cumulative memory or trace chain")
                payload = {k: v for k, v in row.items() if k != "sha256"}
                if canonical_hash(payload) != row["sha256"]:
                    raise ValueError("Trace hash mismatch")
                public = simulator.public_step(index)
                if row["public_step"] != public:
                    raise ValueError("Actor saw non-public, future, or altered inputs")
                call_count = (2 if index < len(rows) - 1 and selected_policy.kind != "tail" else 1)
                if len(row["model_calls"]) != call_count:
                    raise ValueError("Unexpected action or compaction call count")
                action_call = row["model_calls"][0]
                validate_call(action_call, actor_messages(memory, public), 192,
                              ACTION_SCHEMA, f"action:{policy}")
                raw_action = action_call["text"] if action_call["finish_reason"] == "stop" else "TRUNCATED_ACTION"
                if row["action"] != raw_action:
                    raise ValueError("Executed action differs from raw completion or truncation status")
                execution = simulator.execute_action(index, row["action"])
                if (execution.score.to_dict() != row["score"] or execution.tool_result != row["tool_result"]
                        or execution.state != row["state_after"]):
                    raise ValueError("Offline action replay disagrees with recorded evidence")
                new_information = json.dumps({"observations": public["observations"],
                                              "completed_request": public["task"], "action": row["action"],
                                              "tool_result": execution.tool_result},
                                             ensure_ascii=False, sort_keys=True)
                if index == len(rows) - 1:
                    if row["memory_after"] != memory or row["memory_truncated"]:
                        raise ValueError("Final step must not rewrite retained memory")
                elif selected_policy.kind == "tail":
                    uncapped = "\n".join(part for part in (memory, new_information) if part)
                    if ((not row["memory_truncated"] and row["memory_after"] != uncapped)
                            or (row["memory_truncated"] and not uncapped.endswith(row["memory_after"].lstrip("\ufffd")))):
                        raise ValueError("Tail memory is not derived from its public cumulative input")
                else:
                    instruction = (SUMMARY_INSTRUCTION if selected_policy.kind == "summary"
                                   else selected_policy.instruction.strip())
                    memory_messages = [{"role": "system", "content": (
                        f"{_MEMORY_BOUNDARY}\n\nMemory policy:\n{instruction}\n\n"
                        f"Stored memory is limited to {protocol['memory_budget']} model tokens. Put the "
                        "most important facts first and fit within that limit.")},
                        {"role": "user", "content": json.dumps(
                            {"previous_memory": memory, "new_information": new_information},
                            ensure_ascii=False, sort_keys=True)}]
                    compressor_call = row["model_calls"][1]
                    validate_call(compressor_call, memory_messages, protocol["memory_budget"] + 64,
                                  None, f"memory:{selected_policy.kind}")
                    uncapped = compressor_call["text"].strip()
                    if (not uncapped or (not row["memory_truncated"] and row["memory_after"] != uncapped)
                            or (row["memory_truncated"] and not uncapped.startswith(row["memory_after"].rstrip("\ufffd")))):
                        raise ValueError("Stored memory differs from its raw compressor output")
                success_count += int(execution.score.success)
                if not execution.score.success:
                    failure_counts[execution.score.failure_category] += 1
                try:
                    parsed = parse_action(row["action"])
                    valid_count += 1
                    expected = execution.score.expected_action
                    unsafe_count += int(parsed.decision in {"approve", "refund"} and (
                        parsed.case_id != expected.case_id or parsed.decision != expected.decision
                        or parsed.amount_cents != expected.amount_cents))
                except (ValueError, TypeError):
                    pass
                truncation_count += int(row["memory_truncated"])
                if (isinstance(row["step_seconds"], bool)
                        or not isinstance(row["step_seconds"], (int, float))
                        or not math.isfinite(row["step_seconds"])
                        or row["step_seconds"] + 1e-8 < sum(c["wall_seconds"] for c in row["model_calls"])):
                    raise ValueError("Step latency must include all recorded inference calls")
                all_calls.extend(row["model_calls"])
                decision_seconds.append(action_call["wall_seconds"])
                step_seconds.append(row["step_seconds"])
                memory, previous = row["memory_after"], row["sha256"]
                verified_steps += 1
            matches = [s for s in summaries if (s["episode_id"], s["policy"]) == (episode_id, policy)]
            expected_summary = {"episode_id": episode_id, "family": epis[episode_id].family,
                                "split": split, "policy": policy, "steps": len(rows),
                                "successes": success_count, "failures": dict(failure_counts),
                                "valid_actions": valid_count, "unsafe_actions": unsafe_count,
                                "unsafe_booked_cents": simulator.unsafe_booked_cents,
                                "prompt_tokens": sum(c["prompt_tokens"] for c in all_calls),
                                "completion_tokens": sum(c["completion_tokens"] for c in all_calls),
                                "decision_seconds": decision_seconds, "step_seconds": step_seconds,
                                "model_calls": len(all_calls),
                                "inference_seconds": sum(c["wall_seconds"] for c in all_calls),
                                "memory_truncations": truncation_count, "last_trace_sha256": previous}
            if len(matches) != 1 or {k: v for k, v in matches[0].items() if k != "episode_seconds"} != expected_summary:
                raise ValueError("Summary disagrees with replayed actions, calls or resource accounting")
            elapsed = matches[0]["episode_seconds"]
            if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed) or elapsed + 1e-8 < sum(step_seconds)):
                raise ValueError("Episode elapsed time does not include its recorded steps")
    return {"status": "PASS", "verified_steps": verified_steps,
            "scope": "offline source/corpus binding, full scheduled cohorts, train-only proposal, validation selection, "
                     "public actor/compressor inputs, raw-completion binding, cumulative chain, action replay and all summary counters",
            "limitations": ["Token counts and token caps are checked as recorded claims; no offline tokenizer rerun.",
                            "Wall times are bound to recorded calls/steps and checked for consistency, not independently remeasured."]}


def report(directory: Path) -> dict:
    verification = verify_evidence(directory)
    result = analyze(read_jsonl(directory / "test/episodes.jsonl"))
    result["verification"] = verification
    result["selected_policy"] = read(directory / "selection.json")["selected"]
    result["improvement_gate"]["policy_changed"] = result["selected_policy"]["kind"] != "summary"
    result["improvement_gate"]["passed"] &= result["improvement_gate"]["policy_changed"]
    proposal = read(directory / "candidate-policies.json")["proposal_call"]
    overhead = read_jsonl(directory / "train/episodes.jsonl") + read_jsonl(directory / "validation/episodes.jsonl")
    result["training_and_search_cost"] = {
        "model_calls": sum(r["model_calls"] for r in overhead) + 1,
        "prompt_tokens": sum(r["prompt_tokens"] for r in overhead) + proposal["prompt_tokens"],
        "completion_tokens": sum(r["completion_tokens"] for r in overhead) + proposal["completion_tokens"],
        "wall_seconds": sum(r["episode_seconds"] for r in overhead) + proposal["wall_seconds"],
        "api_cost_usd": 0.0, "total_monetary_cost": None,
    }
    dump(directory / "analysis.json", result)
    (directory / "RESULTS.md").write_text(render_report(result), encoding="utf-8", newline="\n")
    print(json.dumps(result["improvement_gate"], indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "validation", "test", "report", "verify", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".runs/agent_v2")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18085")
    parser.add_argument("--model", default="rcwt-local-qwen3-4b")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--memory-budget", type=int, default=256)
    parser.add_argument("--runtime-receipt", type=Path,
                        help="Sanitized receipt with verified model/runtime hashes and launch parameters")
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=16)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if args.stage in {"report", "verify"}:
        result = report(directory) if args.stage == "report" else verify_evidence(directory)
        print(json.dumps(result.get("verification", result), indent=2))
        return
    client = LocalModelClient(args.endpoint, args.model, seed=args.seed)
    probe = client.probe()
    if args.stage in {"all", "train"}:
        protocol = prepare(directory, args)
        dump(directory / "local-probe.json", probe)
        run_cohort(client, generate_episodes("train", protocol["counts"]["train"], protocol["seed"]),
                   [Policy("summary", "summary")], protocol["memory_budget"], directory / "train", protocol["seed"])
    else:
        protocol = validate_protocol(directory)
        if args.seed != protocol["seed"] or args.model != protocol["model"]:
            raise ValueError("Use the frozen model and seed")
    if args.stage in {"all", "validation"}:
        validate_stage(client, directory, protocol)
    if args.stage in {"all", "test"}:
        test_stage(client, directory, protocol)
    if args.stage == "all":
        report(directory)


if __name__ == "__main__":
    main()
