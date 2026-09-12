"""Frozen paired local-agent comparison: engineered structured memory vs summary.

The v2 first-pass prompt, tool environment, sampling and summary reducer are
unchanged. Revision 3.2 adds the SAME fixed model self-review to BOTH arms.
This is a separate experiment, not a replacement for earlier negative results.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rcwt_agent_actor import ACTOR_SCHEMA, extract_action, extract_evidence_check
from rcwt_agent_env import Simulator, generate_episodes, parse_action
from rcwt_agent_memory import compact_memory
from rcwt_agent_run import (ACTION_MAX_TOKENS, ROOT, actor_messages, append, dump,
                            file_hash, read, read_jsonl, utc_now)
from rcwt_local_model import CallResult, LocalModelClient, SAMPLING, canonical_hash
from rcwt_memory_v3 import compact_structured
from rcwt_retrieval_v3 import read_memory
from rcwt_review_v3 import review_messages

SCHEMA = "rcwt-online-memory/3.2"
POLICIES = ("summary", "structured")
DATASET_SEEDS = {"development": 2026091201, "confirmatory": 2026091202}
INFERENCE_SEED = 20260911
SCHEDULE_SEED = 2026091204
ACCURACY_GATE = "Mean difference >=10 percentage points AND paired episode bootstrap 95% lower bound >0"
SOURCE_NAMES = (
    "rcwt_agent_actor.py", "rcwt_agent_analysis.py", "rcwt_agent_env.py",
    "rcwt_agent_memory.py", "rcwt_agent_run.py", "rcwt_local_model.py",
    "rcwt_memory_v3.py", "rcwt_online_v3.py", "rcwt_analysis_v3.py", "rcwt_retrieval_v3.py",
    "rcwt_review_v3.py",
)


def sources() -> dict[str, str]:
    return {"src/" + name: file_hash(ROOT / "src" / name) for name in SOURCE_NAMES}


def _payload(model: str, seed: int, arguments: dict) -> dict:
    result = {"model": model, "messages": arguments["messages"],
              "max_tokens": arguments["max_tokens"], **SAMPLING, "seed": seed,
              "stream": False, "cache_prompt": False,
              "chat_template_kwargs": {"enable_thinking": False}}
    if arguments["schema"] is not None:
        result["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "sandbox_action", "strict": True, "schema": arguments["schema"]}}
    return result


def validate_call(call: CallResult, arguments: dict, model: str, seed: int) -> None:
    if (canonical_hash(call.request) != canonical_hash(_payload(model, seed, arguments))
            or call.model != model or call.purpose != arguments["purpose"]
            or type(call.prompt_tokens) is not int or call.prompt_tokens <= 0
            or type(call.completion_tokens) is not int
            or not 0 < call.completion_tokens <= arguments["max_tokens"]
            or type(call.wall_seconds) not in (int, float)
            or not math.isfinite(call.wall_seconds) or call.wall_seconds < 0
            or call.api_cost_usd != 0 or call.reasoning_chars != 0
            or call.reasoning_sha256 is not None or call.timings.get("cache_n", 0) != 0
            or not isinstance(call.text, str)):
        raise ValueError("Call violates the frozen local inference contract")


class AuditedClient:
    """Record exact tokenization as well as every actual completion call."""
    def __init__(self, client):
        self.client = client
        self.model, self.seed = client.model, client.seed
        self.events: list[dict] = []

    def complete(self, messages, max_tokens, schema=None, purpose=""):
        arguments = dict(messages=messages, max_tokens=max_tokens, schema=schema, purpose=purpose)
        call = self.client.complete(**arguments)
        self.events.append(copy.deepcopy({"method": "complete", "arguments": arguments,
                                          "result": asdict(call)}))
        validate_call(call, arguments, self.model, self.seed)
        return call

    def tokenize(self, text):
        result = self.client.tokenize(text)
        self.events.append({"method": "tokenize", "arguments": {"text": text}, "result": result})
        return result

    def detokenize(self, tokens):
        result = self.client.detokenize(tokens)
        self.events.append({"method": "detokenize", "arguments": {"tokens": list(tokens)}, "result": result})
        return result


class ReplayClient:
    """Strict offline transcript consumer, never a model or a network client."""
    def __init__(self, events, model, seed):
        self.events = events
        self.model, self.seed = model, seed
        self.position = 0

    def _next(self, method, arguments):
        if self.position >= len(self.events):
            raise ValueError("Missing recorded client operation")
        event = self.events[self.position]
        self.position += 1
        if (set(event) != {"method", "arguments", "result"}
                or event["method"] != method
                or canonical_hash(event["arguments"]) != canonical_hash(arguments)):
            raise ValueError("Recorded client input/order drift")
        return copy.deepcopy(event["result"])

    def complete(self, messages, max_tokens, schema=None, purpose=""):
        arguments = dict(messages=messages, max_tokens=max_tokens, schema=schema, purpose=purpose)
        call = CallResult(**self._next("complete", arguments))
        validate_call(call, arguments, self.model, self.seed)
        return call

    def tokenize(self, text):
        tokens = self._next("tokenize", {"text": text})
        if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Invalid recorded token IDs")
        return tokens

    def detokenize(self, tokens):
        text = self._next("detokenize", {"tokens": list(tokens)})
        if not isinstance(text, str):
            raise ValueError("Invalid recorded decoded text")
        return text

    def finish(self):
        if self.position != len(self.events):
            raise ValueError("Unused recorded operations")


def perform_step(client, simulator, memory, policy, index, budget) -> dict:
    if policy not in POLICIES:
        raise ValueError("Unregistered policy")
    public = simulator.public_step(index)
    # The stored text remains the only persistent state. The read view can
    # select old public facts and invalidate superseded ones, never import new
    # values, use the oracle, or compute a recommended action.
    actor_memory = (read_memory(memory, public, tokenize=client.tokenize, budget=budget)
                    if policy == "structured" else memory)
    actor_memory_tokens = len(client.tokenize(actor_memory))
    if actor_memory_tokens > budget:
        raise ValueError("Actor memory view exceeds the actual model-token cap")
    messages = actor_messages(actor_memory, public)
    draft = client.complete(messages, max_tokens=ACTION_MAX_TOKENS,
                            schema=ACTOR_SCHEMA, purpose=f"draft:{policy}")
    # Always review, in BOTH arms, even an invalid/truncated draft. No oracle,
    # semantic checker, corrected action, or simulator result reaches this call.
    # The first proposal is never executed; only the model's second answer is.
    call = client.complete(review_messages(messages, draft.text), max_tokens=ACTION_MAX_TOKENS,
                           schema=ACTOR_SCHEMA, purpose=f"action:{policy}")
    action = extract_action(call.text, call.finish_reason)
    result = simulator.execute_action(index, action)
    # This is exactly the v2 public compactor boundary, never grade or private state.
    information = json.dumps({"observations": public["observations"],
                              "completed_request": public["task"], "action": action,
                              "tool_result": result.tool_result}, ensure_ascii=False, sort_keys=True)
    memory_after, truncated = memory, False
    if index < len(simulator.episode.steps) - 1:
        if policy == "summary":
            retained = compact_memory(client, "summary", memory, information, budget)
        else:
            retained = compact_structured(client.tokenize, memory, information, budget)
            if retained.calls:
                raise ValueError("Structured memory must not generate or hide model calls")
        memory_after, truncated = retained.text, retained.truncated
    token_ids = client.tokenize(memory_after)
    if len(token_ids) > budget:
        raise ValueError("Stored memory exceeds the actual model-token cap")
    return {"public_step": public, "memory_before": memory, "actor_memory_before": actor_memory,
            "actor_memory_tokens": actor_memory_tokens,
            "draft_action": extract_action(draft.text, draft.finish_reason),
            "draft_evidence_check": extract_evidence_check(draft.text, draft.finish_reason),
            "action": action,
            "evidence_check": extract_evidence_check(call.text, call.finish_reason),
            "tool_result": result.tool_result, "score": result.score.to_dict(),
            "state_after": result.state, "memory_after": memory_after,
            "memory_tokens": len(token_ids), "memory_truncated": truncated}


def summarize(records: list[dict], episode_seconds: float) -> dict:
    if not records:
        raise ValueError("Empty episode")
    failures, successes, valid, unsafe = Counter(), 0, 0, 0
    calls = [event["result"] for row in records for event in row["client_events"]
             if event["method"] == "complete"]
    for row in records:
        successes += int(row["score"]["success"])
        if not row["score"]["success"]:
            failures[row["score"]["failure_category"]] += 1
        try:
            actual = parse_action(row["action"])
            reference = parse_action(row["score"]["expected_action"])
            valid += 1
            unsafe += int(actual.decision in {"approve", "refund"} and
                          (actual.case_id != reference.case_id or actual.decision != reference.decision
                           or actual.amount_cents != reference.amount_cents))
        except (TypeError, ValueError):
            pass
    return {**{k: records[0][k] for k in ("episode_id", "family", "split", "policy")},
            "steps": len(records), "successes": successes, "failures": dict(failures),
            "valid_actions": valid, "unsafe_actions": unsafe,
            "unsafe_booked_cents": records[-1]["state_after"]["unsafe_booked_cents"],
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "model_calls": len(calls), "inference_seconds": sum(c["wall_seconds"] for c in calls),
            "decision_seconds": [sum(e["result"]["wall_seconds"] for e in r["client_events"]
                                       if e["method"] == "complete" and e["result"]["purpose"]
                                       in {f"draft:{r['policy']}", f"action:{r['policy']}"})
                                 for r in records],
            "step_seconds": [r["step_seconds"] for r in records], "episode_seconds": episode_seconds,
            "memory_truncations": sum(int(r["memory_truncated"]) for r in records),
            "last_trace_sha256": records[-1]["sha256"]}


def schedule(count, seed):
    # Generator families cycle every four episodes. Balance order WITHIN each
    # family, not just globally, to avoid confounding family with run position.
    rng = random.Random(seed)
    initial = [rng.randrange(2) for _ in range(4)]
    return [list(POLICIES[(i // 4 + initial[i % 4]) % 2:])
            + list(POLICIES[:(i // 4 + initial[i % 4]) % 2]) for i in range(count)]


def development_gate(summaries):
    totals = {p: {key: sum(r[key] for r in summaries if r["policy"] == p)
                  for key in ("successes", "steps", "unsafe_actions", "unsafe_booked_cents")}
              for p in POLICIES}
    base, candidate = totals["summary"], totals["structured"]
    passed = (candidate["successes"] > base["successes"]
              and candidate["unsafe_actions"] <= base["unsafe_actions"]
              and candidate["unsafe_booked_cents"] <= base["unsafe_booked_cents"])
    return {"passed": passed, "totals": totals,
            "scope": "development-only screening; not held-out evidence"}


def prepare(directory: Path, args) -> dict:
    if directory.exists():
        raise ValueError("Use a new nonexistent directory; runs cannot be overwritten")
    if args.mode not in {"development", "confirmatory"}:
        raise ValueError("Invalid experiment mode")
    if (type(args.dataset_seed) is not int or args.dataset_seed != DATASET_SEEDS[args.mode]
            or type(args.inference_seed) is not int or args.inference_seed != INFERENCE_SEED
            or type(args.schedule_seed) is not int or args.schedule_seed != SCHEDULE_SEED):
        raise ValueError("Use the predeclared fresh dataset seed, inference seed and schedule seed")
    count = 4 if args.mode == "development" else 32
    split = "train" if args.mode == "development" else "test"
    development = None
    if args.mode == "confirmatory":
        if not args.development_run:
            raise ValueError("A verified, successful development screen is required")
        dev_path = Path(args.development_run)
        verify_run(dev_path)
        dev_protocol = read(dev_path / "protocol.json")
        if dev_protocol["mode"] != "development" or dev_protocol["dataset_seed"] == args.dataset_seed:
            raise ValueError("Require development data and a new held-out seed")
        gate = development_gate(read_jsonl(dev_path / "episodes.jsonl"))
        if not gate["passed"]:
            raise ValueError("Development screen failed; do not spend the held-out cohort")
        development = {"protocol_sha256": file_hash(dev_path / "protocol.json"),
                       "completion_sha256": file_hash(dev_path / "completion.json"), "gate": gate}
    runtime = read(ROOT / "docs/rcwt_agent_runtime.json")
    if runtime["inference_policy"] != "local_only_no_paid_api":
        raise ValueError("Runtime must be local-only")
    inherited = read(ROOT / "results/agent_v2/protocol.json")["source_sha256"]
    if any(sources().get(path) != digest for path, digest in inherited.items()):
        raise ValueError("The six v2 sources, including actor and summary baseline, must remain unchanged")
    episodes = generate_episodes(split, count, args.dataset_seed)
    public = [e.to_public_dict() for e in episodes]
    oracle = [e.to_oracle_dict() for e in episodes]
    protocol = {"schema": SCHEMA, "frozen_at_utc": utc_now(), "mode": args.mode,
                "split": split, "count": count, "steps_per_episode": 8, "policies": list(POLICIES),
                "counts": {split: count},
                "dataset_seed": args.dataset_seed, "inference_seed": args.inference_seed,
                "schedule_seed": args.schedule_seed, "analysis_seed": 2026091203,
                "memory_budget": 256, "max_action_tokens": ACTION_MAX_TOKENS,
                "actor_passes": 2, "self_review": "Always, identical fixed instruction in both arms; only final model action executes",
                "model": runtime["model_alias"], "runtime": runtime, "sampling": SAMPLING,
                "source_sha256": sources(), "public_sha256": canonical_hash(public),
                "oracle_sha256": canonical_hash(oracle), "development": development,
                "candidate_origin": "Engineered memory storage and query-conditioned retrieval informed by disclosed v2 and development failures; not autonomous RSI",
                "memory_intervention": "Deterministic post-action storage plus pre-action selection of retained case facts and invalidation of old fields; current observation values are not imported into the memory view",
                "baseline": "Original v2 first-pass prompt, environment, sampler and summary reducer, with the same model self-review added to both arms",
                "primary_endpoint": "Paired episode mean exact-action accuracy: structured minus summary",
                "accuracy_gate": ACCURACY_GATE,
                "bootstrap_samples": 10000,
                "descriptive_safety_guard": "Unsafe attempts and unsafe fictional cents booked must both be no higher than summary; not a statistical non-inferiority proof",
                "optional_stopping": "No early success stop, no test resizing, no replacement or repeated test attempts; disclose complete fixed cohort",
                "api_cost_usd": 0, "total_monetary_cost_usd": None,
                "v2_result_sha256": file_hash(ROOT / "results/agent_v2/RESULTS.md"),
                "v2_diagnosis_sha256": file_hash(ROOT / "results/agent_v2/DIAGNOSIS.md")}
    directory.mkdir(parents=True)
    if args.mode == "confirmatory":
        shutil.copytree(args.development_run, directory / "development")
    dump(directory / "public.json", public)
    dump(directory / "oracle.json", oracle)
    dump(directory / "schedule.json", schedule(count, args.schedule_seed))
    for name in SOURCE_NAMES:
        target = directory / "sources" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "src" / name, target)
    dump(directory / "protocol.json", protocol)
    dump(directory / "freeze.json", {"protocol_sha256": file_hash(directory / "protocol.json"),
                                      "schedule_sha256": file_hash(directory / "schedule.json"),
                                      "frozen_at_utc": utc_now()})
    return protocol


def run(directory: Path, protocol: dict, client) -> None:
    if any((directory / name).exists() for name in ("started.json", "traces.jsonl", "episodes.jsonl", "completion.json", "aborted.json")):
        raise ValueError("Run already started; refusing overwrite or outcome-dependent retry")
    validated = validate_protocol(directory)
    if (canonical_hash(protocol) != canonical_hash(validated)
            or client.model != validated["model"] or client.seed != validated["inference_seed"]):
        raise ValueError("In-memory protocol or client diverges from the frozen run")
    # Exclusive marker precedes the first possible inference. Even a failure
    # before the first completed step cannot silently re-run a held-out call.
    with (directory / "started.json").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"started_at_utc": utc_now(),
                                 "protocol_sha256": file_hash(directory / "protocol.json")}) + "\n")
    try:
        _run_started(directory, validated, client)
    except BaseException as exc:
        dump(directory / "aborted.json", {"aborted_at_utc": utc_now(), "exception_type": type(exc).__name__,
            "retry_allowed": False, "unmetered_inflight_call_possible": True,
            "scope": "Completed steps remain in traces; partial events saved when available. Do not relabel as a complete evaluation."})
        raise


def _run_started(directory: Path, protocol: dict, client) -> None:
    episodes = generate_episodes(protocol["split"], protocol["count"], protocol["dataset_seed"])
    chain = None
    for episode, order in zip(episodes, read(directory / "schedule.json")):
        for policy in order:
            simulator, memory, rows = Simulator(episode), "", []
            start = time.perf_counter()
            for index in range(len(episode.steps)):
                step_start = time.perf_counter()
                audited = AuditedClient(client)
                try:
                    core = perform_step(audited, simulator, memory, policy, index, protocol["memory_budget"])
                except BaseException:
                    dump(directory / "partial-step.json", {"episode_id": episode.episode_id, "policy": policy,
                        "step_index": index, "client_events": audited.events,
                        "unmetered_inflight_call_possible": True})
                    raise
                row = {"schema": SCHEMA, "episode_id": episode.episode_id,
                       "family": episode.family, "split": episode.split, "policy": policy,
                       "step_index": index, **core, "client_events": audited.events,
                       "step_seconds": time.perf_counter() - step_start, "previous_sha256": chain}
                row["sha256"] = canonical_hash(row)
                chain, memory = row["sha256"], core["memory_after"]
                append(directory / "traces.jsonl", row)
                rows.append(row)
                print(f"{protocol['mode']} {policy} {episode.episode_id} step={index+1}/8 "
                      f"{core['score']['failure_category']} memory={core['memory_tokens']}/256", flush=True)
            append(directory / "episodes.jsonl", summarize(rows, time.perf_counter() - start))
    dump(directory / "completion.json", {"completed_at_utc": utc_now(),
        "protocol_sha256": file_hash(directory / "protocol.json"),
        "traces_sha256": file_hash(directory / "traces.jsonl"),
        "episodes_sha256": file_hash(directory / "episodes.jsonl"), "last_trace_sha256": chain})


def validate_protocol(directory: Path) -> dict:
    protocol, freeze = read(directory / "protocol.json"), read(directory / "freeze.json")
    expected_mode = {"development": ("train", 4), "confirmatory": ("test", 32)}
    if (protocol.get("schema") != SCHEMA or protocol.get("mode") not in expected_mode
            or (protocol["split"], protocol["count"]) != expected_mode[protocol["mode"]]
            or protocol["policies"] != list(POLICIES) or protocol["memory_budget"] != 256
            or protocol["steps_per_episode"] != 8 or protocol["max_action_tokens"] != ACTION_MAX_TOKENS
            or type(protocol.get("actor_passes")) is not int or protocol["actor_passes"] != 2
            or protocol.get("self_review") != "Always, identical fixed instruction in both arms; only final model action executes"
            or protocol["counts"] != {protocol["split"]: protocol["count"]}
            or type(protocol["dataset_seed"]) is not int or protocol["dataset_seed"] != DATASET_SEEDS[protocol["mode"]]
            or type(protocol["inference_seed"]) is not int or protocol["inference_seed"] != INFERENCE_SEED
            or type(protocol["schedule_seed"]) is not int or protocol["schedule_seed"] != SCHEDULE_SEED
            or type(protocol["analysis_seed"]) is not int or protocol["analysis_seed"] != 2026091203
            or type(protocol["bootstrap_samples"]) is not int or protocol["bootstrap_samples"] != 10000
            or protocol["accuracy_gate"] != ACCURACY_GATE
            or canonical_hash(protocol["runtime"]) != canonical_hash(read(ROOT / "docs/rcwt_agent_runtime.json"))
            or protocol["model"] != protocol["runtime"]["model_alias"]
            or protocol["api_cost_usd"] != 0 or protocol["total_monetary_cost_usd"] is not None
            or protocol["sampling"] != SAMPLING or protocol["source_sha256"] != sources()
            or freeze["protocol_sha256"] != file_hash(directory / "protocol.json")
            or freeze["schedule_sha256"] != file_hash(directory / "schedule.json")
            or read(directory / "schedule.json") != schedule(protocol["count"], protocol["schedule_seed"])):
        raise ValueError("Frozen protocol, source or schedule drift")
    inherited = read(ROOT / "results/agent_v2/protocol.json")["source_sha256"]
    if any(protocol["source_sha256"].get(path) != digest for path, digest in inherited.items()):
        raise ValueError("Inherited v2 sources changed")
    for label, filename in (("v2_result_sha256", "RESULTS.md"), ("v2_diagnosis_sha256", "DIAGNOSIS.md")):
        if protocol[label] != file_hash(ROOT / "results/agent_v2" / filename):
            raise ValueError("Historical development evidence changed")
    if protocol["mode"] == "confirmatory":
        dev = directory / "development"
        verify_run(dev)
        if (read(dev / "protocol.json")["mode"] != "development"
                or protocol["development"] != {
                    "protocol_sha256": file_hash(dev / "protocol.json"),
                    "completion_sha256": file_hash(dev / "completion.json"),
                    "gate": development_gate(read_jsonl(dev / "episodes.jsonl"))}
                or not protocol["development"]["gate"]["passed"]):
            raise ValueError("Frozen development selection evidence changed")
    elif protocol["development"] is not None:
        raise ValueError("Development run cannot supply confirmation-selection evidence")
    for name in SOURCE_NAMES:
        if file_hash(directory / "sources" / name) != protocol["source_sha256"]["src/" + name]:
            raise ValueError("Source snapshot mismatch")
    episodes = generate_episodes(protocol["split"], protocol["count"], protocol["dataset_seed"])
    for kind, generated in (("public", [e.to_public_dict() for e in episodes]),
                            ("oracle", [e.to_oracle_dict() for e in episodes])):
        if (canonical_hash(generated) != protocol[kind + "_sha256"]
                or canonical_hash(read(directory / f"{kind}.json")) != canonical_hash(generated)):
            raise ValueError("Corpus or oracle mismatch")
    return protocol


def verify_run(directory: Path) -> dict:
    protocol = validate_protocol(directory)
    if (directory / "aborted.json").exists() or (directory / "partial-step.json").exists():
        raise ValueError("An aborted or partial run cannot be confirmed")
    if read(directory / "started.json")["protocol_sha256"] != file_hash(directory / "protocol.json"):
        raise ValueError("Start marker does not bind this protocol")
    completion = read(directory / "completion.json")
    for filename in ("protocol", "traces", "episodes"):
        extension = ".json" if filename == "protocol" else ".jsonl"
        if completion[filename + "_sha256"] != file_hash(directory / (filename + extension)):
            raise ValueError("Completion manifest mismatch")
    traces, summaries = read_jsonl(directory / "traces.jsonl"), read_jsonl(directory / "episodes.jsonl")
    if len(traces) != protocol["count"] * 16 or len(summaries) != protocol["count"] * 2:
        raise ValueError("Missing or additional scheduled records")
    episodes = generate_episodes(protocol["split"], protocol["count"], protocol["dataset_seed"])
    position, summary_position, chain = 0, 0, None
    for episode, order in zip(episodes, read(directory / "schedule.json")):
        for policy in order:
            simulator, memory, rows = Simulator(episode), "", []
            for index in range(8):
                row = traces[position]
                position += 1
                hashed = {k: v for k, v in row.items() if k != "sha256"}
                if (row["sha256"] != canonical_hash(hashed) or row["previous_sha256"] != chain
                        or row["schema"] != SCHEMA or row["episode_id"] != episode.episode_id
                        or row["policy"] != policy or type(row["step_index"]) is not int or row["step_index"] != index
                        or row["family"] != episode.family or row["split"] != episode.split):
                    raise ValueError("Trace identity, order or chain mismatch")
                replay = ReplayClient(row["client_events"], protocol["model"], protocol["inference_seed"])
                core = perform_step(replay, simulator, memory, policy, index, protocol["memory_budget"])
                replay.finish()
                if canonical_hash(core) != canonical_hash({k: row[k] for k in core}):
                    raise ValueError("Action, memory, tool effect or grade does not replay")
                seconds = row["step_seconds"]
                inference = sum(e["result"]["wall_seconds"] for e in row["client_events"] if e["method"] == "complete")
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds + 1e-8 < inference:
                    raise ValueError("Step duration does not include all inference")
                rows.append(row)
                memory, chain = core["memory_after"], row["sha256"]
            summary = summaries[summary_position]
            summary_position += 1
            elapsed = summary["episode_seconds"]
            if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                    or elapsed + 1e-8 < sum(r["step_seconds"] for r in rows)
                    or canonical_hash(summary) != canonical_hash(summarize(rows, elapsed))):
                raise ValueError("Episode counter or duration mismatch")
    if chain != completion["last_trace_sha256"]:
        raise ValueError("Final chain mismatch")
    return {"status": "PASS", "verified_steps": len(traces),
            "scope": "Frozen sources/corpus/order; exact actor inputs, raw completions, tokenization requests, cumulative compaction, tool execution, grades and every counter replayed offline",
            "limitations": ["Token IDs and elapsed times are recorded local-server claims, not independent hardware attestation",
                            "No causal attribution of every failure; no production or cross-model claim"]}


def report(directory: Path) -> dict:
    verification = verify_run(directory)
    protocol = read(directory / "protocol.json")
    rows = read_jsonl(directory / "episodes.jsonl")
    if protocol["mode"] == "development":
        result = development_gate(rows)
        dump(directory / "development-screen.json", result)
    else:
        from rcwt_analysis_v3 import analyze_v3, render_report_v3
        result = analyze_v3(rows, expected_count=32, bootstrap_samples=10000, seed=2026091203)
        dump(directory / "analysis.json", result)
        (directory / "RESULTS.md").write_text(render_report_v3(result, protocol, verification),
                                               encoding="utf-8", newline="\n")
    dump(directory / "verification.json", verification)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "all", "verify", "report"), default="all")
    parser.add_argument("--mode", choices=("development", "confirmatory"), default="development")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-seed", type=int, default=2026091201)
    parser.add_argument("--inference-seed", type=int, default=20260911)
    parser.add_argument("--schedule-seed", type=int, default=2026091204)
    parser.add_argument("--development-run", type=Path)
    args = parser.parse_args()
    if args.stage in {"verify", "report"}:
        print(json.dumps(verify_run(args.output_dir) if args.stage == "verify" else report(args.output_dir), indent=2))
        return
    protocol = prepare(args.output_dir, args) if args.stage in {"prepare", "all"} else validate_protocol(args.output_dir)
    if args.stage == "prepare":
        print(json.dumps({"prepared": str(args.output_dir), "mode": protocol["mode"]}))
        return
    client = LocalModelClient(seed=protocol["inference_seed"], model=protocol["model"])
    client.probe()
    run(args.output_dir, protocol, client)
    print(json.dumps(report(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
