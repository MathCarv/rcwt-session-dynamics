"""Verify an interrupted v4 confirmation prefix offline, without evaluating gain.

Only the current trusted project sources are imported, after all fifteen match
the frozen source manifest and both archived snapshots. No model client is
constructed and sockets are blocked during validation/replay. The CLI writes
only its JSON receipt to stdout; original inputs are checked again before return.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from pathlib import Path
import socket
import stat
import sys

# Importing the frozen modules must not create __pycache__ in original sources.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "rcwt_agent_actor.py", "rcwt_agent_analysis.py", "rcwt_agent_env.py",
    "rcwt_agent_memory.py", "rcwt_agent_run.py", "rcwt_local_model.py",
    "rcwt_memory_v3.py", "rcwt_online_v3.py", "rcwt_analysis_v3.py",
    "rcwt_retrieval_v3.py", "rcwt_review_v3.py", "rcwt_context_v4.py",
    "rcwt_decision_v4.py", "rcwt_online_v4.py", "rcwt_analysis_v4.py",
)
INPUTS = ("protocol.json", "freeze.json", "started.json", "schedule.json",
          "public.json", "oracle.json", "traces.jsonl", "episodes.jsonl")
HISTORY = ("docs/rcwt_agent_runtime.json", "results/agent_v2/protocol.json",
           "results/agent_v2/RESULTS.md", "results/agent_v2/DIAGNOSIS.md",
           "results/agent_v3_development/attempt_05/protocol.json",
           "results/agent_v3_development/development.json",
           "results/agent_v3_development/RESULTS.md")
MARKERS = ("completion.json", "aborted.json", "partial-step.json")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=pairs)
    # Also rejects overflow literals such as 1e999, including nested metering.
    json.dumps(value, allow_nan=False)
    return value


def _jsonl(data):
    return [_json(line) for line in data.splitlines()]


def _local(path):
    boundary = ROOT.resolve(strict=True)
    original = Path(path).absolute()
    try:
        parts = original.relative_to(boundary).parts
    except ValueError as exc:
        raise ValueError("Only trusted local project inputs are accepted") from exc
    current = boundary
    for part in parts:
        current = current / part
        info = current.lstat()
        if (current.is_symlink() or getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError("Input links and reparse points are forbidden")
    resolved = original.resolve(strict=True)
    if not resolved.is_relative_to(boundary):
        raise ValueError("Input path leaves the trusted project")
    return resolved


def _read(path, observed):
    path = _local(path)
    if not path.is_file():
        raise ValueError("Expected a regular input file")
    data = path.read_bytes()
    if path in observed and data != observed[path]:
        raise ValueError("Input changed during partial verification")
    observed[path] = data
    return data


@contextmanager
def _offline():
    def blocked(*args, **kwargs):
        raise RuntimeError("Network is forbidden during partial offline verification")

    targets = [(socket, "create_connection"), (socket, "getaddrinfo"),
               (socket.socket, "connect"), (socket.socket, "connect_ex"),
               (socket.socket, "sendto")]
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    try:
        for owner, name in targets:
            setattr(owner, name, blocked)
        yield
    finally:
        for owner, name, value in originals:
            setattr(owner, name, value)


def _load_runner(expected):
    # No archived Python is executed. Check already-imported module files too,
    # so a different checkout on sys.path cannot silently supply the replay.
    sys.path.insert(0, str(ROOT / "src"))
    try:
        runner = importlib.import_module("rcwt_online_v4")
        for name in NAMES:
            module = sys.modules.get(name[:-3])
            if module is not None:
                path = Path(module.__file__).resolve(strict=True)
                if _digest(path.read_bytes()) != expected["src/" + name]:
                    raise ValueError("Loaded module differs from the frozen source")
        return runner
    finally:
        sys.path.pop(0)


def _finite_nonnegative(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid recorded " + label)
    return value


def _episodes(public, oracle):
    # validate_protocol already regenerated and authenticated these manifests;
    # reconstruct their dataclasses instead of generating another corpus here.
    from rcwt_agent_env import Episode, OracleSnapshot, Step

    if len(public) != 32 or len(oracle) != 32:
        raise ValueError("Expected the complete frozen 32-episode corpus")
    result, seen = [], set()
    for item, private in zip(public, oracle):
        identity = item["episode_id"]
        if (identity in seen or identity != private["episode_id"]
                or len(item["steps"]) != 8 or len(private["oracle_steps"]) != 8):
            raise ValueError("Corpus identity or step count mismatch")
        seen.add(identity)
        result.append(Episode(identity, item["family"], item["split"], item["seed"],
                              tuple(Step(**step) for step in item["steps"]),
                              tuple(OracleSnapshot(**step) for step in private["oracle_steps"])))
    return result


def _verify_prefix(runner, protocol, public, oracle, order, traces, summaries):
    expected_steps = 512
    if not 0 < len(traces) < expected_steps:
        raise ValueError("Require a nonempty interrupted prefix with fewer than 512 steps")
    if len(summaries) != len(traces) // 8:
        raise ValueError("Require exactly one summary per fully completed trajectory")
    episodes = _episodes(public, oracle)
    if order != runner.schedule(32, protocol["schedule_seed"]):
        raise ValueError("Frozen schedule order mismatch")
    position, summary_position, chain = 0, 0, None
    partial = None
    for episode, policies in zip(episodes, order):
        for policy in policies:
            if position == len(traces):
                break
            simulator, memory, rows = runner.Simulator(episode), "", []
            for index in range(8):
                if position == len(traces):
                    break
                row = traces[position]
                position += 1
                hashed = {key: value for key, value in row.items() if key != "sha256"}
                if (row["sha256"] != runner.canonical_hash(hashed)
                        or row["previous_sha256"] != chain or row["schema"] != runner.SCHEMA
                        or row["episode_id"] != episode.episode_id or row["family"] != episode.family
                        or row["split"] != episode.split or row["policy"] != policy
                        or type(row["step_index"]) is not int or row["step_index"] != index):
                    raise ValueError("Trace identity, order or chain mismatch")
                replay = runner.ReplayClient(row["client_events"], protocol["model"], protocol["inference_seed"])
                core = runner.perform_step(replay, simulator, memory, policy, index, protocol["memory_budget"])
                replay.finish()
                metadata = {"schema", "episode_id", "family", "split", "policy", "step_index",
                            "client_events", "step_seconds", "previous_sha256", "sha256"}
                if (set(row) != set(core) | metadata
                        or runner.canonical_hash(core) != runner.canonical_hash({key: row[key] for key in core})):
                    raise ValueError("Action, memory, tool effect or grade does not replay")
                seconds = _finite_nonnegative(row["step_seconds"], "step duration")
                calls = [event["result"] for event in row["client_events"] if event["method"] == "complete"]
                for call in calls:
                    _finite_nonnegative(call["api_cost_usd"], "API charge")
                    _finite_nonnegative(call["wall_seconds"], "inference duration")
                inference = sum(call["wall_seconds"] for call in calls)
                if seconds + 1e-8 < inference:
                    raise ValueError("Step duration does not include all inference")
                rows.append(row)
                memory, chain = core["memory_after"], row["sha256"]
            if len(rows) == 8:
                summary = summaries[summary_position]
                summary_position += 1
                elapsed = _finite_nonnegative(summary["episode_seconds"], "episode duration")
                if (elapsed + 1e-8 < sum(row["step_seconds"] for row in rows)
                        or runner.canonical_hash(summary) != runner.canonical_hash(runner.summarize(rows, elapsed))):
                    raise ValueError("Episode counter, order or duration mismatch")
            elif rows:
                partial = {"episode_id": episode.episode_id, "policy": policy,
                           "completed_steps": len(rows), "remaining_steps": 8 - len(rows),
                           "next_step_index": len(rows)}
    if position != len(traces) or summary_position != len(summaries):
        raise ValueError("Unused trace or summary records")
    calls = [event["result"] for row in traces for event in row["client_events"] if event["method"] == "complete"]
    totals = {"completed_steps": len(traces), "generation_calls": len(calls),
              "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
              "completion_tokens": sum(call["completion_tokens"] for call in calls),
              "model_tokens": sum(call["prompt_tokens"] + call["completion_tokens"] for call in calls),
              "inference_seconds": sum(call["wall_seconds"] for call in calls),
              "sum_step_seconds": sum(row["step_seconds"] for row in traces),
              "tokenizer_operations": sum(event["method"] in {"tokenize", "detokenize"}
                                          for row in traces for event in row["client_events"]),
              "api_cost_usd": sum(call["api_cost_usd"] for call in calls),
              "total_monetary_cost_usd": None}
    for key in ("inference_seconds", "sum_step_seconds", "api_cost_usd"):
        _finite_nonnegative(totals[key], "aggregate " + key)
    return {"status": "PARTIAL_VERIFIED", "confirmation_status": "INCOMPLETE", "gain": "NOT_EVALUATED",
            "verified_steps": len(traces), "scheduled_steps": expected_steps,
            "missing_steps": expected_steps - len(traces), "verified_episode_summaries": len(summaries),
            "scheduled_episode_trajectories": 64, "partial_last_trajectory": partial,
            "last_trace_sha256": chain, "recorded_completed_step_resources": totals}


def verify_partial(directory: Path) -> dict:
    directory = _local(directory)
    relative = directory.relative_to(ROOT.resolve())
    if len(relative.parts) < 2 or relative.parts[0] not in {".runs", "results"} or not directory.is_dir():
        raise ValueError("Use a trusted project bundle below .runs/ or results/")
    markers = {name: (directory / name).exists() for name in MARKERS}
    if markers["completion.json"]:
        raise ValueError("A completion manifest is forbidden for partial verification")
    observed = {}
    files = {name: _read(directory / name, observed) for name in INPUTS}
    _json(files["freeze.json"])
    for name in MARKERS[1:]:
        if markers[name]:
            _json(_read(directory / name, observed))
    protocol = _json(files["protocol.json"])
    if (protocol.get("schema") != "rcwt-online-memory/4" or protocol.get("mode") != "confirmatory"
            or type(protocol.get("count")) is not int or protocol["count"] != 32
            or protocol.get("split") != "test"):
        raise ValueError("Require the frozen v4 confirmation with 32 episodes")
    expected = protocol.get("source_sha256")
    if (not isinstance(expected, dict) or set(expected) != {"src/" + name for name in NAMES}
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value) for value in expected.values())):
        raise ValueError("Require exactly fifteen frozen source hashes")
    for name in NAMES:
        for path in (ROOT / "src" / name, directory / "sources" / name,
                     directory / "development/sources" / name):
            if _digest(_read(path, observed)) != expected["src/" + name]:
                raise ValueError("Frozen source hash mismatch: " + name)
    for name in (*INPUTS, "completion.json"):
        data = _read(directory / "development" / name, observed)
        _jsonl(data) if name.endswith(".jsonl") else _json(data)
    development_markers = {name: (directory / "development" / name).exists() for name in MARKERS[1:]}
    if any(development_markers.values()):
        raise ValueError("Copied development must be complete")
    for name in HISTORY:
        _read(ROOT / name, observed)
    protocol_hash = _digest(files["protocol.json"])
    if _json(files["started.json"]).get("protocol_sha256") != protocol_hash:
        raise ValueError("Start marker does not bind this protocol")
    with _offline():
        runner = _load_runner(expected)
        validated = runner.validate_protocol(directory)
        if runner.canonical_hash(validated) != runner.canonical_hash(protocol):
            raise ValueError("Validated protocol differs from initial input")
        result = _verify_prefix(runner, protocol, _json(files["public.json"]), _json(files["oracle.json"]),
                                _json(files["schedule.json"]), _jsonl(files["traces.jsonl"]),
                                _jsonl(files["episodes.jsonl"]))
    for path, data in list(observed.items()):
        if _read(path, observed) != data:
            raise ValueError("Input changed during partial verification")
    if (markers != {name: (directory / name).exists() for name in MARKERS}
            or development_markers != {name: (directory / "development" / name).exists() for name in MARKERS[1:]}):
        raise ValueError("Completion or interruption markers changed during verification")
    return {**result, "protocol_sha256": protocol_hash, "started_sha256": _digest(files["started.json"]),
            "input_sha256": {path.relative_to(ROOT.resolve()).as_posix(): _digest(data)
                             for path, data in sorted(observed.items())},
            "inference_calls": 0, "read_only": True, "retry_allowed": False,
            "unmetered_inflight_call_possible": True,
            "scope": "Exact frozen prefix replay; completed-step resources only; no confirmation analysis or gain estimate",
            "limitations": ["Unfinished or unlogged operations are excluded from resource totals",
                            "Recorded tokens and times are runtime claims, not independent attestation",
                            "Hashes establish integrity, not authorship or proof of no other attempts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_partial(args.run_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
