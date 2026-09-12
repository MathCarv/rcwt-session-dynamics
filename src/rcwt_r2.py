"""Finite R2 policy-learning campaign; unchanged actor and a separate safety kernel.

Prepare seals code without generating any corpus. Test is generated only after
all ten policy proposals and all five validation decisions have been sealed.
Every inference stage is exclusive and terminal on interruption; verification
consumes recorded client operations offline and never repairs an artifact.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import stat
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rcwt_online_v4 as core
from rcwt_agent_env import score_action
from rcwt_policy_r2 import (DEFAULT_POLICY, POLICY_SCHEMA, compact_policy,
                            build_feedback, shuffle_feedback, feedback_json,
                            parse_policy, validate_policy)
from rcwt_safety_r2 import SafetyExecutor
from rcwt_analysis_r2 import summarize_episode, selection, analyze_r2, render_report
from tools.verify_online_v4_partial import _json as strict_json, _offline

SCHEMA = "rcwt-policy-learning/2"
REGISTRATION = "docs/rcwt_r2_protocol.md"
R1 = "results/agent_v4_replication"
R1_PROTOCOL_SHA256 = "5fdb1fcb9b81bd4fc073d9429c90e199a44688dfbdf9b2bdfedc74f1c9092524"
NEW_SOURCES = ("src/rcwt_r2.py", "src/rcwt_policy_r2.py", "src/rcwt_safety_r2.py",
               "src/rcwt_analysis_r2.py", "tools/rcwt_r2_runtime.ps1", "tools/verify_r2_accounting.py")
OLD_ORCHESTRATORS = ("src/rcwt_replication_v4.py", "tools/verify_online_v4_partial.py",
                    "tools/verify_v4_replication_report.py")
ARMS = ("fixed", "learned", "shuffled")
STAGES = ("train", "propose", "validate", "test")
PHASE = {"train": "train", "validate": "validation", "test": "test"}
EXPECTED_CALLS = {"train": 320, "propose": 10, "validate": 960, "test": 960}
PROPOSAL_TOKENS = 256


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _local(path: Path, *, missing=False) -> Path:
    original = Path(path).absolute()
    boundary = ROOT.resolve(strict=True)
    try:
        parts = original.relative_to(boundary).parts
    except ValueError as exc:
        raise ValueError("R2 paths must remain inside this checkout") from exc
    current = boundary
    for part in parts:
        if part in {".", ".."}:
            raise ValueError("Path traversal is forbidden")
        current = current / part
        if not current.exists() and not current.is_symlink() and missing:
            continue
        info = current.lstat()
        if current.is_symlink() or (getattr(info, "st_file_attributes", 0)
                                  & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError("Links and reparse points are forbidden")
    resolved = original.resolve(strict=not missing)
    if not resolved.is_relative_to(boundary):
        raise ValueError("R2 path leaves the checkout")
    return resolved


def _bytes(path):
    checked = _local(path)
    if not checked.is_file():
        raise ValueError("Expected a regular input file")
    return checked.read_bytes()


def _output(path, evidence_kind):
    target = _local(path, missing=True)
    parts = target.relative_to(ROOT.resolve(strict=True)).parts
    real = (evidence_kind == "real_local_model" and len(parts) == 2
            and parts[0] == ".runs" and parts[1].startswith("r2_"))
    synthetic = (evidence_kind == "EXPLICIT_SYNTHETIC_TEST_ONLY" and len(parts) >= 3
                 and parts[0] == ".runs" and parts[1].startswith("testing-r2-"))
    if not (real or synthetic):
        raise ValueError("R2 writes require a fresh .runs/r2_* campaign or segregated synthetic fixture")
    return target


def read(path):
    return strict_json(_bytes(path))


def read_jsonl(path):
    return [strict_json(line) for line in _bytes(path).splitlines()]


def _write(path, value, *, raw=False):
    target = _local(path, missing=True)
    with target.open("xb") as handle:
        handle.write(value if raw else json_bytes(value))


def _append(path, row):
    target = _local(path, missing=True)
    with target.open("ab") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
        handle.flush()


def _snapshot(directory):
    root = _local(directory)
    result = {}
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        checked = _local(path)
        if checked.is_dir():
            for name, value in _snapshot(checked).items():
                result[path.name + "/" + name] = value
        elif checked.is_file():
            result[path.name] = digest(_bytes(checked))
        else:
            raise ValueError("Nonregular evidence entry")
    return result


def _frozen_inputs():
    raw = _bytes(ROOT / R1 / "protocol.json")
    if digest(raw) != R1_PROTOCOL_SHA256:
        raise ValueError("Published R1 protocol changed")
    protocol = strict_json(raw)
    expected = dict(protocol["historical_evidence"]["input_sha256"])
    expected.update(protocol["source_sha256"])
    expected.update(protocol["orchestrator_sha256"])
    if set(protocol["source_sha256"]) != {"src/" + n for n in core.SOURCE_NAMES}:
        raise ValueError("Require all fifteen original sources")
    if set(protocol["orchestrator_sha256"]) != set(OLD_ORCHESTRATORS):
        raise ValueError("Require the three original R1 orchestrators")
    if any(digest(_bytes(ROOT / name)) != value for name, value in expected.items()):
        raise ValueError("Original sources or historical evidence changed")
    r1 = _snapshot(ROOT / R1)
    if len(r1) != 32:
        raise ValueError("Require the complete original R1 archive")
    expected.update({R1 + "/" + name: value for name, value in r1.items()})
    return dict(sorted(expected.items()))


def _sources():
    result = {name: digest(_bytes(ROOT / name)) for name in NEW_SOURCES}
    for name in (*NEW_SOURCES, *("src/" + n for n in core.SOURCE_NAMES)):
        module = sys.modules.get(Path(name).stem)
        if module is not None and Path(module.__file__).resolve() != (ROOT / name).resolve():
            raise ValueError("Imported source came from a different checkout")
    return result


def replicas():
    return [{"replica": r, "train_seed": 2026091300 + r * 100 + 1,
             "validation_seed": 2026091300 + r * 100 + 2,
             "test_seed": 2026091300 + r * 100 + 3,
             "inference_seed": 2026091300 + r * 100 + 10,
             "schedule_seed": 2026091300 + r * 100 + 20,
             "shuffle_seed": 2026091300 + r * 100 + 30} for r in range(5)]


def _contract():
    return {"schema": SCHEMA, "replication_id": "R2", "replicas": replicas(),
            "evidence_kind": "real_local_model",
            "arms": list(ARMS), "episodes_per_split_per_replica": 4,
            "steps_per_episode": 8, "memory_budget": 256, "actor_passes": 2,
            "max_action_tokens": 512, "max_proposal_tokens": PROPOSAL_TOKENS,
            "expected_calls": EXPECTED_CALLS, "expected_total_calls": 2250,
            "expected_total_steps": 1120, "sampling": core.SAMPLING,
            "actor_instruction_variant": core.ACTOR_INSTRUCTION_VARIANT,
            "test_generation": "After all proposals and validation selection are frozen",
            "retry_allowed": False, "historical_pooling": False,
            "api_cost_usd": 0, "total_monetary_cost_usd": None}


def prepare(directory: Path):
    directory = _output(directory, _contract()["evidence_kind"])
    if directory.exists():
        raise ValueError("Use a nonexistent output directory")
    inputs, sources = _frozen_inputs(), _sources()
    registration = _bytes(ROOT / REGISTRATION)
    runtime = read(ROOT / "docs/rcwt_agent_runtime.json")
    if runtime["inference_policy"] != "local_only_no_paid_api":
        raise ValueError("Only the pinned local runtime is allowed")
    protocol = {**_contract(), "frozen_at_utc": core.utc_now(), "runtime": runtime,
                "model": runtime["model_alias"], "source_sha256": sources,
                "frozen_input_sha256": inputs, "registration_sha256": digest(registration)}
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "sources").mkdir()
    for name in NEW_SOURCES:
        _write(directory / "sources" / Path(name).name, _bytes(ROOT / name), raw=True)
    _write(directory / "registration.md", registration, raw=True)
    _write(directory / "protocol.json", protocol)
    _write(directory / "freeze.json", {"protocol_sha256": digest(_bytes(directory / "protocol.json")),
                                         "frozen_at_utc": core.utc_now()})
    validate_protocol(directory)
    return protocol


def validate_protocol(directory):
    directory = _local(directory)
    before = _snapshot(directory)
    protocol = read(directory / "protocol.json")
    extra = {"frozen_at_utc", "runtime", "model", "source_sha256", "frozen_input_sha256",
             "registration_sha256"}
    fixed = _contract()
    if set(protocol) != set(fixed) | extra:
        raise ValueError("Unexpected or missing R2 protocol field")
    for key, value in fixed.items():
        if core.canonical_hash(protocol[key]) != core.canonical_hash(value):
            raise ValueError("R2 fixed parameter drift: " + key)
    if (protocol["source_sha256"] != _sources() or protocol["frozen_input_sha256"] != _frozen_inputs()
            or protocol["runtime"] != read(ROOT / "docs/rcwt_agent_runtime.json")
            or protocol["model"] != protocol["runtime"]["model_alias"]
            or protocol["registration_sha256"] != digest(_bytes(ROOT / REGISTRATION))
            or protocol["registration_sha256"] != digest(_bytes(directory / "registration.md"))
            or read(directory / "freeze.json")["protocol_sha256"] != digest(_bytes(directory / "protocol.json"))):
        raise ValueError("R2 source, history, runtime or registration drift")
    for name, expected in protocol["source_sha256"].items():
        if digest(_bytes(directory / "sources" / Path(name).name)) != expected:
            raise ValueError("New source snapshot drift")
    if before != _snapshot(directory):
        raise ValueError("Evidence changed while validating the protocol")
    return protocol


def schedule(replica, phase):
    if phase == "train":
        return [["fixed"] for _ in range(4)]
    # A single predeclared offset preserves Latin balance across all replicas:
    # per-family position counts 2/2/1 and global position counts 7/7/6.
    offset = int(core.canonical_hash([r["schedule_seed"] for r in replicas()]), 16) % 3
    return [list(ARMS[(i + replica["replica"] + offset) % 3:])
            + list(ARMS[:(i + replica["replica"] + offset) % 3]) for i in range(4)]


def all_rows(directory, phase="test"):
    if phase not in {"train", "validation", "test"}:
        raise ValueError("Unknown trace phase")
    return read_jsonl(Path(directory) / phase / "traces.jsonl")


def _policies(directory):
    rows = read_jsonl(directory / "proposals.jsonl")
    result = {(r, "fixed"): copy.deepcopy(DEFAULT_POLICY) for r in range(5)}
    for row in rows:
        key = (row["replica"], row["arm"])
        if key in result:
            raise ValueError("Duplicate policy proposal")
        result[key] = validate_policy(row["policy"])
    if set(result) != {(r, arm) for r in range(5) for arm in ARMS}:
        raise ValueError("Require exactly ten frozen policy proposals")
    return result


def perform_step(client, episode, executor, memory, policy, arm, index, budget):
    public = episode.public_step(index)
    actor_memory = core.build_context(memory, public, tokenize=client.tokenize, budget=budget)
    actor_tokens = len(client.tokenize(actor_memory))
    if actor_tokens > budget:
        raise ValueError("Actor memory cap exceeded")
    messages = core.actor_messages(actor_memory, public)
    draft = client.complete(core.planning_messages(messages), max_tokens=512,
                            schema=None, purpose="draft:" + arm)
    final = client.complete(core.review_messages(messages, draft.text), max_tokens=512,
                            schema=core.task_schema(public), purpose="action:" + arm)
    action = core.extract_action(final.text, final.finish_reason)
    completed_before = set(executor.completed)
    result = executor.execute_step(public, action)
    information = json.dumps({"observations": public["observations"],
                              "completed_request": public["task"], "action": action,
                              "tool_result": result.tool_result}, ensure_ascii=False, sort_keys=True)
    memory_after, truncated, candidates = memory, False, []
    if index < 7:
        retained, candidates = compact_policy(client.tokenize, memory, information, budget, policy)
        if retained.calls:
            raise ValueError("Retention must never perform inference")
        memory_after, truncated = retained.text, retained.truncated
    tokens = client.tokenize(memory_after)
    if len(tokens) > budget:
        raise ValueError("Persistent memory cap exceeded")
    # Private grading is downstream of the actor, safety kernel and writer.
    grade = score_action(episode, index, action, completed_before).to_dict()
    return {"public_step": public, "memory_before": memory,
            "actor_memory_before": actor_memory, "actor_memory_tokens": actor_tokens,
            "draft": draft.text, "raw_action": action, "proposal_score": grade, "score": copy.deepcopy(grade),
            "executor_result": result.to_dict(), "state_after": executor.state_dict(),
            "memory_after": memory_after, "memory_tokens": len(tokens),
            "memory_truncated": truncated, "retention_candidates": candidates}


def proposer_messages(feedback):
    return [{"role": "system", "content":
             "Return only the JSON retention policy matching the supplied schema. "
             "Six integer weights range from -4 to 4: invoice, account, payment, return, booked, incomplete. "
             "A component's retention score is the dot product of its Boolean features and these weights. "
             "The lowest score is evicted first, with oldest recency breaking ties. All-zero weights "
             "give the existing oldest-first writer. Feedback contains aggregated TRAIN associations only: "
             "Choose weights that prioritize retaining components useful to upcoming actor decisions, "
             "using the observed requested_later, later_raw_failed and evicted associations. "
             "invoice/account/payment/return/booked mean source-record presence, not authorization; "
             "incomplete means missing or unknown fields, not an approval signal. "
             "later failures are not proof that eviction caused them. Propose one policy; no code, IDs, "
             "extra keys, explanations or markdown. The actor, safety rules and token budget cannot change."},
            {"role": "user", "content": feedback_json(feedback)}]


def _proposal(client, feedback, arm):
    call = client.complete(proposer_messages(feedback), max_tokens=PROPOSAL_TOKENS,
                           schema=POLICY_SCHEMA,
                           purpose="policy:train-real" if arm == "learned" else "policy:train-shuffled")
    policy = parse_policy(call.text, call.finish_reason)
    return {"feedback": feedback, "policy": policy, "call": asdict(call)}


class _Meter:
    def __init__(self, directory, stage):
        self.path = directory / (stage + "-requests.jsonl")
        self.limit = EXPECTED_CALLS[stage]
        self.count = 0
        self.stage = stage

    def wrap(self, client, replica):
        meter = self

        class Metered:
            model, seed = client.model, client.seed

            def tokenize(self, text):
                return client.tokenize(text)

            def detokenize(self, tokens):
                return client.detokenize(tokens)

            def complete(self, messages, max_tokens, schema=None, purpose=""):
                if meter.count >= meter.limit:
                    raise ValueError("Finite stage inference budget exhausted")
                meter.count += 1
                arguments = dict(messages=messages, max_tokens=max_tokens, schema=schema, purpose=purpose)
                identity = {"number": meter.count, "request_sha256": core.canonical_hash(
                    core._payload(self.model, self.seed, arguments)), "purpose": purpose,
                    "replica": replica, "phase": PHASE.get(meter.stage, meter.stage)}
                _append(meter.path, {**identity, "status": "started"})
                try:
                    call = client.complete(**arguments)
                except BaseException as exc:
                    _append(meter.path, {**identity, "status": "failed",
                                          "exception_type": type(exc).__name__})
                    raise
                _append(meter.path, {**identity, "status": "completed",
                                      "result_sha256": core.canonical_hash(asdict(call))})
                return call

        return Metered()


def _client(protocol, replica):
    runtime = protocol["runtime"]
    client = core.LocalModelClient(model=protocol["model"], seed=replica["inference_seed"],
                                  endpoint=f"http://{runtime['host']}:{runtime['port']}")
    client.probe()
    return client


def _generate(protocol, phase):
    return [core.generate_episodes(phase, 4, replica[phase + "_seed"])
            for replica in protocol["replicas"]]


def _stage_inputs(directory, stage):
    result = {"protocol_sha256": digest(_bytes(directory / "protocol.json"))}
    position = STAGES.index(stage)
    if position:
        previous = STAGES[position - 1]
        result["previous_completion_sha256"] = digest(_bytes(directory / (previous + "-completion.json")))
    if stage in {"validate", "test"}:
        result["proposals_sha256"] = digest(_bytes(directory / "proposals.jsonl"))
    if stage == "test":
        result["selection_sha256"] = digest(_bytes(directory / "selection.json"))
    return result


def _run_trajectories(directory, protocol, stage, factory, meter):
    phase = PHASE[stage]
    destination = directory / phase
    destination.mkdir(exist_ok=False)
    corpus = _generate(protocol, phase)
    public = [[e.to_public_dict() for e in group] for group in corpus]
    oracle = [[e.to_oracle_dict() for e in group] for group in corpus]
    orders = [schedule(replica, phase) for replica in protocol["replicas"]]
    _write(destination / "public.json", public)
    _write(destination / "oracle.json", oracle)
    _write(destination / "schedule.json", orders)
    _write(destination / "freeze.json", {**_stage_inputs(directory, stage),
           "public_sha256": digest(_bytes(destination / "public.json")),
           "oracle_sha256": digest(_bytes(destination / "oracle.json")),
           "schedule_sha256": digest(_bytes(destination / "schedule.json"))})
    policies = ({(r, "fixed"): DEFAULT_POLICY for r in range(5)}
                if phase == "train" else _policies(directory))
    chain = None
    for replica, episodes, order in zip(protocol["replicas"], corpus, orders):
        client = meter.wrap(factory(protocol, replica), replica["replica"])
        if client.model != protocol["model"] or client.seed != replica["inference_seed"]:
            raise ValueError("Client differs from registered model or replica seed")
        for episode, arms in zip(episodes, order):
            for arm in arms:
                executor, memory, rows = SafetyExecutor(), "", []
                for index in range(8):
                    audited = core.AuditedClient(client)
                    start = time.perf_counter()
                    try:
                        actual = perform_step(audited, episode, executor, memory,
                                              policies[replica["replica"], arm], arm, index, 256)
                    except BaseException:
                        _write(directory / (stage + "-partial-step.json"), {
                            "replica": replica["replica"], "episode_id": episode.episode_id,
                            "arm": arm, "step_index": index, "client_events": audited.events,
                            "unmetered_inflight_call_possible": True})
                        raise
                    row = {"schema": SCHEMA, "replica": replica["replica"], "phase": phase,
                           "episode_id": episode.episode_id, "family": episode.family, "split": episode.split,
                           "arm": arm, "step_index": index, **actual, "client_events": audited.events,
                           "step_seconds": time.perf_counter() - start, "previous_sha256": chain}
                    row["sha256"] = core.canonical_hash(row)
                    chain, memory = row["sha256"], actual["memory_after"]
                    _append(destination / "traces.jsonl", row)
                    rows.append(row)
                    print(f"R2 {phase} replica={replica['replica']} {arm} step={index + 1}/8", flush=True)
                _append(destination / "episodes.jsonl", summarize_episode(rows))
    if phase == "validation":
        _write(directory / "selection.json", selection(all_rows(directory, phase)))


def _run_proposals(directory, protocol, factory, meter):
    train_rows = all_rows(directory, "train")
    chain = None
    for replica in protocol["replicas"]:
        number = replica["replica"]
        real = build_feedback([row for row in train_rows if row["replica"] == number])
        shuffled = shuffle_feedback(real, replica["shuffle_seed"])
        client = meter.wrap(factory(protocol, replica), number)
        if client.model != protocol["model"] or client.seed != replica["inference_seed"]:
            raise ValueError("Policy client differs from registered model or seed")
        for arm, feedback in (("learned", real), ("shuffled", shuffled)):
            audited = core.AuditedClient(client)
            start = time.perf_counter()
            try:
                actual = _proposal(audited, feedback, arm)
            except BaseException:
                _write(directory / "propose-partial-step.json", {"replica": number, "arm": arm,
                    "client_events": audited.events, "unmetered_inflight_call_possible": True})
                raise
            row = {"schema": SCHEMA, "replica": number, "arm": arm, **actual,
                   "client_events": audited.events, "step_seconds": time.perf_counter() - start,
                   "previous_sha256": chain}
            row["sha256"] = core.canonical_hash(row)
            chain = row["sha256"]
            _append(directory / "proposals.jsonl", row)


def _stage_artifacts(directory, stage):
    result = {}
    if stage in PHASE:
        result.update({PHASE[stage] + "/" + name: value
                       for name, value in _snapshot(directory / PHASE[stage]).items()})
    else:
        result["proposals.jsonl"] = digest(_bytes(directory / "proposals.jsonl"))
    if stage == "validate":
        result["selection.json"] = digest(_bytes(directory / "selection.json"))
    result[stage + "-requests.jsonl"] = digest(_bytes(directory / (stage + "-requests.jsonl")))
    if read(directory / "protocol.json")["evidence_kind"] == "real_local_model":
        for label in ("before", "after"):
            name = stage + "-runtime-" + label + ".json"
            result[name] = digest(_bytes(directory / name))
    return result


def _inventory(directory, stages, protocol):
    allowed = {"protocol.json", "freeze.json", "registration.md"}
    allowed.update("sources/" + Path(name).name for name in NEW_SOURCES)
    for stage in stages:
        allowed.update(stage + suffix for suffix in ("-started.json", "-completion.json", "-requests.jsonl"))
        if protocol["evidence_kind"] == "real_local_model":
            allowed.update(stage + "-runtime-" + label + ".json" for label in ("before", "after"))
        if stage in PHASE:
            allowed.update(PHASE[stage] + "/" + name for name in
                           ("public.json", "oracle.json", "schedule.json", "freeze.json", "traces.jsonl", "episodes.jsonl"))
        else:
            allowed.add("proposals.jsonl")
        if stage == "validate":
            allowed.add("selection.json")
    observed = set(_snapshot(directory))
    if stages == list(STAGES):
        report_files = {"analysis.json", "verification.json", "RESULTS.md", "report-started.json"}
        allowed.update(observed & report_files)
    if observed != allowed:
        raise ValueError("Missing, unexpected or prematurely exposed stage artifacts")
    # Empty future-stage directories are also exposure/state, not ignored data.
    dirs = {path.name for path in directory.iterdir() if path.is_dir()}
    if dirs != {"sources"} | {PHASE[s] for s in stages if s in PHASE}:
        raise ValueError("Unexpected or premature phase directory")


RUNTIME_FIELDS = {"schema", "pass", "status", "read_only", "inference_calls", "health_requests",
                  "pid", "process_start_time_utc", "executable_sha256", "model_sha256", "model_bytes",
                  "runtime_config_sha256", "helper_sha256", "started_receipt_sha256",
                  "intent_receipt_sha256", "arguments_sha256", "health_status", "host", "port"}


def _validate_runtime_status(value, protocol):
    runtime = protocol["runtime"]
    if not isinstance(value, dict) or set(value) != RUNTIME_FIELDS:
        raise ValueError("Unexpected or missing runtime status field")
    fixed = {"schema": "rcwt-r2-runtime-status/1", "pass": True, "status": "READY", "read_only": True,
             "inference_calls": 0, "health_requests": 1, "health_status": "ok",
             "host": runtime["host"], "port": runtime["port"],
             "executable_sha256": runtime["executable_sha256"], "model_sha256": runtime["model_sha256"],
             "model_bytes": runtime["model_bytes"],
             "runtime_config_sha256": digest(_bytes(ROOT / "docs/rcwt_agent_runtime.json")),
             "helper_sha256": protocol["source_sha256"]["tools/rcwt_r2_runtime.ps1"]}
    if any(core.canonical_hash(value[key]) != core.canonical_hash(expected) for key, expected in fixed.items()):
        raise ValueError("Runtime process, model, helper or configuration is not the pinned local runtime")
    if type(value["pid"]) is not int or value["pid"] <= 0:
        raise ValueError("Runtime PID must be a positive integer")
    if not isinstance(value["process_start_time_utc"], str) or not value["process_start_time_utc"]:
        raise ValueError("Runtime process creation time is required")
    for key in ("started_receipt_sha256", "intent_receipt_sha256", "arguments_sha256"):
        if not isinstance(value[key], str) or len(value[key]) != 64 or any(c not in "0123456789abcdef" for c in value[key]):
            raise ValueError("Invalid runtime ownership hash")
    return value


def _runtime_status(protocol, evidence):
    if evidence is None:
        raise ValueError("Real stages require --runtime-evidence with a verified owned local process")
    evidence = _local(evidence)
    if not evidence.is_dir():
        raise ValueError("Runtime evidence must be an owned directory")
    result = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-File",
        str(ROOT / "tools/rcwt_r2_runtime.ps1"), "-Mode", "status", "-OutputDirectory", str(evidence)],
        capture_output=True, text=True, encoding="utf-8", timeout=240, check=False)
    if result.returncode != 0:
        raise ValueError("Owned runtime physical verification failed")
    value = strict_json(result.stdout.lstrip("\ufeff").encode("utf-8"))
    return _validate_runtime_status(value, protocol)


def run_stage(directory, stage, client_factory=None, runtime_evidence=None):
    if stage not in STAGES:
        raise ValueError("Unknown inference stage")
    directory = _local(directory)
    if any((directory / (s + "-aborted.json")).exists() for s in STAGES):
        raise ValueError("Campaign aborted; no retry or resumption")
    if (directory / (stage + "-started.json")).exists():
        raise ValueError("Stage already started; no overwrite or retry")
    protocol = validate_protocol(directory)
    _output(directory, protocol["evidence_kind"])
    if client_factory is not None and protocol["evidence_kind"] != "EXPLICIT_SYNTHETIC_TEST_ONLY":
        raise ValueError("A fake/injected transport cannot produce real-model evidence")
    if protocol["evidence_kind"] == "EXPLICIT_SYNTHETIC_TEST_ONLY" and client_factory is None:
        raise ValueError("Synthetic evidence must never construct a real client")
    if protocol["evidence_kind"] == "EXPLICIT_SYNTHETIC_TEST_ONLY" and not (
            ".runs" in directory.parts and any(p.startswith("testing-r2-") for p in directory.parts)):
        raise ValueError("Synthetic evidence requires an explicitly segregated test directory")
    previous = STAGES[:STAGES.index(stage)]
    if any(not (directory / (s + "-completion.json")).exists() for s in previous):
        raise ValueError("All previous stages must be completely verified")
    verify_run(directory, require_complete=False)
    _write(directory / (stage + "-started.json"), {**_stage_inputs(directory, stage),
           "started_at_utc": core.utc_now(), "expected_generation_calls": EXPECTED_CALLS[stage]})
    meter = _Meter(directory, stage)
    try:
        if protocol["evidence_kind"] == "real_local_model":
            before_runtime = _runtime_status(protocol, runtime_evidence)
            _write(directory / (stage + "-runtime-before.json"), before_runtime)
        if stage == "propose":
            _run_proposals(directory, protocol, client_factory or _client, meter)
        else:
            _run_trajectories(directory, protocol, stage, client_factory or _client, meter)
        if meter.count != EXPECTED_CALLS[stage]:
            raise ValueError("Stage did not consume exactly its registered inference budget")
        if protocol["evidence_kind"] == "real_local_model":
            after_runtime = _runtime_status(protocol, runtime_evidence)
            _write(directory / (stage + "-runtime-after.json"), after_runtime)
            if core.canonical_hash(before_runtime) != core.canonical_hash(after_runtime):
                raise ValueError("Owned runtime process or byte identity changed during the stage")
        if validate_protocol(directory) != protocol:
            raise ValueError("Sources or protocol changed during inference")
        _write(directory / (stage + "-completion.json"), {**_stage_inputs(directory, stage),
               "generation_calls": meter.count, "artifact_sha256": _stage_artifacts(directory, stage),
               "completed_at_utc": core.utc_now()})
    except BaseException as exc:
        _write(directory / (stage + "-aborted.json"), {"exception_type": type(exc).__name__,
               "attempted_generation_calls": meter.count, "retry_allowed": False,
               "unmetered_inflight_call_possible": True, "aborted_at_utc": core.utc_now()})
        raise
    return verify_run(directory, require_complete=stage == "test")


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid finite nonnegative metering value")
    return value


def _trace_identity(row, metadata, chain):
    if (row.get("sha256") != core.canonical_hash({k: v for k, v in row.items() if k != "sha256"})
            or row.get("previous_sha256") != chain
            or any(core.canonical_hash(row.get(k)) != core.canonical_hash(v) for k, v in metadata.items())):
        raise ValueError("Trace identity, order or hash chain mismatch")


def _match_actual(row, actual, metadata):
    if (set(row) != set(actual) | set(metadata) | {"client_events", "step_seconds", "previous_sha256", "sha256"}
            or core.canonical_hash(actual) != core.canonical_hash({k: row[k] for k in actual})):
        raise ValueError("Actor input, proposal, safety effect, score or retention does not replay")
    calls = [e["result"] for e in row["client_events"] if e["method"] == "complete"]
    if _finite(row["step_seconds"]) + 1e-8 < sum(_finite(c["wall_seconds"]) for c in calls):
        raise ValueError("Step time excludes inference")
    return calls


def _replay_phase(directory, protocol, stage):
    phase = PHASE[stage]
    path = directory / phase
    corpus = _generate(protocol, phase)
    expected_public = [[e.to_public_dict() for e in group] for group in corpus]
    expected_oracle = [[e.to_oracle_dict() for e in group] for group in corpus]
    orders = [schedule(replica, phase) for replica in protocol["replicas"]]
    if (read(path / "public.json") != expected_public or read(path / "oracle.json") != expected_oracle
            or read(path / "schedule.json") != orders):
        raise ValueError("Sealed corpus or schedule changed")
    expected_freeze = {**_stage_inputs(directory, stage),
        **{name + "_sha256": digest(_bytes(path / (name + ".json"))) for name in ("public", "oracle", "schedule")}}
    if read(path / "freeze.json") != expected_freeze:
        raise ValueError("Stage corpus or policy freeze changed")
    rows, summaries = all_rows(directory, phase), read_jsonl(path / "episodes.jsonl")
    count = 160 if phase == "train" else 480
    if len(rows) != count or len(summaries) != count // 8:
        raise ValueError("Missing or additional scheduled trajectories")
    policies = ({(r, "fixed"): DEFAULT_POLICY for r in range(5)} if phase == "train" else _policies(directory))
    position, summary_position, chain, calls = 0, 0, None, []
    for replica, episodes, order in zip(protocol["replicas"], corpus, orders):
        for episode, arms in zip(episodes, order):
            for arm in arms:
                executor, memory, episode_rows = SafetyExecutor(), "", []
                for index in range(8):
                    row = rows[position]
                    position += 1
                    metadata = {"schema": SCHEMA, "replica": replica["replica"], "phase": phase,
                        "episode_id": episode.episode_id, "family": episode.family, "split": episode.split,
                        "arm": arm, "step_index": index}
                    _trace_identity(row, metadata, chain)
                    replay = core.ReplayClient(row["client_events"], protocol["model"], replica["inference_seed"])
                    actual = perform_step(replay, episode, executor, memory, policies[replica["replica"], arm], arm, index, 256)
                    replay.finish()
                    step_calls = _match_actual(row, actual, metadata)
                    if [c["purpose"] for c in step_calls] != ["draft:" + arm, "action:" + arm]:
                        raise ValueError("Every action requires exactly two unchanged actor calls")
                    calls.extend(step_calls)
                    chain, memory = row["sha256"], actual["memory_after"]
                    episode_rows.append(row)
                if core.canonical_hash(summaries[summary_position]) != core.canonical_hash(summarize_episode(episode_rows)):
                    raise ValueError("Episode metrics do not replay")
                summary_position += 1
    if stage == "validate" and core.canonical_hash(read(directory / "selection.json")) != core.canonical_hash(selection(rows)):
        raise ValueError("Validation deployment selection changed")
    return calls, len(rows)


def _replay_proposals(directory, protocol):
    rows = read_jsonl(directory / "proposals.jsonl")
    if len(rows) != 10:
        raise ValueError("Exactly ten proposals are required")
    train = all_rows(directory, "train")
    chain, position, calls = None, 0, []
    for replica in protocol["replicas"]:
        real = build_feedback([r for r in train if r["replica"] == replica["replica"]])
        shuffled = shuffle_feedback(real, replica["shuffle_seed"])
        for arm, feedback in (("learned", real), ("shuffled", shuffled)):
            row = rows[position]
            position += 1
            metadata = {"schema": SCHEMA, "replica": replica["replica"], "arm": arm}
            _trace_identity(row, metadata, chain)
            replay = core.ReplayClient(row["client_events"], protocol["model"], replica["inference_seed"])
            actual = _proposal(replay, feedback, arm)
            replay.finish()
            step_calls = _match_actual(row, actual, metadata)
            if len(step_calls) != 1:
                raise ValueError("Each policy must have exactly one generating call")
            calls.extend(step_calls)
            chain = row["sha256"]
    return calls, 0


def _verify_requests(directory, stage, calls, protocol):
    rows = read_jsonl(directory / (stage + "-requests.jsonl"))
    expected = []
    for number, call in enumerate(calls, 1):
        identity = {"number": number, "request_sha256": core.canonical_hash(call["request"]),
                    "purpose": call["purpose"], "phase": PHASE.get(stage, stage),
                    "replica": next(r["replica"] for r in protocol["replicas"]
                                    if r["inference_seed"] == call["request"]["seed"])}
        expected.extend([{**identity, "status": "started"},
                         {**identity, "status": "completed", "result_sha256": core.canonical_hash(call)}])
    if core.canonical_hash(rows) != core.canonical_hash(expected):
        raise ValueError("Attempted/completed inference journal does not reconcile")


def verify_run(directory, require_complete=True):
    directory = _local(directory)
    before, frozen_before = _snapshot(directory), _frozen_inputs()
    with _offline():
        protocol = validate_protocol(directory)
        if any((directory / (s + suffix)).exists() for s in STAGES
               for suffix in ("-aborted.json", "-partial-step.json")):
            raise ValueError("Aborted or partial campaign cannot pass complete integrity")
        present = [s for s in STAGES if (directory / (s + "-completion.json")).exists()]
        _inventory(directory, present, protocol)
        calls, steps, stages, missing = [], 0, [], False
        for stage in STAGES:
            if (directory / (stage + "-aborted.json")).exists() or (directory / (stage + "-partial-step.json")).exists():
                raise ValueError("Aborted or partial campaign cannot pass complete integrity")
            completion_path = directory / (stage + "-completion.json")
            if not completion_path.exists():
                missing = True
                if (directory / (stage + "-started.json")).exists():
                    raise ValueError("Started stage has no complete terminal receipt")
                continue
            if missing:
                raise ValueError("Completed stages are not a contiguous prefix")
            inputs = _stage_inputs(directory, stage)
            started = read(directory / (stage + "-started.json"))
            if (set(started) != set(inputs) | {"started_at_utc", "expected_generation_calls"}
                    or any(started[k] != v for k, v in inputs.items())
                    or started["expected_generation_calls"] != EXPECTED_CALLS[stage]):
                raise ValueError("Exclusive stage start binding changed")
            completed = read(completion_path)
            if (set(completed) != set(inputs) | {"generation_calls", "artifact_sha256", "completed_at_utc"}
                    or any(completed[k] != v for k, v in inputs.items())
                    or completed["artifact_sha256"] != _stage_artifacts(directory, stage)):
                raise ValueError("Stage completion artifact binding changed")
            if protocol["evidence_kind"] == "real_local_model":
                before_runtime = _validate_runtime_status(read(directory / (stage + "-runtime-before.json")), protocol)
                after_runtime = _validate_runtime_status(read(directory / (stage + "-runtime-after.json")), protocol)
                if core.canonical_hash(before_runtime) != core.canonical_hash(after_runtime):
                    raise ValueError("Runtime before/after ownership differs")
            stage_calls, stage_steps = (_replay_proposals(directory, protocol) if stage == "propose"
                                        else _replay_phase(directory, protocol, stage))
            if len(stage_calls) != EXPECTED_CALLS[stage] or completed["generation_calls"] != len(stage_calls):
                raise ValueError("Exact finite inference budget differs")
            _verify_requests(directory, stage, stage_calls, protocol)
            calls.extend(stage_calls)
            steps += stage_steps
            stages.append(stage)
        if require_complete and stages != list(STAGES):
            raise ValueError("All four stages are required for complete R2 verification")
        _inventory(directory, stages, protocol)
    if before != _snapshot(directory) or frozen_before != _frozen_inputs():
        raise ValueError("Input or source changed during read-only replay")
    return {"status": "PASS", "schema": SCHEMA, "integrity_only": True, "gain": "NOT_EVALUATED",
            "evidence_kind": protocol["evidence_kind"],
            "complete": stages == list(STAGES), "completed_stages": stages,
            "verified_steps": steps, "generation_calls": len(calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "inference_seconds": sum(c["wall_seconds"] for c in calls),
            "protocol_sha256": digest(_bytes(directory / "protocol.json")),
            "inference_calls": 0, "read_only": True}


def report(directory):
    directory = _local(directory)
    _output(directory, read(directory / "protocol.json")["evidence_kind"])
    before, old_before, sources_before = _snapshot(directory), _frozen_inputs(), _sources()
    verification = verify_run(directory)
    analysis = analyze_r2(all_rows(directory), train_rows=all_rows(directory, "train"),
        validation_rows=all_rows(directory, "validation"),
        proposal_calls=[row["call"] for row in read_jsonl(directory / "proposals.jsonl")])
    artifacts = {"analysis.json": json_bytes(analysis), "verification.json": json_bytes(verification),
                 "RESULTS.md": render_report(analysis, verification).encode("utf-8")}
    if before != _snapshot(directory) or old_before != _frozen_inputs() or sources_before != _sources():
        raise ValueError("Inputs or sources changed during report recomputation")
    present = [name for name in artifacts if (directory / name).exists()]
    if present:
        if len(present) != len(artifacts) or any(_bytes(directory / n) != b for n, b in artifacts.items()):
            raise ValueError("Existing report differs; no overwrite")
    else:
        marker = {"protocol_sha256": verification["protocol_sha256"]}
        _write(directory / "report-started.json", marker)
        for name, data in artifacts.items():
            _write(directory / name, data, raw=True)
        before = {**before, **{name: digest(data) for name, data in artifacts.items()},
                  "report-started.json": digest(json_bytes(marker))}
    if (before != _snapshot(directory) or old_before != _frozen_inputs() or sources_before != _sources()
            or any(_bytes(directory / name) != data for name, data in artifacts.items())):
        raise ValueError("Report outputs or original evidence changed during publication to the local directory")
    validate_protocol(directory)
    if before != _snapshot(directory):
        raise ValueError("Report evidence changed during final protocol validation")
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("prepare", *STAGES, "verify", "report"))
    parser.add_argument("--runtime-evidence", type=Path,
                        help="Private owned runtime directory; mandatory only for real inference stages")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output_dir)
        result = {"status": "PREPARED", "corpora_generated": 0, "inference_calls": 0}
    elif args.stage == "verify":
        result = verify_run(args.output_dir)
    elif args.stage == "report":
        result = report(args.output_dir)
    else:
        result = run_stage(args.output_dir, args.stage, runtime_evidence=args.runtime_evidence)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
