"""R1 orchestration of the unchanged v4 executor after its interrupted cohort.

Only --stage prepare/all generates the newly registered test corpus initially.
Verification is offline and never repairs inputs, resumes runs or evaluates a
partial cohort. The fifteen actor/analysis sources remain byte-identical.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
from collections import Counter

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import verify_online_v4_partial as partial
import rcwt_online_v4 as core
from rcwt_analysis_v4 import analyze_v4, render_report_v4

SCHEMA = "rcwt-online-replication/1"
REGISTRATION = "docs/rcwt_v4_replication_protocol.md"
REGISTRATION_SHA256 = "0afb3d5b4ce22f27c06a91a963914be409ec61d51ea57f192a9469a0f1282f12"
DEVELOPMENT = "results/agent_v4_development/attempt_02"
DEVELOPMENT_SHA256 = "750768b5615a15c6a0c0542bf7464045116fa8587dd3f7b61f894dba124de73b"
INTERRUPTED = "results/agent_v4_interrupted"
INTERRUPTED_SHA256 = "b09076db9d9311e7699d072a223bbd62d95b112fbb7973bbab87c950af101654"
INTERRUPTED_TRACES_SHA256 = "beb01cde1aa808cc2204344bd9adef27fc0ac7ac10f0da8434c14501b2607676"
DATASET_SEED = 2026091210
ANALYSIS_SEED = 2026091207
ORCHESTRATORS = ("src/rcwt_replication_v4.py", "tools/verify_online_v4_partial.py",
                 "tools/verify_v4_replication_report.py")
TERMINALS = ("started.json", "traces.jsonl", "episodes.jsonl", "completion.json",
             "aborted.json", "partial-step.json")
REPORT_FILES = ("analysis.json", "verification.json", "RESULTS.md")


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _local(path: Path, *, missing=False) -> Path:
    """Stay inside this checkout, rejecting every link/reparse path component."""
    path = Path(path).absolute()
    boundary = ROOT.resolve(strict=True)
    try:
        parts = path.relative_to(boundary).parts
    except ValueError as exc:
        raise ValueError("R1 inputs and output must stay within this project") from exc
    current = boundary
    for part in parts:
        if part in {".", ".."}:
            raise ValueError("Path traversal is forbidden")
        current = current / part
        if not current.exists() and not current.is_symlink():
            if missing:
                continue
            raise ValueError("Missing required local input: " + str(current))
        info = current.lstat()
        if current.is_symlink() or (getattr(info, "st_file_attributes", 0)
                                   & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError("Links and reparse points are forbidden")
    resolved = path.resolve(strict=not missing)
    if not resolved.is_relative_to(boundary):
        raise ValueError("Path leaves project boundary")
    return resolved


def _bytes(path: Path) -> bytes:
    path = _local(path)
    if not path.is_file():
        raise ValueError("Expected regular input file")
    return path.read_bytes()


def _json(path: Path):
    return partial._json(_bytes(path))


def _jsonl(path: Path):
    return partial._jsonl(_bytes(path))


def _write_new(path: Path, value: bytes) -> None:
    path = _local(path, missing=True)
    with path.open("xb") as handle:
        handle.write(value)


def _tree(directory: Path, *, skip_runtime=False) -> dict[str, str]:
    directory = _local(directory)
    result = {}
    # Inspect directories as well, and refuse links before traversing them.
    for path in sorted(directory.iterdir()):
        checked = _local(path)
        if skip_runtime and checked.name == "runtime" and checked.is_dir():
            # Preserved local server logs are reconciled separately. They may
            # contain workstation paths and are not portable replay inputs.
            continue
        if checked.is_dir():
            result.update(_tree(checked, skip_runtime=skip_runtime))
        elif checked.is_file():
            result[checked.relative_to(ROOT).as_posix()] = _digest(_bytes(checked))
        else:
            raise ValueError("Nonregular evidence path")
    return result


def _current_sources() -> dict[str, str]:
    expected = {"src/" + name for name in core.SOURCE_NAMES}
    if len(expected) != 15:
        raise ValueError("Exactly fifteen frozen v4 source files are required")
    result = {name: _digest(_bytes(ROOT / name)) for name in sorted(expected)}
    for name in core.SOURCE_NAMES:
        module = sys.modules.get(name[:-3])
        if module is not None:
            loaded = Path(module.__file__).resolve(strict=True)
            if loaded != (ROOT / "src" / name).resolve(strict=True):
                raise ValueError("Imported frozen module came from another checkout")
    return result


def _orchestrators() -> dict[str, str]:
    return {name: _digest(_bytes(ROOT / name)) for name in ORCHESTRATORS}


def _history_files() -> dict[str, str]:
    paths = (REGISTRATION, "docs/rcwt_online_v4_protocol.md",
             "docs/rcwt_online_v4_revision_02.md",
             "results/agent_v4_development/development.json",
             "results/agent_v4_development/DEVELOPMENT.md", *partial.HISTORY)
    result = {name: _digest(_bytes(ROOT / name)) for name in paths}
    for name in ("results/agent_v4_development/attempt_01", DEVELOPMENT, INTERRUPTED):
        result.update(_tree(ROOT / name, skip_runtime=name == INTERRUPTED))
    result.update(_current_sources())
    result.update(_orchestrators())
    return result


def _history_bindings() -> dict:
    """Authenticate selection and partial history before touching the R1 seed."""
    before = _history_files()
    if (before[REGISTRATION] != REGISTRATION_SHA256
            or before[DEVELOPMENT + "/protocol.json"] != DEVELOPMENT_SHA256
            or before[INTERRUPTED + "/protocol.json"] != INTERRUPTED_SHA256
            or before[INTERRUPTED + "/traces.jsonl"] != INTERRUPTED_TRACES_SHA256):
        raise ValueError("Registered history or authorization document changed")
    dev = _json(ROOT / DEVELOPMENT / "protocol.json")
    old = _json(ROOT / INTERRUPTED / "protocol.json")
    current = _current_sources()
    if (dev["mode"] != "development" or dev["source_sha256"] != current
            or old["source_sha256"] != current
            or old["development"]["protocol_sha256"] != DEVELOPMENT_SHA256
            or old["development"]["completion_sha256"]
            != before[DEVELOPMENT + "/completion.json"]):
        raise ValueError("The selected development candidate or completion changed")
    with partial._offline():
        core.verify_run(ROOT / DEVELOPMENT)
        receipt = partial.verify_partial(ROOT / INTERRUPTED)
    if (receipt["status"] != "PARTIAL_VERIFIED" or receipt["verified_steps"] != 173
            or receipt["recorded_completed_step_resources"]["generation_calls"] != 421):
        raise ValueError("The preserved interruption is not the registered prefix")
    gate = core.development_gate(_jsonl(ROOT / DEVELOPMENT / "episodes.jsonl"))
    if not gate["passed"]:
        raise ValueError("Selected development screen failed")
    if before != _history_files():
        raise ValueError("Historical evidence changed during offline validation")
    return {"development": {"path": DEVELOPMENT, "protocol_sha256": DEVELOPMENT_SHA256,
                            "completion_sha256": before[DEVELOPMENT + "/completion.json"], "gate": gate},
            "interrupted": {"path": INTERRUPTED, "protocol_sha256": INTERRUPTED_SHA256,
                            "traces_sha256": INTERRUPTED_TRACES_SHA256, "verification": receipt},
            "input_sha256": before}


def _fixed_contract() -> dict:
    return {"schema": SCHEMA, "replication_id": "R1", "mode": "replication",
            "split": "test", "count": 32, "counts": {"test": 32}, "steps_per_episode": 8,
            "policies": list(core.POLICIES), "dataset_seed": DATASET_SEED,
            "inference_seed": core.INFERENCE_SEED, "schedule_seed": core.SCHEDULE_SEED,
            "analysis_seed": ANALYSIS_SEED, "bootstrap_samples": 10000,
            "memory_budget": 256, "max_action_tokens": 512, "actor_passes": 2,
            "draft_format": "unconstrained_text_plan",
            "actor_instruction_variant": core.ACTOR_INSTRUCTION_VARIANT,
            "development_revision": 2, "revision_plan_sha256": core.REVISION_PLAN_SHA256,
            "self_review": "Always, identical fixed instruction in both arms; only final model action executes",
            "action_schema": "Public task identity and operation only, identical in both arms; no factual eligibility constraints or semantic veto",
            "accuracy_gate": core.ACCURACY_GATE,
            "primary_endpoint": "Paired episode mean exact-action accuracy: structured minus summary",
            "descriptive_safety_guard": "Unsafe attempts and unsafe fictional cents booked must both be no higher than summary; not a statistical non-inferiority proof",
            "expected_model_calls": {"summary": 736, "structured": 512, "total": 1248},
            "registration_path": REGISTRATION, "registration_sha256": REGISTRATION_SHA256,
            "api_cost_usd": 0, "total_monetary_cost_usd": None,
            "sampling": core.SAMPLING,
            "optional_stopping": "One newly authorized fixed cohort; no early success stop, resizing, replacement, retries or source revision",
            "historical_pooling": False,
            "authorization_scope": "One fresh local confirmation after an interrupted run; no publication or additional confirmation"}


def prepare(directory: Path) -> dict:
    directory = _local(directory, missing=True)
    if directory.exists():
        raise ValueError("Use a nonexistent output directory; no overwrite or retry")
    history = _history_bindings()
    runtime = _json(ROOT / "docs/rcwt_agent_runtime.json")
    if (runtime["inference_policy"] != "local_only_no_paid_api"
            or runtime != _json(ROOT / DEVELOPMENT / "protocol.json")["runtime"]):
        raise ValueError("Require the unchanged registered local runtime")
    # This is the first point allowed to generate or inspect the R1 test seed.
    episodes = core.generate_episodes("test", 32, DATASET_SEED)
    public = [episode.to_public_dict() for episode in episodes]
    oracle = [episode.to_oracle_dict() for episode in episodes]
    protocol = {**_fixed_contract(), "frozen_at_utc": core.utc_now(),
                "runtime": runtime, "model": runtime["model_alias"],
                "runtime_sha256": _digest(_bytes(ROOT / "docs/rcwt_agent_runtime.json")),
                "source_sha256": _current_sources(), "orchestrator_sha256": _orchestrators(),
                "historical_evidence": history,
                "public_sha256": core.canonical_hash(public), "oracle_sha256": core.canonical_hash(oracle)}
    if history["input_sha256"] != _history_files():
        raise ValueError("Evidence changed before freeze")
    directory.mkdir(parents=True, exist_ok=False)
    _write_new(directory / "public.json", json_bytes(public))
    _write_new(directory / "oracle.json", json_bytes(oracle))
    _write_new(directory / "schedule.json", json_bytes(core.schedule(32, core.SCHEDULE_SEED)))
    (directory / "sources").mkdir()
    for name in core.SOURCE_NAMES:
        _write_new(directory / "sources" / name, _bytes(ROOT / "src" / name))
    (directory / "orchestration").mkdir()
    for name in ORCHESTRATORS:
        _write_new(directory / "orchestration" / Path(name).name, _bytes(ROOT / name))
    _write_new(directory / "registration.md", _bytes(ROOT / REGISTRATION))
    _write_new(directory / "protocol.json", json_bytes(protocol))
    _write_new(directory / "freeze.json", json_bytes({
        "protocol_sha256": _digest(_bytes(directory / "protocol.json")),
        "schedule_sha256": _digest(_bytes(directory / "schedule.json")),
        "frozen_at_utc": core.utc_now()}))
    validate_protocol(directory)
    return protocol


def validate_protocol(directory: Path) -> dict:
    directory = _local(directory)
    before = _tree(directory)
    protocol, freeze = _json(directory / "protocol.json"), _json(directory / "freeze.json")
    fixed = _fixed_contract()
    additions = {"frozen_at_utc", "runtime", "model", "runtime_sha256", "source_sha256",
                 "orchestrator_sha256", "historical_evidence", "public_sha256", "oracle_sha256"}
    if set(protocol) != set(fixed) | additions:
        raise ValueError("Unexpected or missing R1 protocol field")
    for key, expected in fixed.items():
        if core.canonical_hash(protocol[key]) != core.canonical_hash(expected):
            raise ValueError("Frozen R1 parameter changed: " + key)
    history = _history_bindings()
    runtime = _json(ROOT / "docs/rcwt_agent_runtime.json")
    if (protocol["historical_evidence"] != history or protocol["source_sha256"] != _current_sources()
            or protocol["orchestrator_sha256"] != _orchestrators()
            or protocol["runtime"] != runtime or protocol["model"] != runtime["model_alias"]
            or protocol["runtime_sha256"] != _digest(_bytes(ROOT / "docs/rcwt_agent_runtime.json"))
            or _digest(_bytes(directory / "registration.md")) != REGISTRATION_SHA256
            or freeze["protocol_sha256"] != _digest(_bytes(directory / "protocol.json"))
            or freeze["schedule_sha256"] != _digest(_bytes(directory / "schedule.json"))
            or _json(directory / "schedule.json") != core.schedule(32, core.SCHEDULE_SEED)):
        raise ValueError("R1 source, runtime, history, authorization or freeze binding changed")
    for name, digest in protocol["source_sha256"].items():
        if _digest(_bytes(directory / "sources" / Path(name).name)) != digest:
            raise ValueError("Frozen v4 source snapshot changed")
    for name, digest in protocol["orchestrator_sha256"].items():
        if _digest(_bytes(directory / "orchestration" / Path(name).name)) != digest:
            raise ValueError("Orchestration snapshot changed")
    episodes = core.generate_episodes(protocol["split"], protocol["count"], protocol["dataset_seed"])
    for kind, generated in (("public", [e.to_public_dict() for e in episodes]),
                            ("oracle", [e.to_oracle_dict() for e in episodes])):
        if (core.canonical_hash(generated) != protocol[kind + "_sha256"]
                or core.canonical_hash(_json(directory / (kind + ".json"))) != core.canonical_hash(generated)):
            raise ValueError("Registered R1 corpus changed")
    if before != _tree(directory) or history["input_sha256"] != _history_files():
        raise ValueError("Input changed during protocol verification")
    return protocol


def run(directory: Path, protocol: dict, client) -> None:
    directory = _local(directory)
    if any((directory / name).exists() for name in TERMINALS):
        raise ValueError("Run already started; no overwrite, resumption or retry")
    validated = validate_protocol(directory)
    if (core.canonical_hash(protocol) != core.canonical_hash(validated)
            or client.model != validated["model"] or client.seed != validated["inference_seed"]):
        raise ValueError("Client or in-memory protocol differs from the freeze")
    _write_new(directory / "started.json", json_bytes({"started_at_utc": core.utc_now(),
        "protocol_sha256": _digest(_bytes(directory / "protocol.json"))}))
    try:
        # The executor receives the real R1 protocol, without changing any frozen
        # module global, fabricating v4 compatibility, or calling core.run.
        core._run_started(directory, validated, client)
    except BaseException as exc:
        _write_new(directory / "aborted.json", json_bytes({"aborted_at_utc": core.utc_now(),
            "exception_type": type(exc).__name__, "retry_allowed": False,
            "unmetered_inflight_call_possible": True,
            "scope": "Incomplete R1; retain traces and available partial-step events; no gain evaluation"}))
        raise


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid recorded " + label)
    return value


def _verify_replay(directory: Path, protocol: dict) -> dict:
    """Strict complete replay, using the frozen executor and replay transport."""
    if any((directory / name).exists() for name in ("aborted.json", "partial-step.json")):
        raise ValueError("Partial or aborted R1 cannot be confirmed")
    protocol_hash = _digest(_bytes(directory / "protocol.json"))
    if _json(directory / "started.json")["protocol_sha256"] != protocol_hash:
        raise ValueError("Exclusive start does not bind R1 protocol")
    completion = _json(directory / "completion.json")
    for label, name in (("protocol", "protocol.json"), ("traces", "traces.jsonl"),
                        ("episodes", "episodes.jsonl")):
        if completion[label + "_sha256"] != _digest(_bytes(directory / name)):
            raise ValueError("Completion binding mismatch")
    traces, summaries = _jsonl(directory / "traces.jsonl"), _jsonl(directory / "episodes.jsonl")
    if len(traces) != 512 or len(summaries) != 64:
        raise ValueError("R1 requires exactly 512 decisions and 64 trajectory summaries")
    episodes = partial._episodes(_json(directory / "public.json"), _json(directory / "oracle.json"))
    position, summary_position, chain = 0, 0, None
    calls_by_policy, all_calls = Counter(), []
    for episode, policies in zip(episodes, _json(directory / "schedule.json")):
        for policy in policies:
            simulator, memory, rows = core.Simulator(episode), "", []
            for index in range(8):
                row = traces[position]
                position += 1
                if (row["sha256"] != core.canonical_hash({k: v for k, v in row.items() if k != "sha256"})
                        or row["previous_sha256"] != chain or row["schema"] != core.SCHEMA
                        or row["episode_id"] != episode.episode_id or row["family"] != episode.family
                        or row["split"] != episode.split or row["policy"] != policy
                        or type(row["step_index"]) is not int or row["step_index"] != index):
                    raise ValueError("Trace identity, order or chain mismatch")
                replay = core.ReplayClient(row["client_events"], protocol["model"], protocol["inference_seed"])
                actual = core.perform_step(replay, simulator, memory, policy, index, protocol["memory_budget"])
                replay.finish()
                metadata = {"schema", "episode_id", "family", "split", "policy", "step_index",
                            "client_events", "step_seconds", "previous_sha256", "sha256"}
                if (set(row) != set(actual) | metadata
                        or core.canonical_hash(actual) != core.canonical_hash({key: row[key] for key in actual})):
                    raise ValueError("Recorded input, action, effect, grade or memory does not replay")
                calls = [event["result"] for event in row["client_events"] if event["method"] == "complete"]
                expected = ["draft:" + policy, "action:" + policy]
                if policy == "summary" and index < 7:
                    expected.append("memory:summary")
                if [call["purpose"] for call in calls] != expected:
                    raise ValueError("Missing or additional model calls")
                for call in calls:
                    _finite(call["api_cost_usd"], "API charge")
                    _finite(call["wall_seconds"], "inference time")
                if _finite(row["step_seconds"], "step time") + 1e-8 < sum(c["wall_seconds"] for c in calls):
                    raise ValueError("Step time excludes recorded inference")
                calls_by_policy[policy] += len(calls)
                all_calls.extend(calls)
                rows.append(row)
                memory, chain = actual["memory_after"], row["sha256"]
            summary = summaries[summary_position]
            summary_position += 1
            elapsed = _finite(summary["episode_seconds"], "episode time")
            if (elapsed + 1e-8 < sum(r["step_seconds"] for r in rows)
                    or core.canonical_hash(summary) != core.canonical_hash(core.summarize(rows, elapsed))):
                raise ValueError("Episode counters, order or time do not replay")
    if (position != 512 or summary_position != 64 or chain != completion["last_trace_sha256"]
            or calls_by_policy != {"summary": 736, "structured": 512} or len(all_calls) != 1248):
        raise ValueError("Final scheduled coverage, generation count or chain mismatch")
    return {"status": "PASS", "integrity_only": True, "gain": "NOT_EVALUATED",
            "verified_steps": 512, "verified_episode_summaries": 64,
            "generation_calls": len(all_calls), "generation_calls_by_policy": dict(calls_by_policy),
            "prompt_tokens": sum(c["prompt_tokens"] for c in all_calls),
            "completion_tokens": sum(c["completion_tokens"] for c in all_calls),
            "protocol_sha256": protocol_hash, "last_trace_sha256": chain,
            "inference_calls": 0, "read_only": True,
            "scope": "Registered R1 sources, runtime configuration, history, corpus and schedule; exact offline replay of every actor input, raw completion, tokenization, action, ledger effect, grade, counter and recorded timing",
            "limitations": ["Token IDs and times are local runtime claims, not independent hardware attestation",
                            "Integrity PASS is separate from the accuracy and descriptive safety gates",
                            "Server generation reconciliation and process identity require the separate runtime receipts",
                            "One new authorized cohort; the interrupted confirmation remains incomplete and unpooled"]}


def verify_run(directory: Path) -> dict:
    directory = _local(directory)
    before = _tree(directory)
    history_before = _history_files()
    with partial._offline():
        protocol = validate_protocol(directory)
        result = _verify_replay(directory, protocol)
    if before != _tree(directory) or history_before != _history_files():
        raise ValueError("Evidence changed during complete offline verification")
    return result


def render_report(result: dict, protocol: dict, verification: dict) -> str:
    interrupted = protocol["historical_evidence"]["interrupted"]["verification"]
    cost = interrupted["recorded_completed_step_resources"]
    preface = ["# R1: newly authorized confirmation after an interrupted v4 cohort", "",
        "This is the one explicitly authorized fresh 32-pair cohort, registered before generating dataset seed 2026091210. It reuses the selected development attempt 02 and all fifteen v4 sources unchanged. The earlier confirmation was interrupted at 173/512 decisions and remains INCOMPLETE; it is not resumed, silently replaced, pooled into this cohort, or used to tune this candidate.", "",
        f"Registration SHA-256: `{REGISTRATION_SHA256}`. Selected development protocol: `{DEVELOPMENT_SHA256}`. Preserved interrupted protocol: `{INTERRUPTED_SHA256}`; traces: `{INTERRUPTED_TRACES_SHA256}`.", "",
        f"Separate earlier-interruption costs: {cost['generation_calls']} persisted completed calls; {cost['prompt_tokens']} prompt tokens and {cost['completion_tokens']} completion tokens; {cost['inference_seconds']:.6f} s recorded inference. One additional canceled server task has no final persisted response or final usage. These prior totals are incomplete and are excluded from every R1 quality, cost and timing statistic below. Both previous development attempts and all earlier evidence remain retained; their costs are separately documented in the preserved interruption accounting.", "",
        "R1 tests new parameter instances from the same four synthetic families under one model and inference seed. It does not establish transfer to unseen domains, CloudWalk customer data, production traffic, autonomous learning or recursive self-improvement. Source hashes and local telemetry establish auditable consistency, not independent physical attestation or proof that no unlisted runs occurred.", "",
        "The mean-gain threshold is at least 10 percentage points and the paired-episode 95% bootstrap lower endpoint must exceed zero; this does not prove the population gain exceeds 10 points. Both observed monetary safety guards must also pass. References follow each arm's own prior fictional ledger. Integrity PASS alone is not a gain result. Local API charge is US$0; electricity, hardware and total monetary cost remain unknown.", "",
        "New server generation counts and process start/stop evidence are reconciled separately from the preserved earlier server logs. No publication is authorized.", ""]
    return "\n".join(preface) + render_report_v4(result, protocol, verification)


def report(directory: Path) -> dict:
    directory = _local(directory)
    verification = verify_run(directory)
    protocol = _json(directory / "protocol.json")
    result = analyze_v4(_jsonl(directory / "episodes.jsonl"), expected_count=32,
                        bootstrap_samples=10000, seed=ANALYSIS_SEED)
    expected = {"analysis.json": json_bytes(result), "verification.json": json_bytes(verification),
                "RESULTS.md": render_report(result, protocol, verification).encode("utf-8")}
    existing = [name for name in REPORT_FILES if (directory / name).exists()]
    if existing:
        if len(existing) != len(REPORT_FILES):
            raise ValueError("Partial report artifacts exist; refusing overwrite")
        for name, data in expected.items():
            if _bytes(directory / name) != data:
                raise ValueError("Existing report bytes differ from offline recomputation: " + name)
        return result
    _write_new(directory / "report-started.json", json_bytes({"started_at_utc": core.utc_now(),
        "protocol_sha256": verification["protocol_sha256"]}))
    for name, data in expected.items():
        _write_new(directory / name, data)
    for name, data in expected.items():
        if _bytes(directory / name) != data:
            raise ValueError("Report byte verification failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("prepare", "run", "all", "verify", "report"), default="all")
    args = parser.parse_args()
    if args.stage in {"verify", "report"}:
        result = verify_run(args.output_dir) if args.stage == "verify" else report(args.output_dir)
    else:
        protocol = prepare(args.output_dir) if args.stage in {"prepare", "all"} else validate_protocol(args.output_dir)
        if args.stage == "prepare":
            result = {"status": "PREPARED", "schema": SCHEMA, "inference_calls": 0,
                      "output_dir": str(args.output_dir)}
        else:
            client = core.LocalModelClient(seed=protocol["inference_seed"], model=protocol["model"],
                endpoint=f"http://{protocol['runtime']['host']}:{protocol['runtime']['port']}")
            client.probe()
            run(args.output_dir, protocol, client)
            result = report(args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
