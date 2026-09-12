"""Verify the complete R1 report offline, without changing its evidence.

PASS is report integrity only. The accuracy and descriptive safety verdicts
are returned separately, including for honestly reported negative results.
This command neither starts a model nor generates a replacement cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SOURCE_NAMES = (
    "rcwt_agent_actor.py", "rcwt_agent_analysis.py", "rcwt_agent_env.py",
    "rcwt_agent_memory.py", "rcwt_agent_run.py", "rcwt_local_model.py",
    "rcwt_memory_v3.py", "rcwt_online_v3.py", "rcwt_analysis_v3.py",
    "rcwt_retrieval_v3.py", "rcwt_review_v3.py", "rcwt_context_v4.py",
    "rcwt_decision_v4.py", "rcwt_online_v4.py", "rcwt_analysis_v4.py",
)
ORCHESTRATOR_NAMES = (
    "src/rcwt_replication_v4.py", "tools/verify_online_v4_partial.py",
    "tools/verify_v4_replication_report.py",
)
DEVELOPMENT_PROTOCOL = ROOT / "results/agent_v4_development/attempt_02/protocol.json"
DEVELOPMENT_PROTOCOL_SHA256 = "750768b5615a15c6a0c0542bf7464045116fa8587dd3f7b61f894dba124de73b"
REGISTRATION = ROOT / "docs/rcwt_v4_replication_protocol.md"
REGISTRATION_SHA256 = "0afb3d5b4ce22f27c06a91a963914be409ec61d51ea57f192a9469a0f1282f12"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(directory: Path) -> dict[str, str]:
    """Bind every original artifact before and after verification, without writes."""
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Require an existing, ordinary R1 evidence directory")
    if hasattr(directory, "is_junction") and directory.is_junction():
        raise ValueError("R1 evidence directory must not be a junction")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("R1 evidence must not contain symbolic links or junctions")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = _hash(path)
    return files


def _runner_module():
    return importlib.import_module("rcwt_replication_v4")


def verify_run(directory: Path) -> dict:
    return _runner_module().verify_run(directory)


def validate_protocol(directory: Path) -> dict:
    return _runner_module().validate_protocol(directory)


def _analysis_module():
    return importlib.import_module("rcwt_analysis_v4")


def _fixed_protocol(protocol: dict) -> None:
    integers = {
        "count": 32, "steps_per_episode": 8, "dataset_seed": 2026091210,
        "inference_seed": 20260911, "schedule_seed": 2026091208,
        "analysis_seed": 2026091207, "bootstrap_samples": 10000,
        "memory_budget": 256, "max_action_tokens": 512, "actor_passes": 2,
        "development_revision": 2,
    }
    if (not isinstance(protocol, dict)
            or protocol.get("schema") != "rcwt-online-replication/1"
            or protocol.get("mode") != "replication"
            or protocol.get("replication_id") != "R1"
            or protocol.get("split") != "test"
            or protocol.get("counts") != {"test": 32}
            or protocol.get("policies") != ["summary", "structured"]
            or protocol.get("registration_sha256") != REGISTRATION_SHA256
            or any(type(protocol.get(key)) is not int or protocol[key] != value
                   for key, value in integers.items())):
        raise ValueError("Require the fixed R1 protocol: 32 test pairs, fresh seed and frozen endpoints")


def _source_pins(protocol: dict) -> dict:
    """Check this candidate against the independently pinned selected development."""
    expected = {"src/" + name for name in SOURCE_NAMES}
    pins = protocol.get("source_sha256")
    if not isinstance(pins, dict) or set(pins) != expected:
        raise ValueError("Require exactly the fifteen frozen v4 source pins")
    if _hash(DEVELOPMENT_PROTOCOL) != DEVELOPMENT_PROTOCOL_SHA256:
        raise ValueError("Selected attempt_02 development protocol changed")
    development = _read(DEVELOPMENT_PROTOCOL)
    if pins != development.get("source_sha256"):
        raise ValueError("R1 source pins differ from the immutable attempt_02 candidate")
    actual = {"src/" + name: _hash(ROOT / "src" / name) for name in SOURCE_NAMES}
    if pins != actual:
        raise ValueError("Current candidate source drift; refusing statistical recomputation")
    orchestrator = protocol.get("orchestrator_sha256")
    if not isinstance(orchestrator, dict) or set(orchestrator) != set(ORCHESTRATOR_NAMES):
        raise ValueError("Require exactly the registered orchestration source pins")
    actual_orchestrator = {name: _hash(ROOT / name) for name in ORCHESTRATOR_NAMES}
    if orchestrator != actual_orchestrator:
        raise ValueError("Current orchestration source drift")
    if _hash(REGISTRATION) != REGISTRATION_SHA256:
        raise ValueError("R1 written registration changed")
    return {"candidate": actual, "orchestration": actual_orchestrator,
            "development_protocol": DEVELOPMENT_PROTOCOL_SHA256,
            "registration": REGISTRATION_SHA256}


def verify_report(directory: Path) -> dict:
    """Replay all 512 decisions, recompute statistics, and compare exact report bytes."""
    directory = Path(directory)
    before = _snapshot(directory)
    protocol = _read(directory / "protocol.json")
    _fixed_protocol(protocol)
    pins_before = _source_pins(protocol)
    verifier_before = _hash(Path(__file__))
    verification = verify_run(directory)
    if (not isinstance(verification, dict) or verification.get("status") != "PASS"
            or verification.get("read_only") is not True
            or verification.get("integrity_only") is not True
            or verification.get("gain") != "NOT_EVALUATED"
            or any(type(verification.get(key)) is not int or verification[key] != value
                   for key, value in {"verified_steps": 512, "verified_episode_summaries": 64,
                                      "inference_calls": 0, "generation_calls": 1248}.items())):
        raise ValueError("Require a complete offline R1 replay: 512 decisions, 64 summaries, 1248 recorded calls")
    if verification.get("protocol_sha256") != before.get("protocol.json"):
        raise ValueError("Replay verification receipt does not bind the original protocol bytes")
    if validate_protocol(directory) != protocol:
        raise ValueError("Validated R1 protocol differs from the original report inputs")
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    analysis = _analysis_module().analyze_v4(rows, expected_count=32, bootstrap_samples=10000, seed=2026091207)
    expected = {
        "analysis.json": _json_bytes(analysis),
        "RESULTS.md": _runner_module().render_report(analysis, protocol, verification).encode("utf-8"),
        "verification.json": _json_bytes(verification),
    }
    for name, content in expected.items():
        if (directory / name).read_bytes() != content:
            raise ValueError(f"{name} bytes do not match the recomputed frozen R1 report")
    after = _snapshot(directory)
    if before != after or pins_before != _source_pins(protocol) or verifier_before != _hash(Path(__file__)):
        raise ValueError("R1 evidence, report or source files changed during verification")
    return {
        "status": "PASS", "integrity_only": True,
        "scope": "complete R1 recorded evidence and exact report integrity; gain is a separate verdict",
        "schema": protocol["schema"], "mode": protocol["mode"], "replication_id": "R1",
        "split": "test", "dataset_seed": 2026091210,
        "paired_episodes": 32, "verified_steps": 512, "recorded_generation_calls": 1248,
        "inference_calls": 0, "writes": 0,
        "writes_scope": "original R1 evidence and report files; no result is regenerated or overwritten",
        "files_sha256": after, "current_source_sha256": pins_before["candidate"],
        "orchestrator_sha256": pins_before["orchestration"],
        "report_verifier_sha256": verifier_before,
        "recomputed_input_sha256": analysis["input_sha256"],
        "accuracy_gate_passed": analysis["accuracy_gate"]["passed"],
        "descriptive_safety_guard_passed": analysis["descriptive_safety_guard"]["passed"],
        "improvement_gate_passed": analysis["improvement_gate"]["passed"],
        "limitations": [
            "Recorded consistency is not independent physical attestation of inference or proof that no unlisted runs occurred.",
            "Integrity PASS also applies to an honestly reported negative R1 result.",
            "R1 is not pooled with development or the interrupted 173/512-decision confirmation.",
            "This one-generator, one-model, one-inference-seed cohort does not establish production safety or autonomous learning.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    print(json.dumps(verify_report(args.run_dir), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
