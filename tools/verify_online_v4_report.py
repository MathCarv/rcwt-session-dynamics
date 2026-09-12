"""Verify a v4 confirmation report against replayed evidence; no writes or inference.

PASS means artifact/report consistency, not that the experiment demonstrated a
gain. A correctly reported negative result also passes this verifier. The
archive gate establishes transcript integrity; this tool additionally binds the
published statistics and Markdown to the recorded episode summaries.
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
sys.path.insert(0, str(ROOT / "tools"))
ARCHIVE_VERIFIER_PATH = ROOT / "tools" / "verify_online_v4_archive.py"
SOURCE_NAMES = (
    "rcwt_agent_actor.py", "rcwt_agent_analysis.py", "rcwt_agent_env.py",
    "rcwt_agent_memory.py", "rcwt_agent_run.py", "rcwt_local_model.py",
    "rcwt_memory_v3.py", "rcwt_online_v3.py", "rcwt_analysis_v3.py", "rcwt_retrieval_v3.py",
    "rcwt_review_v3.py", "rcwt_context_v4.py", "rcwt_decision_v4.py", "rcwt_online_v4.py",
    "rcwt_analysis_v4.py",
)


def verify_archive(directory: Path) -> dict:
    from verify_online_v4_archive import verify_archive as verify
    return verify(directory)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value) -> bytes:
    """Exactly the frozen runner's dump serialization, including final LF."""
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _source_pins(protocol: dict) -> dict[str, str]:
    expected = {"src/" + name for name in SOURCE_NAMES}
    pins = protocol.get("source_sha256")
    if not isinstance(pins, dict) or set(pins) != expected:
        raise ValueError("Require exactly the fifteen frozen v4 source pins")
    actual = {"src/" + name: _hash(ROOT / "src" / name) for name in SOURCE_NAMES}
    if pins != actual:
        raise ValueError("Current source drift: refusing to recompute a report with different code")
    return actual


def _analysis_module():
    # Called only after all current source files match the archived protocol.
    # The CLI runs in a fresh process; the trusted project sources are unchanged.
    return importlib.import_module("rcwt_analysis_v4")


def verify_report(directory: Path) -> dict:
    """Recompute and compare confirmation outputs without overwriting anything."""
    archive = verify_archive(directory)
    if (not isinstance(archive, dict) or archive.get("status") != "PASS"
            or archive.get("inference_calls") != 0):
        raise ValueError("Require a complete offline-verified v4 archive")
    verification = archive.get("verification")
    if (not isinstance(verification, dict) or verification.get("status") != "PASS"
            or type(verification.get("verified_steps")) is not int
            or verification["verified_steps"] != 512):
        raise ValueError("Require the original verification receipt for all 512 confirmation decisions")

    if archive.get("protocol_sha256") != _hash(directory / "protocol.json"):
        raise ValueError("Archive verification receipt does not bind the current protocol bytes")
    protocol = _read(directory / "protocol.json")
    if (not isinstance(protocol, dict) or protocol.get("schema") != "rcwt-online-memory/4"
            or protocol.get("mode") != "confirmatory" or protocol.get("split") != "test"
            or type(protocol.get("count")) is not int or protocol["count"] != 32
            or protocol.get("counts") != {"test": 32}
            or protocol.get("policies") != ["summary", "structured"]
            or protocol.get("steps_per_episode") != 8
            or type(protocol.get("bootstrap_samples")) is not int or protocol["bootstrap_samples"] != 10000
            or type(protocol.get("analysis_seed")) is not int or protocol["analysis_seed"] != 2026091207):
        raise ValueError("Require the fixed v4 confirmation protocol: 32 test pairs and declared bootstrap")
    pins = _source_pins(protocol)
    names = ("protocol.json", "completion.json", "episodes.jsonl", "analysis.json", "RESULTS.md", "verification.json")
    before = {name: _hash(directory / name) for name in names}
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    module = _analysis_module()
    analysis = module.analyze_v4(rows, expected_count=32, bootstrap_samples=10000, seed=2026091207)
    expected = {
        "analysis.json": _json_bytes(analysis),
        "RESULTS.md": module.render_report_v4(analysis, protocol, verification).encode("utf-8"),
        "verification.json": _json_bytes(verification),
    }
    for name, content in expected.items():
        if (directory / name).read_bytes() != content:
            raise ValueError(f"{name} bytes do not match the recomputed frozen report")
    after = {name: _hash(directory / name) for name in names}
    if before != after or pins != _source_pins(protocol):
        raise ValueError("Report inputs or source files changed during verification")
    return {
        "status": "PASS", "scope": "v4 recorded confirmation report integrity; not an improvement verdict",
        "schema": protocol["schema"], "mode": "confirmatory", "split": "test",
        "paired_episodes": 32, "verified_steps": 512, "inference_calls": 0, "writes": 0,
        "writes_scope": "original evidence and report files; the archive gate uses a temporary verification copy",
        "files_sha256": after, "current_source_sha256": pins,
        "report_verifier_sha256": _hash(Path(__file__)),
        "archive_verifier_sha256": _hash(ARCHIVE_VERIFIER_PATH),
        "recomputed_input_sha256": analysis["input_sha256"],
        "accuracy_gate_passed": analysis["accuracy_gate"]["passed"],
        "descriptive_safety_guard_passed": analysis["descriptive_safety_guard"]["passed"],
        "improvement_gate_passed": analysis["improvement_gate"]["passed"],
        "limitations": [
            "Recorded-artifact consistency is not independent attestation of physical model inference.",
            "This verifier does not prove that no other run or preview occurred outside the supplied archive.",
            "A passing report-integrity check also applies to honestly reported negative results.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    # Receipt goes to stdout only. Callers may explicitly choose to save it.
    print(json.dumps(verify_report(args.run_dir), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
