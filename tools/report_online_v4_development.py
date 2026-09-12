"""Index explicitly supplied, trusted v4 development archives without pooling quality.

Usage: python tools/report_online_v4_development.py --runs RUN [RUN ...] --output-dir OUTPUT
Add --verify to reverify inputs and compare both existing report files exactly.

The archive helper executes the project's trusted source snapshots in offline
verify mode. Do not use downloaded/untrusted bundles. No model is called here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import verify_online_v4_archive as archive_verifier


SCHEMA = "rcwt-online-development-index/4"
JSON_NAME = "development.json"
MARKDOWN_NAME = "DEVELOPMENT.md"
POLICIES = ("summary", "structured")
CORE_INPUTS = ("protocol.json", "public.json", "oracle.json", "schedule.json", "freeze.json",
               "started.json", "traces.jsonl", "episodes.jsonl", "completion.json", "development-screen.json")
COST_FIELDS = ("model_calls", "prompt_tokens", "completion_tokens", "total_tokens",
               "inference_seconds", "episode_wall_seconds")
LIMITATIONS = [
    "Development-only index, not a confirmatory analysis, confidence interval, or generalization claim.",
    "Each attempt is reported separately. Reused training episodes and adaptive revisions must not be pooled as independent quality evidence.",
    "PASS/FAIL denotes the development screen, not held-out success. Failed attempts remain visible.",
    "Only expenditure is summed across attempts; no pooled accuracy, unsafe-attempt rate, or unsafe-booking total is reported.",
    "Episode wall time is the sum of recorded episode_seconds, including episode overhead; it is not elapsed calendar time across runs.",
    "Token/call/inference totals include all completion calls recorded by the archived runner, including compression and draft/review stages when present.",
    "API charges are zero under the local-only protocol; energy, hardware and total monetary cost are not measured.",
    "Verification replays trusted archived project code and recorded local telemetry, not independent attestation of physical inference or token counts.",
    "The index covers exactly the explicitly supplied run directories; it does not discover or silently omit additional attempts.",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError("Non-finite JSON number")


def _loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_object, parse_constant=_reject_constant)


def _read(path: Path) -> Any:
    return _loads(path.read_text(encoding="utf-8"))


def _relative(path: Path, output_dir: Path) -> str:
    try:
        relative = Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()
    except ValueError as exc:
        raise ValueError("Inputs, tools and output must support portable relative paths on one volume") from exc
    if Path(relative).is_absolute() or ":" in relative:
        raise ValueError("Absolute report paths are not allowed")
    return relative


def _file_record(path: Path, output_dir: Path) -> dict:
    return {"path": _relative(path, output_dir), "sha256": digest(path)}


def _manifest(directory: Path, protocol: dict, output_dir: Path) -> list[dict]:
    paths = [directory / name for name in CORE_INPUTS]
    sources = protocol.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Protocol lacks its archived source manifest")
    for reference, expected in sorted(sources.items()):
        source = Path(reference)
        if (source.as_posix() != reference or len(source.parts) != 2 or source.parts[0] != "src"
                or source.suffix != ".py" or not isinstance(expected, str)):
            raise ValueError("Unsafe source archive reference")
        path = directory / "sources" / source.name
        if digest(path) != expected:
            raise ValueError("Archived source does not match protocol")
        paths.append(path)
    return [_file_record(path, output_dir) for path in paths]


def _provenance(output_dir: Path) -> dict:
    baseline = archive_verifier.ROOT / "results" / "agent_v2"
    return {
        "report_source": _file_record(Path(__file__), output_dir),
        "archive_verifier_source": _file_record(Path(archive_verifier.__file__), output_dir),
        "archive_verifier_baseline_inputs": [_file_record(baseline / name, output_dir)
                                             for name in ("protocol.json", "RESULTS.md", "DIAGNOSIS.md")],
        "historical_v3_inputs": [_file_record(archive_verifier.ROOT / "results/agent_v3_development" / name, output_dir)
                                 for name in ("attempt_05/protocol.json", "development.json", "RESULTS.md")],
        "registered_series_plan": _file_record(archive_verifier.ROOT / "docs/rcwt_online_v4_protocol.md", output_dir),
        "registered_final_revision": _file_record(archive_verifier.ROOT / "docs/rcwt_online_v4_revision_02.md", output_dir),
    }


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid nonnegative integer: {label}")
    return value


def _seconds(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid duration: {label}")
    return float(value)


def _metrics(records: list[dict], protocol: dict) -> dict:
    expected_count = _integer(protocol.get("count"), "episode count", 1)
    expected_steps = _integer(protocol.get("steps_per_episode"), "episode steps", 1)
    groups = {policy: {} for policy in POLICIES}
    for record in records:
        if (not isinstance(record, dict) or record.get("split") != "train"
                or not isinstance(record.get("policy"), str) or record["policy"] not in groups
                or not isinstance(record.get("episode_id"), str)
                or not record["episode_id"] or not isinstance(record.get("family"), str)):
            raise ValueError("Episode outside the declared development cohort")
        group = groups[record["policy"]]
        if record["episode_id"] in group:
            raise ValueError("Duplicate episode/policy")
        group[record["episode_id"]] = record
        steps = _integer(record.get("steps"), "steps", 1)
        successes = _integer(record.get("successes"), "successes")
        unsafe = _integer(record.get("unsafe_actions"), "unsafe_actions")
        if steps != expected_steps or successes > steps or unsafe > steps:
            raise ValueError("Episode count outside valid bounds")
        for field in ("unsafe_booked_cents", "model_calls", "prompt_tokens", "completion_tokens"):
            _integer(record.get(field), field)
        inference = _seconds(record.get("inference_seconds"), "inference_seconds")
        wall = _seconds(record.get("episode_seconds"), "episode_seconds")
        if inference > wall + 1e-8:
            raise ValueError("Episode wall duration omits inference")
    if (any(len(group) != expected_count for group in groups.values())
            or groups["summary"].keys() != groups["structured"].keys()):
        raise ValueError("Missing paired development cohort")
    for identifier, baseline in groups["summary"].items():
        candidate = groups["structured"][identifier]
        if (baseline["family"], baseline["steps"]) != (candidate["family"], candidate["steps"]):
            raise ValueError("Paired family or step mismatch")
    result = {}
    for policy, group in groups.items():
        values = list(group.values())
        totals = {field: sum(row[field] for row in values) for field in (
            "steps", "successes", "unsafe_actions", "unsafe_booked_cents", "model_calls",
            "prompt_tokens", "completion_tokens")}
        totals.update(episodes=len(values), accuracy=totals["successes"] / totals["steps"],
                      total_tokens=totals["prompt_tokens"] + totals["completion_tokens"],
                      inference_seconds=sum(float(row["inference_seconds"]) for row in values),
                      episode_wall_seconds=sum(float(row["episode_seconds"]) for row in values))
        result[policy] = totals
    return result


def _screen(metrics: dict) -> dict:
    totals = {policy: {key: metrics[policy][key] for key in (
        "successes", "steps", "unsafe_actions", "unsafe_booked_cents")} for policy in POLICIES}
    baseline, candidate = totals["summary"], totals["structured"]
    passed = (candidate["successes"] > baseline["successes"]
              and candidate["unsafe_actions"] <= baseline["unsafe_actions"]
              and candidate["unsafe_booked_cents"] <= baseline["unsafe_booked_cents"])
    return {"passed": passed, "totals": totals,
            "scope": "development-only screening; not held-out evidence"}


def _sum_costs(rows: list[dict]) -> dict:
    return {field: sum(row[field] for row in rows) for field in COST_FIELDS}


def _attempt(directory: Path, output_dir: Path) -> dict:
    protocol = _read(directory / "protocol.json")
    # Reject before archive replay or reading episode/oracle/trace content.
    if (not isinstance(protocol, dict) or protocol.get("mode") != "development"
            or protocol.get("split") != "train" or protocol.get("policies") != list(POLICIES)
            or not isinstance(protocol.get("schema"), str)
            or protocol["schema"] != "rcwt-online-memory/4"):
        raise ValueError("Only declared TRAIN development archives are accepted")
    if (type(protocol.get("api_cost_usd")) not in (int, float) or protocol["api_cost_usd"] != 0
            or protocol.get("total_monetary_cost_usd", "missing") is not None):
        raise ValueError("Expected zero local API charges and unmeasured total monetary cost")
    before = _manifest(directory, protocol, output_dir)
    verification = archive_verifier.verify_archive(directory)
    if (not isinstance(verification, dict) or verification.get("status") != "PASS"
            or type(verification.get("inference_calls")) is not int or verification["inference_calls"] != 0
            or verification.get("protocol_sha256") != before[0]["sha256"]
            or not isinstance(verification.get("verification"), dict)
            or verification["verification"].get("status") != "PASS"):
        raise ValueError("Archive verifier did not return a matching offline PASS")
    lines = (directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Empty or blank episode records")
    metrics = _metrics([_loads(line) for line in lines], protocol)
    screen = _screen(metrics)
    if _read(directory / "development-screen.json") != screen:
        raise ValueError("Saved development screen differs from recomputed episode totals")
    if _manifest(directory, protocol, output_dir) != before:
        raise ValueError("Archive inputs changed during report construction")
    return {
        "attempt": directory.name,
        "run_path": _relative(directory, output_dir),
        "protocol_schema": protocol["schema"],
        "protocol_sha256": before[0]["sha256"],
        "dataset_seed": protocol.get("dataset_seed"),
        "inference_seed": protocol.get("inference_seed"),
        "model": protocol.get("model"),
        "memory_budget": protocol.get("memory_budget"),
        "intervention": protocol.get("memory_intervention", protocol.get("candidate_origin")),
        "baseline": protocol.get("baseline"),
        "archive_verification": verification,
        "screen_status": "PASS" if screen["passed"] else "FAIL",
        "calculated_screen": screen,
        "by_policy": metrics,
        "expenditure_total": _sum_costs(list(metrics.values())),
        "api_cost_usd": 0,
        "total_monetary_cost_usd": None,
        "input_manifest": before,
    }


def build_index(runs: list[Path], output_dir: Path) -> dict:
    if not runs or len(runs) > 2:
        raise ValueError("Supply one or two declared v4 development attempts")
    directories = [Path(directory).resolve(strict=True) for directory in runs]
    if any(not directory.is_dir() for directory in directories) or len(set(directories)) != len(directories):
        raise ValueError("Development run paths must be distinct directories")
    if len({directory.name for directory in directories}) != len(directories):
        raise ValueError("Attempt labels must be unique")
    output_dir = Path(output_dir).resolve()
    provenance = _provenance(output_dir)
    attempts = [_attempt(directory, output_dir) for directory in directories]
    if len({attempt["protocol_sha256"] for attempt in attempts}) != len(attempts):
        raise ValueError("Duplicate archive content cannot be counted as another attempt")
    if any(attempt["screen_status"] == "PASS" for attempt in attempts[:-1]):
        raise ValueError("The first passing development candidate must be selected; no later search")
    expenditure_by_policy = {policy: _sum_costs([attempt["by_policy"][policy] for attempt in attempts])
                             for policy in POLICIES}
    if _provenance(output_dir) != provenance:
        raise ValueError("Report or verifier dependencies changed during construction")
    return {
        "schema": SCHEMA,
        "scope": "adaptive development accounting only; no pooled quality or confirmatory inference",
        "attempt_count": len(attempts),
        "attempts": attempts,
        "development_expenditure": {
            "scope": "Costs only; repeated tasks are counted as actual recorded work, not independent evidence",
            "by_policy": expenditure_by_policy,
            "all_policies": _sum_costs(list(expenditure_by_policy.values())),
            "api_cost_usd": 0,
            "total_monetary_cost_usd": None,
        },
        "new_model_calls_for_this_report": 0,
        "provenance": provenance,
        "limitations": LIMITATIONS,
    }


def _cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_report(index: dict) -> str:
    lines = ["# Online v4: development attempts", "",
             "Adaptive TRAIN results only. Every supplied attempt is retained; screen PASS is not held-out confirmation.", "",
             "| Attempt / schema | Arm | Exact actions | Unsafe attempts | Unsafe booked cents (fictional) | Calls | Input + output tokens | Inference s | Episode wall s | Screen |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
             ]
    for attempt in index["attempts"]:
        link = quote(attempt["run_path"] + "/protocol.json", safe="/.")
        name = _cell(attempt["attempt"]).replace("[", "\\[").replace("]", "\\]")
        label = f"[{name}]({link}) / {_cell(attempt['protocol_schema'])}"
        for policy in POLICIES:
            row = attempt["by_policy"][policy]
            lines.append(f"| {label} | {policy} | {row['successes']}/{row['steps']} ({100 * row['accuracy']:.2f}%) "
                         f"| {row['unsafe_actions']} | {row['unsafe_booked_cents']} | {row['model_calls']} "
                         f"| {row['prompt_tokens']} + {row['completion_tokens']} "
                         f"| {row['inference_seconds']:.3f} | {row['episode_wall_seconds']:.3f} | {attempt['screen_status']} |")
        total = attempt["expenditure_total"]
        lines.append(f"| {label} | TOTAL cost only | — | — | — | {total['model_calls']} "
                     f"| {total['prompt_tokens']} + {total['completion_tokens']} "
                     f"| {total['inference_seconds']:.3f} | {total['episode_wall_seconds']:.3f} | {attempt['screen_status']} |")
    lines += ["", "Archive integrity: every attempt above returned offline replay PASS. Screen status is a separate development criterion.",
              "", "## Development expenditure (not pooled performance)", "",
              "| Arm | Calls | Input tokens | Output tokens | Total tokens | Inference s | Episode wall s |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    costs = index["development_expenditure"]
    for policy, row in [*costs["by_policy"].items(), ("TOTAL", costs["all_policies"])]:
        lines.append(f"| {policy} | {row['model_calls']} | {row['prompt_tokens']} | {row['completion_tokens']} "
                     f"| {row['total_tokens']} | {row['inference_seconds']:.3f} | {row['episode_wall_seconds']:.3f} |")
    lines += ["", "API charges: US$0. Total monetary cost: unmeasured, not zero.", "",
              f"Exact counters, calculated screens, verifier receipts and relative-path SHA-256 manifests: [{JSON_NAME}]({JSON_NAME}).",
              "", "## Limits", ""]
    lines.extend("- " + limit for limit in index["limitations"])
    return "\n".join(lines) + "\n"


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def generate(runs: list[Path], output_dir: Path) -> dict:
    output_dir = Path(output_dir).resolve()
    targets = [output_dir / JSON_NAME, output_dir / MARKDOWN_NAME]
    if any(path.exists() for path in targets):
        raise FileExistsError("Report files already exist; use --verify or a new output location")
    index = build_index(runs, output_dir)
    payloads = [_json_bytes(index), render_report(index).encode("utf-8")]
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        # Exclusive creation never overwrites user files, even after the precheck.
        for target, payload in zip(targets, payloads):
            with target.open("xb") as stream:
                created.append(target)
                stream.write(payload)
    except Exception:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    return {"status": "PASS", "attempts": len(index["attempts"]), "new_model_calls": 0,
            "files": [JSON_NAME, MARKDOWN_NAME]}


def verify_index(runs: list[Path], output_dir: Path) -> dict:
    output_dir = Path(output_dir).resolve(strict=True)
    expected = build_index(runs, output_dir)
    if (output_dir / JSON_NAME).read_bytes() != _json_bytes(expected):
        raise ValueError("Saved development JSON differs from reverified inputs or provenance")
    if (output_dir / MARKDOWN_NAME).read_bytes() != render_report(expected).encode("utf-8"):
        raise ValueError("Saved development Markdown differs from its exact rendering")
    return {"status": "PASS", "attempts": len(expected["attempts"]), "new_model_calls": 0,
            "scope": "Exact read-only development index verification; no writes"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    result = (verify_index if args.verify else generate)(args.runs, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

