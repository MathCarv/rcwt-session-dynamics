"""Export a small, fixed R1 illustration after full offline report verification.

This creates a derived display, not new experiment evidence. Selection is always
public manifest index 0 and step index 4. No model or endpoint is contacted.
An existing destination is never overwritten; --verify performs no writes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
from pathlib import Path
import stat
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True
from verify_v4_replication_report import (  # noqa: E402
    _hash, _json_bytes, _snapshot, _source_pins, verify_report,
)

POLICIES = ("summary", "structured")
EPISODE_INDEX = 0
STEP_INDEX = 4
MAX_ARTIFACT_BYTES = 256 * 1024


def _ordinary_path(path: Path) -> Path:
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and
                getattr(part.stat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError("Symbolic links and reparse points are not allowed")
    return path.resolve()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[tuple[int, dict]]:
    # Keep physical source line numbers, not the index after removing blank lines.
    with path.open(encoding="utf-8") as handle:
        return [(line_number, json.loads(line)) for line_number, line in enumerate(handle, 1) if line.strip()]


def _check_receipt(receipt: dict) -> None:
    expected = {"paired_episodes": 32, "verified_steps": 512,
                "recorded_generation_calls": 1248, "inference_calls": 0, "writes": 0}
    if (not isinstance(receipt, dict) or receipt.get("status") != "PASS"
            or receipt.get("integrity_only") is not True
            or receipt.get("replication_id") != "R1"
            or receipt.get("schema") != "rcwt-online-replication/1"
            or any(type(receipt.get(key)) is not int or receipt[key] != value
                   for key, value in expected.items())):
        raise ValueError("Require full offline report integrity: 32 R1 pairs, 512 decisions, 1248 recorded calls")


@dataclass(frozen=True)
class DemoBundle:
    """In-memory derived bytes and the original inputs to which they are bound."""

    artifacts: dict[str, bytes]
    run_dir: Path
    evidence_sha256: dict[str, str]
    protocol: dict
    source_pins: dict
    exporter_sha256: str

    def assert_unchanged(self) -> None:
        if (self.evidence_sha256 != _snapshot(self.run_dir)
                or self.source_pins != _source_pins(self.protocol)
                or self.exporter_sha256 != _hash(Path(__file__))):
            raise ValueError("Original R1 evidence or source files changed during demo export/verification")


def _cell(value) -> str:
    return html.escape(str(value), quote=False).replace("|", "&#124;").replace("`", "&#96;").replace("\r", " ").replace("\n", " ")


def _action_label(action) -> str:
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except json.JSONDecodeError:
            return "unparsed action (see original trace)"
    if not isinstance(action, dict) or not isinstance(action.get("arguments"), dict):
        return "invalid action (see original trace)"
    args = action["arguments"]
    return _cell(f"{args.get('decision')} / {args.get('amount_cents')} cents / {args.get('reason_code')}")


def _render_markdown(document: dict) -> str:
    episode = document["selection"]["episode_id"]
    source = "../" + quote(document["provenance"]["source_directory_name"], safe="")
    pair = document["selected_pair_totals"]
    cohort = document["whole_cohort"]
    summary, structured = (cohort["policies"][policy] for policy in POLICIES)
    comparison = cohort["comparison"]
    low, high = comparison["ci95_percentage_points"]
    lines = [
        "# R1: one fixed recorded episode", "",
        "Derived illustration, not a new run. The exporter first verifies all 512 recorded decisions and the exact R1 report bytes offline.", "",
        f"Selection: **manifest index 0**, `{_cell(episode)}` ({_cell(document['selection']['family'])}); focused **step 5 (index 4)**. Neither choice searches scores.", "",
        "All accounts, observations, money and tool effects are fictional. Private expected actions and grades shown below were not actor inputs.", "",
        "## All eight decisions, both policies", "",
        "Each cell is `decision / amount in cents / reason`; the mark is the recorded exact-reference grade.", "",
        "| Step | Request / case | Summary | Structured |",
        "| --- | --- | --- | --- |",
    ]
    for step in document["decisions"]:
        cells = []
        for policy in POLICIES:
            arm = step["policies"][policy]
            mark = "correct" if arm["score"]["success"] else "incorrect: " + arm["score"]["failure_category"]
            cells.append(_action_label(arm["action"]) + " — " + _cell(mark))
        lines.append(f"| {step['step_index'] + 1} | {_cell(step['task']['operation'])} / {_cell(step['task']['case_id'])} | {cells[0]} | {cells[1]} |")
    lines += [
        "", "## Selected-pair totals", "",
        f"Summary: **{pair['summary']['successes']}/8 correct**; structured: **{pair['structured']['successes']}/8 correct**.", "",
        "Expected actions follow each arm's own prior simulated ledger. Later reference actions can therefore differ between policies; this is not grading against one common outcome.", "",
        "## Fixed step 5", "",
        "[step5.json](step5.json) contains the exact public observation/request, retained memory, actor memory/context, recorded final action, tool receipt, and private score for both arms. Text fields are copied without rewriting. It omits raw generation plans and request envelopes; those remain in the full trace.", "",
    ]
    for policy in POLICIES:
        arm = document["focused_step"]["policies"][policy]
        lines += [
            f"- **{policy}**: {_action_label(arm['action'])}; expected {_action_label(arm['score']['expected_action'])}. "
            f"Source: `traces.jsonl`, line {arm['trace_line']}; trace-chain SHA-256 `{arm['trace_sha256']}`.",
        ]
    lines += [
        "", f"The large [original trace]({source}/traces.jsonl) is the authority; if the hosting UI cannot render it, download it and use the recorded physical line numbers. The trace-chain hash is not the hash of the rendered JSON excerpt.", "",
        "## Whole-cohort evidence, not an inference from this pair", "",
        f"Across all 32 paired episodes: **{summary['successes']}/256 → {structured['successes']}/256 correct**, "
        f"delta **{comparison['delta_percentage_points']:+.2f} percentage points**, paired-episode bootstrap 95% CI **[{low:.2f}, {high:.2f}]**.", "",
        f"Observed unsafe attempts: **{summary['unsafe_actions']} → {structured['unsafe_actions']}**. Aggregate guards are descriptive, not a production-safety certificate; family-level regressions remain visible in the complete report.", "",
        f"Recorded generation calls: **{cohort['evaluation_cost']['model_calls']:,}**. Total model tokens: "
        f"**{summary['total_tokens']:,} → {structured['total_tokens']:,}**; median full-step time: "
        f"**{summary['step_latency_seconds']['p50']:.2f} → {structured['step_latency_seconds']['p50']:.2f} seconds**. "
        "Local timings depend on hardware/load; API charge is US$0, while total monetary cost is unknown.", "",
        f"See the [complete report]({source}/RESULTS.md), [machine-readable analysis]({source}/analysis.json), "
        f"[full pair summaries]({source}/episodes.jsonl), and [protocol]({source}/protocol.json). "
        "Earlier development and the interrupted confirmation are not pooled into R1.", "",
        "This same-generator, one-model, one-inference-seed comparison evaluates engineered memory plus public-context organization under a shared actor. It does not establish intrinsic model improvement, recursive self-improvement, unseen-domain transfer, or production readiness.", "",
        "## Reproduce this display without inference", "", "```sh",
        f"python tools/export_r1_demo.py --run-dir results/{document['provenance']['source_directory_name']} --output-dir results/agent_v4_replication_demo --verify",
        "```", "",
        "Omit `--verify` only to create a previously absent sibling directory. Existing artifacts are never overwritten. Verification rebuilds expected display bytes in memory and checks both files exactly; it does not rewrite the archive or the display.", "",
        f"Input protocol SHA-256: `{document['provenance']['files_sha256']['protocol.json']}`. "
        f"Full trace-file SHA-256: `{document['provenance']['files_sha256']['traces.jsonl']}`. "
        "The JSON includes the complete archive hash snapshot and exporter hash for provenance.", "",
    ]
    return "\n".join(lines)


def build_demo(run_dir: Path) -> DemoBundle:
    """Verify the full report, then derive fixed display bytes in memory only."""
    run_dir = _ordinary_path(run_dir)
    exporter_sha256 = _hash(Path(__file__))
    verified = verify_report(run_dir)  # Must precede any display-data read.
    _check_receipt(verified)
    before = _snapshot(run_dir)
    if before != verified.get("files_sha256"):
        raise ValueError("Display input snapshot differs from the full report verification receipt")
    protocol = _read(run_dir / "protocol.json")
    pins = _source_pins(protocol)
    if (pins["candidate"] != verified.get("current_source_sha256")
            or pins["orchestration"] != verified.get("orchestrator_sha256")
            or _hash(ROOT / "tools/verify_v4_replication_report.py") != verified.get("report_verifier_sha256")):
        raise ValueError("Display source pins differ from the full report verification receipt")
    public = _read(run_dir / "public.json")
    rows = _read_jsonl(run_dir / "traces.jsonl")
    summaries = _read_jsonl(run_dir / "episodes.jsonl")
    analysis = _read(run_dir / "analysis.json")
    if (len(public) != 32 or len(rows) != 512 or len(summaries) != 64
            or analysis["input_sha256"] != verified["recomputed_input_sha256"]):
        raise ValueError("Require the complete verified R1 display inputs")
    selected = public[EPISODE_INDEX]
    episode_id = selected["episode_id"]
    if len(selected["steps"]) != 8:
        raise ValueError("The fixed first episode requires eight steps")
    pairs = {}
    for line_number, row in rows:
        if row["episode_id"] != episode_id:
            continue
        key = (row["policy"], row["step_index"])
        if (key in pairs or row["policy"] not in POLICIES or type(row["step_index"]) is not int
                or not 0 <= row["step_index"] < 8
                or row["public_step"] != selected["steps"][row["step_index"]]
                or type(row["score"]["success"]) is not bool):
            raise ValueError("Duplicate or mismatched fixed first-episode trajectory")
        pairs[key] = {"trace_line": line_number, "trace_sha256": row["sha256"],
                      **{name: row[name] for name in ("memory_before", "actor_memory_before", "action",
                                                     "evidence_check", "tool_result", "score")}}
    if set(pairs) != {(policy, step) for policy in POLICIES for step in range(8)}:
        raise ValueError("Both complete fixed first-episode trajectories are required")
    pair_totals = {}
    for policy in POLICIES:
        found = [row for _, row in summaries if row["episode_id"] == episode_id and row["policy"] == policy]
        successes = sum(pairs[(policy, step)]["score"]["success"] for step in range(8))
        if len(found) != 1 or found[0]["steps"] != 8 or found[0]["successes"] != successes:
            raise ValueError("Selected pair summary does not match its complete trajectory")
        pair_totals[policy] = {"steps": 8, "successes": successes}
    document = {
        "schema": "rcwt-r1-fixed-demo/1",
        "scope": "Derived display from fully verified recorded R1 evidence; not a new experiment or standalone gain verdict.",
        "selection": {"manifest_episode_index": EPISODE_INDEX, "episode_id": episode_id,
                      "family": selected["family"], "step_index": STEP_INDEX, "score_search": False},
        # Sort POSIX-path strings, not platform-dependent Path objects.
        "provenance": {"source_directory_name": run_dir.name, "files_sha256": dict(sorted(before.items())),
                       "exporter_sha256": exporter_sha256,
                       "full_report_integrity": "PASS", "verified_steps": 512,
                       "recorded_generation_calls": 1248, "export_inference_calls": 0,
                       "recomputed_analysis_input_sha256": verified["recomputed_input_sha256"]},
        "selected_pair_totals": pair_totals,
        "decisions": [{"step_index": step, "task": selected["steps"][step]["task"],
                       "policies": {policy: {name: pairs[(policy, step)][name] for name in
                                             ("action", "score", "trace_line", "trace_sha256")}
                                    for policy in POLICIES}} for step in range(8)],
        "focused_step": {"public_step": selected["steps"][STEP_INDEX],
                         "policies": {policy: pairs[(policy, STEP_INDEX)] for policy in POLICIES}},
        "whole_cohort": {
            "paired_episodes": 32,
            "policies": {policy: {name: analysis["policies"][policy][name] for name in
                                   ("steps", "successes", "unsafe_actions", "total_tokens", "step_latency_seconds")}
                         for policy in POLICIES},
            "comparison": {name: value for name, value in analysis["comparisons"]["structured_vs_summary"].items()
                           if name != "by_family"},
            "evaluation_cost": analysis["evaluation_cost"],
            "accuracy_gate": analysis["accuracy_gate"],
            "descriptive_safety_guard": analysis["descriptive_safety_guard"],
            "improvement_gate": analysis["improvement_gate"],
        },
    }
    artifacts = {"README.md": _render_markdown(document).encode("utf-8"), "step5.json": _json_bytes(document)}
    if any(len(value) > MAX_ARTIFACT_BYTES for value in artifacts.values()):
        raise ValueError("Fixed demo exceeds the 256 KiB per-file display limit")
    bundle = DemoBundle(artifacts, run_dir, before, protocol, pins, exporter_sha256)
    bundle.assert_unchanged()
    return bundle


def export_demo(run_dir: Path, output_dir: Path, *, verify: bool = False) -> dict:
    """Exclusively create, or read-only byte-verify, the two-file sibling display."""
    run_dir, output_dir = _ordinary_path(run_dir), _ordinary_path(output_dir)
    if output_dir == run_dir or output_dir.parent != run_dir.parent:
        raise ValueError("Demo destination must be a distinct sibling of the original R1 archive")
    if not verify and output_dir.exists():
        raise FileExistsError("Demo destination already exists; use --verify, never overwrite")
    bundle = build_demo(run_dir)
    bundle.assert_unchanged()
    if verify:
        before = _snapshot(output_dir)
        if set(before) != set(bundle.artifacts) or any(path.is_dir() for path in output_dir.iterdir()):
            raise ValueError("Demo must contain exactly README.md and step5.json, without extra entries")
    else:
        output_dir.mkdir(exist_ok=False)
        for name, content in bundle.artifacts.items():
            _ordinary_path(output_dir)
            with (output_dir / name).open("xb") as handle:
                handle.write(content)
    for name, content in bundle.artifacts.items():
        if (output_dir / name).read_bytes() != content:
            raise ValueError(f"{name} differs from the exact verified fixed-demo bytes")
    after = _snapshot(output_dir)
    if set(after) != set(bundle.artifacts) or (verify and before != after):
        raise ValueError("Demo artifacts changed during creation or verification")
    bundle.assert_unchanged()
    return {"status": "PASS", "mode": "verified" if verify else "created",
            "scope": "derived display byte integrity, not a separate gain verdict",
            "manifest_episode_index": EPISODE_INDEX, "step_index": STEP_INDEX,
            "verified_steps": 512, "inference_calls": 0, "original_evidence_writes": 0,
            "display_files_written": 0 if verify else len(bundle.artifacts),
            "files_sha256": after, "bytes": {name: len(value) for name, value in bundle.artifacts.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/agent_v4_replication")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/agent_v4_replication_demo")
    parser.add_argument("--verify", action="store_true", help="Compare exact expected bytes without creating or modifying files")
    args = parser.parse_args()
    print(json.dumps(export_demo(args.run_dir, args.output_dir, verify=args.verify), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
