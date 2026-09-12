"""Offline, post-hoc evidence-check diagnosis; never a correction or agent input.

Usage: python tools/inspect_agent_traces.py --run-dir RUN --output-dir OUTPUT
Verify saved artifacts without writing: add --verify to the same command.

The CLI requires a held-out freeze and successful offline evidence verification
before reading its diagnostic inputs. It performs no inference. Its labels
describe recorded check/action agreement, not the causal origin of memory errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcwt_agent_actor import extract_action, extract_evidence_check
from rcwt_agent_env import Episode, OracleSnapshot, Simulator, Step, expected_action, parse_action
from rcwt_agent_run import verify_evidence


SCHEMA_VERSION = "rcwt-agent-trace-diagnosis/1"
CHECK_FIELDS = (
    "invoice_amount_cents", "account_status", "ownership_match",
    "payment_status", "return_status", "operation_already_booked",
)
CATEGORIES = (
    "failed_matching_reference_check", "failed_divergent_check", "invalid_envelope",
    "correct_matching_reference_check", "correct_divergent_check",
)
LIMITATIONS = [
    "This is post-hoc diagnosis of verified recorded artifacts, not a new confirmatory endpoint or an independent attestation of physical inference.",
    "A self-reported evidence check may be wrong. Divergence does not distinguish compression loss, extraction, linkage, stale-state resolution, or unsupported inference.",
    "Reference facts describe all public evidence seen so far, which may exceed the retained memory. A missing fact never supplied is not a compression failure.",
    "All six fields are compared, including facts that may be irrelevant to this operation or overridden by higher-priority rules. A divergent check need not cause the action error.",
    "A matching check with a failed action demonstrates inconsistency with the reference rule application, not a hidden-reasoning or neural-mechanism diagnosis.",
    "The action implied by the claimed check is computed for offline comparison only. It is never executed, fed back, substituted into the trace, or used to rewrite the original score.",
    "Paired policy decisions share a scheduled task but may have different prior bookings. A learned-correct/summary-incorrect example is not by itself a controlled causal memory experiment.",
    "Examples are selected deterministically after evaluation; they are illustrations, not unbiased frequency estimates. Both directions of paired disagreement are counted.",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    # Runner JSONL contains no blank records; keeping that invariant preserves
    # the physical, one-based line references used by this diagnostic artifact.
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("Blank JSONL line would make trace references ambiguous")
    return [json.loads(line) for line in lines]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def reference_check(snapshot: OracleSnapshot, completed_before: set[tuple[str, str]]) -> dict[str, Any]:
    """Render six reference fields using actual bookings *before* this action."""

    return {
        "invoice_amount_cents": snapshot.amount_cents,
        "account_status": snapshot.account_status or "unknown",
        "ownership_match": "unknown" if snapshot.ownership_match is None else "yes" if snapshot.ownership_match else "no",
        "payment_status": snapshot.payment_status or "unknown",
        "return_status": snapshot.return_status or "unknown",
        "operation_already_booked": "yes" if (snapshot.case_id, snapshot.operation) in completed_before else "no_record",
    }


def action_from_claimed_check(check: dict[str, Any], case_id: str, operation: str) -> dict[str, Any]:
    """Apply reference rules to the *claimed* facts only; never execute the result.

    The actual completed ledger and private reference snapshot are deliberately
    not accepted as arguments. Only the public case/operation identity is added.
    """

    ownership = {"unknown": None, "yes": True, "no": False}[check["ownership_match"]]
    snapshot = OracleSnapshot(
        case_id=case_id,
        operation=operation,
        amount_cents=check["invoice_amount_cents"],
        account_status=None if check["account_status"] == "unknown" else check["account_status"],
        ownership_match=ownership,
        payment_status=None if check["payment_status"] == "unknown" else check["payment_status"],
        return_status=None if check["return_status"] == "unknown" else check["return_status"],
    )
    claimed_completed = {(case_id, operation)} if check["operation_already_booked"] == "yes" else set()
    return expected_action(snapshot, claimed_completed).to_dict()


def _episodes(public_records: list[dict], oracle_records: list[dict]) -> dict[str, Episode]:
    public = {record["episode_id"]: record for record in public_records}
    oracle = {record["episode_id"]: record for record in oracle_records}
    if len(public) != len(public_records) or len(oracle) != len(oracle_records) or public.keys() != oracle.keys():
        raise ValueError("Public/oracle episode identities must be unique and identical")
    result = {}
    for episode_id, record in public.items():
        if record["split"] != "test":
            raise ValueError("Post-hoc held-out diagnosis cannot pool other splits")
        steps = tuple(Step(step["step_index"], tuple(step["observations"]), step["task"])
                      for step in record["steps"])
        snapshots = tuple(OracleSnapshot(**snapshot) for snapshot in oracle[episode_id]["oracle_steps"])
        if not steps or len(steps) != len(snapshots) or [step.step_index for step in steps] != list(range(len(steps))):
            raise ValueError("Public and reference step sequences differ")
        for step, snapshot in zip(steps, snapshots):
            if (step.task["case_id"], step.task["operation"]) != (snapshot.case_id, snapshot.operation):
                raise ValueError("Reference snapshot does not match public task identity")
        result[episode_id] = Episode(episode_id, record["family"], "test", record["seed"], steps, snapshots)
    return result


def _summary(decisions: list[dict]) -> dict[str, Any]:
    categories = Counter(decision["category"] for decision in decisions)
    deltas = Counter(delta["field"] for decision in decisions for delta in decision["field_deltas"])
    valid = sum(decision["claimed_check"] is not None for decision in decisions)
    consistent = sum(decision["action_consistent_with_claimed_evidence"] is True for decision in decisions)
    inconsistent = sum(decision["action_consistent_with_claimed_evidence"] is False for decision in decisions)
    return {
        "decisions": len(decisions),
        "exact_action_successes": sum(decision["success"] for decision in decisions),
        "category_counts": {category: categories[category] for category in CATEGORIES},
        "valid_envelopes": valid,
        "reference_check_matches": sum(decision["check_matches_reference"] is True for decision in decisions),
        "reference_check_divergences": sum(decision["check_matches_reference"] is False for decision in decisions),
        "field_divergence_counts": {field: deltas[field] for field in CHECK_FIELDS},
        "action_consistent_with_claimed_evidence": consistent,
        "action_inconsistent_with_claimed_evidence": inconsistent,
        "action_consistency_not_evaluable": len(decisions) - valid,
        "action_consistency_rate_among_valid_envelopes": consistent / valid if valid else None,
    }


def _examples(decisions: list[dict]) -> tuple[list[dict], dict]:
    examples = []
    for category in ("failed_matching_reference_check", "failed_divergent_check"):
        first = next((decision for decision in decisions if decision["category"] == category), None)
        if first:
            examples.append({"kind": category, "decision_ids": [first["decision_id"]]})
    pairs: dict[tuple[str, int], dict[str, dict]] = {}
    for decision in decisions:
        if decision["policy"] in {"summary", "learned"}:
            pairs.setdefault((decision["episode_id"], decision["step_index"]), {})[decision["policy"]] = decision
    counts = Counter()
    first_forward = first_reverse = None
    for key in sorted(pairs):
        pair = pairs[key]
        if set(pair) != {"summary", "learned"}:
            raise ValueError("Missing paired summary/learned diagnostic decision")
        summary, learned = pair["summary"], pair["learned"]
        if learned["success"] and not summary["success"]:
            counts["learned_correct_summary_incorrect"] += 1
            if first_forward is None:
                first_forward = [summary["decision_id"], learned["decision_id"]]
        elif summary["success"] and not learned["success"]:
            counts["summary_correct_learned_incorrect"] += 1
            if first_reverse is None:
                first_reverse = [summary["decision_id"], learned["decision_id"]]
        elif learned["success"]:
            counts["both_correct"] += 1
        else:
            counts["both_incorrect"] += 1
    if first_forward:
        examples.append({
            "kind": "paired_learned_correct_summary_incorrect",
            "decision_ids": first_forward,
            "opposite_direction_decision_ids": first_reverse or [],
        })
    elif first_reverse:
        examples.append({"kind": "paired_summary_correct_learned_incorrect", "decision_ids": first_reverse})
    paired = {key: counts[key] for key in (
        "learned_correct_summary_incorrect", "summary_correct_learned_incorrect", "both_correct", "both_incorrect"
    )}
    paired["paired_tasks"] = len(pairs)
    paired["note"] = "Descriptive decision counts, not independent samples or a replacement for episode-cluster analysis."
    return examples, paired


def diagnose_records(traces: list[dict], public_records: list[dict], oracle_records: list[dict]) -> dict[str, Any]:
    """Pure diagnostic core for already-verified records and isolated unit fixtures.

    This helper does not write artifacts or claim verification. Real-run file
    access and the verify-before-read gate belong to ``inspect_run`` below.
    """

    episodes = _episodes(public_records, oracle_records)
    groups: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    for line, row in enumerate(traces, 1):
        if row["split"] != "test" or row["episode_id"] not in episodes or row["policy"] not in {"tail", "summary", "learned"}:
            raise ValueError("Diagnostic trace outside the held-out three-policy cohort")
        groups.setdefault((row["episode_id"], row["policy"]), []).append((line, row))
    expected_pairs = {(episode_id, policy) for episode_id in episodes for policy in ("tail", "summary", "learned")}
    if set(groups) != expected_pairs:
        raise ValueError("Incomplete diagnostic episode/policy cohort")
    decisions = []
    for (episode_id, policy), rows in groups.items():
        episode = episodes[episode_id]
        if [row["step_index"] for _, row in rows] != list(range(len(episode.steps))):
            raise ValueError("Missing or reordered diagnostic steps")
        simulator = Simulator(episode)
        for index, (line, row) in enumerate(rows):
            public = simulator.public_step(index)
            if row["public_step"] != public:
                raise ValueError("Diagnostic current-step input disagrees with public corpus")
            call = row["model_calls"][0]
            check = extract_evidence_check(call["text"], call["finish_reason"])
            if row["action"] != extract_action(call["text"], call["finish_reason"]) or row["evidence_check"] != check:
                raise ValueError("Diagnostic extraction differs from recorded raw output")
            completed_before = set(simulator.completed)
            snapshot = episode.oracle_steps[index]
            reference = reference_check(snapshot, completed_before)
            # Only this branch reads private reference facts. The independent
            # claimed-check branch below does not receive them or actual state.
            from_reference = expected_action(snapshot, completed_before).to_dict()
            execution = simulator.execute_action(index, row["action"])
            if (execution.score.to_dict() != row["score"] or execution.state != row["state_after"]
                    or execution.tool_result != row["tool_result"]):
                raise ValueError("Diagnostic replay disagrees with the verified original outcome")
            if check is None:
                actual, implied, matches, consistent, deltas = row["action"], None, None, None, []
                category = "invalid_envelope"
            else:
                actual = parse_action(row["action"]).to_dict()
                implied = action_from_claimed_check(check, public["task"]["case_id"], public["task"]["operation"])
                matches, consistent = check == reference, actual == implied
                deltas = [{"field": field, "reference": reference[field], "claimed": check[field]}
                          for field in CHECK_FIELDS if check[field] != reference[field]]
                category = ("correct_" if execution.score.success else "failed_") + (
                    "matching_reference_check" if matches else "divergent_check"
                )
            decisions.append({
                "decision_id": f"{episode_id}/{policy}/{index}",
                "episode_id": episode_id, "family": episode.family, "policy": policy,
                "step_index": index, "success": execution.score.success,
                "original_failure_category": execution.score.failure_category,
                "category": category, "claimed_check": check, "reference_check": reference,
                "check_matches_reference": matches, "field_deltas": deltas,
                "actual_action": actual, "expected_action_from_reference": from_reference,
                "expected_action_from_claimed_evidence": implied,
                "action_consistent_with_claimed_evidence": consistent,
                "completed_before": [list(key) for key in sorted(completed_before)],
                "public_input": {"memory_before": row["memory_before"], "current_step": public},
                "source": {"trace_file": "test/traces.jsonl", "line": line,
                           "trace_sha256": row["sha256"], "public_file": "test-public.json",
                           "oracle_file": "test-oracle.json", "oracle_step_index": index},
            })
    decisions.sort(key=lambda decision: decision["source"]["line"])
    examples, paired = _examples(decisions)
    return {
        "schema_version": SCHEMA_VERSION, "split": "test", "post_hoc": True,
        "feedback_to_agent": False, "causal_memory_attribution": False,
        "n_episodes": len(episodes), "decisions": decisions,
        "policy_summaries": {policy: _summary([decision for decision in decisions if decision["policy"] == policy])
                             for policy in ("tail", "summary", "learned")},
        "paired_disagreements": paired, "examples": examples,
        "example_selection": "First recorded matching-check failure, first recorded divergent-check failure; then first episode-ID/step-sorted learned-correct contrast with opposite-direction contrast if available.",
        "limitations": list(LIMITATIONS),
    }


def _trace_link(decision: dict, relative_run: str) -> str:
    source = decision["source"]
    target = f"{relative_run}/{source['trace_file']}#L{source['line']}"
    return f"[{decision['episode_id']} / {decision['policy']} / step {decision['step_index']}](<{target}>)"


def render_report(diagnosis: dict[str, Any], relative_run: str = ".") -> str:
    """Render bounded examples while keeping full records in diagnosis.json."""

    lines = [
        "# Post-hoc evidence-check diagnosis", "",
        "This artifact describes recorded evidence/action agreement. It is not an additional success gate, a corrected agent run, or proof that memory caused an error.", "",
        f"Episodes: {diagnosis['n_episodes']}; split: test; feedback to agent: none.", "",
        "| Policy | Decisions | Exact successes | Failed with matching check | Failed with divergent check | Invalid envelope | Action consistent with own check |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, summary in diagnosis["policy_summaries"].items():
        counts = summary["category_counts"]
        lines.append(f"| {policy} | {summary['decisions']} | {summary['exact_action_successes']} | {counts['failed_matching_reference_check']} | {counts['failed_divergent_check']} | {counts['invalid_envelope']} | {summary['action_consistent_with_claimed_evidence']}/{summary['valid_envelopes']} valid |")
    lines += ["", "Matching reference check means all six fields match the reference snapshot and actual pre-action booking state. Divergent check is a descriptive mismatch, not a causal memory-failure label. Invalid envelopes are excluded from the own-check consistency denominator.", "", "## Field disagreements", "", "| Policy | Amount | Account | Ownership | Payment | Return | Already booked |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for policy, summary in diagnosis["policy_summaries"].items():
        values = " | ".join(str(summary["field_divergence_counts"][field]) for field in CHECK_FIELDS)
        lines.append(f"| {policy} | {values} |")
    paired = diagnosis["paired_disagreements"]
    lines += ["", "## Paired task outcomes", "", f"Across {paired['paired_tasks']} scheduled summary/learned task pairs: learned correct / summary incorrect = {paired['learned_correct_summary_incorrect']}; summary correct / learned incorrect = {paired['summary_correct_learned_incorrect']}; both correct = {paired['both_correct']}; both incorrect = {paired['both_incorrect']}.", "", "These are descriptive counts, not independent samples. Prior actual bookings may differ between arms. The principal result remains the separately computed paired episode analysis.", "", "## Deterministically selected examples", "", diagnosis["example_selection"], ""]
    indexed = {decision["decision_id"]: decision for decision in diagnosis["decisions"]}
    if not diagnosis["examples"]:
        lines += ["No decision matches the specified example categories; no example was fabricated.", ""]
    for example_number, example in enumerate(diagnosis["examples"], 1):
        lines += [f"### Example {example_number}: {example['kind']}", ""]
        selected_ids = example["decision_ids"] + example.get("opposite_direction_decision_ids", [])
        for decision_id in selected_ids:
            decision = indexed[decision_id]
            deltas = "; ".join(f"{delta['field']}: claimed={json.dumps(delta['claimed'])}, reference={json.dumps(delta['reference'])}" for delta in decision["field_deltas"]) or "none"
            lines += [
                f"{_trace_link(decision, relative_run)} — `{decision['category']}`.", "",
                f"- Field deltas: {deltas}.",
                f"- Recorded action: `{json.dumps(decision['actual_action'], ensure_ascii=False, sort_keys=True)}`.",
                f"- Reference action using actual prior bookings: `{json.dumps(decision['expected_action_from_reference'], ensure_ascii=False, sort_keys=True)}`.",
                f"- Action implied by the actor's check alone: `{json.dumps(decision['expected_action_from_claimed_evidence'], ensure_ascii=False, sort_keys=True)}`.",
                f"- Consistent with own check: `{decision['action_consistent_with_claimed_evidence']}`. Prior actual booked keys: `{json.dumps(decision['completed_before'])}`.",
                "", "Retained public memory before this action:", "", "```text",
                decision["public_input"]["memory_before"].replace("```", "[backticks escaped]") or "(empty)",
                "```", "",
                "Current public observations and the complete check are retained in diagnosis.json and the linked trace. No missing historical fact is automatically classified as a compression defect.", "",
            ]
    lines += ["## Provenance and limits", ""]
    if "verification" in diagnosis:
        lines += [f"Offline evidence verification: `{diagnosis['verification']['status']}`. This is consistency verification of recorded artifacts, not independent inference attestation.", ""]
    if "provenance" in diagnosis:
        provenance = diagnosis["provenance"]
        lines += [f"Inspector SHA-256: `{provenance['inspector_sha256']}`. Input-manifest SHA-256: `{provenance['input_manifest_sha256']}`. Per-file hashes and frozen-source hashes are in diagnosis.json.", ""]
    lines.extend(f"- {limitation}" for limitation in diagnosis["limitations"])
    lines += [""]
    return "\n".join(lines)


def _input_files(run_dir: Path) -> dict[str, str]:
    paths = ["protocol.json", "test-freeze.json", "selection.json", "candidate-policies.json", "training-failures.json"]
    for split in ("train", "validation", "test"):
        paths.extend([f"{split}-public.json", f"{split}-oracle.json"])
        paths.extend(f"{split}/{name}" for name in ("traces.jsonl", "episodes.jsonl", "completion.json", "schedule.json"))
    return {relative: _hash(run_dir / relative) for relative in sorted(paths)}


def _verified_facts(run_dir: Path) -> dict[str, Any]:
    """Recompute diagnostic facts and provenance only after the core gate passes."""

    if not (run_dir / "test-freeze.json").is_file():
        raise ValueError("Held-out freeze is absent; diagnostic inspection is not allowed")
    if not (run_dir / "test/completion.json").is_file():
        raise ValueError("Held-out evaluation is incomplete; wait for completion before diagnosis")
    verification = verify_evidence(run_dir)
    if verification.get("status") != "PASS":
        raise ValueError("Offline evidence verification must PASS before diagnostic reads")
    input_hashes = _input_files(run_dir)
    protocol = _read_json(run_dir / "protocol.json")
    diagnosis = diagnose_records(_read_jsonl(run_dir / "test/traces.jsonl"),
                                 _read_json(run_dir / "test-public.json"),
                                 _read_json(run_dir / "test-oracle.json"))
    if _input_files(run_dir) != input_hashes:
        raise ValueError("Diagnostic inputs changed during inspection")
    frozen_sources = {relative: _hash(ROOT / relative) for relative in protocol["source_sha256"]}
    if frozen_sources != protocol["source_sha256"]:
        raise ValueError("Frozen evaluator sources changed during inspection")
    diagnosis["verification"] = verification
    diagnosis["provenance"] = {
        "inspector_path": "tools/inspect_agent_traces.py", "inspector_sha256": _hash(Path(__file__)),
        "files_sha256": input_hashes, "input_manifest_sha256": _canonical_hash(input_hashes),
        "frozen_sources_sha256": frozen_sources,
    }
    return diagnosis


def _relative_run(run_dir: Path, output_dir: Path) -> str:
    # pathlib has no relpath on Python versions supported by the project.
    import os
    return Path(os.path.relpath(run_dir, output_dir)).as_posix()


def inspect_run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Verify first, inspect offline, and write a fresh diagnosis without overwrite."""

    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    diagnosis = _verified_facts(run_dir)
    diagnosis["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    destinations = [output_dir / "diagnosis.json", output_dir / "DIAGNOSIS.md"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Diagnostic artifacts already exist; choose a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(diagnosis, _relative_run(run_dir, output_dir))
    destinations[0].write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    destinations[1].write_text(report, encoding="utf-8", newline="\n")
    return diagnosis


def _saved_diagnosis(path: Path) -> dict[str, Any]:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON field in saved diagnosis")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError("Non-finite JSON number in saved diagnosis")

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                       parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Saved diagnosis must be a JSON object")
    return value


def verify_diagnosis(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Read-only regeneration check of every factual field and report byte.

    The original generation timestamp is validated as an observed UTC ISO value,
    not remeasured or independently attested. Everything else, including the
    inspector and input/source hashes, must equal freshly recomputed evidence.
    """

    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    expected = _verified_facts(run_dir)
    diagnosis_path, report_path = output_dir / "diagnosis.json", output_dir / "DIAGNOSIS.md"
    if not diagnosis_path.is_file() or not report_path.is_file():
        raise ValueError("Both saved diagnostic artifacts are required for verification")
    saved = _saved_diagnosis(diagnosis_path)
    observed_at = saved.get("generated_at_utc")
    if not isinstance(observed_at, str):
        raise ValueError("Diagnostic generation time must be an observed UTC ISO timestamp")
    try:
        timestamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Diagnostic generation time must be an observed UTC ISO timestamp") from exc
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("Diagnostic generation time must have a UTC offset")
    expected["generated_at_utc"] = observed_at
    # Canonical JSON comparison distinguishes booleans from integers, unlike
    # ordinary Python dict equality. Unknown or omitted fields also fail closed.
    if _canonical_hash(saved) != _canonical_hash(expected):
        raise ValueError("Saved diagnostic facts or provenance differ from recomputed evidence")
    rendered = render_report(saved, _relative_run(run_dir, output_dir)).encode("utf-8")
    if report_path.read_bytes() != rendered:
        raise ValueError("Saved DIAGNOSIS.md differs from the verified deterministic rendering")
    return {
        "status": "PASS", "scope": "recomputed post-hoc facts, source/input/inspector hashes, core verification, observed UTC timestamp format, exact Markdown rendering",
        "n_episodes": saved["n_episodes"], "decisions": len(saved["decisions"]),
        "diagnosis_sha256": _hash(diagnosis_path), "report_sha256": _hash(report_path),
        "read_only": True, "feedback_to_agent": False,
        "timestamp_limit": "The saved generation time is format-validated, not independently remeasured or attested.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="Recompute and validate saved diagnosis without writing")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_diagnosis(args.run_dir, args.output_dir), indent=2))
        return
    result = inspect_run(args.run_dir, args.output_dir)
    print(json.dumps({"status": "POST_HOC_DIAGNOSIS_WRITTEN", "episodes": result["n_episodes"],
                      "decisions": len(result["decisions"]), "feedback_to_agent": False}, indent=2))


if __name__ == "__main__":
    main()
