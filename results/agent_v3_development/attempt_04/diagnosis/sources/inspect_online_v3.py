"""Post-hoc offline diagnosis of completed v3 summary/structured recordings.

No inference, feedback, corrected actions, or new success endpoint is produced.
The CLI requires completion, freezing and verify_run PASS before diagnostic
trace reads. --verify recomputes the saved facts and Markdown without writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import inspect_agent_traces as reference_helpers
from inspect_agent_traces import CHECK_FIELDS, action_from_claimed_check, reference_check
from rcwt_agent_actor import extract_action, extract_evidence_check
from rcwt_agent_env import Episode, OracleSnapshot, Simulator, Step, expected_action, parse_action
from rcwt_local_model import canonical_hash
from rcwt_online_v3 import POLICIES, verify_run


SCHEMA_VERSION = "rcwt-online-trace-diagnosis/3"
CATEGORIES = (
    "failed_matching_reference_check", "failed_divergent_check", "invalid_envelope",
    "correct_matching_reference_check", "correct_divergent_check",
)
PAIR_CATEGORIES = (
    "structured_correct_summary_incorrect", "summary_correct_structured_incorrect",
    "both_correct", "both_incorrect",
)
LIMITATIONS = [
    "Post-hoc descriptive diagnosis of verified recordings, not a new confirmatory endpoint, new inference, or independent attestation of physical inference.",
    "Checks are actor self-reports. A divergent field does not identify compression loss, extraction error, wrong linkage, stale-state resolution, or unsupported inference as the cause.",
    "Reference fields represent evidence available across the episode and actual pre-action bookings, not only stored memory or the query-conditioned view supplied to the actor. A fact never supplied is not a memory-compression failure.",
    "All six check fields are compared, including fields irrelevant to an operation or overridden by higher-priority rules. A check mismatch need not cause an action error.",
    "Failure despite a matching reference check describes inconsistent rule application in the recorded answer, not hidden reasoning or a neural mechanism.",
    "The action implied by the claimed check is computed only for offline comparison. It is never executed, substituted, fed back, or used to change the original grade.",
    "Paired task counts are not independent statistical samples. The two arms may have different prior actual bookings; their contrast is not a controlled causal attribution to memory.",
    "Examples follow a frozen descriptive selection rule: first occurrence in public-manifest order, then step, then summary/structured. Both disagreement directions are shown when present.",
    "The structured intervention combines engineered storage and query-conditioned retrieval, including deterministic selection, joins and stale-field invalidation. It is not compaction alone and not autonomous policy learning or recursive self-improvement. This diagnosis does not isolate which component caused an outcome or establish production safety or cross-domain transfer.",
    "When present, a draft is an unexecuted model output. Only the completion labelled action:policy is executed and graded; the same mandatory two-pass actor is used in both arms. Draft quality is not counted as final performance, and the diagnosis does not isolate the effect of self-review.",
]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _decode(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _read_json(path: Path) -> Any:
    return _decode(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Trace JSONL must have nonempty physical lines for unambiguous references")
    return [_decode(line) for line in lines]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episodes(public_records: list[dict], oracle_records: list[dict]) -> dict[str, Episode]:
    if not isinstance(public_records, list) or not public_records or not isinstance(oracle_records, list):
        raise ValueError("Nonempty public and oracle manifests are required")
    public = {record["episode_id"]: record for record in public_records}
    oracle = {record["episode_id"]: record for record in oracle_records}
    if len(public) != len(public_records) or len(oracle) != len(oracle_records) or public.keys() != oracle.keys():
        raise ValueError("Public/oracle episode identities must be unique and identical")
    splits = {record["split"] for record in public_records}
    if len(splits) != 1 or not splits <= {"train", "test"}:
        raise ValueError("Diagnose one complete v3 development or confirmation split, never pool them")
    episodes = {}
    for episode_id, record in public.items():
        steps = tuple(Step(row["step_index"], tuple(row["observations"]), row["task"]) for row in record["steps"])
        snapshots = tuple(OracleSnapshot(**row) for row in oracle[episode_id]["oracle_steps"])
        if (len(steps) != 8 or len(snapshots) != 8
                or any(type(step.step_index) is not int or step.step_index != index for index, step in enumerate(steps))):
            raise ValueError("Public and oracle manifests must contain eight ordered steps")
        for step, snapshot in zip(steps, snapshots):
            if (step.task["case_id"], step.task["operation"]) != (snapshot.case_id, snapshot.operation):
                raise ValueError("Private reference identity differs from the public task")
        episodes[episode_id] = Episode(episode_id, record["family"], record["split"], record["seed"], steps, snapshots)
    return episodes


def _summary(decisions: list[dict]) -> dict[str, Any]:
    categories = Counter(row["category"] for row in decisions)
    fields = Counter(delta["field"] for row in decisions for delta in row["field_deltas"])
    valid = sum(row["claimed_check"] is not None for row in decisions)
    consistent = sum(row["action_consistent_with_claimed_evidence"] is True for row in decisions)
    return {
        "decisions": len(decisions),
        "exact_action_successes": sum(row["success"] for row in decisions),
        "drafts_recorded_not_executed": sum(row["draft"] is not None for row in decisions),
        "category_counts": {category: categories[category] for category in CATEGORIES},
        "valid_envelopes": valid,
        "reference_check_matches": sum(row["check_matches_reference"] is True for row in decisions),
        "reference_check_divergences": sum(row["check_matches_reference"] is False for row in decisions),
        "invalid_checks": len(decisions) - valid,
        "failures_despite_matching_check": categories["failed_matching_reference_check"],
        "field_divergence_counts": {field: fields[field] for field in CHECK_FIELDS},
        "action_consistent_with_claimed_evidence": consistent,
        "action_inconsistent_with_claimed_evidence": sum(row["action_consistent_with_claimed_evidence"] is False for row in decisions),
        "action_consistency_not_evaluable": len(decisions) - valid,
        "action_consistency_rate_among_valid_envelopes": consistent / valid if valid else None,
    }


def _contrasts(decisions: list[dict], episode_order: list[str]) -> tuple[dict, list[dict]]:
    indexed = {(row["episode_id"], row["step_index"], row["policy"]): row for row in decisions}
    counts: Counter = Counter()
    first_pairs = {}
    for episode_id in episode_order:
        for step in range(8):
            baseline, candidate = (indexed[(episode_id, step, policy)] for policy in POLICIES)
            if candidate["success"] and not baseline["success"]:
                category = PAIR_CATEGORIES[0]
            elif baseline["success"] and not candidate["success"]:
                category = PAIR_CATEGORIES[1]
            else:
                category = "both_correct" if candidate["success"] else "both_incorrect"
            counts[category] += 1
            first_pairs.setdefault(category, [baseline["decision_id"], candidate["decision_id"]])
    examples = []
    for category in ("failed_matching_reference_check", "failed_divergent_check", "invalid_envelope"):
        first = next((row for row in decisions if row["category"] == category), None)
        if first:
            examples.append({"kind": category, "decision_ids": [first["decision_id"]]})
    for category in PAIR_CATEGORIES[:2]:
        if category in first_pairs:
            examples.append({"kind": "paired_" + category, "decision_ids": first_pairs[category]})
    paired = {category: counts[category] for category in PAIR_CATEGORIES}
    paired.update({"paired_tasks": len(episode_order) * 8,
                   "independent_samples": False,
                   "note": "Descriptive paired task counts; use paired episodes, not decisions, for the primary inference."})
    return paired, examples


def diagnose_records(traces: list[dict], public_records: list[dict], oracle_records: list[dict]) -> dict[str, Any]:
    """Pure post-hoc core for verified inputs or explicitly fake unit fixtures."""
    episodes = _episodes(public_records, oracle_records)
    groups: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    for line, row in enumerate(traces, 1):
        if (row["episode_id"] not in episodes or row["policy"] not in POLICIES
                or row["split"] != episodes[row["episode_id"]].split
                or row["family"] != episodes[row["episode_id"]].family):
            raise ValueError("Trace outside the complete two-arm diagnostic cohort")
        if row["sha256"] != canonical_hash({key: value for key, value in row.items() if key != "sha256"}):
            raise ValueError("Diagnostic trace hash differs from its recorded content")
        groups.setdefault((row["episode_id"], row["policy"]), []).append((line, row))
    if set(groups) != {(episode_id, policy) for episode_id in episodes for policy in POLICIES}:
        raise ValueError("Missing diagnostic episode/policy pair")
    decisions = []
    for episode_id, episode in episodes.items():
        for policy in POLICIES:
            rows = groups[(episode_id, policy)]
            if ([row["step_index"] for _, row in rows] != list(range(8))
                    or any(type(row["step_index"]) is not int for _, row in rows)):
                raise ValueError("Missing, duplicate or reordered diagnostic steps")
            simulator = Simulator(episode)
            for index, (line, row) in enumerate(rows):
                public = simulator.public_step(index)
                if canonical_hash(public) != canonical_hash(row["public_step"]):
                    raise ValueError("Diagnostic public input differs from the frozen manifest")
                events = row["client_events"]
                completions = [event["result"] for event in events if event["method"] == "complete"]
                final_calls = [(position, call) for position, call in enumerate(completions) if call.get("purpose") == f"action:{policy}"]
                draft_calls = [(position, call) for position, call in enumerate(completions) if call.get("purpose") == f"draft:{policy}"]
                if len(final_calls) != 1 or len(draft_calls) > 1:
                    raise ValueError("Exactly one final action completion and at most one draft are required")
                final_position, call = final_calls[0]
                draft = None
                if draft_calls:
                    draft_position, draft_call = draft_calls[0]
                    if draft_position >= final_position:
                        raise ValueError("Draft must precede the final action completion")
                    draft = {"text": draft_call["text"], "finish_reason": draft_call["finish_reason"],
                             "proposed_action": extract_action(draft_call["text"], draft_call["finish_reason"]),
                             "evidence_check": extract_evidence_check(draft_call["text"], draft_call["finish_reason"]),
                             "executed": False, "purpose": draft_call["purpose"]}
                    if (("draft_action" in row and row["draft_action"] != draft["proposed_action"])
                            or ("draft_evidence_check" in row and canonical_hash(row["draft_evidence_check"]) != canonical_hash(draft["evidence_check"]))):
                        raise ValueError("Draft metadata differs from the unexecuted raw draft")
                elif "draft_action" in row or "draft_evidence_check" in row:
                    raise ValueError("Recorded draft metadata requires a draft completion")
                stored_memory = row["memory_before"]
                actor_memory = row.get("actor_memory_before", stored_memory)
                if not isinstance(stored_memory, str) or not isinstance(actor_memory, str):
                    raise ValueError("Stored and actor-view memories must be recorded strings")
                actor_tokens = row.get("actor_memory_tokens")
                if actor_tokens is not None and (type(actor_tokens) is not int or actor_tokens < 0):
                    raise ValueError("Recorded actor-view token count must be a nonnegative integer")
                # Real call records include the exact request. The optional
                # branch also supports older deliberately minimal unit fixtures.
                bound_to_request = "request" in call
                if bound_to_request:
                    try:
                        payload = _decode(call["request"]["messages"][1]["content"])
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        raise ValueError("Recorded actor request payload is malformed") from exc
                    if canonical_hash(payload) != canonical_hash({"retained_memory": actor_memory, "current_step": public}):
                        raise ValueError("Actor memory view or current step differs from the recorded request")
                    if draft is not None and "request" in draft_call:
                        messages = call["request"]["messages"]
                        if (len(messages) != 4 or canonical_hash(messages[:2]) != canonical_hash(draft_call["request"]["messages"])
                                or messages[2] != {"role": "assistant", "content": draft["text"]}):
                            raise ValueError("Final review request is not bound to the original input and raw draft")
                check = extract_evidence_check(call["text"], call["finish_reason"])
                action = extract_action(call["text"], call["finish_reason"])
                if row["action"] != action or canonical_hash(row["evidence_check"]) != canonical_hash(check):
                    raise ValueError("Action or check is not bound to the recorded raw completion")
                completed = set(simulator.completed)
                snapshot = episode.oracle_steps[index]
                reference = reference_check(snapshot, completed)
                expected = expected_action(snapshot, completed).to_dict()
                execution = simulator.execute_action(index, action)
                if canonical_hash([execution.score.to_dict(), execution.state, execution.tool_result]) != canonical_hash([
                    row["score"], row["state_after"], row["tool_result"]
                ]):
                    raise ValueError("Diagnostic execution differs from the original score or ledger")
                if check is None:
                    actual, implied, matches, consistent, deltas = action, None, None, None, []
                    category = "invalid_envelope"
                else:
                    actual = parse_action(action).to_dict()
                    # This branch receives only claimed facts and public identity,
                    # not the real snapshot or completed ledger used above.
                    implied = action_from_claimed_check(check, public["task"]["case_id"], public["task"]["operation"])
                    matches, consistent = check == reference, actual == implied
                    deltas = [{"field": field, "claimed": check[field], "reference": reference[field]}
                              for field in CHECK_FIELDS if check[field] != reference[field]]
                    category = ("correct_" if execution.score.success else "failed_") + (
                        "matching_reference_check" if matches else "divergent_check")
                decisions.append({
                    "decision_id": f"{episode_id}/{policy}/{index}", "episode_id": episode_id,
                    "family": episode.family, "split": episode.split, "policy": policy,
                    "step_index": index, "success": execution.score.success,
                    "original_failure_category": execution.score.failure_category,
                    "category": category, "claimed_check": check, "reference_check": reference,
                    "draft": draft, "executed_completion_purpose": call["purpose"],
                    "check_matches_reference": matches, "field_deltas": deltas,
                    "actual_action": actual, "expected_action_from_reference": expected,
                    "expected_action_from_claimed_evidence": implied,
                    "action_consistent_with_claimed_evidence": consistent,
                    "completed_before": [list(key) for key in sorted(completed)],
                    "public_input": {"memory_before": stored_memory, "actor_memory_before": actor_memory,
                                     "actor_memory_tokens": actor_tokens,
                                     "actor_memory_source": "recorded_actor_view" if "actor_memory_before" in row else "legacy_stored_memory_fallback",
                                     "actor_input_bound_to_request": bound_to_request, "current_step": public},
                    "source": {"trace_file": "traces.jsonl", "line": line, "trace_sha256": row["sha256"],
                               "public_file": "public.json", "oracle_file": "oracle.json", "oracle_step_index": index},
                })
    positions = {episode_id: index for index, episode_id in enumerate(episodes)}
    decisions.sort(key=lambda row: (positions[row["episode_id"]], row["step_index"], POLICIES.index(row["policy"])))
    paired, examples = _contrasts(decisions, list(episodes))
    return {
        "schema_version": SCHEMA_VERSION, "split": next(iter(episodes.values())).split,
        "post_hoc": True, "feedback_to_agent": False, "causal_memory_attribution": False,
        "n_episodes": len(episodes), "decisions": decisions,
        "policy_summaries": {policy: _summary([row for row in decisions if row["policy"] == policy]) for policy in POLICIES},
        "paired_disagreements": paired, "examples": examples,
        "example_selection": "First occurrence in public-manifest order, then step, then summary/structured: matching-check failure, divergent-check failure, invalid envelope, and each direction of paired outcome disagreement when present.",
        "limitations": list(LIMITATIONS),
    }


def render_report(diagnosis: dict[str, Any], relative_run: str = ".") -> str:
    if diagnosis.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported v3 diagnosis schema")
    lines = [
        "# V3 post-hoc evidence-check diagnosis", "",
        "OFFLINE RECORDED DIAGNOSIS / NO MODEL CALLS / NO FEEDBACK", "",
        "This describes reference-check and action agreement after a completed run. It is not an extra success endpoint, a repaired trajectory, or evidence that memory caused an error.", "",
        "Only the final completion labelled `action:policy` is executed and graded. A `draft:policy` completion, when present, is shown solely as an unexecuted draft; there is no fallback to it.", "",
        f"Episodes: {diagnosis['n_episodes']}; split: `{diagnosis['split']}`; arms: summary and structured.", "",
        "| Policy | Decisions | Exact successes | Matching checks | Divergent checks | Invalid checks | Failures despite matching check |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in POLICIES:
        summary = diagnosis["policy_summaries"][policy]
        lines.append(f"| {policy} | {summary['decisions']} | {summary['exact_action_successes']} | {summary['reference_check_matches']} | {summary['reference_check_divergences']} | {summary['invalid_checks']} | {summary['failures_despite_matching_check']} |")
    lines += ["", "A matching check requires all six fields to match the reference, including actual prior bookings. Invalid envelopes are not treated as evidence of memory failure.",
              "", "## Field mismatches", "", "| Policy | Amount | Account | Ownership | Payment | Return | Already booked |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for policy in POLICIES:
        summary = diagnosis["policy_summaries"][policy]
        values = " | ".join(str(summary["field_divergence_counts"][field]) for field in CHECK_FIELDS)
        lines.append(f"| {policy} | {values} |")
    lines += ["", "## Action agreement with the actor's own check", ""]
    for policy in POLICIES:
        summary = diagnosis["policy_summaries"][policy]
        lines.append(f"- `{policy}`: {summary['action_consistent_with_claimed_evidence']} consistent and {summary['action_inconsistent_with_claimed_evidence']} inconsistent among {summary['valid_envelopes']} valid envelopes; {summary['action_consistency_not_evaluable']} not evaluable.")
    paired = diagnosis["paired_disagreements"]
    lines += ["", "## Paired task outcomes (descriptive)", "",
              f"Across {paired['paired_tasks']} task pairs: structured correct / summary incorrect = {paired[PAIR_CATEGORIES[0]]}; summary correct / structured incorrect = {paired[PAIR_CATEGORIES[1]]}; both correct = {paired['both_correct']}; both incorrect = {paired['both_incorrect']}.",
              "", "These counts are not independent samples. Prior bookings can differ by arm. The primary result remains the separate paired-episode analysis.",
              "", "## Deterministic examples", "", diagnosis["example_selection"], ""]
    indexed = {row["decision_id"]: row for row in diagnosis["decisions"]}
    for number, example in enumerate(diagnosis["examples"], 1):
        lines += [f"### Example {number}: {example['kind']}", ""]
        for decision_id in example["decision_ids"]:
            row = indexed[decision_id]
            source = row["source"]
            link = f"{relative_run}/{source['trace_file']}#L{source['line']}"
            deltas = "; ".join(f"{delta['field']}: claimed={json.dumps(delta['claimed'])}, reference={json.dumps(delta['reference'])}" for delta in row["field_deltas"]) or "none"
            lines += [f"[{decision_id}](<{link}>) — `{row['category']}`.", "",
                      f"- Trace line/hash: {source['line']} / `{source['trace_sha256']}`.",
                      f"- Original failure category: `{row['original_failure_category']}`; field mismatches: {deltas}.",
                      f"- Actual action: `{json.dumps(row['actual_action'], ensure_ascii=False, sort_keys=True)}`.",
                      f"- Reference action: `{json.dumps(row['expected_action_from_reference'], ensure_ascii=False, sort_keys=True)}`.",
                      f"- Action implied only by the claimed check: `{json.dumps(row['expected_action_from_claimed_evidence'], ensure_ascii=False, sort_keys=True)}`.",
                      f"- Executed completion purpose: `{row['executed_completion_purpose']}`.",
                      "", "Stored memory before this action (persistent writer state):", "", "```text",
                      row["public_input"]["memory_before"].replace("```", "[backticks escaped]") or "(empty)", "```", "",
                      "Actual memory view supplied to the actor:", "", "```text",
                      row["public_input"]["actor_memory_before"].replace("```", "[backticks escaped]") or "(empty)", "```", "",
                      f"Memory-view source: `{row['public_input']['actor_memory_source']}`; bound to recorded request: `{row['public_input']['actor_input_bound_to_request']}`; recorded actor-view tokens: `{row['public_input']['actor_memory_tokens']}`.", ""]
            if row["draft"] is not None:
                lines += ["Draft model output (NOT EXECUTED):", "", "```text",
                          row["draft"]["text"].replace("```", "[backticks escaped]"), "```", "",
                          f"Draft finish reason: `{row['draft']['finish_reason']}`; proposed action, not executed: `{json.dumps(row['draft']['proposed_action'], ensure_ascii=False)}`.", ""]
    if not diagnosis["examples"]:
        lines += ["No example meets the declared categories; none was fabricated.", ""]
    lines += ["## Provenance and limits", ""]
    if "verification" in diagnosis:
        lines += [f"Original-run offline verification: `{diagnosis['verification']['status']}`. This is recorded-artifact consistency verification, not independent inference attestation.", ""]
    if "provenance" in diagnosis:
        provenance = diagnosis["provenance"]
        lines += [f"Inspector SHA-256: `{provenance['inspector_sha256']}`. Input-manifest SHA-256: `{provenance['input_manifest_sha256']}`. Per-file, frozen-source and reused-helper hashes are in diagnosis.json.", ""]
    lines.extend(f"- {limitation}" for limitation in diagnosis["limitations"])
    return "\n".join(lines) + "\n"


def _input_files(run_dir: Path) -> dict[str, str]:
    names = ("protocol.json", "freeze.json", "started.json", "completion.json", "schedule.json",
             "public.json", "oracle.json", "traces.jsonl", "episodes.jsonl")
    return {name: _hash(run_dir / name) for name in sorted(names)}


def _diagnostic_sources() -> dict[str, str]:
    return {"tools/inspect_online_v3.py": _hash(Path(__file__)),
            "tools/inspect_agent_traces.py": _hash(Path(reference_helpers.__file__))}


def _verified_facts(run_dir: Path) -> dict[str, Any]:
    if not (run_dir / "freeze.json").is_file():
        raise ValueError("Frozen run required before diagnostic inspection")
    if not (run_dir / "completion.json").is_file():
        raise ValueError("Completed run required; no diagnosis during evaluation")
    verification = verify_run(run_dir)
    if not isinstance(verification, dict) or verification.get("status") != "PASS":
        raise ValueError("verify_run must PASS before diagnostic reads")
    inputs, diagnostic_sources = _input_files(run_dir), _diagnostic_sources()
    protocol = _read_json(run_dir / "protocol.json")
    diagnosis = diagnose_records(_read_jsonl(run_dir / "traces.jsonl"),
                                 _read_json(run_dir / "public.json"), _read_json(run_dir / "oracle.json"))
    if (diagnosis["split"] != protocol["split"] or diagnosis["n_episodes"] != protocol["count"]
            or protocol["policies"] != list(POLICIES)):
        raise ValueError("Diagnosis cohort differs from the verified protocol")
    frozen_sources = {name: _hash(ROOT / name) for name in protocol["source_sha256"]}
    if frozen_sources != protocol["source_sha256"]:
        raise ValueError("Frozen evaluator sources changed during inspection")
    if inputs != _input_files(run_dir) or diagnostic_sources != _diagnostic_sources():
        raise ValueError("Diagnostic input or inspector source changed during inspection")
    diagnosis["verification"] = verification
    diagnosis["provenance"] = {
        "inspector_path": "tools/inspect_online_v3.py",
        "inspector_sha256": diagnostic_sources["tools/inspect_online_v3.py"],
        "diagnostic_sources_sha256": diagnostic_sources,
        "files_sha256": inputs, "input_manifest_sha256": canonical_hash(inputs),
        "frozen_sources_sha256": frozen_sources,
    }
    return diagnosis


def _relative_run(run_dir: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(run_dir, output_dir)).as_posix()


def inspect_run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    diagnosis = _verified_facts(run_dir)
    destinations = (output_dir / "diagnosis.json", output_dir / "DIAGNOSIS.md")
    if any(path.exists() for path in destinations):
        raise FileExistsError("Diagnostic artifacts already exist; use a fresh output directory")
    report = render_report(diagnosis, _relative_run(run_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in zip(destinations, (json.dumps(diagnosis, indent=2, ensure_ascii=False, allow_nan=False) + "\n", report)):
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    return diagnosis


def verify_diagnosis(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Recompute every fact/hash and require exact Markdown bytes, without writes."""
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    expected = _verified_facts(run_dir)
    diagnosis_path, report_path = output_dir / "diagnosis.json", output_dir / "DIAGNOSIS.md"
    if not diagnosis_path.is_file() or not report_path.is_file():
        raise ValueError("Both saved diagnosis artifacts are required")
    saved = _read_json(diagnosis_path)
    if not isinstance(saved, dict) or canonical_hash(saved) != canonical_hash(expected):
        raise ValueError("Saved diagnosis facts or provenance differ from recomputed evidence")
    if report_path.read_bytes() != render_report(expected, _relative_run(run_dir, output_dir)).encode("utf-8"):
        raise ValueError("Saved Markdown differs from deterministic diagnostic rendering")
    return {"status": "PASS", "read_only": True, "feedback_to_agent": False,
            "scope": "recomputed post-hoc facts, original-run verification, input/source/inspector/helper hashes, and exact Markdown bytes",
            "n_episodes": expected["n_episodes"], "decisions": len(expected["decisions"]),
            "diagnosis_sha256": _hash(diagnosis_path), "report_sha256": _hash(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="Recompute and verify saved diagnosis without writing")
    args = parser.parse_args()
    if args.verify:
        result = verify_diagnosis(args.run_dir, args.output_dir)
    else:
        diagnosis = inspect_run(args.run_dir, args.output_dir)
        result = {"status": "POST_HOC_DIAGNOSIS_WRITTEN", "n_episodes": diagnosis["n_episodes"],
                  "decisions": len(diagnosis["decisions"]), "feedback_to_agent": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
