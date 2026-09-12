"""Post-hoc offline diagnosis of completed v4 summary/structured recordings.

No inference, feedback, corrected actions, or new success endpoint is produced.
The CLI requires completion, freezing and verify_archive PASS before diagnostic
trace reads. --verify recomputes the saved facts and Markdown without writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from rcwt_local_model import CallResult, canonical_hash
from rcwt_online_v4 import POLICIES, SCHEMA, SOURCE_NAMES, ACTION_MAX_TOKENS, actor_messages, validate_call
from rcwt_decision_v4 import task_schema
from rcwt_review_v3 import planning_messages, review_messages
import verify_online_v4_archive as archive_helpers
from verify_online_v4_archive import verify_archive


SCHEMA_VERSION = "rcwt-online-trace-diagnosis/4"
DIAGNOSTIC_DEPENDENCIES = (
    "rcwt_agent_actor.py", "rcwt_agent_env.py", "rcwt_agent_run.py", "rcwt_agent_analysis.py",
    "rcwt_agent_memory.py", "rcwt_local_model.py", "rcwt_review_v3.py", "rcwt_decision_v4.py",
    "rcwt_online_v4.py",
)
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
    "The structured intervention combines retained storage and a transient context that overlays current public records before literal joins. It may change the first-step input. It is not compaction alone and not autonomous policy learning or recursive self-improvement. This diagnosis does not isolate which component caused an outcome or establish production safety or cross-domain transfer.",
    "Every first pass is an unexecuted schema-free text plan and is never parsed as an action or evidence check, even when it resembles JSON. Only action:policy is executed and graded. The two arms share planning/review instructions and a schema bound solely to public case identity and operation, never factual eligibility. Plan quality is not counted as final performance.",
    "Unsafe attempts and actual unsafe fictional amounts booked are separate measures: an idempotency rejection may block a repeated unsafe attempt. No real payment occurs.",
    "Costs include every recorded plan, final answer and summary-compaction completion. Local API-provider charges are zero; electricity, hardware depreciation, setup, downloads and other unmetered costs are unknown, not zero. Token counts and elapsed times are recorded claims, not independent hardware attestation.",
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
        raise ValueError("Diagnose one complete v4 development or confirmation split, never pool them")
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
    matching_fields = Counter(field for row in decisions for field in row["matched_fields"])
    valid = sum(row["claimed_check"] is not None for row in decisions)
    consistent = sum(row["action_consistent_with_claimed_evidence"] is True for row in decisions)
    return {
        "decisions": len(decisions),
        "exact_action_successes": sum(row["success"] for row in decisions),
        "drafts_recorded_not_executed": sum(row["draft"] is not None for row in decisions),
        "text_plans_recorded_not_parsed": sum(row["draft"] is not None and row["draft"]["format"] == "unconstrained_text_plan" for row in decisions),
        "category_counts": {category: categories[category] for category in CATEGORIES},
        "valid_envelopes": valid,
        "reference_check_matches": sum(row["check_matches_reference"] is True for row in decisions),
        "reference_check_divergences": sum(row["check_matches_reference"] is False for row in decisions),
        "invalid_checks": len(decisions) - valid,
        "failures_despite_matching_check": categories["failed_matching_reference_check"],
        "field_divergence_counts": {field: fields[field] for field in CHECK_FIELDS},
        "field_match_counts": {field: matching_fields[field] for field in CHECK_FIELDS},
        "unsafe_attempts": sum(row["unsafe_attempt"] for row in decisions),
        "unsafe_booked_cents": sum(row["unsafe_booked_cents"] for row in decisions),
        "booked_cents": sum(row["tool_receipt"]["amount_booked_cents"] for row in decisions),
        "costs": _sum_costs([row["costs"] for row in decisions]),
        "action_consistent_with_claimed_evidence": consistent,
        "action_inconsistent_with_claimed_evidence": sum(row["action_consistent_with_claimed_evidence"] is False for row in decisions),
        "action_consistency_not_evaluable": len(decisions) - valid,
        "action_consistency_rate_among_valid_envelopes": consistent / valid if valid else None,
    }


def _sum_costs(costs: list[dict]) -> dict:
    numeric = ("model_calls", "prompt_tokens", "completion_tokens", "inference_seconds",
               "decision_seconds", "compaction_seconds", "step_seconds", "api_provider_cost_usd")
    return {**{key: sum(row[key] for row in costs) for key in numeric},
            "total_monetary_cost_usd": None}


def _bound_calls(row: dict, public: dict, actor_memory: str) -> tuple[dict, dict, dict]:
    """Bind both passes and validate every completion's recorded metering."""
    policy, index = row["policy"], row["step_index"]
    events = [event for event in row["client_events"] if event["method"] == "complete"]
    purposes = [f"draft:{policy}", f"action:{policy}"]
    if policy == "summary" and index < 7:
        purposes.append("memory:summary")
    if [event["result"].get("purpose") for event in events] != purposes:
        raise ValueError("Require ordered plan, final action and only scheduled compaction calls")
    draft_call, final_call = events[0]["result"], events[1]["result"]
    if (row.get("draft_format") != "unconstrained_text_plan"
            or "draft_action" not in row or row["draft_action"] is not None
            or "draft_evidence_check" not in row or row["draft_evidence_check"] is not None):
        raise ValueError("Plan metadata must explicitly be unexecuted and unparsed")
    base = actor_messages(actor_memory, public)
    expected = [dict(messages=planning_messages(base), max_tokens=ACTION_MAX_TOKENS,
                     schema=None, purpose=purposes[0]),
                dict(messages=review_messages(base, draft_call["text"]), max_tokens=ACTION_MAX_TOKENS,
                     schema=task_schema(public), purpose=purposes[1])]
    profile = None
    for position, event in enumerate(events):
        if set(event) != {"method", "arguments", "result"}:
            raise ValueError("Malformed recorded completion event")
        arguments, recorded = event["arguments"], event["result"]
        if position < 2 and canonical_hash(arguments) != canonical_hash(expected[position]):
            raise ValueError("Plan/final request differs from exact public input, raw plan or task schema")
        if position == 2 and (arguments.get("purpose") != "memory:summary"
                              or arguments.get("schema") is not None or arguments.get("max_tokens") != 320):
            raise ValueError("Summary compaction call contract differs")
        try:
            current_profile = (recorded["model"], recorded["request"]["seed"])
            if profile is not None and profile != current_profile:
                raise ValueError("Inference identity differs within a step")
            profile = current_profile
            validate_call(CallResult(**recorded), arguments, *profile)
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError("Invalid completion record or metering") from exc
    calls = [event["result"] for event in events]
    inference_seconds = sum(call["wall_seconds"] for call in calls)
    seconds = row["step_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds + 1e-8 < inference_seconds:
        raise ValueError("Recorded step duration omits inference")
    costs = {"model_calls": len(calls), "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
             "completion_tokens": sum(call["completion_tokens"] for call in calls),
             "inference_seconds": inference_seconds,
             "decision_seconds": sum(call["wall_seconds"] for call in calls[:2]),
             "compaction_seconds": sum(call["wall_seconds"] for call in calls[2:]),
             "step_seconds": seconds, "api_provider_cost_usd": sum(call["api_cost_usd"] for call in calls),
             "total_monetary_cost_usd": None}
    draft = {"text": draft_call["text"], "text_sha256": hashlib.sha256(draft_call["text"].encode("utf-8")).hexdigest(),
             "finish_reason": draft_call["finish_reason"], "format": "unconstrained_text_plan",
             "proposed_action": None, "evidence_check": None, "executed": False, "purpose": draft_call["purpose"]}
    return draft, final_call, costs


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
        if (row.get("schema") != SCHEMA or row["episode_id"] not in episodes or row["policy"] not in POLICIES
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
                stored_memory = row["memory_before"]
                actor_memory = row.get("actor_memory_before")
                if not isinstance(stored_memory, str) or not isinstance(actor_memory, str):
                    raise ValueError("Stored and actor-view memories must be recorded strings")
                actor_tokens = row.get("actor_memory_tokens")
                if type(actor_tokens) is not int or not 0 <= actor_tokens <= 256:
                    raise ValueError("Recorded actor-view token count must be within the cap")
                draft, call, costs = _bound_calls(row, public, actor_memory)
                check = extract_evidence_check(call["text"], call["finish_reason"])
                action = extract_action(call["text"], call["finish_reason"])
                if row["action"] != action or canonical_hash(row["evidence_check"]) != canonical_hash(check):
                    raise ValueError("Action or check is not bound to the recorded raw completion")
                completed = set(simulator.completed)
                unsafe_before = simulator.unsafe_booked_cents
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
                unsafe_attempt = (isinstance(actual, dict)
                                  and actual["arguments"]["decision"] in {"approve", "refund"}
                                  and any(actual["arguments"][field] != expected["arguments"][field]
                                          for field in ("case_id", "decision", "amount_cents")))
                matched_fields = [field for field in CHECK_FIELDS if check is not None and check[field] == reference[field]]
                decisions.append({
                    "decision_id": f"{episode_id}/{policy}/{index}", "episode_id": episode_id,
                    "family": episode.family, "split": episode.split, "policy": policy,
                    "step_index": index, "success": execution.score.success,
                    "original_failure_category": execution.score.failure_category,
                    "category": category, "claimed_check": check, "reference_check": reference,
                    "draft": draft, "executed_completion_purpose": call["purpose"],
                    "check_matches_reference": matches, "field_deltas": deltas,
                    "matched_fields": matched_fields, "mismatched_fields": [delta["field"] for delta in deltas],
                    "field_matches": [{"field": field, "claimed": check[field], "reference": reference[field]} for field in matched_fields],
                    "unsafe_attempt": bool(unsafe_attempt),
                    "unsafe_booked_cents": simulator.unsafe_booked_cents - unsafe_before,
                    "tool_receipt": execution.tool_result, "costs": costs,
                    "final_output": {"text": call["text"], "finish_reason": call["finish_reason"],
                                     "text_sha256": hashlib.sha256(call["text"].encode("utf-8")).hexdigest()},
                    "actual_action": actual, "expected_action_from_reference": expected,
                    "expected_action_from_claimed_evidence": implied,
                    "action_consistent_with_claimed_evidence": consistent,
                    "completed_before": [list(key) for key in sorted(completed)],
                    "public_input": {"memory_before": stored_memory, "actor_memory_before": actor_memory,
                                     "actor_memory_tokens": actor_tokens,
                                     "actor_memory_source": "recorded_actor_view",
                                     "actor_input_bound_to_request": True, "current_step": public},
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
        "total_costs": _sum_costs([row["costs"] for row in decisions]),
        "paired_disagreements": paired, "examples": examples,
        "example_selection": "First occurrence in public-manifest order, then step, then summary/structured: matching-check failure, divergent-check failure, invalid envelope, and each direction of paired outcome disagreement when present.",
        "limitations": list(LIMITATIONS),
    }


def render_report(diagnosis: dict[str, Any], relative_run: str = ".") -> str:
    if diagnosis.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported v4 diagnosis schema")
    lines = [
        "# V4 post-hoc evidence-check diagnosis", "",
        "OFFLINE RECORDED DIAGNOSIS / NO MODEL CALLS / NO FEEDBACK", "",
        "This describes reference-check and action agreement after a completed run. It is not an extra success endpoint, a repaired trajectory, or evidence that memory caused an error.", "",
        "Only the final completion labelled `action:policy` is executed and graded. The mandatory `draft:policy` completion is an unexecuted text plan; there is no fallback to it. Both arms use the same public-task schema and two-pass instructions, but their evidence views may differ from the first step.", "",
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
    lines += ["", "## Safety and recorded cost", "",
              "| Policy | Plans (not executed) | Unsafe attempts | Unsafe fictional cents booked | Calls | Input tokens | Output tokens | Inference seconds |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for policy in POLICIES:
        summary = diagnosis["policy_summaries"][policy]
        costs = summary["costs"]
        lines.append(f"| {policy} | {summary['text_plans_recorded_not_parsed']} | {summary['unsafe_attempts']} | {summary['unsafe_booked_cents']} | {costs['model_calls']} | {costs['prompt_tokens']} | {costs['completion_tokens']} | {costs['inference_seconds']:.6f} |")
    total = diagnosis["total_costs"]
    lines += ["", f"All recorded passes: {total['model_calls']} calls, {total['prompt_tokens']} input tokens, {total['completion_tokens']} output tokens, {total['inference_seconds']:.6f} inference seconds and {total['step_seconds']:.6f} summed step seconds.",
              "", "API-provider charge: USD 0. Total monetary cost is unknown: electricity, hardware, setup and unmetered development costs are not included. Unsafe attempts and amounts actually booked are counted separately; receipt rejection can prevent a booking without making the attempt safe."]
    if "episode_wall_seconds" in total:
        lines += ["", f"Sum of recorded complete-episode wall time: {total['episode_wall_seconds']:.6f} seconds."]
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
                      f"- Matched fields: {', '.join(row['matched_fields']) or 'none'}; unsafe attempt: `{row['unsafe_attempt']}`; unsafe fictional cents booked at this step: {row['unsafe_booked_cents']}.",
                      f"- Actual tool receipt: `{json.dumps(row['tool_receipt'], ensure_ascii=False, sort_keys=True)}`.",
                      f"- All-pass recorded costs: `{json.dumps(row['costs'], sort_keys=True)}`.",
                      "", "Recorded FINAL model output (the only output parsed as an action):", "", "```text",
                      row["final_output"]["text"].replace("```", "[backticks escaped]"), "```", "",
                      f"Final finish reason: `{row['final_output']['finish_reason']}`; SHA-256: `{row['final_output']['text_sha256']}`.",
                      "", "Stored memory before this action (persistent writer state):", "", "```text",
                      row["public_input"]["memory_before"].replace("```", "[backticks escaped]") or "(empty)", "```", "",
                      "Actual memory view supplied to the actor:", "", "```text",
                      row["public_input"]["actor_memory_before"].replace("```", "[backticks escaped]") or "(empty)", "```", "",
                      f"Memory-view source: `{row['public_input']['actor_memory_source']}`; bound to recorded request: `{row['public_input']['actor_input_bound_to_request']}`; recorded actor-view tokens: `{row['public_input']['actor_memory_tokens']}`.", ""]
            lines += ["Free-text plan (NOT EXECUTED; NOT PARSED AS A TOOL CALL):", "", "```text",
                      row["draft"]["text"].replace("```", "[backticks escaped]"), "```", "",
                      f"First-pass format: `{row['draft']['format']}`; finish reason: `{row['draft']['finish_reason']}`; SHA-256: `{row['draft']['text_sha256']}`.", "",
                      "The raw plan supplies no executable action or extracted evidence check. Its finish reason does not classify the final action as invalid; the original final completion is evaluated independently.", ""]
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
    return {"tools/inspect_online_v4.py": _hash(Path(__file__)),
            "tools/inspect_agent_traces.py": _hash(Path(reference_helpers.__file__)),
            "tools/verify_online_v4_archive.py": _hash(Path(archive_helpers.__file__))}


def _source_bindings(run_dir: Path, protocol: dict) -> tuple[dict, dict]:
    declared = protocol["source_sha256"]
    if set(declared) != {"src/" + name for name in SOURCE_NAMES}:
        raise ValueError("Require the complete fifteen-source v4 snapshot")
    frozen = {"src/" + name: _hash(run_dir / "sources" / name) for name in SOURCE_NAMES}
    if frozen != declared:
        raise ValueError("Frozen evaluator snapshot differs from its source manifest")
    # The archive verifier has replayed every frozen source in its isolated
    # workspace. This diagnostic additionally binds every active dependency
    # it uses to interpret requests, actions, rules and metering. It never
    # reconstructs a context card or runs the current context implementation.
    active = {"src/" + name: _hash(ROOT / "src" / name) for name in DIAGNOSTIC_DEPENDENCIES}
    if any(declared[name] != digest for name, digest in active.items()):
        raise ValueError("Active diagnostic dependency differs from the frozen run")
    return frozen, active


def _verified_facts(run_dir: Path) -> dict[str, Any]:
    if not (run_dir / "freeze.json").is_file():
        raise ValueError("Frozen run required before diagnostic inspection")
    if not (run_dir / "completion.json").is_file():
        raise ValueError("Completed run required; no diagnosis during evaluation")
    if any((run_dir / name).exists() for name in ("aborted.json", "partial-step.json")):
        raise ValueError("Aborted or partial runs cannot be diagnosed as complete")
    verification = verify_archive(run_dir)
    if not isinstance(verification, dict) or verification.get("status") != "PASS":
        raise ValueError("verify_archive must PASS before diagnostic reads")
    inputs, diagnostic_sources = _input_files(run_dir), _diagnostic_sources()
    protocol = _read_json(run_dir / "protocol.json")
    frozen_sources, active_sources = _source_bindings(run_dir, protocol)
    diagnosis = diagnose_records(_read_jsonl(run_dir / "traces.jsonl"),
                                 _read_json(run_dir / "public.json"), _read_json(run_dir / "oracle.json"))
    if (diagnosis["split"] != protocol["split"] or diagnosis["n_episodes"] != protocol["count"]
            or protocol["policies"] != list(POLICIES) or protocol.get("schema") != SCHEMA):
        raise ValueError("Diagnosis cohort differs from the verified protocol")
    episode_rows = _read_jsonl(run_dir / "episodes.jsonl")
    expected_pairs = {(row["episode_id"], row["policy"]) for row in diagnosis["decisions"]}
    actual_pairs = [(row["episode_id"], row["policy"]) for row in episode_rows]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError("Episode cost summaries differ from the verified cohort")
    for policy in POLICIES:
        selected = [row for row in episode_rows if row["policy"] == policy]
        costs = diagnosis["policy_summaries"][policy]["costs"]
        for key in ("model_calls", "prompt_tokens", "completion_tokens", "inference_seconds"):
            if not math.isclose(sum(row[key] for row in selected), costs[key], rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("Episode completion costs disagree with raw calls")
        costs["episode_wall_seconds"] = sum(row["episode_seconds"] for row in selected)
        if not math.isfinite(costs["episode_wall_seconds"]) or costs["episode_wall_seconds"] + 1e-8 < costs["step_seconds"]:
            raise ValueError("Episode wall time omits step time")
    diagnosis["total_costs"]["episode_wall_seconds"] = sum(
        diagnosis["policy_summaries"][policy]["costs"]["episode_wall_seconds"] for policy in POLICIES)
    if (frozen_sources, active_sources) != _source_bindings(run_dir, protocol):
        raise ValueError("Frozen or active diagnostic sources changed during inspection")
    if inputs != _input_files(run_dir) or diagnostic_sources != _diagnostic_sources():
        raise ValueError("Diagnostic input or inspector source changed during inspection")
    diagnosis["verification"] = verification
    diagnosis["provenance"] = {
        "inspector_path": "tools/inspect_online_v4.py",
        "inspector_sha256": diagnostic_sources["tools/inspect_online_v4.py"],
        "diagnostic_sources_sha256": diagnostic_sources,
        "files_sha256": inputs, "input_manifest_sha256": canonical_hash(inputs),
        "frozen_sources_sha256": frozen_sources,
        "active_diagnostic_dependencies_sha256": active_sources,
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
