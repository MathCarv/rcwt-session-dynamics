"""Post-hoc inspector contracts with invented fixtures, not model evidence.

No generated held-out corpus or real run is inspected. Fixtures construct small
fictional episodes directly; model construction and network use are forbidden.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rcwt_agent_run as runner
from rcwt_agent_actor import extract_action, extract_evidence_check
from rcwt_agent_env import Action, Episode, OracleSnapshot, Simulator, Step, expected_action
from rcwt_local_model import canonical_hash
from rcwt_review_v3 import ORIGINAL_JSON_REQUIREMENT, planning_messages, review_messages


SPEC = importlib.util.spec_from_file_location("online_v3_diagnosis", runner.ROOT / "tools/inspect_online_v3.py")
inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspector)


def _fictional_episode(name, index):
    steps, snapshots = [], []
    for step in range(8):
        case_id = f"FAKE-{index}-INVOICE-{0 if step == 1 else step}"
        account, merchant, order = (f"FAKE-{index}-{kind}-{0 if step == 1 else step}" for kind in ("ACCOUNT", "MERCHANT", "ORDER"))
        observations = [] if step == 1 else [
            {"source": "tool", "tool": "read_invoice", "content": {
                "invoice_id": case_id, "account_id": account, "merchant_id": merchant,
                "order_id": order, "total_cents": 1000, "currency": "BRL"}},
            {"source": "tool", "tool": "read_account", "content": {
                "account_id": account, "verification_status": "active", "holder_merchant_id": merchant}},
            {"source": "tool", "tool": "read_payment", "content": {"invoice_id": case_id, "status": "cleared"}},
        ]
        steps.append(Step(step, tuple(observations), {"case_id": case_id, "operation": "payout"}))
        snapshots.append(OracleSnapshot(case_id, "payout", 1000, "active", True, "cleared", None))
    return Episode(name, "updated-state", "train", index, tuple(steps), tuple(snapshots))


def _fake_records():
    # Manifest order deliberately differs from lexicographic ID and trace order.
    episodes = [_fictional_episode("Z-FAKE-FIRST", 0), _fictional_episode("A-FAKE-SECOND", 1)]
    groups = {}
    for episode in episodes:
        for policy in inspector.POLICIES:
            simulator, rows = Simulator(episode), []
            for step, snapshot in enumerate(episode.oracle_steps):
                public_step = simulator.public_step(step)
                check = inspector.reference_check(snapshot, set(simulator.completed))
                action = expected_action(snapshot, set(simulator.completed)).to_dict()
                finish_reason = "stop"
                if policy == "summary":
                    if step == 0:
                        action = Action(snapshot.case_id, "ask_info", 0, "missing_evidence").to_dict()
                    elif step == 2:
                        check["invoice_amount_cents"] = 1007
                        action = inspector.action_from_claimed_check(check, snapshot.case_id, snapshot.operation)
                    elif step == 5:
                        check["payment_status"] = "pending"  # Correct action, divergent claimed field.
                    elif step == 6:
                        action = Action(snapshot.case_id, "hold", 0, "funds_unsettled").to_dict()
                else:
                    if step == 1:
                        check["operation_already_booked"] = "no_record"
                        action = inspector.action_from_claimed_check(check, snapshot.case_id, snapshot.operation)
                    elif step == 3:
                        check["return_status"] = "accepted"  # Irrelevant to payout, but still a mismatch.
                    elif step == 5:
                        action = Action(snapshot.case_id, "hold", 0, "funds_unsettled").to_dict()
                    elif step == 6:
                        check["payment_status"] = "pending"
                        action = inspector.action_from_claimed_check(check, snapshot.case_id, snapshot.operation)
                    elif step == 7:
                        finish_reason = "length"
                envelope = {"evidence_check": check, **action}
                text = "{fake malformed" if policy == "summary" and step == 3 else json.dumps(envelope)
                actual = extract_action(text, finish_reason)
                extracted = extract_evidence_check(text, finish_reason)
                execution = simulator.execute_action(step, actual)
                stored_memory = "" if step == 0 else f"FAKE STORED MEMORY {episode.episode_id}/{policy}/{step}"
                actor_memory = (f"FAKE ACTOR VIEW {episode.episode_id}/{policy}/{step}"
                                if policy == "structured" and step > 0 else stored_memory)
                row = {"schema": "FAKE-DIAGNOSTIC-CONTRACT", "episode_id": episode.episode_id,
                       "family": episode.family, "split": episode.split, "policy": policy, "step_index": step,
                       "public_step": public_step, "action": actual, "evidence_check": extracted,
                       "tool_result": execution.tool_result, "score": execution.score.to_dict(),
                       "state_after": execution.state, "memory_before": stored_memory,
                       "actor_memory_before": actor_memory, "actor_memory_tokens": 43 if actor_memory else 0,
                       "client_events": [{"method": "tokenize", "arguments": {"text": actor_memory}, "result": [21]},
                                         {"method": "complete", "result": {
                           "text": text, "finish_reason": finish_reason, "purpose": f"action:{policy}",
                           "request": {"messages": [{"role": "system", "content": "FAKE CONTRACT ONLY"},
                               {"role": "user", "content": json.dumps({"retained_memory": actor_memory, "current_step": public_step})}]}}}]}
                row["sha256"] = canonical_hash(row)
                rows.append(row)
            groups[(episode.episode_id, policy)] = rows
    traces = []
    for episode in reversed(episodes):
        for policy in reversed(inspector.POLICIES):
            traces.extend(groups[(episode.episode_id, policy)])
    return traces, [episode.to_public_dict() for episode in episodes], [episode.to_oracle_dict() for episode in episodes]


def _reseal(row):
    row["sha256"] = canonical_hash({key: value for key, value in row.items() if key != "sha256"})


def _with_drafts(original):
    """Add fictional unexecuted drafts without changing any final action."""
    traces = copy.deepcopy(original)
    for row in traces:
        final = next(event["result"] for event in row["client_events"] if event["method"] == "complete")
        check = {"invoice_amount_cents": 999, "account_status": "unknown", "ownership_match": "unknown",
                 "payment_status": "unknown", "return_status": "unknown", "operation_already_booked": "no_record"}
        draft_text = json.dumps({"evidence_check": check,
                                 **Action("FAKE-DRAFT-NOT-EXECUTED", "approve", 999, "authorized_payout").to_dict()})
        draft = {**copy.deepcopy(final), "purpose": f"draft:{row['policy']}", "text": draft_text, "finish_reason": "stop"}
        draft["request"]["response_format"] = {"type": "json_schema", "json_schema": {"name": "FAKE-CONTRACT-ONLY"}}
        final["request"]["messages"].extend([{"role": "assistant", "content": draft_text},
                                               {"role": "user", "content": "FAKE FIXED REVIEW INSTRUCTION"}])
        row["client_events"].insert(1, {"method": "complete", "result": draft})
        row["draft_action"] = extract_action(draft_text, "stop")
        row["draft_evidence_check"] = extract_evidence_check(draft_text, "stop")
        _reseal(row)
    return traces


def _with_plans(original, text="FAKE plain-text plan: defer because evidence is missing.", finish_reason="stop"):
    """Invent a schema-free first pass without changing any recorded final output."""
    traces = copy.deepcopy(original)
    for row in traces:
        final = next(event["result"] for event in row["client_events"] if event["method"] == "complete")
        original_messages = copy.deepcopy(final["request"]["messages"])
        original_messages[0]["content"] += " " + ORIGINAL_JSON_REQUIREMENT
        plan = {**copy.deepcopy(final), "purpose": f"draft:{row['policy']}", "text": text,
                "finish_reason": finish_reason, "request": {"messages": planning_messages(original_messages)}}
        final["request"]["messages"] = review_messages(original_messages, text)
        row["client_events"].insert(1, {"method": "complete", "result": plan})
        row.update(draft_format="unconstrained_text_plan", draft_action=None, draft_evidence_check=None)
        _reseal(row)
    return traces


class OnlineV3DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.traces, self.public, self.oracle = _fake_records()
        network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Diagnosis attempted network access"))
        network.start(); self.addCleanup(network.stop)
        model = patch("rcwt_local_model.LocalModelClient.__init__", side_effect=AssertionError("Diagnosis constructed a model client"))
        model.start(); self.addCleanup(model.stop)

    def diagnose(self):
        return inspector.diagnose_records(self.traces, self.public, self.oracle)

    def test_counts_matching_divergent_invalid_and_failures_despite_matching_check(self):
        result = self.diagnose()
        self.assertEqual(result["n_episodes"], 2)
        self.assertEqual(len(result["decisions"]), 32)
        summary, structured = (result["policy_summaries"][policy] for policy in inspector.POLICIES)
        self.assertEqual(summary["exact_action_successes"], 8)
        self.assertEqual(summary["reference_check_matches"], 10)
        self.assertEqual(summary["reference_check_divergences"], 4)
        self.assertEqual(summary["invalid_checks"], 2)
        self.assertEqual(summary["failures_despite_matching_check"], 4)
        self.assertEqual(summary["category_counts"], {
            "failed_matching_reference_check": 4, "failed_divergent_check": 2, "invalid_envelope": 2,
            "correct_matching_reference_check": 6, "correct_divergent_check": 2})
        self.assertEqual(structured["exact_action_successes"], 8)
        self.assertEqual(structured["reference_check_matches"], 8)
        self.assertEqual(structured["reference_check_divergences"], 6)
        self.assertEqual(structured["invalid_checks"], 2)
        self.assertEqual(structured["failures_despite_matching_check"], 2)
        self.assertEqual(summary["field_divergence_counts"]["invoice_amount_cents"], 2)
        self.assertEqual(summary["field_divergence_counts"]["payment_status"], 2)
        self.assertEqual(structured["field_divergence_counts"]["operation_already_booked"], 2)
        self.assertEqual(structured["field_divergence_counts"]["return_status"], 2)
        self.assertEqual(summary["action_consistency_not_evaluable"], 2)
        self.assertTrue(result["post_hoc"])
        self.assertFalse(result["feedback_to_agent"])
        self.assertFalse(result["causal_memory_attribution"])

    def test_pair_counts_and_both_examples_use_manifest_then_step_not_ids_or_trace_order(self):
        result = self.diagnose()
        paired = result["paired_disagreements"]
        self.assertEqual(paired["structured_correct_summary_incorrect"], 6)
        self.assertEqual(paired["summary_correct_structured_incorrect"], 6)
        self.assertEqual(paired["both_correct"], 2)
        self.assertEqual(paired["both_incorrect"], 2)
        self.assertEqual(paired["paired_tasks"], 16)
        self.assertFalse(paired["independent_samples"])
        examples = {item["kind"]: item["decision_ids"] for item in result["examples"]}
        self.assertEqual(examples["failed_matching_reference_check"], ["Z-FAKE-FIRST/summary/0"])
        self.assertEqual(examples["failed_divergent_check"], ["Z-FAKE-FIRST/structured/1"])
        self.assertEqual(examples["paired_structured_correct_summary_incorrect"], [
            "Z-FAKE-FIRST/summary/0", "Z-FAKE-FIRST/structured/0"])
        self.assertEqual(examples["paired_summary_correct_structured_incorrect"], [
            "Z-FAKE-FIRST/summary/1", "Z-FAKE-FIRST/structured/1"])
        first = result["decisions"][0]
        self.assertEqual(first["decision_id"], "Z-FAKE-FIRST/summary/0")
        source = first["source"]
        original = self.traces[source["line"] - 1]
        self.assertEqual(source["line"], 25)
        self.assertEqual(source["trace_sha256"], original["sha256"])

    def test_reference_booking_is_per_arm_and_claimed_check_branch_does_not_get_actual_ledger(self):
        indexed = {row["decision_id"]: row for row in self.diagnose()["decisions"]}
        summary, structured = (indexed[f"Z-FAKE-FIRST/{policy}/1"] for policy in inspector.POLICIES)
        self.assertEqual(summary["reference_check"]["operation_already_booked"], "no_record")
        self.assertEqual(structured["reference_check"]["operation_already_booked"], "yes")
        self.assertEqual(structured["claimed_check"]["operation_already_booked"], "no_record")
        self.assertEqual(structured["expected_action_from_reference"]["arguments"]["decision"], "hold")
        self.assertEqual(structured["expected_action_from_claimed_evidence"]["arguments"]["decision"], "approve")
        self.assertEqual(structured["original_failure_category"], "unsafe_execution")
        self.assertFalse(structured["success"])
        self.assertTrue(structured["action_consistent_with_claimed_evidence"])

    def test_diagnosis_is_deterministic_does_not_modify_traces_or_repair_actions(self):
        before = copy.deepcopy((self.traces, self.public, self.oracle))
        first = self.diagnose()
        self.assertEqual(first, self.diagnose())
        self.assertEqual((self.traces, self.public, self.oracle), before)
        failure = first["decisions"][0]
        self.assertNotEqual(failure["actual_action"], failure["expected_action_from_claimed_evidence"])
        self.assertFalse(failure["success"])

    def test_actual_actor_view_and_stored_memory_are_separate_bound_evidence(self):
        result = self.diagnose()
        indexed = {row["decision_id"]: row for row in result["decisions"]}
        observed = indexed["Z-FAKE-FIRST/structured/1"]["public_input"]
        self.assertEqual(observed["memory_before"], "FAKE STORED MEMORY Z-FAKE-FIRST/structured/1")
        self.assertEqual(observed["actor_memory_before"], "FAKE ACTOR VIEW Z-FAKE-FIRST/structured/1")
        self.assertEqual(observed["actor_memory_tokens"], 43)
        self.assertEqual(observed["actor_memory_source"], "recorded_actor_view")
        self.assertTrue(observed["actor_input_bound_to_request"])
        first = [indexed[f"Z-FAKE-FIRST/{policy}/0"]["public_input"]["actor_memory_before"] for policy in inspector.POLICIES]
        self.assertEqual(first, ["", ""])
        report = inspector.render_report(result)
        self.assertIn("Stored memory before this action (persistent writer state):", report)
        self.assertIn("Actual memory view supplied to the actor:", report)
        self.assertIn("FAKE STORED MEMORY Z-FAKE-FIRST/structured/1", report)
        self.assertIn("FAKE ACTOR VIEW Z-FAKE-FIRST/structured/1", report)

    def test_legacy_rows_fall_back_explicitly_to_stored_actor_memory(self):
        traces = copy.deepcopy(self.traces)
        for row in traces:
            del row["actor_memory_before"]
            del row["actor_memory_tokens"]
            action_call = next(event["result"] for event in row["client_events"] if event["method"] == "complete")
            user_message = action_call["request"]["messages"][1]
            payload = json.loads(user_message["content"])
            payload["retained_memory"] = row["memory_before"]
            user_message["content"] = json.dumps(payload)
            _reseal(row)
        result = inspector.diagnose_records(traces, self.public, self.oracle)
        for decision in result["decisions"]:
            observed = decision["public_input"]
            self.assertEqual(observed["actor_memory_before"], observed["memory_before"])
            self.assertEqual(observed["actor_memory_source"], "legacy_stored_memory_fallback")
            self.assertIsNone(observed["actor_memory_tokens"])

    def test_two_pass_draft_is_not_executed_and_only_final_call_defines_diagnosis(self):
        traces = _with_drafts(self.traces)
        baseline = self.diagnose()
        result = inspector.diagnose_records(traces, self.public, self.oracle)
        self.assertEqual(result["paired_disagreements"], baseline["paired_disagreements"])
        for policy in inspector.POLICIES:
            self.assertEqual(result["policy_summaries"][policy]["category_counts"], baseline["policy_summaries"][policy]["category_counts"])
            self.assertEqual(result["policy_summaries"][policy]["drafts_recorded_not_executed"], 16)
        for row in result["decisions"]:
            self.assertFalse(row["draft"]["executed"])
            self.assertEqual(row["executed_completion_purpose"], f"action:{row['policy']}")
            self.assertNotEqual(row["actual_action"], row["draft"]["proposed_action"])
        final_invalid = next(row for row in result["decisions"] if row["decision_id"] == "Z-FAKE-FIRST/structured/7")
        self.assertEqual(final_invalid["category"], "invalid_envelope")
        self.assertEqual(final_invalid["actual_action"], "TRUNCATED_ACTION")
        self.assertIsNotNone(final_invalid["draft"]["evidence_check"])
        report = inspector.render_report(result)
        self.assertIn("Draft model output (NOT EXECUTED):", report)
        self.assertIn("no fallback to it", report)

    def test_draft_metadata_and_final_review_input_are_bound_after_reseal(self):
        for change in ("draft_metadata", "review_input", "duplicate_final", "missing_final"):
            traces = _with_drafts(self.traces)
            row = traces[0]
            finals = [event for event in row["client_events"] if event["method"] == "complete" and event["result"]["purpose"] == f"action:{row['policy']}"]
            final = finals[0]["result"]
            if change == "draft_metadata":
                row["draft_action"] = "UNBOUND DRAFT"
            elif change == "review_input":
                final["request"]["messages"][2]["content"] = "DIFFERENT RAW DRAFT"
            elif change == "duplicate_final":
                row["client_events"].append(copy.deepcopy(finals[0]))
            else:
                final["purpose"] = "memory:summary"
            _reseal(row)
            with self.subTest(change=change), self.assertRaises(ValueError):
                inspector.diagnose_records(traces, self.public, self.oracle)

    def test_schema_free_plans_are_never_parsed_or_counted_as_invalid_final_outputs(self):
        baseline = self.diagnose()
        json_shaped_plan = next(event["result"]["text"] for row in self.traces for event in row["client_events"]
                               if event["method"] == "complete" and event["result"]["finish_reason"] == "stop")
        for text, finish in (("", "stop"), ("FAKE unfinished plan", "length"), ("FAKE invalid JSON {", "stop"),
                             (json_shaped_plan, "stop")):
            traces = _with_plans(self.traces, text, finish)
            with self.subTest(text=text, finish=finish), patch.object(inspector, "extract_action", wraps=extract_action) as actions, patch.object(inspector, "extract_evidence_check", wraps=extract_evidence_check) as checks:
                result = inspector.diagnose_records(traces, self.public, self.oracle)
                self.assertEqual(actions.call_count, 32, "Only the 32 FINAL outputs may be parsed")
                self.assertEqual(checks.call_count, 32, "Only the 32 FINAL checks may be parsed")
            self.assertEqual(result["paired_disagreements"], baseline["paired_disagreements"])
            for policy in inspector.POLICIES:
                actual = result["policy_summaries"][policy]
                expected = baseline["policy_summaries"][policy]
                self.assertEqual(actual["category_counts"], expected["category_counts"])
                self.assertEqual(actual["invalid_checks"], expected["invalid_checks"])
                self.assertEqual(actual["text_plans_recorded_not_parsed"], 16)
            for decision in result["decisions"]:
                plan = decision["draft"]
                self.assertEqual(plan["format"], "unconstrained_text_plan")
                self.assertEqual((plan["text"], plan["finish_reason"]), (text, finish))
                self.assertIsNone(plan["proposed_action"])
                self.assertIsNone(plan["evidence_check"])
                self.assertFalse(plan["executed"])
            failed_final = next(row for row in result["decisions"] if row["decision_id"] == "Z-FAKE-FIRST/structured/7")
            self.assertEqual(failed_final["actual_action"], "TRUNCATED_ACTION")
            self.assertEqual(failed_final["category"], "invalid_envelope")
            report = inspector.render_report(result)
            self.assertIn("Free-text plan (NOT EXECUTED; NOT PARSED AS A TOOL CALL):", report)
            self.assertNotIn("Draft proposed action, not executed:", report)
            self.assertIn("Its finish reason does not classify the final action as invalid", report)

    def test_planning_input_and_null_metadata_remain_bound_after_reseal(self):
        for change in ("unmodified_base", "plan_system", "plan_payload", "raw_plan", "parsed_action", "parsed_check", "format", "response_format"):
            traces = _with_plans(self.traces)
            row = traces[0]
            plan, final = [event["result"] for event in row["client_events"] if event["method"] == "complete"]
            if change == "unmodified_base":
                plan["request"]["messages"] = copy.deepcopy(final["request"]["messages"][:2])
            elif change == "plan_system":
                plan["request"]["messages"][0]["content"] += " FAKE UNBOUND RULE"
            elif change == "plan_payload":
                plan["request"]["messages"][1]["content"] = "FAKE DIFFERENT EVIDENCE"
            elif change == "raw_plan":
                final["request"]["messages"][2]["content"] += " FAKE CHANGED PLAN"
            elif change == "parsed_action":
                row["draft_action"] = "INVALID_ACTOR_OUTPUT"
            elif change == "parsed_check":
                row["draft_evidence_check"] = {}
            elif change == "format":
                row["draft_format"] = "json_action_draft"
            else:
                plan["request"]["response_format"] = {"type": "json_schema"}
            _reseal(row)
            with self.subTest(change=change), self.assertRaises(ValueError):
                inspector.diagnose_records(traces, self.public, self.oracle)

    def test_tampered_raw_binding_or_grade_is_rejected_even_after_row_reseal(self):
        for field in ("action", "evidence_check", "score", "state_after", "public_step", "actor_memory_before"):
            traces = copy.deepcopy(self.traces)
            row = traces[0]
            if field == "action":
                row[field] = "CHANGED_ACTION"
            elif field == "evidence_check":
                row[field]["invoice_amount_cents"] = 7777
            elif field == "score":
                row[field]["success"] = not row[field]["success"]
            elif field == "state_after":
                row[field]["unsafe_booked_cents"] = 7777
            elif field == "actor_memory_before":
                row[field] = "WRONG FAKE ACTOR VIEW"
            else:
                row[field]["task"]["case_id"] = "WRONG_FAKE_CASE"
            _reseal(row)
            with self.subTest(field=field), self.assertRaises(ValueError):
                inspector.diagnose_records(traces, self.public, self.oracle)

    def test_missing_duplicate_reordered_policy_and_mixed_split_fail_closed(self):
        bad_traces = [self.traces[:-1], self.traces + [copy.deepcopy(self.traces[0])]]
        reordered = copy.deepcopy(self.traces)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        bad_traces.append(reordered)
        for field, value in (("policy", "learned"), ("split", "test"), ("step_index", True)):
            changed = copy.deepcopy(self.traces)
            changed[0][field] = value
            _reseal(changed[0])
            bad_traces.append(changed)
        for traces in bad_traces:
            with self.subTest(traces=traces), self.assertRaises(ValueError):
                inspector.diagnose_records(traces, self.public, self.oracle)
        duplicate = self.public + [copy.deepcopy(self.public[0])]
        with self.assertRaises(ValueError):
            inspector.diagnose_records(self.traces, duplicate, self.oracle)
        mixed = copy.deepcopy(self.public)
        mixed[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "never pool"):
            inspector.diagnose_records(self.traces, mixed, self.oracle)

    def test_report_has_source_lines_both_directions_and_causal_limits(self):
        report = inspector.render_report(self.diagnose(), "../fake-run")
        for text in ("NO MODEL CALLS / NO FEEDBACK", "not independent samples", "not autonomous policy learning",
                     "not a controlled causal attribution", "structured correct / summary incorrect = 6",
                     "summary correct / structured incorrect = 6", "../fake-run/traces.jsonl#L25",
                     "Trace line/hash: 25", "does not identify compression loss"):
            with self.subTest(text=text):
                self.assertIn(text, report)
        self.assertIn("paired_structured_correct_summary_incorrect", report)
        self.assertIn("paired_summary_correct_structured_incorrect", report)


class OnlineV3DiagnosisArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_dir, self.output_dir = self.root / "explicit-fake-run", self.root / "fake-diagnosis"
        self.run_dir.mkdir()
        traces, public, oracle = _fake_records()
        self.protocol = {"split": "train", "count": 2, "policies": list(inspector.POLICIES),
                         "source_sha256": {"src/rcwt_agent_env.py": inspector._hash(runner.ROOT / "src/rcwt_agent_env.py")}}
        objects = {"protocol.json": self.protocol, "public.json": public, "oracle.json": oracle,
                   "freeze.json": {"explicit_fake_fixture": True}, "completion.json": {"explicit_fake_fixture": True},
                   "started.json": {"explicit_fake_fixture": True}, "schedule.json": [], "episodes.jsonl": []}
        for name, value in objects.items():
            (self.run_dir / name).write_text(json.dumps(value), encoding="utf-8")
        (self.run_dir / "traces.jsonl").write_text("\n".join(json.dumps(row) for row in traces) + "\n", encoding="utf-8")
        self.verifier = patch.object(inspector, "verify_run", return_value={
            "status": "PASS", "scope": "Patched invented fixture, not real inference evidence"})
        self.verify_mock = self.verifier.start()
        self.addCleanup(self.verifier.stop)
        for patcher in (
            patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("No network allowed")),
            patch("rcwt_local_model.LocalModelClient.__init__", side_effect=AssertionError("No model allowed")),
        ):
            patcher.start(); self.addCleanup(patcher.stop)

    def test_completion_and_freeze_required_before_verifier_or_trace_read(self):
        for name in ("completion.json", "freeze.json"):
            path = self.run_dir / name
            saved = path.read_bytes()
            path.unlink()
            with patch.object(inspector, "_input_files") as inputs, patch.object(inspector, "_read_jsonl") as traces:
                with self.assertRaises(ValueError):
                    inspector.inspect_run(self.run_dir, self.output_dir)
            self.verify_mock.assert_not_called()
            inputs.assert_not_called(); traces.assert_not_called()
            self.assertFalse(self.output_dir.exists())
            path.write_bytes(saved)

    def test_failed_verification_cannot_read_traces_or_write_artifacts(self):
        for receipt in ({"status": "FAIL"}, {"status": "WARN"}, {}, None):
            self.verify_mock.return_value = receipt
            with patch.object(inspector, "_input_files") as inputs, patch.object(inspector, "_read_jsonl") as traces:
                with self.subTest(receipt=receipt), self.assertRaisesRegex(ValueError, "must PASS"):
                    inspector.inspect_run(self.run_dir, self.output_dir)
            inputs.assert_not_called(); traces.assert_not_called()
            self.assertFalse(self.output_dir.exists())

    def test_pass_precedes_diagnostic_reads_and_output_has_provenance(self):
        passed = False
        original = inspector._read_jsonl

        def verify(directory):
            nonlocal passed
            passed = True
            return {"status": "PASS", "scope": "fake receipt"}

        def read_traces(path):
            self.assertTrue(passed)
            return original(path)

        self.verify_mock.side_effect = verify
        with patch.object(inspector, "_read_jsonl", side_effect=read_traces):
            diagnosis = inspector.inspect_run(self.run_dir, self.output_dir)
        provenance = diagnosis["provenance"]
        self.assertEqual(provenance["inspector_sha256"], inspector._hash(Path(inspector.__file__)))
        self.assertIn("tools/inspect_agent_traces.py", provenance["diagnostic_sources_sha256"])
        self.assertEqual(provenance["files_sha256"]["traces.jsonl"], inspector._hash(self.run_dir / "traces.jsonl"))
        self.assertEqual(provenance["input_manifest_sha256"], canonical_hash(provenance["files_sha256"]))
        self.assertTrue((self.output_dir / "diagnosis.json").is_file())
        self.assertTrue((self.output_dir / "DIAGNOSIS.md").is_file())
        with self.assertRaises(FileExistsError):
            inspector.inspect_run(self.run_dir, self.output_dir)

    def test_verify_recomputes_without_writes_and_preserves_every_artifact_byte(self):
        inspector.inspect_run(self.run_dir, self.output_dir)
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        original_open = Path.open

        def readonly_open(path, mode="r", *args, **kwargs):
            if any(flag in mode for flag in ("w", "a", "x", "+")):
                raise AssertionError("--verify attempted a write")
            return original_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", new=readonly_open), patch.object(Path, "mkdir", side_effect=AssertionError("--verify attempted mkdir")):
            receipt = inspector.verify_diagnosis(self.run_dir, self.output_dir)
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(receipt["read_only"])
        self.assertFalse(receipt["feedback_to_agent"])
        self.assertEqual(receipt["decisions"], 32)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_saved_fact_hash_and_markdown_tampering_are_rejected(self):
        inspector.inspect_run(self.run_dir, self.output_dir)
        facts = self.output_dir / "diagnosis.json"
        report = self.output_dir / "DIAGNOSIS.md"
        original_facts, original_report = facts.read_bytes(), report.read_bytes()
        for mutate in (
            lambda saved: saved["paired_disagreements"].update(structured_correct_summary_incorrect=999),
            lambda saved: saved["provenance"].update(inspector_sha256="0" * 64),
            lambda saved: saved.update(feedback_to_agent=0),  # false vs 0 must not compare equal.
        ):
            saved = json.loads(original_facts)
            mutate(saved)
            facts.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "facts or provenance"):
                inspector.verify_diagnosis(self.run_dir, self.output_dir)
            facts.write_bytes(original_facts)
        report.write_bytes(original_report + b"\nUnverified gain claim\n")
        with self.assertRaisesRegex(ValueError, "Markdown differs"):
            inspector.verify_diagnosis(self.run_dir, self.output_dir)

    def test_changed_run_bytes_fail_provenance_even_if_patched_verifier_still_passes(self):
        inspector.inspect_run(self.run_dir, self.output_dir)
        traces = self.run_dir / "traces.jsonl"
        traces.write_text(traces.read_text(encoding="utf-8").replace('"schema": ', '"schema" : '), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "facts or provenance"):
            inspector.verify_diagnosis(self.run_dir, self.output_dir)

    def test_input_race_is_rejected_before_writing(self):
        actual = inspector._input_files(self.run_dir)
        changed = {**actual, "traces.jsonl": "f" * 64}
        with patch.object(inspector, "_input_files", side_effect=[actual, changed]):
            with self.assertRaisesRegex(ValueError, "changed during inspection"):
                inspector.inspect_run(self.run_dir, self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_blank_jsonl_lines_and_duplicate_saved_json_keys_are_rejected(self):
        traces = self.run_dir / "traces.jsonl"
        traces.write_text(traces.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "physical lines"):
            inspector.inspect_run(self.run_dir, self.output_dir)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
            inspector._decode('{"feedback_to_agent": false, "feedback_to_agent": true}')


if __name__ == "__main__":
    unittest.main()
