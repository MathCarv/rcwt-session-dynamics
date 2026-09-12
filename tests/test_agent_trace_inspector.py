"""Explicitly synthetic offline inspector fixtures; never real-model evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import rcwt_agent_run as runner
from rcwt_agent_actor import extract_action, extract_evidence_check
from rcwt_agent_env import Action, Episode, OracleSnapshot, Simulator, Step, expected_action, generate_episodes
from test_rcwt_agent_pipeline_smoke import OfflineTransport
from tools import inspect_agent_traces as inspector


def _fixture_episode():
    snapshots = (
        OracleSnapshot("case-A", "payout", 100, "active", True, "pending", None),
        OracleSnapshot("case-B", "payout", 200, "active", True, "cleared", None),
        OracleSnapshot("case-B", "payout", 200, "active", True, "cleared", None),
        OracleSnapshot("case-B", "refund", 200, "active", True, "cleared", "accepted"),
    )
    steps = tuple(Step(index, ({"fixture": "explicit synthetic test", "state": snapshot.to_dict()},),
                       {"case_id": snapshot.case_id, "operation": snapshot.operation})
                  for index, snapshot in enumerate(snapshots))
    return Episode("synthetic-test-episode", "test-fixture-only", "test", 31, steps, snapshots)


def _traces(episode, overrides=None):
    overrides = overrides or {}
    traces = []
    for policy in ("tail", "summary", "learned"):
        simulator = Simulator(episode)
        for index, snapshot in enumerate(episode.oracle_steps):
            public = simulator.public_step(index)
            check = inspector.reference_check(snapshot, set(simulator.completed))
            action = expected_action(snapshot, set(simulator.completed))
            override = overrides.get((policy, index), {})
            check.update(override.get("check", {}))
            action = override.get("action", action)
            text = json.dumps({"evidence_check": check, **action.to_dict()})
            text = override.get("text", text)
            finish = override.get("finish_reason", "stop")
            extracted = extract_action(text, finish)
            execution = simulator.execute_action(index, extracted)
            traces.append({
                "episode_id": episode.episode_id, "family": episode.family, "split": "test",
                "policy": policy, "step_index": index, "public_step": public,
                "memory_before": "Synthetic retained memory; no inference was performed.",
                "action": extracted, "evidence_check": extract_evidence_check(text, finish),
                "score": execution.score.to_dict(), "tool_result": execution.tool_result,
                "state_after": execution.state,
                "model_calls": [{"text": text, "finish_reason": finish}],
                "sha256": "f" * 64,
            })
    return traces


def _diagnose(episode, overrides=None):
    return inspector.diagnose_records(_traces(episode, overrides), [episode.to_public_dict()], [episode.to_oracle_dict()])


class TraceDiagnosticCoreTests(unittest.TestCase):
    def test_matching_facts_and_wrong_action_is_rule_mismatch_not_memory_label(self):
        episode = _fixture_episode()
        result = _diagnose(episode, {
            ("summary", 0): {"action": Action("case-A", "approve", 100, "authorized_payout")},
        })
        decision = next(item for item in result["decisions"] if item["policy"] == "summary" and item["step_index"] == 0)
        self.assertEqual(decision["category"], "failed_matching_reference_check")
        self.assertTrue(decision["check_matches_reference"])
        self.assertFalse(decision["action_consistent_with_claimed_evidence"])
        self.assertEqual(decision["field_deltas"], [])
        self.assertFalse(result["causal_memory_attribution"])
        self.assertFalse(result["feedback_to_agent"])

    def test_divergent_check_can_produce_an_action_consistent_with_claimed_facts(self):
        result = _diagnose(_fixture_episode(), {
            ("summary", 0): {"check": {"payment_status": "cleared"},
                             "action": Action("case-A", "approve", 100, "authorized_payout")},
        })
        decision = next(item for item in result["decisions"] if item["policy"] == "summary" and item["step_index"] == 0)
        self.assertEqual(decision["category"], "failed_divergent_check")
        self.assertTrue(decision["action_consistent_with_claimed_evidence"])
        self.assertEqual(decision["field_deltas"], [{"field": "payment_status", "reference": "pending", "claimed": "cleared"}])

    def test_correct_action_with_divergent_irrelevant_field_is_reported(self):
        result = _diagnose(_fixture_episode(), {("summary", 0): {"check": {"return_status": "accepted"}}})
        decision = next(item for item in result["decisions"] if item["policy"] == "summary" and item["step_index"] == 0)
        self.assertEqual(decision["category"], "correct_divergent_check")
        self.assertTrue(decision["success"])
        self.assertTrue(decision["action_consistent_with_claimed_evidence"])

    def test_actual_pre_action_ledger_is_separate_from_claimed_ledger_and_operation(self):
        result = _diagnose(_fixture_episode(), {
            ("summary", 2): {"check": {"operation_already_booked": "no_record"},
                             "action": Action("case-B", "approve", 200, "authorized_payout")},
        })
        summary = [item for item in result["decisions"] if item["policy"] == "summary"]
        self.assertEqual(summary[1]["reference_check"]["operation_already_booked"], "no_record")
        self.assertEqual(summary[2]["reference_check"]["operation_already_booked"], "yes")
        self.assertEqual(summary[2]["expected_action_from_reference"]["arguments"]["reason_code"], "already_completed")
        self.assertEqual(summary[2]["expected_action_from_claimed_evidence"]["arguments"]["decision"], "approve")
        self.assertTrue(summary[2]["action_consistent_with_claimed_evidence"])
        self.assertEqual(summary[3]["reference_check"]["operation_already_booked"], "no_record")
        self.assertEqual(summary[3]["expected_action_from_reference"]["arguments"]["decision"], "refund")

    def test_claimed_unknowns_do_not_fall_back_to_reference_facts(self):
        result = _diagnose(_fixture_episode(), {
            ("summary", 1): {"check": {"invoice_amount_cents": None},
                             "action": Action("case-B", "ask_info", 0, "missing_evidence")},
        })
        decision = next(item for item in result["decisions"] if item["policy"] == "summary" and item["step_index"] == 1)
        self.assertEqual(decision["expected_action_from_claimed_evidence"]["arguments"]["decision"], "ask_info")
        self.assertEqual(decision["expected_action_from_reference"]["arguments"]["decision"], "approve")

    def test_invalid_and_truncated_envelopes_are_not_evaluable(self):
        result = _diagnose(_fixture_episode(), {
            ("summary", 0): {"text": "not JSON"},
            ("summary", 1): {"finish_reason": "length"},
        })
        summary = result["policy_summaries"]["summary"]
        self.assertEqual(summary["category_counts"]["invalid_envelope"], 2)
        self.assertEqual(summary["action_consistency_not_evaluable"], 2)
        decisions = [item for item in result["decisions"] if item["category"] == "invalid_envelope"]
        self.assertTrue(all(item["expected_action_from_claimed_evidence"] is None for item in decisions))

    def test_examples_are_deterministic_and_paired_outcomes_include_both_directions(self):
        episode = _fixture_episode()
        overrides = {
            ("tail", 0): {"check": {"payment_status": "cleared"}, "action": Action("case-A", "approve", 100, "authorized_payout")},
            ("summary", 0): {"action": Action("case-A", "approve", 100, "authorized_payout")},
            ("learned", 1): {"action": Action("case-B", "ask_info", 0, "missing_evidence")},
        }
        result = _diagnose(episode, overrides)
        self.assertEqual(result, _diagnose(episode, overrides))
        self.assertEqual(len(result["examples"]), 3)
        self.assertGreater(result["paired_disagreements"]["learned_correct_summary_incorrect"], 0)
        self.assertGreater(result["paired_disagreements"]["summary_correct_learned_incorrect"], 0)
        self.assertTrue(result["examples"][2]["opposite_direction_decision_ids"])
        report = inspector.render_report(result, "../fake-run")
        for phrase in ("post-hoc", "not independent samples", "reference", "claimed", "../fake-run/test/traces.jsonl#L"):
            self.assertIn(phrase.lower(), report.lower())

    def test_core_does_not_mutate_inputs_and_rejects_unpaired_or_other_split_data(self):
        episode = _fixture_episode()
        traces = _traces(episode)
        before = copy.deepcopy(traces)
        inspector.diagnose_records(traces, [episode.to_public_dict()], [episode.to_oracle_dict()])
        self.assertEqual(traces, before)
        with self.assertRaises(ValueError):
            inspector.diagnose_records(traces[:-1], [episode.to_public_dict()], [episode.to_oracle_dict()])
        with self.assertRaises(ValueError):
            inspector.diagnose_records(traces, [replace(episode, split="train").to_public_dict()], [episode.to_oracle_dict()])


class TraceInspectorFileGateTests(unittest.TestCase):
    def test_missing_freeze_blocks_before_verification_or_diagnostic_file_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "unstarted"
            run.mkdir()
            with patch.object(inspector, "verify_evidence") as verify, patch.object(inspector, "_read_jsonl") as read:
                with self.assertRaisesRegex(ValueError, "freeze is absent"):
                    inspector.inspect_run(run, Path(temporary) / "output")
            verify.assert_not_called()
            read.assert_not_called()

    def test_incomplete_test_or_failed_verification_blocks_diagnostic_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "fake-run"
            run.mkdir()
            (run / "test-freeze.json").write_text("{}", encoding="utf-8")
            with patch.object(inspector, "verify_evidence") as verify:
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    inspector.inspect_run(run, Path(temporary) / "output")
                verify.assert_not_called()
            (run / "test").mkdir()
            (run / "test/completion.json").write_text("{}", encoding="utf-8")
            with patch.object(inspector, "verify_evidence", return_value={"status": "BLOCKED"}), patch.object(inspector, "_read_jsonl") as read:
                with self.assertRaisesRegex(ValueError, "must PASS"):
                    inspector.inspect_run(run, Path(temporary) / "output")
                read.assert_not_called()

    def test_verified_fake_pipeline_writes_provenance_and_does_not_change_inputs(self):
        with tempfile.TemporaryDirectory() as temporary, patch("builtins.print"):
            run = Path(temporary) / "explicit-fake-run"
            output = Path(temporary) / "post-hoc-output"
            args = SimpleNamespace(train_count=4, validation_count=4, test_count=4, seed=31,
                                   memory_budget=256, model="rcwt-local-qwen35-4b",
                                   runtime_receipt=runner.ROOT / "docs/rcwt_agent_runtime.json")
            protocol = runner.prepare(run, args)
            client = OfflineTransport(seed=args.seed)
            runner.run_cohort(client, generate_episodes("train", 4, args.seed),
                              [runner.Policy("summary", "summary")], 256, run / "train", args.seed)
            runner.validate_stage(client, run, protocol)
            runner.test_stage(client, run, protocol)
            input_hash = hashlib.sha256((run / "test/traces.jsonl").read_bytes()).hexdigest()
            calls_before = len(client.calls)
            result = inspector.inspect_run(run, output)
            self.assertEqual(result["verification"]["status"], "PASS")
            self.assertEqual(len(result["decisions"]), 96)
            self.assertEqual(len(client.calls), calls_before)
            self.assertEqual(result["provenance"]["files_sha256"]["test/traces.jsonl"], input_hash)
            self.assertEqual(hashlib.sha256((run / "test/traces.jsonl").read_bytes()).hexdigest(), input_hash)
            self.assertEqual(result["provenance"]["inspector_sha256"], hashlib.sha256(Path(inspector.__file__).read_bytes()).hexdigest())
            self.assertEqual(json.loads((output / "diagnosis.json").read_text(encoding="utf-8")), result)
            self.assertTrue((output / "DIAGNOSIS.md").is_file())
            self.assertIn("../explicit-fake-run/test/traces.jsonl#L", (output / "DIAGNOSIS.md").read_text(encoding="utf-8"))
            artifact_bytes = [(output / name).read_bytes() for name in ("diagnosis.json", "DIAGNOSIS.md")]
            verified = inspector.verify_diagnosis(run, output)
            self.assertEqual(verified["status"], "PASS")
            self.assertTrue(verified["read_only"])
            self.assertEqual(verified["decisions"], 96)
            self.assertEqual(artifact_bytes, [(output / name).read_bytes() for name in ("diagnosis.json", "DIAGNOSIS.md")])
            with self.assertRaises(FileExistsError):
                inspector.inspect_run(run, output)


class SavedDiagnosisVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.run_dir = Path(cls.temporary.name) / "explicit-fake-diagnosis-verification"
        cls.output = Path(cls.temporary.name) / "diagnosis"
        args = SimpleNamespace(train_count=4, validation_count=4, test_count=4, seed=31,
                               memory_budget=256, model="rcwt-local-qwen35-4b",
                               runtime_receipt=runner.ROOT / "docs/rcwt_agent_runtime.json")
        with patch("builtins.print"):
            protocol = runner.prepare(cls.run_dir, args)
            client = OfflineTransport(seed=args.seed)
            runner.run_cohort(client, generate_episodes("train", 4, args.seed),
                              [runner.Policy("summary", "summary")], 256, cls.run_dir / "train", args.seed)
            runner.validate_stage(client, cls.run_dir, protocol)
            runner.test_stage(client, cls.run_dir, protocol)
            inspector.inspect_run(cls.run_dir, cls.output)
        cls.original_json = (cls.output / "diagnosis.json").read_bytes()
        cls.original_markdown = (cls.output / "DIAGNOSIS.md").read_bytes()

    def setUp(self):
        (self.output / "diagnosis.json").write_bytes(self.original_json)
        (self.output / "DIAGNOSIS.md").write_bytes(self.original_markdown)

    def _save(self, diagnosis, rerender=False):
        (self.output / "diagnosis.json").write_text(json.dumps(diagnosis, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        if rerender:
            (self.output / "DIAGNOSIS.md").write_text(inspector.render_report(diagnosis, inspector._relative_run(self.run_dir, self.output)), encoding="utf-8", newline="\n")

    def test_changed_decision_fact_is_rejected(self):
        diagnosis = json.loads(self.original_json)
        diagnosis["decisions"][0]["reference_check"]["invoice_amount_cents"] = 999999999
        self._save(diagnosis)
        with self.assertRaisesRegex(ValueError, "facts or provenance"):
            inspector.verify_diagnosis(self.run_dir, self.output)

    def test_resealed_summary_and_consistent_markdown_cannot_replace_recomputed_counts(self):
        diagnosis = json.loads(self.original_json)
        diagnosis["policy_summaries"]["learned"]["exact_action_successes"] += 1
        self._save(diagnosis, rerender=True)
        with self.assertRaisesRegex(ValueError, "facts or provenance"):
            inspector.verify_diagnosis(self.run_dir, self.output)

    def test_resealed_input_manifest_with_wrong_hash_is_rejected(self):
        diagnosis = json.loads(self.original_json)
        provenance = diagnosis["provenance"]
        provenance["files_sha256"]["test/traces.jsonl"] = "0" * 64
        provenance["input_manifest_sha256"] = inspector._canonical_hash(provenance["files_sha256"])
        self._save(diagnosis, rerender=True)
        with self.assertRaisesRegex(ValueError, "facts or provenance"):
            inspector.verify_diagnosis(self.run_dir, self.output)

    def test_changed_markdown_is_rejected_without_overwriting_it(self):
        changed = self.original_markdown + b"\nUnrecorded result claim.\n"
        (self.output / "DIAGNOSIS.md").write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "deterministic rendering"):
            inspector.verify_diagnosis(self.run_dir, self.output)
        self.assertEqual((self.output / "DIAGNOSIS.md").read_bytes(), changed)

    def test_observed_timestamp_is_validated_not_remeasured(self):
        diagnosis = json.loads(self.original_json)
        diagnosis["generated_at_utc"] = "2020-01-01T00:00:00Z"
        self._save(diagnosis)
        self.assertEqual(inspector.verify_diagnosis(self.run_dir, self.output)["status"], "PASS")
        for invalid in ("not a date", "2020-01-01T00:00:00", "2020-01-01T00:00:00-03:00", None):
            diagnosis["generated_at_utc"] = invalid
            self._save(diagnosis)
            with self.subTest(timestamp=invalid), self.assertRaises(ValueError):
                inspector.verify_diagnosis(self.run_dir, self.output)

    def test_boolean_integer_substitution_and_duplicate_json_fields_fail_closed(self):
        diagnosis = json.loads(self.original_json)
        diagnosis["post_hoc"] = 1
        self._save(diagnosis)
        with self.assertRaisesRegex(ValueError, "facts or provenance"):
            inspector.verify_diagnosis(self.run_dir, self.output)
        original = self.original_json.decode("utf-8")
        (self.output / "diagnosis.json").write_text(original.replace('"post_hoc": true,', '"post_hoc": false, "post_hoc": true,', 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
            inspector.verify_diagnosis(self.run_dir, self.output)

    def test_verify_requires_core_pass_and_cli_does_not_route_to_writer(self):
        with patch.object(inspector, "verify_evidence", return_value={"status": "BLOCKED"}), patch.object(inspector, "_saved_diagnosis") as read:
            with self.assertRaisesRegex(ValueError, "must PASS"):
                inspector.verify_diagnosis(self.run_dir, self.output)
            read.assert_not_called()
        with patch.object(inspector.sys, "argv", ["inspect_agent_traces.py", "--run-dir", str(self.run_dir), "--output-dir", str(self.output), "--verify"]), patch.object(inspector, "verify_diagnosis", return_value={"status": "PASS"}) as verify, patch.object(inspector, "inspect_run") as write, patch("builtins.print"):
            inspector.main()
            verify.assert_called_once_with(self.run_dir, self.output)
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
