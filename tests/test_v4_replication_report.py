"""Invented R1 summaries and FAKE replay gate; never generate a real test corpus."""
from __future__ import annotations

import copy
import importlib.util
import io
import importlib
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rcwt_agent_env import FAMILIES
from rcwt_analysis_v4 import analyze_v4, render_report_v4

SPEC = importlib.util.spec_from_file_location(
    "v4_replication_report", Path(__file__).resolve().parents[1] / "tools/verify_v4_replication_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def fake_rows(candidate=6):
    return [
        {"episode_id": f"EXPLICIT-FAKE-R1-{index:03d}", "family": FAMILIES[index % 4],
         "split": "test", "policy": policy, "steps": 8, "successes": successes,
         "failures": {"wrong_decision": 8 - successes}, "valid_actions": 8,
         "unsafe_actions": 0, "unsafe_booked_cents": 0,
         "prompt_tokens": 100, "completion_tokens": 20,
         "model_calls": 16 if policy == "structured" else 23,
         "decision_seconds": [1.0] * 8, "step_seconds": [2.0] * 8,
         "episode_seconds": 17.0, "memory_truncations": 0}
        for index in range(32)
        for policy, successes in (("summary", 4), ("structured", candidate))
    ]


def fake_render(analysis, protocol, verification):
    # Deliberately not the real R1 preamble: tests exercise the byte comparison
    # through an explicit stub, while preserving the frozen statistical renderer.
    return "# EXPLICIT FAKE R1 display fixture; no inference evidence\n\n" + render_report_v4(analysis, protocol, verification)


def write_json(path, value):
    path.write_bytes(report._json_bytes(value))


class V4ReplicationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="explicit-fake-r1-report-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.workspace = cls.root / "FAKE-WORKSPACE"
        (cls.workspace / "src").mkdir(parents=True)
        (cls.workspace / "tools").mkdir()
        (cls.workspace / "docs").mkdir()
        cls.development = cls.workspace / "development-protocol.json"
        shutil.copyfile(report.DEVELOPMENT_PROTOCOL, cls.development)
        cls.registration = cls.workspace / "docs/rcwt_v4_replication_protocol.md"
        shutil.copyfile(report.REGISTRATION, cls.registration)
        for name in report.SOURCE_NAMES:
            shutil.copyfile(report.ROOT / "src" / name, cls.workspace / "src" / name)
        for name in report.ORCHESTRATOR_NAMES:
            (cls.workspace / name).write_text("# EXPLICIT FAKE source pin for unit tests only\n", encoding="utf-8")
        cls.template = cls.root / "template"
        cls.template.mkdir()
        cls.protocol = {
            "schema": "rcwt-online-replication/1", "mode": "replication", "replication_id": "R1",
            "split": "test", "count": 32, "counts": {"test": 32},
            "policies": ["summary", "structured"], "steps_per_episode": 8,
            "dataset_seed": 2026091210, "inference_seed": 20260911, "schedule_seed": 2026091208,
            "analysis_seed": 2026091207, "bootstrap_samples": 10000,
            "memory_budget": 256, "max_action_tokens": 512, "actor_passes": 2,
            "development_revision": 2, "model": "EXPLICIT-FAKE-NOT-A-MODEL",
            "source_sha256": {"src/" + name: report._hash(cls.workspace / "src" / name) for name in report.SOURCE_NAMES},
            "orchestrator_sha256": {name: report._hash(cls.workspace / name) for name in report.ORCHESTRATOR_NAMES},
            "registration_sha256": report.REGISTRATION_SHA256,
        }
        write_json(cls.template / "protocol.json", cls.protocol)
        cls.verification = {
            "status": "PASS", "verified_steps": 512, "verified_episode_summaries": 64,
            "inference_calls": 0, "generation_calls": 1248, "read_only": True,
            "integrity_only": True, "gain": "NOT_EVALUATED",
            "protocol_sha256": report._hash(cls.template / "protocol.json"),
            "scope": "EXPLICIT FAKE replay gate for display unit tests only",
        }
        cls.rows = fake_rows()
        cls.analysis = analyze_v4(cls.rows)
        write_json(cls.template / "completion.json", {"fixture": "EXPLICIT FAKE; no raw model archive"})
        write_json(cls.template / "analysis.json", cls.analysis)
        write_json(cls.template / "verification.json", cls.verification)
        (cls.template / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in cls.rows), encoding="utf-8", newline="\n")
        (cls.template / "RESULTS.md").write_bytes(fake_render(cls.analysis, cls.protocol, cls.verification).encode("utf-8"))

    def setUp(self):
        self.directory = self.root / self._testMethodName
        shutil.copytree(self.template, self.directory)
        self.gate_receipt = copy.deepcopy(self.verification)

        def fake_gate(directory):
            receipt = copy.deepcopy(self.gate_receipt)
            if isinstance(receipt, dict) and receipt.get("status") == "PASS":
                receipt["protocol_sha256"] = report._hash(directory / "protocol.json")
            return receipt

        self.mocks = {}
        for name, patcher in {
            "root": patch.object(report, "ROOT", self.workspace),
            "development": patch.object(report, "DEVELOPMENT_PROTOCOL", self.development),
            "registration": patch.object(report, "REGISTRATION", self.registration),
            "gate": patch.object(report, "verify_run", side_effect=fake_gate),
            "validate": patch.object(report, "validate_protocol", side_effect=lambda directory: report._read(directory / "protocol.json")),
            "renderer": patch.object(report, "_runner_module", return_value=SimpleNamespace(render_report=fake_render)),
            "network": patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Real HTTP forbidden")),
            "model": patch("rcwt_local_model.LocalModelClient.__init__", side_effect=AssertionError("Model creation forbidden")),
        }.items():
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def verify(self):
        return report.verify_report(self.directory)

    def replace_rows(self, rows):
        analysis = analyze_v4(rows)
        (self.directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        write_json(self.directory / "analysis.json", analysis)
        (self.directory / "RESULTS.md").write_bytes(fake_render(analysis, self.protocol, self.verification).encode("utf-8"))

    def test_exact_report_verifies_read_only_and_separates_gain(self):
        before = report._snapshot(self.directory)
        receipt = self.verify()
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(receipt["integrity_only"])
        self.assertTrue(receipt["improvement_gate_passed"])
        self.assertEqual(receipt["paired_episodes"], 32)
        self.assertEqual(receipt["verified_steps"], 512)
        self.assertEqual(receipt["recorded_generation_calls"], 1248)
        self.assertEqual(receipt["inference_calls"], 0)
        self.assertEqual(receipt["writes"], 0)
        self.assertEqual(receipt["files_sha256"], before)
        self.assertEqual(before, report._snapshot(self.directory))
        self.mocks["gate"].assert_called_once_with(self.directory)
        self.mocks["network"].assert_not_called()
        self.mocks["model"].assert_not_called()

    def test_honest_negative_report_passes_integrity_without_claiming_gain(self):
        self.replace_rows(fake_rows(candidate=4))
        receipt = self.verify()
        self.assertEqual(receipt["status"], "PASS")
        self.assertFalse(receipt["accuracy_gate_passed"])
        self.assertTrue(receipt["descriptive_safety_guard_passed"])
        self.assertFalse(receipt["improvement_gate_passed"])

    def test_safety_failure_is_reported_even_when_accuracy_passes(self):
        rows = fake_rows()
        rows[1]["unsafe_actions"] = 1
        rows[1]["unsafe_booked_cents"] = 17
        self.replace_rows(rows)
        receipt = self.verify()
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(receipt["accuracy_gate_passed"])
        self.assertFalse(receipt["descriptive_safety_guard_passed"])
        self.assertFalse(receipt["improvement_gate_passed"])

    def test_missing_directory_fails_without_inference_or_creation(self):
        directory = self.root / "does-not-exist"
        with self.assertRaisesRegex(ValueError, "existing"):
            report.verify_report(directory)
        self.assertFalse(directory.exists())
        self.mocks["gate"].assert_not_called()

    def test_complete_read_only_replay_is_required_before_analysis(self):
        changes = [None, {"status": "FAIL"}]
        changes += [{**self.verification, key: value} for key, value in (
            ("verified_steps", 173), ("verified_steps", 512.0), ("verified_episode_summaries", 63),
            ("generation_calls", 421), ("inference_calls", 1), ("inference_calls", False),
            ("read_only", False), ("integrity_only", False), ("gain", "PASS"))]
        for receipt in changes:
            self.gate_receipt = receipt
            with self.subTest(receipt=receipt), patch.object(report, "_analysis_module") as analyze, self.assertRaises(ValueError):
                self.verify()
            analyze.assert_not_called()

    def test_gate_exception_propagates_before_analysis(self):
        self.mocks["gate"].side_effect = ValueError("EXPLICIT FAKE broken replay")
        with patch.object(report, "_analysis_module") as analyze, self.assertRaisesRegex(ValueError, "broken replay"):
            self.verify()
        analyze.assert_not_called()

    def test_receipt_must_bind_original_protocol_bytes(self):
        self.mocks["gate"].side_effect = None
        self.mocks["gate"].return_value = {**self.verification, "protocol_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "original protocol bytes"):
            self.verify()

    def test_protocol_validator_cannot_change_inputs(self):
        self.mocks["validate"].side_effect = None
        self.mocks["validate"].return_value = {**self.protocol, "count": 8}
        with self.assertRaisesRegex(ValueError, "differs from the original"):
            self.verify()

    def test_only_registered_r1_parameters_are_accepted(self):
        mutations = (
            ("schema", "rcwt-online-memory/4"), ("mode", "confirmatory"), ("replication_id", "R2"),
            ("split", "train"), ("count", 8), ("count", True), ("counts", {"test": 16}),
            ("steps_per_episode", 7), ("dataset_seed", 2026091206), ("dataset_seed", 2026091205),
            ("inference_seed", 20260912), ("schedule_seed", 2026091209), ("analysis_seed", 123),
            ("bootstrap_samples", 999), ("memory_budget", 512), ("max_action_tokens", 256),
            ("actor_passes", 1), ("development_revision", 1), ("registration_sha256", "0" * 64),
            ("policies", ["summary", "tail"]),
        )
        for key, value in mutations:
            write_json(self.directory / "protocol.json", {**self.protocol, key: value})
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "fixed R1 protocol"):
                self.verify()
        self.mocks["gate"].assert_not_called()

    def test_all_fifteen_candidate_pins_must_match_attempt_02(self):
        for mutation in ("digest", "missing", "extra"):
            protocol = copy.deepcopy(self.protocol)
            if mutation == "digest":
                protocol["source_sha256"]["src/rcwt_analysis_v4.py"] = "0" * 64
            elif mutation == "missing":
                del protocol["source_sha256"]["src/rcwt_context_v4.py"]
            else:
                protocol["source_sha256"]["src/unapproved.py"] = "0" * 64
            write_json(self.directory / "protocol.json", protocol)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.verify()
        self.mocks["gate"].assert_not_called()

    def test_development_and_registration_bytes_are_pinned(self):
        for constant, source in (("DEVELOPMENT_PROTOCOL", self.development), ("REGISTRATION", self.registration)):
            changed = self.directory / (constant + ".fake")
            changed.write_bytes(source.read_bytes() + b"\n")
            with self.subTest(constant=constant), patch.object(report, constant, changed), self.assertRaises(ValueError):
                self.verify()
        self.mocks["gate"].assert_not_called()

    def test_orchestration_sources_are_exactly_pinned(self):
        for pins in ({}, {**self.protocol["orchestrator_sha256"], "extra.py": "0" * 64},
                     {name: "0" * 64 for name in report.ORCHESTRATOR_NAMES}):
            write_json(self.directory / "protocol.json", {**self.protocol, "orchestrator_sha256": pins})
            with self.subTest(pins=pins), self.assertRaisesRegex(ValueError, "orchestration"):
                self.verify()
        self.mocks["gate"].assert_not_called()

    def test_resealed_analysis_does_not_replace_recomputed_statistics(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["policies"]["structured"]["successes"] += 1
        write_json(self.directory / "analysis.json", analysis)
        write_json(self.directory / "completion.json", {"analysis_sha256": report._hash(self.directory / "analysis.json")})
        with self.assertRaisesRegex(ValueError, "analysis.json bytes"):
            self.verify()

    def test_markdown_and_saved_replay_receipt_require_exact_bytes(self):
        for name in ("RESULTS.md", "verification.json"):
            original = (self.directory / name).read_bytes()
            (self.directory / name).write_bytes(original + b"\nEXPLICIT FAKE TAMPERING\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name.replace(".", r"\.")):
                self.verify()
            (self.directory / name).write_bytes(original)

    def test_semantically_equivalent_json_with_different_bytes_is_rejected(self):
        (self.directory / "analysis.json").write_text(json.dumps(self.analysis), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "analysis.json bytes"):
            self.verify()

    def test_missing_and_altered_episode_summaries_fail(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["successes"] -= 1
        rows[1]["failures"]["wrong_decision"] += 1
        for content in (rows, rows[:-1]):
            (self.directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in content), encoding="utf-8")
            with self.subTest(count=len(content)), self.assertRaises(ValueError):
                self.verify()

    def test_recompute_uses_the_exact_registered_statistical_parameters(self):
        with patch.object(report, "_analysis_module", return_value=SimpleNamespace(analyze_v4=analyze_v4)) as module:
            with patch.object(module.return_value, "analyze_v4", wraps=analyze_v4) as analyze:
                self.verify()
        analyze.assert_called_once_with(self.rows, expected_count=32, bootstrap_samples=10000, seed=2026091207)

    def test_before_after_guard_detects_an_added_evidence_file(self):
        def changed_snapshot(directory):
            (directory / "changed-during-verification.txt").write_text("EXPLICIT FAKE concurrent write", encoding="utf-8")
            return copy.deepcopy(self.protocol)
        self.mocks["validate"].side_effect = changed_snapshot
        with self.assertRaisesRegex(ValueError, "changed during verification"):
            self.verify()

    def test_before_after_guard_includes_original_inputs_during_replay(self):
        def changed_gate(directory):
            write_json(directory / "completion.json", {"fixture": "EXPLICIT FAKE concurrent completion replacement"})
            return copy.deepcopy(self.verification)
        self.mocks["gate"].side_effect = changed_gate
        with self.assertRaisesRegex(ValueError, "changed during verification"):
            self.verify()

    def test_source_guard_rechecks_candidate_after_recompute(self):
        changed = self.workspace / "src/rcwt_analysis_v4.py"
        original = changed.read_bytes()
        self.addCleanup(changed.write_bytes, original)
        def changed_analysis(*args, **kwargs):
            result = analyze_v4(*args, **kwargs)
            changed.write_bytes(original + b"\n# EXPLICIT FAKE concurrent source edit\n")
            return result
        with patch.object(report, "_analysis_module", return_value=SimpleNamespace(analyze_v4=changed_analysis)), \
                self.assertRaisesRegex(ValueError, "source drift"):
            self.verify()

    def test_cli_emits_only_receipt_to_stdout(self):
        output = io.StringIO()
        with patch.object(report, "verify_report", return_value={"status": "PASS", "writes": 0}) as verify, \
                patch("sys.argv", ["verify_v4_replication_report.py", "--run-dir", "EXPLICIT-FAKE-CLI"]), \
                patch("sys.dont_write_bytecode", False), redirect_stdout(output):
            report.main()
        verify.assert_called_once_with(Path("EXPLICIT-FAKE-CLI"))
        self.assertEqual(json.loads(output.getvalue()), {"status": "PASS", "writes": 0})

    def test_real_r1_renderer_reuses_frozen_report_and_discloses_prior_costs(self):
        runner = importlib.import_module("rcwt_replication_v4")
        protocol = {**self.protocol, "historical_evidence": {"interrupted": {"verification": {
            "recorded_completed_step_resources": {"generation_calls": 7, "prompt_tokens": 11,
                "completion_tokens": 13, "inference_seconds": 17.0}}}}}
        with patch.object(runner.core, "generate_episodes", side_effect=AssertionError("Real corpus generation forbidden")):
            rendered = runner.render_report(self.analysis, protocol, self.verification)
        self.assertTrue(rendered.startswith("# R1: newly authorized confirmation"))
        self.assertTrue(rendered.endswith(render_report_v4(self.analysis, protocol, self.verification)))
        self.assertIn("7 persisted completed calls; 11 prompt tokens and 13 completion tokens", rendered)
        self.assertIn("173/512", rendered)
        self.assertIn("One additional canceled server task", rendered)
        self.assertIn("does not prove the population gain exceeds 10 points", rendered)
        self.assertEqual(runner.json_bytes(self.verification), report._json_bytes(self.verification))

    def test_runner_fixed_contract_matches_independent_report_requirements(self):
        runner = importlib.import_module("rcwt_replication_v4")
        with patch.object(runner.core, "generate_episodes", side_effect=AssertionError("Real corpus generation forbidden")):
            report._fixed_protocol(runner._fixed_contract())
        self.assertEqual(set(runner.ORCHESTRATORS), set(report.ORCHESTRATOR_NAMES))


if __name__ == "__main__":
    unittest.main()
