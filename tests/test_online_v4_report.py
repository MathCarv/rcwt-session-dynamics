"""Invented confirmation summaries plus FAKE gate; no real corpus or model use."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rcwt_agent_env import FAMILIES
from rcwt_analysis_v4 import analyze_v4, render_report_v4

SPEC = importlib.util.spec_from_file_location(
    "online_v4_report", Path(__file__).resolve().parents[1] / "tools/verify_online_v4_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def fake_rows(candidate=6):
    rows = []
    for index in range(32):
        for policy, successes in (("summary", 4), ("structured", candidate)):
            rows.append({"episode_id": f"EXPLICIT-FAKE-{index:03d}", "family": FAMILIES[index % 4],
                         "split": "test", "policy": policy, "steps": 8, "successes": successes,
                         "failures": {"wrong_decision": 8-successes}, "valid_actions": 8,
                         "unsafe_actions": 0, "unsafe_booked_cents": 0,
                         "prompt_tokens": 100, "completion_tokens": 20, "model_calls": 16 if policy == "structured" else 23,
                         "decision_seconds": [1.0]*8, "step_seconds": [2.0]*8,
                         "episode_seconds": 17.0, "memory_truncations": 0})
    return rows


def write_json(path, value):
    path.write_bytes(report._json_bytes(value))


class OnlineV4ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="explicit-fake-v4-report-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.template = cls.root / "template"
        cls.template.mkdir()
        cls.protocol = {
            "schema": "rcwt-online-memory/4", "mode": "confirmatory", "split": "test", "count": 32,
            "counts": {"test": 32}, "policies": ["summary", "structured"], "steps_per_episode": 8,
            "bootstrap_samples": 10000, "analysis_seed": 2026091207, "model": "EXPLICIT-FAKE-NOT-A-MODEL",
            "source_sha256": {"src/" + name: report._hash(report.ROOT / "src" / name) for name in report.SOURCE_NAMES},
        }
        cls.verification = {"status": "PASS", "verified_steps": 512,
                            "scope": "EXPLICIT FAKE display-test receipt, not inference evidence",
                            "limitations": ["This invented fixture has not replayed physical model calls."]}
        cls.rows = fake_rows()
        cls.analysis = analyze_v4(cls.rows)
        write_json(cls.template / "protocol.json", cls.protocol)
        write_json(cls.template / "completion.json", {"fixture": "EXPLICIT FAKE gate, no raw inference archive"})
        write_json(cls.template / "analysis.json", cls.analysis)
        write_json(cls.template / "verification.json", cls.verification)
        (cls.template / "episodes.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in cls.rows), encoding="utf-8", newline="\n")
        (cls.template / "RESULTS.md").write_bytes(render_report_v4(cls.analysis, cls.protocol, cls.verification).encode("utf-8"))
        cls.fake_helper = cls.root / "fake-archive-helper.py"
        cls.fake_helper.write_text("# EXPLICIT FAKE gate source for unit-test receipt hashing only\n", encoding="utf-8")

    def setUp(self):
        self.directory = self.root / self._testMethodName
        shutil.copytree(self.template, self.directory)
        self.gate_receipt = {"status": "PASS", "inference_calls": 0, "verification": copy.deepcopy(self.verification)}
        def fake_gate(directory):
            result = copy.deepcopy(self.gate_receipt)
            if isinstance(result, dict) and result.get("status") == "PASS":
                result["protocol_sha256"] = report._hash(directory / "protocol.json")
            return result
        self.mocks = {}
        for name, patcher in {
            "gate": patch.object(report, "verify_archive", side_effect=fake_gate),
            "helper": patch.object(report, "ARCHIVE_VERIFIER_PATH", self.fake_helper),
            "network": patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Real HTTP forbidden")),
            "model": patch("rcwt_local_model.LocalModelClient.__init__", side_effect=AssertionError("Model creation forbidden")),
        }.items():
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def verify(self):
        return report.verify_report(self.directory)

    def test_exact_positive_report_verifies_without_writes_or_inference(self):
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        receipt = self.verify()
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(receipt["improvement_gate_passed"])
        self.assertEqual(receipt["paired_episodes"], 32)
        self.assertEqual(receipt["verified_steps"], 512)
        self.assertEqual(receipt["inference_calls"], 0)
        self.assertEqual(receipt["writes"], 0)
        self.assertEqual(receipt["current_source_sha256"], self.protocol["source_sha256"])
        self.assertEqual(receipt["files_sha256"]["analysis.json"], report._hash(self.directory / "analysis.json"))
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.directory.iterdir()})
        self.mocks["gate"].assert_called_once_with(self.directory)
        self.mocks["network"].assert_not_called()
        self.mocks["model"].assert_not_called()

    def test_honest_negative_report_integrity_pass_does_not_claim_gain(self):
        rows = fake_rows(candidate=4)
        analysis = analyze_v4(rows)
        (self.directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        write_json(self.directory / "analysis.json", analysis)
        (self.directory / "RESULTS.md").write_bytes(render_report_v4(analysis, self.protocol, self.verification).encode("utf-8"))
        receipt = self.verify()
        self.assertEqual(receipt["status"], "PASS")
        self.assertFalse(receipt["accuracy_gate_passed"])
        self.assertFalse(receipt["improvement_gate_passed"])

    def test_gate_failure_precedes_report_artifact_reads_and_analysis(self):
        for gate in ({"status": "FAIL"}, {"status": "PASS", "inference_calls": 1}, None):
            self.gate_receipt = gate
            with self.subTest(gate=gate), patch.object(report, "_read") as read, \
                 patch.object(report, "_analysis_module") as analyze, self.assertRaises(ValueError):
                self.verify()
            read.assert_not_called()
            analyze.assert_not_called()

    def test_gate_exception_propagates_without_reading_report(self):
        self.mocks["gate"].side_effect = ValueError("FAKE archive tampering")
        with patch.object(report, "_read") as read, self.assertRaisesRegex(ValueError, "FAKE archive tampering"):
            self.verify()
        read.assert_not_called()

    def test_original_complete_512_step_verification_is_required(self):
        for verification in ({}, {"status": "FAIL", "verified_steps": 512}, {"status": "PASS", "verified_steps": 128}):
            self.gate_receipt["verification"] = verification
            with self.subTest(verification=verification), self.assertRaises(ValueError):
                self.verify()

    def test_archive_receipt_must_bind_the_current_protocol_bytes(self):
        self.mocks["gate"].side_effect = None
        self.mocks["gate"].return_value = {**self.gate_receipt, "protocol_sha256": "0"*64}
        with self.assertRaisesRegex(ValueError, "does not bind the current protocol"):
            self.verify()

    def test_only_fixed_v4_confirmation_protocol_is_accepted(self):
        for key, value in (("schema", "rcwt-online-memory/3"), ("mode", "development"), ("split", "train"),
                           ("count", 8), ("count", True), ("counts", {"test": 16}), ("steps_per_episode", 7),
                           ("bootstrap_samples", 999), ("analysis_seed", 123), ("policies", ["summary", "tail"])):
            write_json(self.directory / "protocol.json", {**self.protocol, key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.verify()

    def test_current_sources_must_match_all_fifteen_frozen_pins_before_analysis(self):
        for mutation in ("digest", "missing", "extra"):
            protocol = copy.deepcopy(self.protocol)
            if mutation == "digest":
                protocol["source_sha256"]["src/rcwt_analysis_v4.py"] = "0"*64
            elif mutation == "missing":
                del protocol["source_sha256"]["src/rcwt_context_v4.py"]
            else:
                protocol["source_sha256"]["src/unapproved.py"] = "0"*64
            write_json(self.directory / "protocol.json", protocol)
            with self.subTest(mutation=mutation), patch.object(report, "_analysis_module") as analyze, self.assertRaises(ValueError):
                self.verify()
            analyze.assert_not_called()

    def test_resealed_analysis_counts_cannot_replace_recomputed_statistics(self):
        analysis = copy.deepcopy(self.analysis)
        analysis["policies"]["structured"]["successes"] += 1
        write_json(self.directory / "analysis.json", analysis)
        completion = {"analysis_sha256": report._hash(self.directory / "analysis.json"), "fixture": "FAKE RESEALED COUNTS"}
        write_json(self.directory / "completion.json", completion)
        with self.assertRaisesRegex(ValueError, "analysis.json bytes"):
            self.verify()

    def test_tampered_markdown_and_verification_receipt_are_rejected(self):
        for name in ("RESULTS.md", "verification.json"):
            original = (self.directory / name).read_bytes()
            (self.directory / name).write_bytes(original + b"\nFAKE TAMPERING\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name.replace(".", r"\.")):
                self.verify()
            (self.directory / name).write_bytes(original)

    def test_equivalent_analysis_with_other_bytes_is_not_the_runner_output(self):
        (self.directory / "analysis.json").write_text(json.dumps(self.analysis), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "analysis.json bytes"):
            self.verify()

    def test_missing_or_changed_episode_summaries_cannot_match_saved_report(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["successes"] -= 1
        rows[1]["failures"]["wrong_decision"] += 1
        (self.directory / "episodes.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "analysis.json bytes"):
            self.verify()
        (self.directory / "episodes.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows[:-1]), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.verify()

    def test_cli_emits_receipt_only_to_stdout(self):
        output = io.StringIO()
        with patch.object(report, "verify_report", return_value={"status": "PASS", "writes": 0}) as verify, \
             patch("sys.argv", ["verify_online_v4_report.py", "--run-dir", "FAKE-CLI"]), redirect_stdout(output):
            report.main()
        verify.assert_called_once_with(Path("FAKE-CLI"))
        self.assertEqual(json.loads(output.getvalue()), {"status": "PASS", "writes": 0})


if __name__ == "__main__":
    unittest.main()
