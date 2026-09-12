"""Invented plumbing fixtures, plus read-only replay of the published TRAIN run.

Fake receipts below are explicitly not model-quality evidence. Only the final
integration test opens the already-completed, public development attempt_01;
it recomputes existing evidence offline, never calls a model or opens held-out.
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "historical_diagnosis_v4", PROJECT / "tools/verify_online_v4_diagnosis_archive.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)
archive = report.archive


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + "\n").encode())


class HistoricalV4DiagnosisUnitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="explicit-fake-diagnosis-archive-v4-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.run = self.root / "results/EXPLICIT-FAKE-development"
        self.diagnosis = self.run / "diagnosis_verified"
        self.sources = {name: ("# EXPLICIT FAKE OLD SOURCE: " + name + "\n").encode() for name in archive.NAMES}
        source_hashes = {"src/" + name: archive._digest(data) for name, data in self.sources.items()}
        for name in archive.INPUTS:
            _write(self.run / name, b"EXPLICIT FAKE ARTIFACT; NOT MODEL EVIDENCE\n")
        self.protocol = {"schema": "rcwt-online-memory/4", "mode": "development", "split": "train",
                         "count": 8, "development": None, "source_sha256": source_hashes}
        _write(self.run / "protocol.json", self.protocol)
        _write(self.run / "schedule.json", [])
        self.protocol_hash = archive.digest(self.run / "protocol.json")
        _write(self.run / "freeze.json", {"protocol_sha256": self.protocol_hash,
                                          "schedule_sha256": archive.digest(self.run / "schedule.json")})
        _write(self.run / "started.json", {"protocol_sha256": self.protocol_hash})
        _write(self.run / "completion.json", {"protocol_sha256": self.protocol_hash,
                                              "traces_sha256": archive.digest(self.run / "traces.jsonl"),
                                              "episodes_sha256": archive.digest(self.run / "episodes.jsonl")})
        for name, data in self.sources.items():
            _write(self.run / "sources" / name, data)
            _write(self.root / "src" / name, b"# CHANGED CURRENT SOURCE; MUST NOT BE USED\n")
        self.tools = {name: ("# EXPLICIT FAKE ARCHIVED TOOL: " + name + "\n").encode() for name in report.TOOLS}
        self.saved = {"schema_version": "rcwt-online-trace-diagnosis/4", "fixture": "EXPLICIT FAKE, NO MODEL",
                      "provenance": {"frozen_sources_sha256": source_hashes}}
        self.refresh_diagnosis()
        for name in (*archive.HISTORY.values(), "results/agent_v2/protocol.json", "docs/rcwt_agent_runtime.json"):
            _write(self.root / name, b"EXPLICIT FAKE HISTORY; GATE IS MOCKED IN UNIT TESTS\n")
        self.gate_receipt = {"status": "PASS", "inference_calls": 0, "protocol_sha256": self.protocol_hash,
                             "fixture": "EXPLICIT FAKE gate, not inference evidence"}
        self.patches = {}
        for name, patcher in {
            "root": patch.object(archive, "ROOT", self.root),
            "gate": patch.object(archive, "verify_archive", side_effect=lambda _: copy.deepcopy(self.gate_receipt)),
            "subprocess": patch.object(report.subprocess, "run", side_effect=self.fake_inspection),
        }.items():
            self.patches[name] = patcher
            setattr(self, name + "_mock", patcher.start())
            self.addCleanup(patcher.stop)

    def refresh_diagnosis(self):
        for name, data in self.tools.items():
            _write(self.diagnosis / "sources" / name, data)
        pins = {"tools/" + name: archive._digest(data) for name, data in self.tools.items()}
        inputs = {name: archive.digest(self.run / name) for name in archive.INPUTS}
        self.saved["provenance"].update({"inspector_path": "tools/inspect_online_v4.py",
            "inspector_sha256": pins["tools/inspect_online_v4.py"], "diagnostic_sources_sha256": pins,
            "files_sha256": inputs, "input_manifest_sha256": archive._digest(archive._canonical(inputs).encode())})
        _write(self.diagnosis / "diagnosis.json", self.saved)
        _write(self.diagnosis / "DIAGNOSIS.md", b"# EXPLICIT FAKE diagnosis\n[run](../protocol.json)\n")

    def fake_inspection(self, command, **kwargs):
        self.assertEqual(command[:5], [sys.executable, "-I", "-B", "-c", report._VERIFY_ONLY])
        self.assertEqual(command[6], "--run-dir")
        self.assertEqual(command[8], "--output-dir")
        self.assertEqual(command[10:], ["--verify"])
        workspace, copied_run, copied_diag = Path(kwargs["cwd"]), Path(command[7]), Path(command[9])
        self.assertEqual(copied_run, workspace / "results/archived-v4-run")
        self.assertEqual(copied_diag.parent, copied_run)
        self.assertEqual(copied_diag.name, self.diagnosis.name)
        self.assertNotEqual(copied_run, self.run)
        self.assertNotIn("PYTHONPATH", kwargs["env"])
        self.assertNotIn("PYTHONHOME", kwargs["env"])
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        self.assertFalse(kwargs["check"])
        self.assertLessEqual(kwargs["timeout"], 60)
        for name, data in self.sources.items():
            self.assertEqual((workspace / "src" / name).read_bytes(), data)
            self.assertEqual((copied_run / "sources" / name).read_bytes(), data)
        for name, data in self.tools.items():
            self.assertEqual((workspace / "tools" / name).read_bytes(), data)
        for name in (*archive.HISTORY.values(), "results/agent_v2/protocol.json", "docs/rcwt_agent_runtime.json"):
            self.assertEqual((workspace / name).read_bytes(), (self.root / name).read_bytes())
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "status": "PASS", "read_only": True, "feedback_to_agent": False, "n_episodes": 8, "decisions": 128,
            "diagnosis_sha256": archive.digest(copied_diag / "diagnosis.json"),
            "report_sha256": archive.digest(copied_diag / "DIAGNOSIS.md"),
            "scope": "EXPLICIT FAKE inspection receipt; not model evidence"}))

    def verify(self):
        return report.verify_diagnosis_archive(self.run, self.diagnosis)

    def test_old_sources_and_tools_used_with_portable_links_no_original_writes(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.dict(os.environ, {"PYTHONPATH": "BAD-PATH", "PYTHONHOME": "BAD-HOME"}):
            receipt = self.verify()
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["verification"]["decisions"], 128)
        self.assertEqual(receipt["frozen_sources_sha256"], self.protocol["source_sha256"])
        self.assertEqual(receipt["inference_calls"], 0)
        self.assertEqual(receipt["original_writes"], 0)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.gate_mock.assert_called_once_with(self.run)
        self.subprocess_mock.assert_called_once()

    def test_gate_failure_precedes_all_diagnostic_reads_or_execution(self):
        for value in (None, {}, {"status": "FAIL"}, {"status": "PASS", "inference_calls": 1},
                      {"status": "PASS", "inference_calls": False}):
            self.gate_receipt = value
            with self.subTest(value=value), patch.object(archive, "_bundle") as bundle, self.assertRaises(ValueError):
                self.verify()
            bundle.assert_not_called()
        self.subprocess_mock.assert_not_called()

    def test_core_receipt_must_bind_protocol_bytes(self):
        self.gate_receipt["protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bind the captured protocol"):
            self.verify()
        self.subprocess_mock.assert_not_called()

    def test_each_archived_tool_hash_and_complete_manifest_are_required(self):
        for name in report.TOOLS:
            path = self.diagnosis / "sources" / name
            path.write_bytes(self.tools[name] + b"# tampered\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "tool snapshots"):
                self.verify()
            path.write_bytes(self.tools[name])
        self.saved["provenance"]["diagnostic_sources_sha256"]["tools/unapproved.py"] = "0" * 64
        _write(self.diagnosis / "diagnosis.json", self.saved)
        with self.assertRaisesRegex(ValueError, "tool snapshots"):
            self.verify()
        self.subprocess_mock.assert_not_called()

    def test_missing_snapshot_and_source_drift_fail_before_execution(self):
        path = self.diagnosis / "sources" / report.TOOLS[0]
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()
        self.refresh_diagnosis()
        _write(self.run / "sources/rcwt_context_v4.py", b"# tampered frozen core\n")
        with self.assertRaisesRegex(ValueError, "Source archive hash"):
            self.verify()
        self.subprocess_mock.assert_not_called()

    def test_all_input_and_source_provenance_bindings_are_checked(self):
        original = copy.deepcopy(self.saved)
        for key in ("files_sha256", "input_manifest_sha256", "frozen_sources_sha256", "inspector_path", "inspector_sha256"):
            saved = copy.deepcopy(original)
            saved["provenance"][key] = "tampered"
            _write(self.diagnosis / "diagnosis.json", saved)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify()
        self.subprocess_mock.assert_not_called()

    def test_external_unscoped_or_non_immediate_diagnoses_never_execute(self):
        sibling = self.run.parent / "sibling"
        nested = self.diagnosis / "nested"
        sibling.mkdir(); nested.mkdir()
        for diagnosis in (sibling, nested, self.run):
            with self.subTest(diagnosis=diagnosis), self.assertRaisesRegex(ValueError, "immediate"):
                report.verify_diagnosis_archive(self.run, diagnosis)
        with tempfile.TemporaryDirectory(prefix="explicit-outside-project-") as outside:
            with self.assertRaisesRegex(ValueError, "trusted local"):
                report.verify_diagnosis_archive(Path(outside), self.diagnosis)
        self.gate_mock.assert_not_called()
        self.subprocess_mock.assert_not_called()

    def test_symlinked_snapshot_is_not_executed(self):
        path = self.diagnosis / "sources" / report.TOOLS[0]
        path.unlink()
        target = self.root / "source.py"
        _write(target, self.tools[report.TOOLS[0]])
        try:
            path.symlink_to(target)
        except OSError:
            self.skipTest("Windows did not allow this test symlink; no elevation requested")
        with self.assertRaisesRegex(ValueError, "links or reparse"):
            self.verify()
        self.subprocess_mock.assert_not_called()

    def test_non_pass_wrong_counts_hashes_or_write_claims_cannot_be_accepted(self):
        def mutate(key, value):
            def replacement(command, **kwargs):
                response = self.fake_inspection(command, **kwargs)
                receipt = json.loads(response.stdout)
                receipt[key] = value
                response.stdout = json.dumps(receipt)
                return response
            return replacement
        for key, value in (("status", "FAIL"), ("n_episodes", 7), ("n_episodes", True), ("decisions", 64),
                           ("read_only", False), ("feedback_to_agent", True), ("diagnosis_sha256", "0"*64),
                           ("report_sha256", "0"*64)):
            self.subprocess_mock.side_effect = mutate(key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "byte-bound PASS"):
                self.verify()

    def test_subprocess_errors_and_malformed_receipts_fail_closed(self):
        self.subprocess_mock.side_effect = None
        for stdout in ("not json", '{"status":"PASS","status":"PASS"}', '{"value":NaN}'):
            self.subprocess_mock.return_value = SimpleNamespace(returncode=0, stdout=stdout, stderr="")
            with self.subTest(stdout=stdout), self.assertRaises(ValueError):
                self.verify()
        self.subprocess_mock.return_value = SimpleNamespace(returncode=1, stdout="", stderr="EXPLICIT FAKE FAILURE")
        with self.assertRaisesRegex(ValueError, "EXPLICIT FAKE FAILURE"):
            self.verify()

    def test_original_input_change_during_inspection_invalidates_receipt(self):
        def mutate(command, **kwargs):
            result = self.fake_inspection(command, **kwargs)
            path = self.diagnosis / "DIAGNOSIS.md"
            path.write_bytes(path.read_bytes() + b"changed during verification")
            return result
        self.subprocess_mock.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "changed during"):
            self.verify()

    def test_actual_bootstrap_blocks_network_and_preserves_verify_only_arguments(self):
        # Executes only this invented fixture script, with a fake core gate.
        self.tools["inspect_online_v4.py"] = b'''import hashlib, json, socket, sys
from pathlib import Path
assert sys.argv[1] == "--run-dir" and sys.argv[3] == "--output-dir" and sys.argv[5] == "--verify"
run, diagnosis = Path(sys.argv[2]), Path(sys.argv[4])
assert diagnosis.parent == run
try:
    socket.create_connection(("127.0.0.1", 1))
except RuntimeError as exc:
    assert "Network is forbidden" in str(exc)
else:
    raise AssertionError("network block missing")
def sha(name): return hashlib.sha256((diagnosis / name).read_bytes()).hexdigest()
print(json.dumps({"status":"PASS", "read_only":True, "feedback_to_agent":False,
 "n_episodes":8, "decisions":128, "diagnosis_sha256":sha("diagnosis.json"),
 "report_sha256":sha("DIAGNOSIS.md"), "scope":"EXPLICIT FAKE bootstrap fixture"}))
'''
        self.refresh_diagnosis()
        self.patches["subprocess"].stop()
        receipt = self.verify()
        self.assertIn("EXPLICIT FAKE", receipt["verification"]["scope"])
        with tempfile.TemporaryDirectory(prefix="explicit-fake-wrong-mode-") as temporary:
            script = Path(temporary) / "do_not_execute.py"
            _write(script, b"raise AssertionError('Wrong-mode script must never execute')\n")
            wrong = subprocess.run([sys.executable, "-I", "-B", "-c", report._VERIFY_ONLY, str(script),
                                    "--run-dir", "fake-run", "--output-dir", "fake-output"],
                                   capture_output=True, text=True, timeout=15, check=False)
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("Only the read-only diagnosis verification", wrong.stderr)

    def test_cli_prints_receipt_and_has_no_write_option(self):
        output = io.StringIO()
        with patch.object(report, "verify_diagnosis_archive", return_value={"status": "PASS"}) as verify, \
             patch("sys.argv", ["tool", "--run-dir", "fake-run", "--diagnosis-dir", "fake-diagnosis"]), redirect_stdout(output):
            report.main()
        verify.assert_called_once_with(Path("fake-run"), Path("fake-diagnosis"))
        self.assertEqual(json.loads(output.getvalue()), {"status": "PASS"})


class HistoricalV4DiagnosisPublicTrainTests(unittest.TestCase):
    def test_completed_published_train_diagnosis_replays_without_current_source_pins(self):
        run = PROJECT / "results/agent_v4_development/attempt_01"
        diagnosis = run / "diagnosis_verified"
        if not (diagnosis / "sources/inspect_online_v4.py").is_file():
            self.skipTest("Published historical TRAIN diagnosis snapshots are unavailable")
        before = {p: p.read_bytes() for p in diagnosis.rglob("*") if p.is_file()}
        receipt = report.verify_diagnosis_archive(run, diagnosis)
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["verification"]["n_episodes"], 8)
        self.assertEqual(receipt["verification"]["decisions"], 128)
        self.assertEqual(receipt["core_verification"]["verified_steps"], 128)
        self.assertEqual(receipt["inference_calls"], 0)
        self.assertEqual(receipt["original_writes"], 0)
        self.assertEqual(before, {p: p.read_bytes() for p in diagnosis.rglob("*") if p.is_file()})

    def test_archived_inspector_rejects_changed_facts_and_markdown_in_a_temporary_train_copy(self):
        original_run = PROJECT / "results/agent_v4_development/attempt_01"
        original_diag = original_run / "diagnosis_verified"
        if not (original_diag / "sources/inspect_online_v4.py").is_file():
            self.skipTest("Published historical TRAIN diagnosis snapshots are unavailable")
        before = {p: p.read_bytes() for p in original_diag.rglob("*") if p.is_file()}
        # This is a copy of completed development only, not a generated corpus.
        # The original files and all current sources remain untouched.
        with tempfile.TemporaryDirectory(prefix="diagnosis-tamper-test-", dir=PROJECT / ".runs") as temporary:
            run = Path(temporary) / "copied-public-train"
            diagnosis = run / "diagnosis_verified"
            observed = {}
            copied, _ = archive._bundle(original_run, observed)
            for name, data in copied.items():
                _write(run / name, data)
            shutil.copytree(original_diag, diagnosis)
            path = diagnosis / "diagnosis.json"
            original = path.read_bytes()
            changed = json.loads(original)
            changed["decisions"][0]["success"] = not changed["decisions"][0]["success"]
            _write(path, changed)
            with self.assertRaisesRegex(ValueError, "Saved diagnosis facts or provenance"):
                report.verify_diagnosis_archive(run, diagnosis)
            path.write_bytes(original)
            markdown = diagnosis / "DIAGNOSIS.md"
            markdown.write_bytes(markdown.read_bytes() + b"\nTAMPERED DISPLAY CLAIM\n")
            with self.assertRaisesRegex(ValueError, "Saved Markdown"):
                report.verify_diagnosis_archive(run, diagnosis)
        self.assertEqual(before, {p: p.read_bytes() for p in original_diag.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
