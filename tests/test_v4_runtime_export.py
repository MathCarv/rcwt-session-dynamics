"""Explicit synthetic export tests; no real R1 or model artifacts are read.

The original-accounting gate is mocked. The unchanged real task-log parser is
used against invented events, tokens and timings, not execution evidence.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import audit_v4_replication_calls as accountant
from tools import export_v4_replication_runtime as exporter
from test_v4_replication_calls import synthetic_log

PROVENANCE = "EXPLICIT SYNTHETIC FIXTURE; no model or actual runtime evidence"
PRIVATE_PREFIX = r"C:\EXPLICIT-SYNTHETIC-PRIVATE\runtime"
MODEL = PRIVATE_PREFIX + r"\models\Qwen3.5-4B-Q4_K_M.gguf"
BINARY = PRIVATE_PREFIX + r"\llama-b10809\llama-server.exe"
MODEL_LINE = ("0.00.000.000 I srv load_model: loading model '" + MODEL + "'\r\n").encode()


class RuntimeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_log = MODEL_LINE + synthetic_log()
        cls.auditor_bytes = Path(accountant.__file__).read_bytes()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="explicit-synthetic-r1-export-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.runtime = self.root / "runtime"
        self.accounting = self.root / "accounting"
        self.output = self.root / "public-runtime"
        for directory in (self.run, self.runtime, self.accounting, self.root / "tools"):
            directory.mkdir()
        (self.root / exporter.AUDITOR_RELATIVE).write_bytes(self.auditor_bytes)
        for name in ("protocol.json", "completion.json", "traces.jsonl", "episodes.jsonl",
                     "analysis.json", "verification.json", "RESULTS.md"):
            (self.run / name).write_bytes((PROVENANCE + "\n").encode())
        self.copies = []
        names = ["logs/server.stderr.log", "logs/server.stdout.log"]
        names += [f"receipts/synthetic-{index}.json" for index in range(16)]
        names += ["launchcontrol/launch.json", "launchcontrol/launch-intent.json",
                  "launchcontrol/stdout.log", "launchcontrol/stderr.log"]
        for name in names:
            path = self.runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = self.original_log if name == "logs/server.stderr.log" else (PROVENANCE + "\n").encode()
            path.write_bytes(content)
            digest = exporter._sha(content)
            self.copies.append({"source": PRIVATE_PREFIX + "\\" + name.replace("/", "\\"),
                                "copy": str(path), "bytes": len(content),
                                "source_sha256_before": digest, "source_sha256_after": digest,
                                "copy_sha256": digest, "source_exclusive_handle": True,
                                "destination_flush_to_disk": True})
        self.custody = {
            "schema": "rcwt-r1-runtime-custody/1", "status": "STOPPED_AND_COPIED_PENDING_CALL_RECONCILIATION",
            "listeners_after": [], "runner_absent_before_stop": True, "server_absent_after_stop": True,
            "input_and_copy_hashes_rechecked": True, "source_handles_exclusive_during_copy": True,
            "copied_files_flush_to_disk": True, "stop_actions": 1, "files_deleted": 0,
            "other_processes_targeted": 0, "endpoint_calls": 0, "inference_calls": 0,
            "run_files_sha256_before_and_after": exporter._snapshot(self.run),
            "copied_files": self.copies, "observed_identity_before_stop": {"Model": MODEL, "Binary": BINARY},
            "provenance": PROVENANCE,
        }
        self.write_custody()
        (self.runtime / "close-intent.json").write_bytes(exporter._json_bytes({"provenance": PROVENANCE}))
        self.accounting_data = {
            "schema": "rcwt-r1-server-call-accounting/1", "status": "PASS", "accounting_only": True,
            "gain": "NOT_EVALUATED", "matched_trace_calls": 1248,
            "inputs": {"audit_helper_sha256": exporter._sha(self.auditor_bytes),
                       "server_log_sha256": exporter._sha(self.original_log),
                       "run_files_sha256": exporter._snapshot(self.run)},
            "provenance": PROVENANCE,
        }
        self.write_accounting()
        (self.accounting / "CALL-ACCOUNTING.md").write_text(PROVENANCE + "\n", encoding="utf-8")
        self.gate = {"status": "PASS", "output_mode": "VERIFIED_EXACT_BYTES", "accounting_only": True,
                     "gain": "NOT_EVALUATED", "matched_trace_calls": 1248, "inference_calls": 0}
        self.patches = {}
        for name, patcher in {
            "root": patch.object(exporter, "ROOT", self.root),
            "gate": patch.object(exporter, "verify_original_accounting", side_effect=lambda *args: copy.deepcopy(self.gate)),
            "auditor": patch.object(exporter, "_auditor", return_value=SimpleNamespace(parse_server_log=accountant.parse_server_log)),
            "network": patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Network forbidden")),
            "socket": patch("socket.create_connection", side_effect=AssertionError("Socket forbidden")),
        }.items():
            self.patches[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def write_custody(self):
        (self.runtime / "custody.json").write_bytes(exporter._json_bytes(self.custody))

    def write_accounting(self):
        (self.accounting / "call-accounting.json").write_bytes(exporter._json_bytes(self.accounting_data))

    def export(self, verify=False):
        return exporter.export_runtime(self.run, self.runtime, self.accounting, self.output, verify=verify)

    def test_complete_synthetic_export_preserves_every_task_and_line(self):
        before = [exporter._snapshot(path) for path in (self.run, self.runtime, self.accounting)]
        result = self.export()
        derived = (self.output / "server.redacted.txt").read_bytes()
        manifest = json.loads((self.output / "manifest.json").read_bytes())
        self.assertEqual(result["preserved_tasks"], 1248)
        self.assertFalse(result["is_original_runtime_log"])
        self.assertEqual(derived, self.original_log.replace(PRIVATE_PREFIX.encode(), b"<LOCAL_RUNTIME>", 1))
        self.assertEqual(manifest["changed_metadata_line_numbers"], [1])
        self.assertEqual(manifest["source_line_count"], manifest["derived_line_count"])
        self.assertEqual(manifest["source_log_sha256"], exporter._sha(self.original_log))
        self.assertTrue(manifest["task_event_bytes_preserved"])
        self.assertNotIn(PRIVATE_PREFIX.encode(), (self.output / "manifest.json").read_bytes())
        self.assertNotIn(PRIVATE_PREFIX.encode(), derived)
        self.assertIn(b"'\r\n", derived)
        self.assertEqual(before, [exporter._snapshot(path) for path in (self.run, self.runtime, self.accounting)])
        self.patches["gate"].assert_called_once_with(self.run, self.runtime / "logs/server.stderr.log", self.accounting)
        self.patches["network"].assert_not_called()
        self.patches["socket"].assert_not_called()

    def test_exact_verify_and_no_overwrite(self):
        self.export()
        before = exporter._snapshot(self.output)
        self.assertEqual(self.export(verify=True)["mode"], "VERIFIED_EXACT_BYTES")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.export()
        self.assertEqual(before, exporter._snapshot(self.output))

    def test_modified_export_bytes_and_extra_files_fail(self):
        self.export()
        path = self.output / "server.redacted.txt"
        path.write_bytes(path.read_bytes() + b"SYNTHETIC EXTRA\n")
        with self.assertRaisesRegex(ValueError, "Saved derived export bytes"):
            self.export(verify=True)

    def test_missing_original_verification_and_unpinned_auditor_fail(self):
        self.gate["matched_trace_calls"] = 1247
        with self.assertRaisesRegex(ValueError, "Complete exact-byte"):
            self.export()
        self.gate["matched_trace_calls"] = 1248
        self.accounting_data["inputs"]["audit_helper_sha256"] = "SYNTHETIC MISMATCH"
        self.write_accounting()
        with self.assertRaisesRegex(ValueError, "unchanged auditor"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_original_gate_failure_prevents_all_export(self):
        self.patches["gate"].side_effect = ValueError("Synthetic gate rejection")
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse(self.output.exists())

    def test_custody_rejects_nonexclusive_copy_and_wrong_hash(self):
        self.custody["copied_files"][0]["source_exclusive_handle"] = False
        self.write_custody()
        with self.assertRaisesRegex(ValueError, "exclusive-copy"):
            self.export()
        self.custody["copied_files"][0]["source_exclusive_handle"] = True
        self.custody["copied_files"][0]["copy_sha256"] = "SYNTHETIC MISMATCH"
        self.write_custody()
        with self.assertRaisesRegex(ValueError, "copy bytes"):
            self.export()

    def test_missing_additional_or_duplicated_custody_inventory_fails(self):
        (self.runtime / "extra.txt").write_text(PROVENANCE)
        with self.assertRaisesRegex(ValueError, "additional"):
            self.export()
        (self.runtime / "extra.txt").unlink()
        self.custody["copied_files"][1] = copy.deepcopy(self.custody["copied_files"][0])
        self.write_custody()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.export()

    def test_custody_copy_cannot_point_outside_bundle(self):
        self.custody["copied_files"][0]["copy"] = str(self.run / "protocol.json")
        self.write_custody()
        with self.assertRaisesRegex(ValueError, "leaves"):
            self.export()

    def test_source_private_paths_are_never_opened(self):
        for entry in self.custody["copied_files"]:
            entry["source"] = r"C:\DO-NOT-OPEN-EXPLICIT-SYNTHETIC\secrets.env"
        self.write_custody()
        self.assertEqual(self.export()["preserved_tasks"], 1248)

    def test_unexpected_metadata_or_task_path_fails_without_general_redaction(self):
        identity = self.custody["observed_identity_before_stop"]
        for data in (self.original_log + b"0.00.000.001 I srv other: C:\\SYNTHETIC\\other\n",
                     self.original_log.replace(b"graphs reused = 3", b"graphs reused = C:\\SYNTHETIC\\other", 1)):
            with self.subTest(), self.assertRaisesRegex(ValueError, "local path"):
                exporter.redact_metadata_prefix(data, identity)

    def test_wrong_model_path_or_duplicate_metadata_fails(self):
        identity = self.custody["observed_identity_before_stop"]
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            exporter.redact_metadata_prefix(self.original_log.replace(b"EXPLICIT-SYNTHETIC-PRIVATE", b"OTHER-SYNTHETIC", 1), identity)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            exporter.redact_metadata_prefix(MODEL_LINE + self.original_log, identity)

    def test_cancel_missing_task_and_modified_timing_are_not_hidden(self):
        identity = self.custody["observed_identity_before_stop"]
        with self.assertRaisesRegex(ValueError, "Canceled"):
            exporter.redact_metadata_prefix(self.original_log + b"0.00.000.000 W srv stop: cancel task, id_task = 1\n", identity)
        with self.assertRaises(ValueError):
            exporter.redact_metadata_prefix(MODEL_LINE + synthetic_log(1247), identity)
        altered = self.original_log.replace(b"prompt eval time = 10.12", b"prompt eval time = INVALID", 1)
        with self.assertRaises(ValueError):
            exporter.redact_metadata_prefix(altered, identity)

    def test_input_mutation_during_gate_fails_before_output(self):
        def mutate(*args):
            (self.run / "protocol.json").write_bytes(b"CHANGED SYNTHETIC INPUT\n")
            return self.gate
        self.patches["gate"].side_effect = mutate
        with self.assertRaisesRegex(ValueError, "changed during export"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_output_cannot_be_inside_originals(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            exporter.export_runtime(self.run, self.runtime, self.accounting, self.run / "new")
        self.patches["gate"].assert_not_called()

    def test_cli_error_suppresses_private_path_values(self):
        args = ["export", "--run-dir", str(self.run), "--runtime-dir", str(self.runtime),
                "--accounting-dir", str(self.accounting), "--output-dir", str(self.output)]
        stdout = io.StringIO()
        with patch.object(exporter, "export_runtime", side_effect=OSError(PRIVATE_PREFIX)), patch("sys.argv", args), contextlib.redirect_stdout(stdout):
            self.assertEqual(exporter.main(), 1)
        self.assertNotIn(PRIVATE_PREFIX, stdout.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
