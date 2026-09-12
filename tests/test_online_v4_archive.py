"""Trusted-archive plumbing with invented bytes, never model-quality evidence.

No real dataset or run is opened. Subprocess receipts are patched except for
one self-contained fixture script that checks the network block and prints a
labelled fake receipt. The real model/runner is never invoked by these tests.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("online_v4_archive", PROJECT / "tools/verify_online_v4_archive.py")
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else _bytes(data))


class OnlineV4ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="explicit-fake-archive-v4-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        root_patch = patch.object(archive, "ROOT", self.root)
        root_patch.start(); self.addCleanup(root_patch.stop)
        self.source = {name: ("# EXPLICIT FAKE SOURCE; NOT MODEL EVIDENCE: " + name + "\n").encode()
                       for name in archive.NAMES}
        self.hashes = {"src/" + name: archive._digest(data) for name, data in self.source.items()}
        self.runtime = {"model_alias": "EXPLICIT-FAKE-ARCHIVE-MODEL",
                        "inference_policy": "local_only_no_paid_api", "fixture_only": True}
        self.refresh_history()
        self.directory = self.root / ".runs/fake-development"
        self.make_bundle(self.directory)
        self.subprocess_patcher = patch.object(archive.subprocess, "run", side_effect=self.fake_verify)
        self.subprocess = self.subprocess_patcher.start()
        self.addCleanup(self.subprocess_patcher.stop)

    def refresh_history(self):
        for path in archive.HISTORY.values():
            _write(self.root / path, b"EXPLICIT FAKE HISTORICAL ARTIFACT; NOT PERFORMANCE EVIDENCE\n")
        _write(self.root / "results/agent_v2/protocol.json",
               {"source_sha256": {"src/" + name: self.hashes["src/" + name] for name in archive.NAMES[:6]}})
        _write(self.root / archive.HISTORY["v3_protocol_sha256"],
               {"source_sha256": {"src/" + name: self.hashes["src/" + name] for name in archive.NAMES[:11]}})
        _write(self.root / "docs/rcwt_agent_runtime.json", self.runtime)

    def make_bundle(self, directory, mode="development"):
        count, split = (8, "train") if mode == "development" else (32, "test")
        protocol = {"schema": "rcwt-online-memory/4", "mode": mode, "split": split, "count": count,
                    "source_sha256": self.hashes, "runtime": self.runtime, "development": None}
        protocol.update({label: archive.digest(self.root / path) for label, path in archive.HISTORY.items()})
        if mode == "confirmatory":
            self.make_bundle(directory / "development")
            protocol["development"] = {
                "protocol_sha256": archive.digest(directory / "development/protocol.json"),
                "completion_sha256": archive.digest(directory / "development/completion.json"),
                "gate": {"passed": True, "note": "explicitly invented selection receipt"},
            }
        for name in archive.INPUTS:
            _write(directory / name, b"EXPLICIT FAKE ARTIFACT\n")
        _write(directory / "protocol.json", protocol)
        _write(directory / "schedule.json", [])
        protocol_hash = archive.digest(directory / "protocol.json")
        _write(directory / "freeze.json", {"protocol_sha256": protocol_hash,
                                          "schedule_sha256": archive.digest(directory / "schedule.json")})
        _write(directory / "started.json", {"protocol_sha256": protocol_hash})
        _write(directory / "completion.json", {"protocol_sha256": protocol_hash,
                                              "traces_sha256": archive.digest(directory / "traces.jsonl"),
                                              "episodes_sha256": archive.digest(directory / "episodes.jsonl")})
        for name, data in self.source.items():
            _write(directory / "sources" / name, data)

    def fake_verify(self, command, **kwargs):
        self.assertEqual(command[:4], [sys.executable, "-I", "-B", "-c"])
        self.assertEqual(command[4], archive._VERIFY_ONLY)
        self.assertEqual(command[6:9], ["--stage", "verify", "--output-dir"])
        workspace, copied_run = Path(kwargs["cwd"]), Path(command[9])
        self.assertEqual(copied_run, workspace / "run")
        self.assertNotEqual(copied_run, self.directory)
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        self.assertFalse(kwargs["check"])
        self.assertLessEqual(kwargs["timeout"], 60)
        self.assertNotIn("PYTHONPATH", kwargs["env"])
        self.assertNotIn("PYTHONHOME", kwargs["env"])
        for name in archive.NAMES:
            self.assertEqual((workspace / "src" / name).read_bytes(), self.source[name])
            self.assertEqual((copied_run / "sources" / name).read_bytes(), self.source[name])
        for path in (*archive.HISTORY.values(), "results/agent_v2/protocol.json", "docs/rcwt_agent_runtime.json"):
            self.assertEqual((workspace / path).read_bytes(), (self.root / path).read_bytes())
        protocol = json.loads((copied_run / "protocol.json").read_text(encoding="utf-8"))
        if protocol["mode"] == "confirmatory":
            self.assertEqual(json.loads((copied_run / "development/protocol.json").read_text())["count"], 8)
            self.assertTrue((copied_run / "development/sources/rcwt_online_v4.py").is_file())
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "status": "PASS", "verified_steps": protocol["count"] * 16,
            "scope": "EXPLICIT FAKE RECEIPT; NOT INFERENCE EVIDENCE"}))

    def test_development_copies_verified_inputs_and_returns_original_receipt_without_writes(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch.dict(os.environ, {"PYTHONPATH": "FAKE-UNTRUSTED-IMPORT-PATH", "PYTHONHOME": "FAKE-HOME"}):
            result = archive.verify_archive(self.directory)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["verified_steps"], 128)
        self.assertEqual(result["inference_calls"], 0)
        self.assertTrue(result["read_only"])
        self.assertEqual(result["verification"], {"status": "PASS", "verified_steps": 128,
                                                   "scope": "EXPLICIT FAKE RECEIPT; NOT INFERENCE EVIDENCE"})
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})
        self.subprocess.assert_called_once()

    def test_confirmation_32_includes_copied_development_8_without_generating_a_corpus(self):
        self.directory = self.root / "results/fake-confirmation"
        self.make_bundle(self.directory, "confirmatory")
        result = archive.verify_archive(self.directory)
        self.assertEqual(result["verified_steps"], 512)
        self.assertEqual(result["verification"]["verified_steps"], 512)
        self.subprocess.assert_called_once()

    def test_modified_source_and_missing_source_fail_before_execution(self):
        source = self.directory / "sources/rcwt_context_v4.py"
        original = source.read_bytes()
        source.write_bytes(original + b"# changed\n")
        with self.assertRaisesRegex(ValueError, "Source archive hash mismatch"):
            archive.verify_archive(self.directory)
        source.unlink()
        with self.assertRaises(FileNotFoundError):
            archive.verify_archive(self.directory)
        self.subprocess.assert_not_called()

    def test_every_pinned_historical_artifact_and_runtime_are_checked(self):
        for label, relative in archive.HISTORY.items():
            path = self.root / relative
            original = path.read_bytes()
            # Preserve JSON parsing for the referenced v3 protocol.
            if label == "v3_protocol_sha256":
                path.write_bytes(original + b"\n")
            else:
                path.write_bytes(original + b"changed")
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "historical"):
                archive.verify_archive(self.directory)
            path.write_bytes(original)
        runtime = {**self.runtime, "model_alias": "CHANGED-FAKE-MODEL"}
        _write(self.root / "docs/rcwt_agent_runtime.json", runtime)
        with self.assertRaisesRegex(ValueError, "runtime"):
            archive.verify_archive(self.directory)
        self.subprocess.assert_not_called()

    def test_v2_inheritance_is_checked_independently_of_v3_manifest(self):
        path = self.root / "results/agent_v2/protocol.json"
        inherited = json.loads(path.read_text())
        inherited["source_sha256"]["src/rcwt_agent_actor.py"] = "0" * 64
        _write(path, inherited)
        with self.assertRaisesRegex(ValueError, "inherited"):
            archive.verify_archive(self.directory)
        self.subprocess.assert_not_called()

    def test_exact_schema_modes_counts_and_source_set_fail_closed(self):
        path = self.directory / "protocol.json"
        original = path.read_bytes()
        for mutate in (
            lambda p: p.update(schema="rcwt-online-memory/3.3"),
            lambda p: p.update(count=True), lambda p: p.update(count=4),
            lambda p: p.update(split="test"), lambda p: p.update(mode="arbitrary"),
            lambda p: p["source_sha256"].pop("src/rcwt_context_v4.py"),
            lambda p: p["source_sha256"].update({"src/unknown.py": "0" * 64}),
        ):
            protocol = json.loads(original)
            mutate(protocol); _write(path, protocol)
            with self.assertRaises(ValueError):
                archive.verify_archive(self.directory)
            path.write_bytes(original)
        self.subprocess.assert_not_called()

    def test_incomplete_mutated_or_aborted_bundles_never_execute(self):
        for filename in ("freeze.json", "started.json", "completion.json", "traces.jsonl", "schedule.json"):
            path = self.directory / filename
            original = path.read_bytes()
            if filename.endswith(".json") and filename != "schedule.json":
                value = json.loads(original); value["protocol_sha256"] = "0" * 64
                _write(path, value)
            else:
                path.write_bytes(original + b"changed")
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                archive.verify_archive(self.directory)
            path.write_bytes(original)
        for filename in ("aborted.json", "partial-step.json"):
            path = self.directory / filename
            _write(path, {})
            with self.assertRaisesRegex(ValueError, "aborted or partial"):
                archive.verify_archive(self.directory)
            path.unlink()
        self.subprocess.assert_not_called()

    def test_copied_development_binding_and_its_source_hashes_are_required(self):
        directory = self.root / ".runs/fake-confirmation"
        self.make_bundle(directory, "confirmatory")
        path = directory / "development/sources/rcwt_context_v4.py"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Source archive"):
            archive.verify_archive(directory)
        self.make_bundle(directory, "confirmatory")
        path = directory / "development/completion.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Copied development"):
            archive.verify_archive(directory)
        self.subprocess.assert_not_called()

    def test_external_unscoped_and_symlink_bundles_are_not_executed(self):
        with tempfile.TemporaryDirectory(prefix="explicit-external-fake-") as external:
            with self.assertRaisesRegex(ValueError, "trusted local"):
                archive.verify_archive(Path(external))
        unscoped = self.root / "unscoped"
        unscoped.mkdir()
        with self.assertRaisesRegex(ValueError, "below .runs"):
            archive.verify_archive(unscoped)
        link = self.root / ".runs/fake-link"
        try:
            link.symlink_to(self.directory, target_is_directory=True)
        except OSError:
            return  # Windows may not permit creation; no elevation requested.
        with self.assertRaisesRegex(ValueError, "links or reparse"):
            archive.verify_archive(link)
        self.subprocess.assert_not_called()

    def test_non_pass_incomplete_or_non_json_receipts_fail(self):
        for receipt in ({"status": "FAIL", "verified_steps": 128}, {"status": "PASS", "verified_steps": 64},
                        {"status": "PASS", "verified_steps": True}, {"status": "PASS"}, []):
            self.subprocess.side_effect = None
            self.subprocess.return_value = SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(receipt))
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                archive.verify_archive(self.directory)
        self.subprocess.return_value = SimpleNamespace(returncode=1, stderr="EXPLICIT FAKE REPLAY FAILURE", stdout="")
        with self.assertRaisesRegex(ValueError, "REPLAY FAILURE"):
            archive.verify_archive(self.directory)

    def test_original_evidence_change_during_subprocess_invalidates_pass(self):
        def mutate(command, **kwargs):
            result = self.fake_verify(command, **kwargs)
            path = self.directory / "traces.jsonl"
            path.write_bytes(path.read_bytes() + b"changed during verification")
            return result
        self.subprocess.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "changed during"):
            archive.verify_archive(self.directory)

    def test_duplicate_nonfinite_json_and_cli_are_explicit(self):
        for content in ('{"mode": "development", "mode": "confirmatory"}', '{"value": NaN}'):
            with self.assertRaises(ValueError):
                archive._json(content)
        with patch.object(archive, "verify_archive", return_value={"status": "PASS"}) as verify, patch("builtins.print"), patch(
                "sys.argv", ["verify_online_v4_archive.py", "--run-dir", "fake-local-argument"]):
            archive.main()
        verify.assert_called_once_with(Path("fake-local-argument"))

    def test_actual_verify_only_bootstrap_blocks_network_with_an_invented_fixture_script(self):
        # This trusted, locally authored fixture prints a fake receipt. It does
        # not import the real runner, generate episodes, or construct a model.
        self.source["rcwt_online_v4.py"] = b'''import json, socket, ssl, sys
assert sys.argv[1:3] == ["--stage", "verify"]
try:
    socket.create_connection(("127.0.0.1", 1))
except RuntimeError as exc:
    assert "Network is forbidden" in str(exc)
else:
    raise AssertionError("Network block was not installed")
print(json.dumps({"status": "PASS", "verified_steps": 128,
                  "scope": "EXPLICIT FAKE SCRIPT: only verifies bootstrap and network block"}))
'''
        self.hashes = {"src/" + name: archive._digest(data) for name, data in self.source.items()}
        self.refresh_history()
        self.make_bundle(self.directory)
        self.subprocess_patcher.stop()
        result = archive.verify_archive(self.directory)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["verified_steps"], 128)
        self.assertIn("EXPLICIT FAKE SCRIPT", result["verification"]["scope"])


if __name__ == "__main__":
    unittest.main()
