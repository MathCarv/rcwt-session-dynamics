"""Strict cross-platform release verification; synthetic cases are unit-only."""
from __future__ import annotations

import builtins
from contextlib import ExitStack
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import unittest
from unittest import mock

from tools import verify_r1_public_accounting as verifier


class CanonicalAccountingTests(unittest.TestCase):
    def setUp(self):
        # Explicitly synthetic values, never written as release artifacts.
        self.saved = {"status": "PASS", "values": {"z": 1, "a": 1.0, "flag": True},
                      "ordered": [{"b": 2, "a": 1}, 3], "nullable": None}

    def test_reordered_nested_maps_pass_without_coercion(self):
        reordered = {"nullable": None, "ordered": [{"a": 1, "b": 2}, 3],
                     "values": {"flag": True, "a": 1.0, "z": 1}, "status": "PASS"}
        verifier.require_same_accounting(reordered, self.saved)

    def test_changed_numbers_booleans_types_values_arrays_and_fields_fail(self):
        mutations = []
        for key, value in (("z", 2), ("z", 1.0), ("a", 1), ("flag", 1),
                           ("flag", False), ("z", "1")):
            changed = deepcopy(self.saved)
            changed["values"][key] = value
            mutations.append(changed)
        changed = deepcopy(self.saved)
        changed["ordered"].reverse()
        mutations.append(changed)
        changed = deepcopy(self.saved)
        del changed["values"]["z"]
        mutations.append(changed)
        changed = deepcopy(self.saved)
        changed["values"]["extra"] = 0
        mutations.append(changed)
        changed = deepcopy(self.saved)
        changed["status"] = "FAIL"
        mutations.append(changed)
        for index, changed in enumerate(mutations):
            with self.subTest(case=index), self.assertRaises(ValueError):
                verifier.require_same_accounting(changed, self.saved)

    def test_nonfinite_or_non_json_types_fail(self):
        for value in (float("nan"), float("inf"), float("-inf"), (1, 2), {1: "a"}):
            with self.subTest(kind=type(value).__name__), self.assertRaises(ValueError):
                verifier.canonical_json(value)


class FixedPublicReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = verifier.ROOT
        cls.accountant = verifier._load_accountant(cls.root)
        released = verifier._released_bytes(cls.root)
        cls.saved = cls.accountant._json(released[verifier.ACCOUNTING + "/call-accounting.json"])

    def test_recomputed_map_order_may_differ_from_pinned_saved_bytes(self):
        recomputed = deepcopy(self.saved)
        mapping = recomputed["inputs"]["run_files_sha256"]
        recomputed["inputs"]["run_files_sha256"] = dict(reversed(list(mapping.items())))
        self.assertNotEqual(self.accountant._json_bytes(recomputed), self.accountant._json_bytes(self.saved))
        with mock.patch.object(self.accountant, "build_accounting", return_value=recomputed):
            result = verifier.verify_public_accounting()
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["original_cross_platform_formatter_claim"])

    def test_every_pinned_release_file_rejects_changed_bytes_before_replay(self):
        original_read = verifier._read
        for relative in verifier.RELEASE_SHA256:
            target = self.root / relative
            def changed_read(path):
                data = original_read(path)
                return data + b"\n" if Path(path) == target else data
            with self.subTest(path=relative), mock.patch.object(verifier, "_read", side_effect=changed_read), \
                    mock.patch.object(self.accountant, "build_accounting") as build:
                with self.assertRaises(ValueError):
                    verifier.verify_public_accounting()
                build.assert_not_called()

    def test_missing_published_file_fails_before_replay(self):
        original_read = verifier._read
        def missing_read(path):
            if Path(path) == self.root / verifier.RUNTIME / "server.redacted.txt":
                raise FileNotFoundError("Synthetic missing artifact")
            return original_read(path)
        with mock.patch.object(verifier, "_read", side_effect=missing_read), \
                mock.patch.object(self.accountant, "build_accounting") as build:
            with self.assertRaises(FileNotFoundError):
                verifier.verify_public_accounting()
            build.assert_not_called()

    def test_additional_public_file_fails_before_replay(self):
        snapshot = self.accountant._snapshot
        def extra_snapshot(directory):
            value = snapshot(directory)
            if Path(directory) == self.root / verifier.RUNTIME:
                value["synthetic-extra.txt"] = "0" * 64
            return value
        with mock.patch.object(self.accountant, "_snapshot", side_effect=extra_snapshot), \
                mock.patch.object(self.accountant, "build_accounting") as build:
            with self.assertRaises(ValueError):
                verifier.verify_public_accounting()
            build.assert_not_called()

    def test_exact_markdown_rendering_is_required(self):
        with mock.patch.object(self.accountant, "build_accounting", return_value=deepcopy(self.saved)), \
                mock.patch.object(self.accountant, "render_markdown", return_value="Synthetic changed rendering\n"):
            with self.assertRaises(ValueError):
                verifier.verify_public_accounting()

    def test_before_after_change_fails(self):
        state = verifier._state
        observed = 0
        def changed_state(*args):
            nonlocal observed
            observed += 1
            value = state(*args)
            if observed == 2:
                value["verifier_sha256"] = "0" * 64
            return value
        with mock.patch.object(self.accountant, "build_accounting", return_value=deepcopy(self.saved)), \
                mock.patch.object(verifier, "_state", side_effect=changed_state):
            with self.assertRaises(ValueError):
                verifier.verify_public_accounting()

    def test_real_release_recomputes_without_writes_network_or_subprocesses(self):
        real_builtin_open, real_io_open, real_os_open = builtins.open, io.open, os.open
        def read_only_open(opener):
            def guarded(file, mode="r", *args, **kwargs):
                if any(flag in mode for flag in "wax+"):
                    raise AssertionError("Unexpected file write")
                return opener(file, mode, *args, **kwargs)
            return guarded
        def guarded_os_open(file, flags, *args, **kwargs):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise AssertionError("Unexpected low-level file write")
            return real_os_open(file, flags, *args, **kwargs)
        with ExitStack() as stack:
            stack.enter_context(mock.patch("builtins.open", side_effect=read_only_open(real_builtin_open)))
            stack.enter_context(mock.patch("io.open", side_effect=read_only_open(real_io_open)))
            stack.enter_context(mock.patch("os.open", side_effect=guarded_os_open))
            for target in ("socket.socket", "socket.create_connection", "socket.getaddrinfo", "subprocess.Popen"):
                stack.enter_context(mock.patch(target, side_effect=AssertionError("Unexpected network/process use")))
            for method in ("write_bytes", "write_text", "mkdir", "unlink", "rename", "replace", "rmdir"):
                stack.enter_context(mock.patch.object(Path, method, side_effect=AssertionError("Unexpected mutation")))
            result = verifier.verify_public_accounting()
        self.assertEqual(result["matched_trace_calls"], 1248)
        self.assertEqual(result["inference_calls"], 0)
        self.assertIs(result["read_only"], True)

    def test_cli_errors_do_not_print_local_path_values(self):
        output = io.StringIO()
        with mock.patch("sys.argv", ["verify_r1_public_accounting.py"]), \
                mock.patch.object(verifier, "verify_public_accounting", side_effect=ValueError("synthetic private detail")), \
                mock.patch("sys.stderr", output):
            self.assertEqual(verifier.main(), 1)
        message = json.loads(output.getvalue())
        self.assertEqual(message["status"], "FAIL")
        self.assertNotIn("synthetic private detail", output.getvalue())


if __name__ == "__main__":
    unittest.main()
