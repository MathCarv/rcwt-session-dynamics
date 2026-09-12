"""Regression coverage for the network boundary of the CI test entrypoint."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("offline_suite_entrypoint", ROOT / "tools/run_tests_offline.py")
entrypoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entrypoint)


class OfflineSuiteTests(unittest.TestCase):
    def test_nested_network_boundary_blocks_calls_and_restores_existing_guard(self):
        original = socket.create_connection
        with entrypoint.offline():
            outer = socket.create_connection
            with self.assertRaisesRegex(ValueError, "fixture failure"):
                with entrypoint.offline():
                    with self.assertRaisesRegex(RuntimeError, "Network access is forbidden"):
                        socket.create_connection(("127.0.0.1", 18085))
                    with self.assertRaisesRegex(RuntimeError, "Network access is forbidden"):
                        socket.getaddrinfo("invalid.example", 443)
                    raise ValueError("fixture failure")
            self.assertIs(socket.create_connection, outer)
        self.assertIs(socket.create_connection, original)

    def test_fresh_checkout_scratch_and_failure_exit_are_explicit(self):
        with tempfile.TemporaryDirectory(prefix="offline-suite-fixture-") as temporary:
            root = Path(temporary)
            with patch.object(entrypoint, "ROOT", root), patch.object(sys, "path", list(sys.path)), \
                    patch.object(entrypoint.unittest.defaultTestLoader, "discover") as discover, \
                    patch.object(entrypoint.unittest, "TextTestRunner") as runner:
                runner.return_value.run.return_value.wasSuccessful.return_value = False
                self.assertEqual(entrypoint.main(), 1)
                self.assertIn(str(root), sys.path)
                self.assertIn(str(root / "src"), sys.path)
                self.assertTrue((root / ".runs").is_dir())
                discover.assert_called_once_with(str(root / "tests"), pattern="test_*.py")


if __name__ == "__main__":
    unittest.main()
