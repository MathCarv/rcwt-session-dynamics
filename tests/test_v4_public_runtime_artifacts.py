"""Public R1 derived-log release checks; no private runtime bundle is required.

These checks fail when the real generated public artifacts are absent. They do
not create substitutes, contact a model, or claim access to the original log.
Complete public replay/accounting is independently verified by the original CLI.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from tools import audit_v4_replication_calls as accountant
from tools import export_v4_replication_runtime as exporter

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RUNTIME = ROOT / "results/agent_v4_replication_public_runtime"
PUBLIC_ACCOUNTING = ROOT / "results/agent_v4_replication_public_accounting"
RUN = ROOT / "results/agent_v4_replication"
SOURCE_LOG_SHA256 = "b138c28f609056c927d4c8e40fa4106ab85e5688e80e0d66080ef966c94c2c2d"
CUSTODY_SHA256 = "2dd16fc9d1b472218c8ba59f15c78fe22bdc68dbdedc49f2856a8d4a6b33b693"
ORIGINAL_ACCOUNTING_SHA256 = "bb9b7e33e46a65773a159d5ffa3c708c98b33eafdd8b2ecdcf11c6f625e1e07f"
ORIGINAL_ACCOUNTING_MARKDOWN_SHA256 = "f64e37fe2db9ea2c9a6eaf8618f517f64949102b55253bb2cff41780d0950461"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class PublicRuntimeArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        expected = [PUBLIC_RUNTIME / "server.redacted.txt", PUBLIC_RUNTIME / "manifest.json",
                    PUBLIC_ACCOUNTING / "call-accounting.json", PUBLIC_ACCOUNTING / "CALL-ACCOUNTING.md"]
        if any(not path.is_file() or path.is_symlink() for path in expected):
            raise AssertionError("Real public runtime/accounting artifacts are required; no synthetic fallback or skip is allowed")
        cls.log = (PUBLIC_RUNTIME / "server.redacted.txt").read_bytes()
        cls.manifest = json.loads((PUBLIC_RUNTIME / "manifest.json").read_bytes())
        cls.accounting = json.loads((PUBLIC_ACCOUNTING / "call-accounting.json").read_bytes())

    def test_manifest_binds_known_originals_and_unchanged_exporter(self):
        manifest = self.manifest
        self.assertEqual(manifest["schema"], "rcwt-r1-derived-runtime-log/1")
        self.assertEqual(manifest["status"], "PASS_DERIVED_EXPORT_INTEGRITY")
        self.assertEqual(manifest["artifact_kind"], "DERIVED_LOG_WITH_METADATA_PATH_PREFIX_REDACTION")
        self.assertIs(manifest["is_original_runtime_log"], False)
        self.assertEqual(manifest["source_log_sha256"], SOURCE_LOG_SHA256)
        self.assertEqual(manifest["custody_sha256"], CUSTODY_SHA256)
        self.assertEqual(manifest["original_accounting_json_sha256"], ORIGINAL_ACCOUNTING_SHA256)
        self.assertEqual(manifest["original_accounting_markdown_sha256"], ORIGINAL_ACCOUNTING_MARKDOWN_SHA256)
        self.assertEqual(manifest["exporter_sha256"], digest(Path(exporter.__file__).read_bytes()))
        self.assertEqual(manifest["original_auditor_sha256"], digest(Path(accountant.__file__).read_bytes()))
        self.assertIs(manifest["original_accounting_verified_exact_bytes"], True)
        self.assertIs(manifest["original_custody_copy_inventory_verified"], True)
        self.assertEqual(manifest["inference_calls"], 0)

    def test_derived_log_has_exactly_one_declared_metadata_edit(self):
        manifest = self.manifest
        self.assertEqual(manifest["derived_file"], "server.redacted.txt")
        self.assertEqual(manifest["derived_log_sha256"], digest(self.log))
        self.assertEqual(manifest["derived_log_bytes"], len(self.log))
        lines = self.log.splitlines(keepends=True)
        self.assertEqual(len(lines), 9253)
        self.assertEqual(manifest["source_line_count"], len(lines))
        self.assertEqual(manifest["derived_line_count"], len(lines))
        self.assertEqual(manifest["unchanged_line_count"], len(lines) - 1)
        self.assertEqual(manifest["changed_metadata_line_numbers"], [4])
        self.assertEqual(manifest["redaction_categories"], [{
            "category": "local_runtime_directory_prefix", "replacement": "<LOCAL_RUNTIME>",
            "occurrences": 1, "metadata_line_count": 1}])
        self.assertEqual(self.log.count(b"<LOCAL_RUNTIME>"), 1)
        self.assertIsNotNone(exporter.METADATA_MODEL.fullmatch(lines[3]))
        self.assertIsNone(exporter.PRIVATE_PATH.search(self.log))
        self.assertIsNone(exporter.CREDENTIAL.search(self.log))
        self.assertIsNone(exporter.PRIVATE_PATH.search((PUBLIC_RUNTIME / "manifest.json").read_bytes()))
        for key in ("line_order_preserved", "line_endings_preserved", "task_event_bytes_preserved"):
            self.assertIs(manifest[key], True)

    def test_all_1248_tasks_are_present_and_task_bytes_bound(self):
        parsed = accountant.parse_server_log(self.log)
        self.assertEqual(len(parsed), 1248)
        self.assertEqual(self.manifest["parsed_task_count"], 1248)
        task_lines = [line for line in self.log.splitlines(keepends=True) if exporter.SLOT_LINE.match(line)]
        self.assertEqual(self.manifest["unchanged_task_event_line_count"], len(task_lines))
        self.assertEqual(self.manifest["unchanged_task_event_bytes_sha256"], digest(b"".join(task_lines)))

    def test_public_accounting_binds_derived_log_and_exact_r1_snapshot(self):
        accounting = self.accounting
        self.assertEqual(accounting["schema"], "rcwt-r1-server-call-accounting/1")
        self.assertEqual(accounting["status"], "PASS")
        self.assertIs(accounting["accounting_only"], True)
        self.assertEqual(accounting["gain"], "NOT_EVALUATED")
        self.assertEqual(accounting["inference_calls"], 0)
        self.assertEqual(accounting["verified_steps"], 512)
        self.assertEqual(accounting["verified_episode_summaries"], 64)
        for key in ("server_task_starts", "server_final_timing_pairs", "matched_trace_calls"):
            self.assertEqual(accounting[key], 1248)
        self.assertEqual(accounting["canceled_tasks"], 0)
        self.assertEqual(accounting["extra_tasks"], 0)
        self.assertEqual(accounting["inputs"]["server_log_sha256"], digest(self.log))
        self.assertEqual(accounting["inputs"]["server_log_bytes"], len(self.log))
        self.assertNotEqual(accounting["inputs"]["server_log_sha256"], SOURCE_LOG_SHA256)
        self.assertEqual(accounting["inputs"]["run_directory"], "results/agent_v4_replication")
        snapshot = accountant._snapshot(RUN)
        self.assertEqual(len(snapshot), 32)
        self.assertEqual(accounting["inputs"]["run_files_sha256"], snapshot)
        self.assertEqual(accounting["inputs"]["audit_helper_sha256"], self.manifest["original_auditor_sha256"])
        self.assertEqual((PUBLIC_ACCOUNTING / "CALL-ACCOUNTING.md").read_bytes(), accountant.render_markdown(accounting).encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
