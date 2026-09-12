"""Campaign accounting on invented manifests/logs, never actual run evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fake_campaign_contract", PROJECT / "tools/audit_online_v4_campaign.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class OnlineV4CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="explicit-fake-v4-campaign-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        patcher = patch.object(audit, "ROOT", self.root)
        patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(audit.archive_verifier, "verify_archive", side_effect=self.fake_verify)
        self.verify = patcher.start(); self.addCleanup(patcher.stop)
        self.log = self.root / "results/campaign/runtime/server.stderr.log"
        self.output = self.root / "results/campaign"
        self.first = self.make_run("v4_development_01", minute=1)
        self.write_log(312)

    def make_run(self, name, *, mode="development", passed=False, minute=1, selected=None):
        directory = self.root / ".runs" / name
        directory.mkdir(parents=True)
        for source in audit.archive_verifier.NAMES:
            path = directory / "sources" / source
            path.parent.mkdir(exist_ok=True)
            path.write_text("# EXPLICIT FAKE SOURCE, NEVER EXECUTED\n" + source, encoding="utf-8")
        count = 8 if mode == "development" else 32
        protocol = {"schema": "rcwt-online-memory/4", "mode": mode, "count": count,
                    "split": "train" if mode == "development" else "test",
                    "frozen_at_utc": f"2026-09-12T00:{minute:02d}:00+00:00", "fixture": name,
                    "source_sha256": {"src/" + source: audit._hash(directory / "sources" / source)
                                      for source in audit.archive_verifier.NAMES},
                    "development": {"protocol_sha256": selected} if mode == "confirmatory" else None}
        for name in audit.CORE:
            (directory / name).write_text("EXPLICIT FAKE MANIFEST\n", encoding="utf-8")
        (directory / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        rows = [{"episode_id": f"FAKE-{i}", "policy": policy, "successes": 6 if passed and policy == "structured" else 4,
                 "steps": 8, "unsafe_actions": 0, "unsafe_booked_cents": 0,
                 "model_calls": 23 if policy == "summary" else 16}
                for i in range(count) for policy in ("summary", "structured")]
        (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return directory

    def write_log(self, count):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.log.write_text("EXPLICIT FAKE SERVER LOG; NOT INFERENCE EVIDENCE\n" +
                            "processing task, is_child = 1\n" +
                            "".join(f"fake timestamp {i}: {audit.MARKER}\n" for i in range(count)), encoding="utf-8")

    def fake_verify(self, directory):
        protocol = json.loads((directory / "protocol.json").read_text())
        return {"status": "PASS", "inference_calls": 0, "verified_steps": protocol["count"] * 16,
                "protocol_sha256": audit._hash(directory / "protocol.json"),
                "verification": {"status": "PASS", "scope": "EXPLICIT FAKE PATCHED RECEIPT"}}

    def facts(self, runs=None):
        return audit.campaign_facts([self.first] if runs is None else runs, self.log)

    def test_one_failed_development_is_fully_accounted_not_renamed_as_gain(self):
        result = self.facts()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["model_calls"], 312)
        self.assertEqual(result["server_log"]["matching_lines"], 312)
        self.assertFalse(result["runs"][0]["development_screen_passed"])
        self.assertIsNone(result["selected_development_protocol_sha256"])
        self.assertFalse(result["global_non_repetition_proven"])
        self.assertIn("not portable global replay", result["scope"])
        self.assertEqual(result["api_cost_usd"], 0)
        self.assertIsNone(result["total_monetary_cost_usd"])
        self.assertIn("not model gain", audit.render_report(result))
        serialized = json.dumps(result)
        self.assertNotIn(str(self.root), serialized)
        self.assertEqual(result["explicit_paths"], [".runs/v4_development_01"])
        self.assertEqual(result["server_log"]["path"], "results/campaign/runtime/server.stderr.log")

    def test_release_copies_are_verified_but_not_counted_twice(self):
        release = self.root / "results/development/attempt_01"
        shutil.copytree(self.first, release)
        result = self.facts([release, self.first, release])
        self.assertEqual(result["unique_protocols"], 1)
        self.assertEqual(result["copies_not_double_counted"], 1)
        self.assertEqual(result["model_calls"], 312)
        self.assertEqual(len(result["runs"][0]["paths"]), 2)
        self.assertEqual(self.verify.call_count, 2)
        (release / "traces.jsonl").write_text("DIFFERENT FAKE TRAJECTORY", encoding="utf-8")
        with self.assertRaisesRegex(audit.CampaignBlocked, "not a release copy"):
            self.facts([release, self.first])

    def test_inventory_requires_every_scoped_run_and_ignores_non_v4_directories(self):
        (self.root / ".runs/not_v4_unrelated").mkdir()
        self.facts()
        second = self.make_run("v4_development_02", minute=2)
        self.write_log(624)
        with self.assertRaisesRegex(audit.CampaignBlocked, "exactly cover"):
            self.facts()
        self.assertEqual(self.facts([self.first, second])["model_calls"], 624)

    def test_aborted_partial_and_active_inventory_block_without_hiding_the_directory(self):
        second = self.make_run("v4_development_02", minute=2)
        for flag in ("aborted.json", "partial-step.json"):
            (second / flag).write_text("{}")
            with self.assertRaisesRegex(audit.CampaignBlocked, "v4_development_02"):
                self.facts()
            (second / flag).unlink()
        (second / "completion.json").unlink()
        with self.assertRaisesRegex(audit.CampaignBlocked, "missing=.*completion.json"):
            self.facts()
        self.verify.assert_not_called()
        self.assertFalse((self.output / "campaign.json").exists())

    def test_first_pass_must_be_last_development_and_limits_are_not_relaxed(self):
        second = self.make_run("v4_development_02", passed=True, minute=2)
        self.write_log(624)
        result = self.facts([second, self.first])
        self.assertEqual(result["selected_development_protocol_sha256"], audit._hash(second / "protocol.json"))
        rows = [json.loads(line) for line in (self.first / "episodes.jsonl").read_text().splitlines()]
        for row in rows:
            if row["policy"] == "structured": row["successes"] = 7
        (self.first / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(audit.CampaignBlocked, "after the first PASS"):
            self.facts([self.first, second])
        third = self.make_run("v4_development_03", minute=3)
        self.write_log(936)
        with self.assertRaisesRegex(audit.CampaignBlocked, "at most two"):
            self.facts([self.first, second, third])

    def test_one_confirmation_must_follow_and_select_a_passing_development(self):
        second = self.make_run("v4_development_02", passed=True, minute=2)
        conf = self.make_run("v4_confirmation", mode="confirmatory", minute=3,
                             selected=audit._hash(second / "protocol.json"))
        self.write_log(1872)
        result = self.facts([conf, self.first, second])
        self.assertTrue(result["confirmation_present"])
        self.assertEqual(result["model_calls"], 1872)
        self.assertEqual(result["unique_protocols"], 3)
        path = conf / "protocol.json"
        protocol = json.loads(path.read_text())
        for selected in (None, {"protocol_sha256": audit._hash(self.first / "protocol.json")}):
            protocol["development"] = selected; path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(audit.CampaignBlocked, "select the first passing"):
                self.facts([conf, self.first, second])

    def test_log_mismatch_both_directions_blocks_and_only_literal_parent_lines_count(self):
        for count in (311, 313):
            self.write_log(count)
            with self.assertRaisesRegex(audit.CampaignBlocked, "accounting mismatch"):
                self.facts()
        self.write_log(312)
        with self.log.open("a") as handle:
            handle.write("child=0\nprocessing task, is_child = 1\n")
        self.assertEqual(self.facts()["server_log"]["matching_lines"], 312)

    def test_creation_is_exclusive_and_verify_is_byte_exact_without_writes(self):
        facts = audit.audit_campaign([self.first], self.log, self.output)
        self.assertEqual(json.loads((self.output / "campaign.json").read_text()), facts)
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with patch.object(Path, "write_bytes", side_effect=AssertionError("Unexpected write")), patch.object(
                Path, "mkdir", side_effect=AssertionError("Unexpected mkdir")):
            receipt = audit.audit_campaign([self.first], self.log, self.output, verify=True)
        self.assertTrue(receipt["read_only"])
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})
        with self.assertRaises(FileExistsError):
            audit.audit_campaign([self.first], self.log, self.output)
        for name in audit.OUTPUTS:
            path = self.output / name; original = path.read_bytes()
            path.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(audit.CampaignBlocked, "differs"):
                audit.audit_campaign([self.first], self.log, self.output, verify=True)
            path.write_bytes(original)

    def test_source_log_and_inventory_changes_during_audit_invalidate_receipt(self):
        baseline = audit._inventory()
        with patch.object(audit, "_inventory", side_effect=[baseline, []]):
            with self.assertRaisesRegex(audit.CampaignBlocked, "changed during"):
                self.facts()
        original_verify = self.fake_verify
        def mutate_log(directory):
            receipt = original_verify(directory)
            with self.log.open("a") as handle: handle.write("new line after read\n")
            return receipt
        self.verify.side_effect = mutate_log
        with self.assertRaisesRegex(audit.CampaignBlocked, "changed during"):
            self.facts()

    def test_bad_archive_pass_and_external_log_are_rejected(self):
        self.verify.side_effect = None
        for receipt in ({"status": "FAIL"}, {"status": "PASS", "inference_calls": 1},
                        {"status": "PASS", "inference_calls": 0, "protocol_sha256": "wrong"}):
            self.verify.return_value = receipt
            with self.assertRaisesRegex(audit.CampaignBlocked, "bind a complete"):
                self.facts()
        with tempfile.TemporaryDirectory(prefix="fake-external-log-") as temporary:
            path = Path(temporary) / "server.log"; path.write_text("fake")
            with self.assertRaisesRegex(audit.CampaignBlocked, "copied inside"):
                audit.campaign_facts([self.first], path)

    def test_cli_blocks_explicitly_without_creating_a_complete_receipt(self):
        with patch.object(audit, "audit_campaign", side_effect=audit.CampaignBlocked("FAKE partial attempt")), patch(
                "sys.argv", ["audit", "--runs", "fake", "--server-log", "fake.log", "--output-dir", "fake-output"]), patch("builtins.print") as output:
            with self.assertRaises(SystemExit) as raised:
                audit.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(json.loads(output.call_args.args[0])["status"], "BLOCKED")

    def test_two_confirmations_are_blocked_even_when_both_select_the_passing_development(self):
        second = self.make_run("v4_development_02", passed=True, minute=2)
        identity = audit._hash(second / "protocol.json")
        runs = [self.first, second] + [self.make_run(f"v4_confirmation_{i}", mode="confirmatory",
                 minute=3 + i, selected=identity) for i in range(2)]
        self.write_log(3120)
        with self.assertRaisesRegex(audit.CampaignBlocked, "one confirmation"):
            self.facts(runs)

    def test_accuracy_improvement_cannot_override_either_development_safety_guard(self):
        second = self.make_run("v4_development_02", passed=True, minute=2)
        path = second / "episodes.jsonl"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        self.write_log(624)
        for field in ("unsafe_actions", "unsafe_booked_cents"):
            rows = copy.deepcopy(original)
            if field == "unsafe_booked_cents": rows[0]["unsafe_actions"] = rows[1]["unsafe_actions"] = 1
            rows[1][field] = 1
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertIsNone(self.facts([self.first, second])["selected_development_protocol_sha256"])


if __name__ == "__main__":
    unittest.main()
