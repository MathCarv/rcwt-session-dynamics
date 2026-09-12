"""Explicitly invented display fixtures with a mocked integrity gate, never R1 evidence."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_r1_demo", ROOT / "tools/export_r1_demo.py")
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


def fake_artifacts() -> dict:
    """Handwritten invented data; do not call any registered corpus generator."""
    public, traces, summaries = [], [], []
    for episode in range(32):
        episode_id = "EXPLICIT-FAKE-FIRST" if episode == 0 else f"EXPLICIT-FAKE-BETTER-{episode}"
        steps = [{"step_index": step, "observations": [{"content": "EXPLICIT FAKE OBSERVATION"}],
                  "task": {"case_id": f"EXPLICIT-FAKE-CASE-{episode}-{step}", "operation": "payout",
                           "instruction": "EXPLICIT FAKE REQUEST"}} for step in range(8)]
        public.append({"episode_id": episode_id, "family": "EXPLICIT-FAKE-FAMILY", "steps": steps})
        for policy in demo.POLICIES:
            successes = 0
            for step_index, step in enumerate(steps):
                action = {"tool": "record_decision", "arguments": {"case_id": step["task"]["case_id"],
                          "decision": "ask_info", "amount_cents": 0, "reason_code": "missing_evidence"}}
                # The fixed first pair is intentionally worse than every later pair.
                success = episode > 0 or (policy == "structured" and step_index == 4)
                successes += success
                traces.append({"episode_id": episode_id, "policy": policy, "step_index": step_index,
                               "public_step": step, "sha256": hashlib.sha256(
                                   f"EXPLICIT-FAKE-NON-EVIDENCE-{episode}-{policy}-{step_index}".encode()).hexdigest(),
                               "memory_before": f"EXPLICIT FAKE STORED {policy} {step_index}\nunchanged",
                               "actor_memory_before": f"EXPLICIT FAKE ACTOR CONTEXT {policy} {step_index}",
                               "action": json.dumps(action), "evidence_check": {"status": "EXPLICIT FAKE"},
                               "tool_result": {"simulated": True, "accepted": False},
                               "score": {"success": success, "failure_category": "" if success else "explicit_fake_failure",
                                         "expected_action": action, "detail": "EXPLICIT FAKE PRIVATE GRADE"}})
            summaries.append({"episode_id": episode_id, "policy": policy, "steps": 8, "successes": successes})
    analysis = {
        "input_sha256": "EXPLICIT-FAKE-ANALYSIS-NOT-EVIDENCE",
        "policies": {policy: {"steps": 256, "successes": 248 + (policy == "structured"),
                              "unsafe_actions": 1, "total_tokens": 100,
                              "step_latency_seconds": {"p50": 2.5}} for policy in demo.POLICIES},
        "comparisons": {"structured_vs_summary": {"delta_percentage_points": 0.390625,
                         "ci95_percentage_points": [-1.0, 2.0], "by_family": {}}},
        "evaluation_cost": {"model_calls": 1248, "api_cost_usd": 0, "total_monetary_cost_usd": None},
        "accuracy_gate": {"passed": False}, "descriptive_safety_guard": {"passed": True},
        "improvement_gate": {"passed": False},
    }
    return {"protocol.json": {"schema": "EXPLICIT-FAKE-PROTOCOL-NOT-EVIDENCE"},
            "public.json": public, "traces.jsonl": traces, "episodes.jsonl": summaries,
            "analysis.json": analysis}


class ExportR1DemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="explicit-fake-r1-demo-")
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.run = self.parent / "EXPLICIT-FAKE-R1"
        self.output = self.parent / "EXPLICIT-FAKE-DISPLAY"
        self.run.mkdir()
        self.data = fake_artifacts()
        self.write_fixture()
        self.pins = {"candidate": {"EXPLICIT-FAKE-SOURCE": "not-evidence"},
                     "orchestration": {"EXPLICIT-FAKE-ORCHESTRATOR": "not-evidence"}}
        self.receipt = {
            "status": "PASS", "integrity_only": True, "replication_id": "R1",
            "schema": "rcwt-online-replication/1", "paired_episodes": 32,
            "verified_steps": 512, "recorded_generation_calls": 1248, "inference_calls": 0, "writes": 0,
            "current_source_sha256": self.pins["candidate"], "orchestrator_sha256": self.pins["orchestration"],
            "report_verifier_sha256": demo._hash(ROOT / "tools/verify_v4_replication_report.py"),
            "recomputed_input_sha256": self.data["analysis.json"]["input_sha256"],
        }
        self.gate_returned = False

        def fake_gate(directory):
            self.assertEqual(directory, self.run.resolve())
            self.gate_returned = True
            return {"files_sha256": demo._snapshot(directory), **copy.deepcopy(self.receipt)}

        real_read, real_jsonl = demo._read, demo._read_jsonl

        def after_gate(function):
            def checked(path):
                self.assertTrue(self.gate_returned, "Display read happened before the full report gate")
                return function(path)
            return checked

        self.mocks = {}
        for name, patcher in {
            "gate": patch.object(demo, "verify_report", side_effect=fake_gate),
            "pins": patch.object(demo, "_source_pins", side_effect=lambda _: copy.deepcopy(self.pins)),
            "read": patch.object(demo, "_read", side_effect=after_gate(real_read)),
            "jsonl": patch.object(demo, "_read_jsonl", side_effect=after_gate(real_jsonl)),
            "network": patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("HTTP forbidden")),
            "model": patch("rcwt_local_model.LocalModelClient.__init__", side_effect=AssertionError("Model forbidden")),
        }.items():
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def write_fixture(self):
        for name, value in self.data.items():
            if name.endswith("jsonl"):
                content = "\n".join(json.dumps(row) for row in value) + "\n"
                (self.run / name).write_bytes(content.encode())
            else:
                (self.run / name).write_bytes(demo._json_bytes(value))

    def test_build_fixed_first_episode_and_fifth_step_without_artifact_writes(self):
        before = demo._snapshot(self.run)
        bundle = demo.build_demo(self.run)
        document = json.loads(bundle.artifacts["step5.json"])
        self.assertEqual(document["selection"]["episode_id"], "EXPLICIT-FAKE-FIRST")
        self.assertEqual(document["selection"]["manifest_episode_index"], 0)
        self.assertEqual(document["selection"]["step_index"], 4)
        self.assertFalse(document["selection"]["score_search"])
        self.assertEqual(len(document["decisions"]), 8)
        self.assertEqual(document["selected_pair_totals"]["summary"]["successes"], 0)
        self.assertEqual(document["selected_pair_totals"]["structured"]["successes"], 1)
        for policy, line in (("summary", 5), ("structured", 13)):
            original = self.data["traces.jsonl"][line - 1]
            focus = document["focused_step"]["policies"][policy]
            self.assertEqual(focus["trace_line"], line)
            self.assertEqual(focus["trace_sha256"], original["sha256"])
            for name in ("memory_before", "actor_memory_before", "action", "score", "tool_result"):
                self.assertEqual(focus[name], original[name])
        self.assertIn("0/8 correct", bundle.artifacts["README.md"].decode())
        self.assertNotIn(b"EXPLICIT-FAKE-BETTER", bundle.artifacts["step5.json"])
        self.assertFalse(document["whole_cohort"]["improvement_gate"]["passed"])
        self.assertEqual(before, demo._snapshot(self.run))
        self.assertFalse(self.output.exists())
        self.assertTrue(all(len(value) < demo.MAX_ARTIFACT_BYTES for value in bundle.artifacts.values()))
        self.mocks["network"].assert_not_called()
        self.mocks["model"].assert_not_called()

    def test_exclusive_creation_and_read_only_exact_byte_verification(self):
        original = demo._snapshot(self.run)
        created = demo.export_demo(self.run, self.output)
        self.assertEqual(created["display_files_written"], 2)
        self.assertEqual(set(created["files_sha256"]), {"README.md", "step5.json"})
        timestamps = {path.name: path.stat().st_mtime_ns for path in self.output.iterdir()}
        verified = demo.export_demo(self.run, self.output, verify=True)
        self.assertEqual(verified["mode"], "verified")
        self.assertEqual(verified["display_files_written"], 0)
        self.assertEqual(created["files_sha256"], verified["files_sha256"])
        self.assertEqual(timestamps, {path.name: path.stat().st_mtime_ns for path in self.output.iterdir()})
        self.assertEqual(original, demo._snapshot(self.run))
        self.assertEqual(self.mocks["gate"].call_count, 2)

    def test_windows_and_posix_snapshot_order_produce_identical_demo_bytes(self):
        self.data["RESULTS.md"] = {"notice": "EXPLICIT FAKE ORDER FIXTURE, NOT R1 EVIDENCE"}
        self.write_fixture()
        original_snapshot = demo._snapshot

        def windows_order(directory):
            return dict(sorted(original_snapshot(directory).items(), key=lambda item: item[0].lower()))

        def posix_order(directory):
            return dict(sorted(original_snapshot(directory).items()))

        self.assertNotEqual(list(windows_order(self.run)), list(posix_order(self.run)))
        with patch.object(demo, "_snapshot", side_effect=windows_order):
            windows_bundle = demo.build_demo(self.run)
            created = demo.export_demo(self.run, self.output)
        saved_before = original_snapshot(self.output)
        with patch.object(demo, "_snapshot", side_effect=posix_order):
            posix_bundle = demo.build_demo(self.run)
            verified = demo.export_demo(self.run, self.output, verify=True)
        self.assertEqual(windows_bundle.artifacts, posix_bundle.artifacts)
        document = json.loads(windows_bundle.artifacts["step5.json"])
        keys = list(document["provenance"]["files_sha256"])
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(created["files_sha256"], verified["files_sha256"])
        self.assertEqual(saved_before, original_snapshot(self.output))
        self.assertEqual(verified["display_files_written"], 0)

    def test_existing_destination_is_never_overwritten(self):
        self.output.mkdir()
        (self.output / "KEEP.txt").write_bytes(b"EXPLICIT FAKE USER DATA")
        before = demo._snapshot(self.output)
        with self.assertRaises(FileExistsError):
            demo.export_demo(self.run, self.output)
        self.assertEqual(before, demo._snapshot(self.output))
        self.mocks["gate"].assert_not_called()

    def test_verify_missing_destination_never_creates_it(self):
        with self.assertRaises(ValueError):
            demo.export_demo(self.run, self.output, verify=True)
        self.assertFalse(self.output.exists())

    def test_partial_failed_or_nonlocal_gate_prevents_display_reads(self):
        receipt = copy.deepcopy(self.receipt)
        for change in ({"status": "FAIL"}, {"verified_steps": 173}, {"paired_episodes": 31},
                       {"inference_calls": 1}, {"writes": 1}, {"verified_steps": True},
                       {"recorded_generation_calls": 1247}, {"replication_id": "EXPLICIT-FAKE-WRONG"}):
            with self.subTest(change=change):
                self.receipt = {**receipt, **change}
                with self.assertRaisesRegex(ValueError, "full offline report integrity"):
                    demo.export_demo(self.run, self.output)
                self.assertFalse(self.output.exists())
        self.mocks["read"].assert_not_called()
        self.mocks["jsonl"].assert_not_called()

    def test_snapshot_must_match_the_gate_receipt_before_any_display_read(self):
        self.receipt["files_sha256"] = {"EXPLICIT-FAKE-MISMATCH": "no"}
        with self.assertRaisesRegex(ValueError, "snapshot differs"):
            demo.build_demo(self.run)
        self.mocks["read"].assert_not_called()
        self.mocks["jsonl"].assert_not_called()

    def test_source_pin_drift_is_rejected(self):
        self.receipt["current_source_sha256"] = {"EXPLICIT-FAKE-DRIFT": "no"}
        with self.assertRaisesRegex(ValueError, "source pins differ"):
            demo.build_demo(self.run)
        self.mocks["jsonl"].assert_not_called()

    def test_mutation_during_display_reads_is_rejected_without_output(self):
        original_reader = self.mocks["read"].side_effect

        def mutate_after_read(path):
            value = original_reader(path)
            if path.name == "analysis.json":
                (self.run / "EXPLICIT-FAKE-MUTATION.txt").write_bytes(b"not evidence")
            return value

        self.mocks["read"].side_effect = mutate_after_read
        with self.assertRaisesRegex(ValueError, "changed during demo"):
            demo.export_demo(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_source_drift_after_read_is_rejected(self):
        self.mocks["pins"].side_effect = [copy.deepcopy(self.pins), {"EXPLICIT-FAKE-DRIFT": "no"}]
        with self.assertRaisesRegex(ValueError, "changed during demo"):
            demo.build_demo(self.run)

    def test_verify_rejects_changed_bytes_missing_files_and_extra_entries(self):
        demo.export_demo(self.run, self.output)
        original = (self.output / "README.md").read_bytes()
        for content in (original + b"\n", original.replace(b"\n", b"\r\n")):
            (self.output / "README.md").write_bytes(content)
            with self.assertRaisesRegex(ValueError, "exact verified fixed-demo bytes"):
                demo.export_demo(self.run, self.output, verify=True)
            self.assertEqual((self.output / "README.md").read_bytes(), content)
        (self.output / "README.md").write_bytes(original)
        for name in ("EXPLICIT-FAKE-EXTRA.txt", "EXPLICIT-FAKE-EMPTY-DIRECTORY"):
            path = self.output / name
            path.write_bytes(b"not evidence") if path.suffix else path.mkdir()
            with self.assertRaisesRegex(ValueError, "exactly README.md and step5.json"):
                demo.export_demo(self.run, self.output, verify=True)
            path.unlink() if path.is_file() else path.rmdir()
        (self.output / "README.md").unlink()
        with self.assertRaisesRegex(ValueError, "exactly README.md and step5.json"):
            demo.export_demo(self.run, self.output, verify=True)

    def test_only_a_distinct_sibling_destination_is_allowed(self):
        for path in (self.run, self.run / "nested-demo", self.parent, self.parent / "other" / "demo"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "distinct sibling"):
                demo.export_demo(self.run, path)
        self.mocks["gate"].assert_not_called()

    def test_path_check_does_not_require_windows_stat_constants(self):
        with patch.object(demo, "stat", SimpleNamespace()):
            self.assertEqual(demo._ordinary_path(self.run), self.run.resolve())

    def test_destination_appearing_during_gate_is_not_overwritten(self):
        original_gate = self.mocks["gate"].side_effect

        def concurrent_destination(directory):
            receipt = original_gate(directory)
            self.output.mkdir()
            (self.output / "KEEP.txt").write_bytes(b"EXPLICIT FAKE USER DATA")
            return receipt

        self.mocks["gate"].side_effect = concurrent_destination
        with self.assertRaises(FileExistsError):
            demo.export_demo(self.run, self.output)
        self.assertEqual((self.output / "KEEP.txt").read_bytes(), b"EXPLICIT FAKE USER DATA")
        self.assertEqual([path.name for path in self.output.iterdir()], ["KEEP.txt"])

    def test_physical_trace_lines_survive_blank_lines_and_different_trace_order(self):
        traces = list(reversed(self.data["traces.jsonl"]))
        content = "\n" + "\n\n".join(json.dumps(row) for row in traces) + "\n"
        (self.run / "traces.jsonl").write_bytes(content.encode())
        document = json.loads(demo.build_demo(self.run).artifacts["step5.json"])
        for policy in demo.POLICIES:
            index = next(i for i, row in enumerate(traces) if row["episode_id"] == "EXPLICIT-FAKE-FIRST"
                         and row["step_index"] == 4 and row["policy"] == policy)
            self.assertEqual(document["focused_step"]["policies"][policy]["trace_line"], 2 + index * 2)

    def test_missing_duplicate_mismatched_trajectory_and_summary_rejected(self):
        pristine = copy.deepcopy(self.data)
        mutations = (
            lambda: self.data["traces.jsonl"].pop(),
            lambda: self.data["traces.jsonl"].__setitem__(0, copy.deepcopy(self.data["traces.jsonl"][1])),
            lambda: self.data["traces.jsonl"][0].__setitem__("public_step", {"EXPLICIT FAKE": "MISMATCH"}),
            lambda: self.data["episodes.jsonl"][0].__setitem__("successes", 8),
        )
        for mutate in mutations:
            self.data = copy.deepcopy(pristine)
            mutate()
            self.write_fixture()
            with self.assertRaises(ValueError):
                demo.build_demo(self.run)

    def test_markdown_table_escapes_observation_identifiers(self):
        value = "EXPLICIT FAKE | <script> `tick`\nline"
        self.data["public.json"][0]["steps"][0]["task"]["case_id"] = value
        self.write_fixture()
        rendered = demo.build_demo(self.run).artifacts["README.md"].decode()
        self.assertIn("EXPLICIT FAKE &#124; &lt;script&gt; &#96;tick&#96; line", rendered)
        self.assertNotIn("<script>", rendered)

    def test_display_size_limit_is_enforced_without_creating_destination(self):
        with patch.object(demo, "MAX_ARTIFACT_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "display limit"):
                demo.export_demo(self.run, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
