"""Invented display fixtures and FAKE gate, never real R1 corpus or inference."""
from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "v4_replication_replay", Path(__file__).resolve().parents[1] / "tools/replay_v4_replication.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def fake_artifacts():
    """Handcrafted display records; never call the registered corpus generator."""
    public, traces, summaries = [], [], []
    protocol = {"schema": "rcwt-online-replication/1", "mode": "replication", "split": "test",
                "replication_id": "R1", "count": 32, "dataset_seed": 2026091210,
                "policies": ["summary", "structured"]}
    for episode_index in range(32):
        episode_id = "EXPLICIT-FAKE-FIRST" if episode_index == 0 else f"EXPLICIT-FAKE-BETTER-{episode_index}"
        steps = [{"task": {"case_id": f"EXPLICIT-FAKE-{episode_index}-{step}", "operation": "payout"},
                  "observations": ["EXPLICIT FAKE PUBLIC OBSERVATION"]} for step in range(8)]
        public.append({"episode_id": episode_id, "steps": steps})
        for policy in ("summary", "structured"):
            summaries.append({"episode_id": episode_id, "policy": policy, "episode_seconds": 17.0})
            for step_index, step in enumerate(steps):
                purposes = ["draft:" + policy, "action:" + policy]
                if policy == "summary" and step_index < 7:
                    purposes.append("memory:summary")
                events = [{"method": "complete", "result": {
                    "purpose": purpose, "text": "EXPLICIT FAKE RAW " + purpose,
                    "finish_reason": "stop", "prompt_tokens": 10, "completion_tokens": 2,
                    "wall_seconds": 0.25, "api_cost_usd": 0}} for purpose in purposes]
                traces.append({
                    "episode_id": episode_id, "policy": policy, "step_index": step_index, "split": "test",
                    "public_step": step, "sha256": "EXPLICIT-FAKE-NON-EVIDENCE-HASH",
                    "memory_before": "EXPLICIT FAKE STORED", "actor_memory_before": "EXPLICIT FAKE CONTEXT",
                    "memory_after": "EXPLICIT FAKE RETAINED", "memory_tokens": 4, "actor_memory_tokens": 5,
                    "memory_truncated": False, "evidence_check": None, "action": None,
                    "tool_result": {"accepted": False, "simulated": True},
                    "score": {"success": episode_index > 0, "failure_category": "explicit_fake_invalid"},
                    "client_events": events, "step_seconds": 2.0,
                })
    # Reversed trace order is intentionally different from the manifest; the
    # display must still pick index zero, not the first trace or a better score.
    return protocol, public, list(reversed(traces)), summaries


class V4ReplicationReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path("EXPLICIT-FAKE-R1-DISPLAY")
        self.protocol, self.public, self.traces, self.summaries = fake_artifacts()
        self.verified = False
        self.receipt = {"status": "PASS", "read_only": True, "integrity_only": True,
                        "inference_calls": 0, "verified_steps": 512,
                        "verified_episode_summaries": 64, "generation_calls": 1248}

        def fake_gate(directory):
            self.assertEqual(directory, self.directory)
            self.verified = True
            return copy.deepcopy(self.receipt)

        def fake_read(path):
            self.assertTrue(self.verified, "Display artifacts read before gate")
            return copy.deepcopy({"protocol.json": self.protocol, "public.json": self.public}[path.name])

        def fake_jsonl(path):
            self.assertTrue(self.verified, "Display traces read before gate")
            return copy.deepcopy({"traces.jsonl": self.traces, "episodes.jsonl": self.summaries}[path.name])

        self.mocks = {}
        for name, patcher in {
            "snapshot": patch.object(replay, "_snapshot", return_value={"EXPLICIT-FAKE": "SAME"}),
            "gate": patch.object(replay, "verify_run", side_effect=fake_gate),
            "read": patch.object(replay, "read", side_effect=fake_read),
            "jsonl": patch.object(replay, "read_jsonl", side_effect=fake_jsonl),
            "network": patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Real HTTP forbidden")),
            "model": patch("rcwt_local_model.LocalModelClient.__init__", side_effect=AssertionError("Real model forbidden")),
        }.items():
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def test_fixed_first_pair_shows_both_arms_raw_failures_and_all_costs(self):
        rendered = replay.render_replay(self.directory)
        self.assertIn("Fixed manifest episode index 0: EXPLICIT-FAKE-FIRST", rendered)
        self.assertNotIn("EXPLICIT-FAKE-BETTER", rendered)
        self.assertEqual(rendered.count("\nSTEP "), 8)
        self.assertEqual(rendered.count("\nSUMMARY\n"), 8)
        self.assertEqual(rendered.count("\nSTRUCTURED\n"), 8)
        for label in ("STORED MEMORY BEFORE:", "EXACT CONTEXT SENT TO ACTOR:", "RAW PLAN (NOT EXECUTED):",
                      "RAW FINAL RESPONSE:", "ACTUAL ACTION (FINAL ONLY): null", "ACTUAL FICTIONAL TOOL RECEIPT:",
                      "PRIVATE EVALUATION:", "RETAINED MEMORY AFTER:"):
            self.assertEqual(rendered.count(label), 16)
        self.assertIn('"success": false', rendered)
        self.assertEqual(rendered.count("RECORDED GENERATION CALL:"), 39)
        self.assertIn('"generation_calls": 736', rendered)
        self.assertIn('"generation_calls": 512', rendered)
        self.assertIn('"accounted_episode_wall_seconds": 544.0', rendered)
        self.assertIn("173/512-decision interruption", rendered)
        self.mocks["gate"].assert_called_once_with(self.directory)
        self.mocks["network"].assert_not_called()
        self.mocks["model"].assert_not_called()

    def test_failed_or_partial_gate_prevents_display_reads(self):
        complete = copy.deepcopy(self.receipt)
        for change in ({"status": "FAIL"}, {"verified_steps": 173}, {"inference_calls": 1}):
            self.receipt = {**complete, **change}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "complete offline-verified R1"):
                replay.render_replay(self.directory)
        self.mocks["read"].assert_not_called()
        self.mocks["jsonl"].assert_not_called()

    def test_mutated_or_missing_recorded_display_evidence_is_rejected(self):
        original = copy.deepcopy(self.traces)
        self.traces.pop()
        with self.assertRaisesRegex(ValueError, "complete-cohort"):
            replay.render_replay(self.directory)
        self.traces = original
        first = next(row for row in self.traces if row["episode_id"] == "EXPLICIT-FAKE-FIRST")
        first["client_events"][0]["result"]["purpose"] = "EXPLICIT-FAKE-UNREGISTERED-PASS"
        with self.assertRaisesRegex(ValueError, "two-pass actor"):
            replay.render_replay(self.directory)
        self.traces = fake_artifacts()[2]
        self.mocks["snapshot"].side_effect = [{"before": "original"}, {"after": "changed"}]
        with self.assertRaisesRegex(ValueError, "changed during display"):
            replay.render_replay(self.directory)


if __name__ == "__main__":
    unittest.main()
