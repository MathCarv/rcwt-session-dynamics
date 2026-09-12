"""Preserve the aborted development record without executing archived sources.

These are offline consistency checks of recorded artifacts, not independent
proof of physical model calls, runtime measurements, or held-out performance.
The aborted run must never be pooled with a later confirmatory evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path


ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "agent_development"
PINNED_HASHES = {
    "protocol_sha256": ("protocol.json", "98171b7ae67b5844f633e2a8517c852d3c8085876a27c9ba05540769e6ef48ae"),
    "training_traces_sha256": ("train/traces.jsonl", "55a09e3b681e15218d38bacfd919cc2b0b797e9bf26c67bd962a585d035e322e"),
    "validation_traces_sha256": ("validation/traces.jsonl", "906d146ff79900ce67aea42e0cd51773daed1c04dfda2d8e61dcce742884e94f"),
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class AbortedDevelopmentEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = _json(ARCHIVE / "ABORTED.json")
        cls.protocol = _json(ARCHIVE / "protocol.json")
        cls.training = _jsonl(ARCHIVE / "train/traces.jsonl")
        cls.validation = _jsonl(ARCHIVE / "validation/traces.jsonl")

    def test_aborted_classification_is_not_confirmatory_evidence(self):
        self.assertEqual(self.receipt["status"], "ABORTED_DEVELOPMENT_RUN_BEFORE_HELD_OUT")
        self.assertIn("not confirmatory evidence", self.receipt["classification"])
        self.assertIn("lower-bound development counters", self.receipt["accounting_limit"])
        self.assertEqual(self.receipt["api_cost_usd"], 0)
        self.assertIsNone(self.receipt["total_monetary_cost_usd"])
        self.assertFalse((ARCHIVE / "analysis.json").exists())
        self.assertFalse((ARCHIVE / "RESULTS.md").exists())

    def test_original_protocol_and_trace_bytes_remain_pinned(self):
        for receipt_key, (relative, expected) in PINNED_HASHES.items():
            with self.subTest(artifact=relative):
                self.assertEqual(self.receipt[receipt_key], expected)
                self.assertEqual(hashlib.sha256((ARCHIVE / relative).read_bytes()).hexdigest(), expected)

    def test_archived_sources_match_original_protocol_without_importing_them(self):
        self.assertEqual(self.receipt["source_snapshot"], "source_snapshot/")
        snapshot = ARCHIVE / "source_snapshot"
        source_hashes = self.protocol["source_sha256"]
        names = [Path(relative).name for relative in source_hashes]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual({path.name for path in snapshot.glob("*.py")}, set(names))
        for relative, expected in source_hashes.items():
            with self.subTest(source=relative):
                self.assertEqual(hashlib.sha256((snapshot / Path(relative).name).read_bytes()).hexdigest(), expected)

    def test_partial_split_counts_and_no_held_out_execution_artifacts(self):
        self.assertEqual(len(self.training), 32)
        self.assertEqual(len(self.validation), 75)
        self.assertEqual(self.receipt["completed_training_steps"], len(self.training))
        self.assertEqual(self.receipt["recorded_validation_steps"], len(self.validation))
        self.assertTrue(all(row["split"] == "train" for row in self.training))
        self.assertTrue(all(row["split"] == "validation" for row in self.validation))
        self.assertEqual(sum(row["score"]["success"] for row in self.training), 6)
        self.assertEqual(self.receipt["held_out_calls"], 0)
        self.assertIs(self.receipt["test_freeze_existed_at_stop"], False)
        self.assertIs(self.receipt["test_directory_existed_at_stop"], False)
        self.assertFalse((ARCHIVE / "test-freeze.json").exists())
        self.assertFalse((ARCHIVE / "test").exists())
        self.assertFalse((ARCHIVE / "selection.json").exists())
        # A planned held-out corpus is not evidence that it was executed.
        self.assertTrue((ARCHIVE / "test-public.json").is_file())
        self.assertTrue((ARCHIVE / "test-oracle.json").is_file())

    def test_lower_bound_recorded_metering_includes_policy_proposal(self):
        rows = self.training + self.validation
        calls = [call for row in rows for call in row["model_calls"]]
        calls.append(_json(ARCHIVE / "candidate-policies.json")["proposal_call"])
        self.assertEqual(len(calls), 202)
        self.assertEqual(self.receipt["recorded_model_calls_including_proposal"], len(calls))
        prompt = sum(call["prompt_tokens"] for call in calls)
        completion = sum(call["completion_tokens"] for call in calls)
        self.assertEqual(prompt, 179899)
        self.assertEqual(completion, 30962)
        self.assertEqual(self.receipt["recorded_prompt_tokens"], prompt)
        self.assertEqual(self.receipt["recorded_completion_tokens"], completion)
        self.assertAlmostEqual(self.receipt["recorded_inference_wall_seconds"], math.fsum(call["wall_seconds"] for call in calls), delta=1e-8)
        self.assertAlmostEqual(self.receipt["recorded_step_seconds"], math.fsum(row["step_seconds"] for row in rows), delta=1e-8)
        self.assertTrue(all(call["api_cost_usd"] == 0 for call in calls))
        self.assertTrue(all(call["model"] == self.protocol["model"] for call in calls))


if __name__ == "__main__":
    unittest.main()
