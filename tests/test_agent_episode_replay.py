"""Recorded-replay contracts using explicit fake fixtures, never LLM evidence."""

from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import rcwt_agent_run as runner
from rcwt_agent_env import generate_episodes
from test_rcwt_agent_pipeline_smoke import OfflineTransport

SPEC = importlib.util.spec_from_file_location("agent_episode_replay", runner.ROOT / "tools/replay_agent_episode.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class RecordedEpisodeReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "explicit-fake-replay-fixture"
        network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Test attempted network access"))
        network.start(); self.addCleanup(network.stop)
        args = SimpleNamespace(train_count=4, validation_count=4, test_count=4, seed=107,
                               memory_budget=256, model="rcwt-local-qwen35-4b",
                               runtime_receipt=runner.ROOT / "docs/rcwt_agent_runtime.json")
        with patch("builtins.print"):
            protocol = runner.prepare(self.directory, args)
            client = OfflineTransport(seed=args.seed)
            runner.run_cohort(client, generate_episodes("train", 4, args.seed),
                              [runner.Policy("summary", "summary")], 256, self.directory / "train", args.seed)
            runner.validate_stage(client, self.directory, protocol)
            runner.test_stage(client, self.directory, protocol)

    def test_defaults_replay_first_manifest_episode_and_recorded_resources(self):
        expected = runner.read(self.directory / "test-public.json")[0]
        rows = runner.read_jsonl(self.directory / "test/traces.jsonl")
        first = next(row for row in rows if row["episode_id"] == expected["episode_id"] and row["policy"] == "summary")
        output = io.StringIO()
        with patch("rcwt_local_model.LocalModelClient.complete", side_effect=AssertionError("Replay called a model")):
            result = replay.replay_episode(self.directory, output=output)
        text = output.getvalue()
        self.assertEqual(result["episode_id"], expected["episode_id"])
        self.assertEqual(result["policy"], "summary")
        self.assertEqual(result["steps"], 8)
        self.assertTrue(text.startswith("RECORDED REPLAY / NO MODEL CALLS\n"))
        self.assertEqual(text.count("STEP "), 8)
        self.assertIn("PRIVATE EVALUATION", text)
        self.assertIn("self-reported; not reference truth", text)
        self.assertIn(f"decision_seconds={first['model_calls'][0]['wall_seconds']:.6f}", text)
        self.assertIn(f"step_seconds={first['step_seconds']:.6f}", text)
        self.assertIn("Model: rcwt-local-qwen35-4b", text)
        self.assertNotIn("Retained memory BEFORE:", text)

    def test_render_is_deterministic_and_memory_is_opt_in(self):
        first, second = io.StringIO(), io.StringIO()
        replay.replay_episode(self.directory, 0, "learned", True, first)
        replay.replay_episode(self.directory, 0, "learned", True, second)
        self.assertEqual(first.getvalue(), second.getvalue())
        self.assertEqual(first.getvalue().count("Retained memory BEFORE:"), 8)
        self.assertEqual(first.getvalue().count("Retained memory AFTER:"), 8)
        self.assertIn("Frozen validation-selected policy:", first.getvalue())

    def test_missing_freeze_prevents_any_render_or_verifier_call(self):
        (self.directory / "test-freeze.json").unlink()
        output = io.StringIO()
        with patch.object(replay, "verify_evidence") as verifier:
            with self.assertRaisesRegex(ValueError, "held-out freeze is required"):
                replay.replay_episode(self.directory, output=output)
        verifier.assert_not_called()
        self.assertEqual(output.getvalue(), "")

    def test_verification_failure_prevents_rendering_even_when_freeze_exists(self):
        for outcome in ({"status": "FAIL"}, {"status": "WARN"}):
            output = io.StringIO()
            with patch.object(replay, "verify_evidence", return_value=outcome):
                with self.assertRaisesRegex(ValueError, "verification did not pass"):
                    replay.replay_episode(self.directory, output=output)
            self.assertEqual(output.getvalue(), "")
        output = io.StringIO()
        with patch.object(replay, "verify_evidence", side_effect=ValueError("tampered trace")):
            with self.assertRaisesRegex(ValueError, "tampered trace"):
                replay.replay_episode(self.directory, output=output)
        self.assertEqual(output.getvalue(), "")

    def test_invalid_index_or_policy_is_not_silently_replaced(self):
        for index in (-1, True, 4):
            with self.subTest(index=index), self.assertRaises(ValueError):
                replay.replay_episode(self.directory, index, output=io.StringIO())
        with self.assertRaisesRegex(ValueError, "policy must be"):
            replay.replay_episode(self.directory, policy="best", output=io.StringIO())


if __name__ == "__main__":
    unittest.main()
