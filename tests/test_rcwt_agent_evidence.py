"""Tamper checks using an explicitly fake pipeline, never real-agent evidence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import rcwt_agent_run as runner
from rcwt_agent_env import generate_episodes
from test_rcwt_agent_pipeline_smoke import OfflineTransport


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in records),
        encoding="utf-8",
        newline="\n",
    )


def _reseal_completion(stage: Path) -> None:
    completion = runner.read(stage / "completion.json")
    completion.update(
        episodes=len(runner.read_jsonl(stage / "episodes.jsonl")),
        traces_sha256=runner.file_hash(stage / "traces.jsonl"),
        episodes_sha256=runner.file_hash(stage / "episodes.jsonl"),
    )
    runner.dump(stage / "completion.json", completion)


class AgentEvidenceTamperTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)
        self.directory = Path(temporary.name) / "explicit-fake-integrity-fixture"
        args = SimpleNamespace(
            train_count=4, validation_count=4, test_count=4, seed=31,
            memory_budget=256, model="rcwt-local-qwen35-4b",
            runtime_receipt=runner.ROOT / "docs/rcwt_agent_runtime.json",
        )
        protocol = runner.prepare(self.directory, args)
        client = OfflineTransport(seed=args.seed)
        runner.run_cohort(
            client, generate_episodes("train", 4, args.seed),
            [runner.Policy("summary", "summary")], 256,
            self.directory / "train", args.seed,
        )
        runner.validate_stage(client, self.directory, protocol)
        runner.test_stage(client, self.directory, protocol)
        self.assertEqual(runner.verify_evidence(self.directory)["verified_steps"], 224)

    def test_deleted_whole_episode_cannot_pass_by_resealing_completion(self):
        stage = self.directory / "test"
        traces = runner.read_jsonl(stage / "traces.jsonl")
        summaries = runner.read_jsonl(stage / "episodes.jsonl")
        removed_id = traces[0]["episode_id"]
        _write_jsonl(stage / "traces.jsonl", [row for row in traces if row["episode_id"] != removed_id])
        _write_jsonl(stage / "episodes.jsonl", [row for row in summaries if row["episode_id"] != removed_id])
        _reseal_completion(stage)
        with self.assertRaisesRegex(ValueError, "scheduled calls|Incomplete episode evidence"):
            runner.verify_evidence(self.directory)

    def test_summary_tokens_must_match_recorded_calls_even_with_new_file_hash(self):
        stage = self.directory / "test"
        summaries = runner.read_jsonl(stage / "episodes.jsonl")
        summaries[0]["prompt_tokens"] += 1
        _write_jsonl(stage / "episodes.jsonl", summaries)
        _reseal_completion(stage)
        with self.assertRaisesRegex(ValueError, "Summary disagrees.*resource accounting"):
            runner.verify_evidence(self.directory)

    def test_action_is_bound_to_raw_model_text_after_trace_chain_rehash(self):
        stage = self.directory / "test"
        traces = runner.read_jsonl(stage / "traces.jsonl")
        summaries = runner.read_jsonl(stage / "episodes.jsonl")
        target = traces[-1]
        pair = (target["episode_id"], target["policy"])
        # Whitespace preserves the parsed JSON action, score and final-step
        # ledger state. Only raw-completion binding should reject this change.
        target["action"] += " "
        previous = None
        for row in traces:
            if (row["episode_id"], row["policy"]) == pair:
                row["previous_sha256"] = previous
                row["sha256"] = runner.canonical_hash({key: value for key, value in row.items() if key != "sha256"})
                previous = row["sha256"]
        for summary in summaries:
            if (summary["episode_id"], summary["policy"]) == pair:
                summary["last_trace_sha256"] = previous
        _write_jsonl(stage / "traces.jsonl", traces)
        _write_jsonl(stage / "episodes.jsonl", summaries)
        _reseal_completion(stage)
        with self.assertRaisesRegex(ValueError, "Executed action differs from raw completion"):
            runner.verify_evidence(self.directory)


if __name__ == "__main__":
    unittest.main()
