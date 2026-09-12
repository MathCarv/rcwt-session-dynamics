"""R1 integrity tests: fake transport and TRAIN data only, never the R1 seed.

The protocol envelope is explicitly mocked for execution/replay fixtures. The
real frozen executor and every strict 512-step replay check still execute.
These tests are not model-quality, tokenizer, latency or confirmation evidence.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import rcwt_replication_v4 as runner
import rcwt_online_v4 as core
from test_rcwt_online_v4 import ExplicitFakeV4Transport, _reseal


class ReplicationV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = patch("urllib.request.OpenerDirector.open",
                            side_effect=AssertionError("Real HTTP is forbidden in R1 tests"))
        cls.network.start()
        cls.addClassCleanup(cls.network.stop)
        cls.temp = tempfile.TemporaryDirectory(prefix="explicit-fake-r1-", dir=runner.ROOT / ".runs")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.fixture = cls.root / "fake-train-complete"
        cls.fixture.mkdir()
        cls.protocol = {"schema": "EXPLICIT-FAKE-TRAIN-TEST-ENVELOPE",
                        "mode": "FAKE-TRAIN-ONLY", "split": "train", "count": 32,
                        "dataset_seed": 202604015, "schedule_seed": core.SCHEDULE_SEED,
                        "memory_budget": 256, "model": "rcwt-local-qwen35-4b",
                        "inference_seed": core.INFERENCE_SEED}
        if cls.protocol["dataset_seed"] == runner.DATASET_SEED:
            raise AssertionError("Never generate the registered R1 cohort in tests")
        episodes = core.generate_episodes("train", 32, cls.protocol["dataset_seed"])
        core.dump(cls.fixture / "protocol.json", cls.protocol)
        core.dump(cls.fixture / "public.json", [episode.to_public_dict() for episode in episodes])
        core.dump(cls.fixture / "oracle.json", [episode.to_oracle_dict() for episode in episodes])
        core.dump(cls.fixture / "schedule.json", core.schedule(32, core.SCHEDULE_SEED))
        cls.initial_sources = core.sources()
        client = ExplicitFakeV4Transport(model=cls.protocol["model"], seed=core.INFERENCE_SEED)
        with patch.object(runner, "validate_protocol", return_value=cls.protocol), contextlib.redirect_stdout(io.StringIO()):
            runner.run(cls.fixture, cls.protocol, client)
        cls.fake_calls = len(client.calls)
        cls.receipt = cls.verify(cls.fixture)

    @classmethod
    def verify(cls, directory):
        with patch.object(runner, "validate_protocol", return_value=cls.protocol), \
                patch.object(runner, "_history_files", return_value={}):
            return runner.verify_run(directory)

    def clone(self):
        directory = self.root / self._testMethodName
        shutil.copytree(self.fixture, directory)
        return directory

    def test_complete_fake_train_replays_exact_512_steps_and_1248_calls(self):
        self.assertEqual(self.receipt["status"], "PASS")
        self.assertEqual(self.receipt["verified_steps"], 512)
        self.assertEqual(self.receipt["verified_episode_summaries"], 64)
        self.assertEqual(self.receipt["generation_calls"], 1248)
        self.assertEqual(self.fake_calls, 1248)
        self.assertEqual(self.receipt["generation_calls_by_policy"], {"summary": 736, "structured": 512})
        self.assertEqual(self.receipt["prompt_tokens"], 1248 * 13)
        self.assertEqual(self.receipt["completion_tokens"], 1248 * 7)
        self.assertEqual(self.receipt["inference_calls"], 0)
        self.assertTrue(self.receipt["read_only"])
        self.assertTrue(self.receipt["integrity_only"])
        self.assertEqual(self.receipt["gain"], "NOT_EVALUATED")
        self.assertEqual(core.sources(), self.initial_sources)
        self.assertTrue(all(row["split"] == "train" for row in core.read_jsonl(self.fixture / "traces.jsonl")))

    def test_second_run_is_refused_before_client_call(self):
        client = ExplicitFakeV4Transport(model=self.protocol["model"], seed=core.INFERENCE_SEED)
        with self.assertRaisesRegex(ValueError, "already started"):
            runner.run(self.fixture, self.protocol, client)
        self.assertEqual(client.calls, [])

    def test_abort_persists_marker_and_no_retry(self):
        directory = self.root / self._testMethodName
        directory.mkdir()
        core.dump(directory / "protocol.json", self.protocol)
        core.dump(directory / "schedule.json", core.schedule(32, core.SCHEDULE_SEED))

        class FailsImmediately:
            model = self.protocol["model"]
            seed = core.INFERENCE_SEED

            def tokenize(self, text):
                if not (directory / "started.json").exists():
                    raise AssertionError("Exclusive marker must precede all inference operations")
                raise RuntimeError("EXPLICIT FAKE INTERRUPT")

        with patch.object(runner, "validate_protocol", return_value=self.protocol):
            with self.assertRaisesRegex(RuntimeError, "EXPLICIT FAKE INTERRUPT"):
                runner.run(directory, self.protocol, FailsImmediately())
        self.assertFalse(core.read(directory / "aborted.json")["retry_allowed"])
        self.assertTrue((directory / "partial-step.json").is_file())
        self.assertFalse((directory / "completion.json").exists())
        with self.assertRaisesRegex(ValueError, "already started"):
            runner.run(directory, self.protocol, FailsImmediately())

    def test_prepare_rejects_existing_directory_without_generating(self):
        with self.assertRaisesRegex(ValueError, "nonexistent"):
            runner.prepare(self.fixture)

    def test_failed_history_blocks_prepare_before_seed_is_generated(self):
        directory = self.root / self._testMethodName
        with patch.object(runner, "_history_bindings", side_effect=ValueError("history failed")):
            with self.assertRaisesRegex(ValueError, "history failed"):
                runner.prepare(directory)
        self.assertFalse(directory.exists())

    def test_fixed_contract_rejects_seed_drift_without_generating(self):
        directory = self.root / self._testMethodName
        directory.mkdir()
        protocol = runner._fixed_contract() | {"frozen_at_utc": "test", "runtime": {}, "model": "fake",
            "runtime_sha256": "", "source_sha256": {}, "orchestrator_sha256": {},
            "historical_evidence": {}, "public_sha256": "", "oracle_sha256": ""}
        protocol["dataset_seed"] = 7
        core.dump(directory / "protocol.json", protocol)
        core.dump(directory / "freeze.json", {})
        with patch.object(runner, "_history_bindings", side_effect=AssertionError("must reject before history")):
            with self.assertRaisesRegex(ValueError, "parameter changed: dataset_seed"):
                runner.validate_protocol(directory)

    def test_rehashed_actor_input_tamper_fails_replay(self):
        directory = self.clone()
        traces = core.read_jsonl(directory / "traces.jsonl")
        event = next(event for event in traces[0]["client_events"] if event["method"] == "complete")
        event["arguments"]["messages"][0]["content"] += " TAMPERED"
        _reseal(directory, traces=traces)
        with self.assertRaisesRegex(ValueError, "input/order drift"):
            self.verify(directory)

    def test_rehashed_effect_tamper_fails_replay(self):
        directory = self.clone()
        traces = core.read_jsonl(directory / "traces.jsonl")
        traces[0]["score"]["success"] = not traces[0]["score"]["success"]
        _reseal(directory, traces=traces)
        with self.assertRaisesRegex(ValueError, "does not replay"):
            self.verify(directory)

    def test_rehashed_additional_event_fails_replay(self):
        directory = self.clone()
        traces = core.read_jsonl(directory / "traces.jsonl")
        traces[0]["client_events"].append(copy.deepcopy(traces[0]["client_events"][-1]))
        _reseal(directory, traces=traces)
        with self.assertRaisesRegex(ValueError, "Unused recorded operations"):
            self.verify(directory)

    def test_rehashed_bad_time_fails_replay(self):
        directory = self.clone()
        traces = core.read_jsonl(directory / "traces.jsonl")
        traces[0]["step_seconds"] = -1
        _reseal(directory, traces=traces)
        with self.assertRaisesRegex(ValueError, "Invalid recorded step time"):
            self.verify(directory)

    def test_rehashed_unknown_trace_field_fails(self):
        directory = self.clone()
        traces = core.read_jsonl(directory / "traces.jsonl")
        traces[0]["hidden_retry"] = 1
        _reseal(directory, traces=traces)
        with self.assertRaisesRegex(ValueError, "does not replay"):
            self.verify(directory)

    def test_extra_or_missing_steps_fail_even_with_resealed_manifests(self):
        directory = self.clone()
        traces = core.read_jsonl(directory / "traces.jsonl")
        _reseal(directory, traces=traces[:-1])
        with self.assertRaisesRegex(ValueError, "exactly 512"):
            self.verify(directory)

    def test_aborted_completed_bundle_never_gets_pass(self):
        directory = self.clone()
        core.dump(directory / "aborted.json", {"retry_allowed": False})
        with self.assertRaisesRegex(ValueError, "aborted"):
            self.verify(directory)

    def test_duplicate_json_fields_and_nonfinite_values_fail(self):
        for data in (b'{"a": 1, "a": 2}', b'{"metering": 1e999}', b'{"a": NaN}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                runner.partial._json(data)

    def test_verification_is_read_only(self):
        before = runner._tree(self.fixture)
        self.assertEqual(self.verify(self.fixture), self.receipt)
        self.assertEqual(before, runner._tree(self.fixture))

    def test_evidence_mutation_during_verify_is_detected(self):
        directory = self.clone()
        replay = runner._verify_replay

        def mutating_replay(path, protocol):
            result = replay(path, protocol)
            (path / "unregistered-input.txt").write_text("mutation", encoding="utf-8")
            return result

        with patch.object(runner, "_verify_replay", side_effect=mutating_replay):
            with self.assertRaisesRegex(ValueError, "changed during complete"):
                self.verify(directory)

    def test_stage_verify_has_no_client_construction(self):
        argv = ["rcwt_replication_v4.py", "--stage", "verify", "--output-dir", str(self.fixture)]
        with patch("sys.argv", argv), patch.object(runner, "verify_run", return_value=self.receipt), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            runner.main()
        self.assertEqual(json.loads(output.getvalue())["inference_calls"], 0)

    def test_paths_cannot_escape_trusted_project(self):
        with self.assertRaises(ValueError):
            runner._local(runner.ROOT.parent / "not-an-r1-input", missing=True)


if __name__ == "__main__":
    unittest.main()
