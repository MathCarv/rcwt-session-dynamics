"""Explicit synthetic R2 orchestration/replay tests; never campaign data or HTTP."""
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

import rcwt_r2 as runner
from test_rcwt_online_v4 import ExplicitFakeV4Transport


class FakeR2Transport(ExplicitFakeV4Transport):
    def _request(self, path, payload=None):
        if (path == "/v1/chat/completions" and "response_format" in payload
                and "weights" in payload["response_format"]["json_schema"]["schema"].get("properties", {})):
            self.transport_requests.append((path, copy.deepcopy(payload)))
            return {"model": self.model, "id": "EXPLICIT_SYNTHETIC_POLICY_NOT_MODEL_EVIDENCE",
                    "choices": [{"message": {"content": json.dumps(runner.DEFAULT_POLICY)}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 13, "completion_tokens": 7}, "timings": {"cache_n": 0}}
        return super()._request(path, payload)


class R2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = contextlib.ExitStack()
        cls.addClassCleanup(cls.stack.close)
        cls.stack.enter_context(patch("urllib.request.OpenerDirector.open",
                                      side_effect=AssertionError("No real HTTP in synthetic R2 tests")))
        cls.root = Path(tempfile.mkdtemp(prefix="testing-r2-", dir=runner.ROOT / ".runs"))
        cls.successful = False
        cls.addClassCleanup(cls.cleanup_success)
        cls.protocol_contract = copy.deepcopy(runner._contract())
        cls.protocol_contract["evidence_kind"] = "EXPLICIT_SYNTHETIC_TEST_ONLY"
        registered = {r[p + "_seed"] for r in runner.replicas() for p in ("train", "validation", "test")}
        for replica in cls.protocol_contract["replicas"]:
            for offset, phase in enumerate(("train", "validation", "test")):
                replica[phase + "_seed"] = 51000 + replica["replica"] * 10 + offset
                if replica[phase + "_seed"] in registered:
                    raise AssertionError("Synthetic fixture must not generate a campaign namespace")
        cls.stack.enter_context(patch.object(runner, "_contract", return_value=cls.protocol_contract))
        cls.stack.enter_context(patch.object(runner, "_frozen_inputs", return_value={}))
        # Runtime helper is not executed by the explicitly synthetic lane.
        cls.stack.enter_context(patch.object(runner, "NEW_SOURCES", tuple(
            name for name in runner.NEW_SOURCES if name.endswith(".py"))))
        original_generate = runner.core.generate_episodes

        def guarded_generate(split, count, seed):
            if seed in registered:
                raise AssertionError("Registered R2 corpus is forbidden in unit tests")
            return original_generate(split, count, seed)

        cls.stack.enter_context(patch.object(runner.core, "generate_episodes", side_effect=guarded_generate))
        cls.fixture = cls.root / "complete"
        cls.clients = []
        with contextlib.redirect_stdout(io.StringIO()):
            runner.prepare(cls.fixture)
            for stage in runner.STAGES:
                cls.receipt = runner.run_stage(cls.fixture, stage, cls.factory)
        cls.successful = True

    @classmethod
    def cleanup_success(cls):
        if cls.successful:
            target = cls.root.resolve(strict=True)
            if not target.is_relative_to((runner.ROOT / ".runs").resolve()) or not target.name.startswith("testing-r2-"):
                raise AssertionError("Refuse cleanup outside the owned synthetic fixture")
            shutil.rmtree(target)
        else:
            print("Preserved failed synthetic fixture: " + str(cls.root))

    def tearDown(self):
        result = self._outcome.result
        if result.errors or result.failures:
            type(self).successful = False

    @classmethod
    def factory(cls, protocol, replica):
        client = FakeR2Transport(model=protocol["model"], seed=replica["inference_seed"])
        cls.clients.append(client)
        return client

    def clone(self):
        destination = self.root / self._testMethodName
        shutil.copytree(self.fixture, destination)
        return destination

    def prepared(self):
        directory = self.root / self._testMethodName
        runner.prepare(directory)
        return directory

    def rewrite(self, path, value, *, jsonl=False):
        path.write_bytes((b"".join(runner.json_bytes(r).replace(b"\n", b" ") + b"\n" for r in value)
                          if jsonl else runner.json_bytes(value)))

    def reseal_test(self, directory, rows):
        chain = None
        for row in rows:
            row["previous_sha256"] = chain
            row["sha256"] = runner.core.canonical_hash({k: v for k, v in row.items() if k != "sha256"})
            chain = row["sha256"]
        self.rewrite(directory / "test/traces.jsonl", rows, jsonl=True)
        completion = runner.read(directory / "test-completion.json")
        completion["artifact_sha256"] = runner._stage_artifacts(directory, "test")
        self.rewrite(directory / "test-completion.json", completion)

    def test_full_synthetic_campaign_exactly_1120_steps_and_2250_calls(self):
        self.assertEqual(self.receipt["status"], "PASS")
        self.assertTrue(self.receipt["complete"])
        self.assertEqual(self.receipt["verified_steps"], 1120)
        self.assertEqual(self.receipt["generation_calls"], 2250)
        self.assertEqual(sum(len(client.calls) for client in self.clients), 2250)
        self.assertEqual(self.receipt["inference_calls"], 0)
        self.assertTrue(self.receipt["read_only"])
        self.assertEqual(runner.read(self.fixture / "protocol.json")["evidence_kind"], "EXPLICIT_SYNTHETIC_TEST_ONLY")

    def test_prepare_generates_no_corpus_and_no_transport(self):
        with patch.object(runner, "_generate", side_effect=AssertionError("Prepare must not generate")):
            directory = self.prepared()
        self.assertFalse((directory / "train").exists())
        self.assertFalse((directory / "test").exists())
        self.assertEqual(runner.verify_run(directory, require_complete=False)["generation_calls"], 0)

    def test_prepare_rejects_original_archive_subdirectory_before_any_write(self):
        target = runner.ROOT / "results/agent_v4_replication/forbidden-r2-test"
        self.assertFalse(target.exists())
        with patch.object(runner, "_frozen_inputs", side_effect=AssertionError("Must reject output before inspecting inputs")):
            with self.assertRaisesRegex(ValueError, "R2 writes require"):
                runner.prepare(target)
        self.assertFalse(target.exists())

    def test_test_before_selection_is_rejected_before_corpus_or_client(self):
        directory = self.prepared()
        with patch.object(runner, "_generate", side_effect=AssertionError("No test before seal")):
            with self.assertRaisesRegex(ValueError, "previous stages"):
                runner.run_stage(directory, "test", self.factory)
        self.assertFalse((directory / "test-started.json").exists())

    def test_future_corpus_and_even_empty_future_directory_are_rejected(self):
        directory = self.prepared()
        (directory / "test").mkdir()
        with self.assertRaisesRegex(ValueError, "premature"):
            runner.verify_run(directory, require_complete=False)
        self.rewrite(directory / "test/public.json", [])
        with self.assertRaisesRegex(ValueError, "premature"):
            runner.verify_run(directory, require_complete=False)

    def test_repeat_stage_is_rejected_without_transport(self):
        with self.assertRaisesRegex(ValueError, "already started"):
            runner.run_stage(self.fixture, "test", lambda *_: self.fail("No repeated client"))

    def test_synthetic_lane_never_constructs_real_client(self):
        directory = self.prepared()
        with patch.object(runner, "_client", side_effect=AssertionError("No real client")):
            with self.assertRaisesRegex(ValueError, "Synthetic evidence"):
                runner.run_stage(directory, "train")

    def test_abort_records_attempt_and_failure_and_forbids_retry(self):
        directory = self.prepared()

        class Broken(FakeR2Transport):
            def complete(self, *args, **kwargs):
                raise RuntimeError("EXPLICIT SYNTHETIC FAILURE")

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "SYNTHETIC FAILURE"):
                runner.run_stage(directory, "train", lambda p, r: Broken(model=p["model"], seed=r["inference_seed"]))
        attempts = runner.read_jsonl(directory / "train-requests.jsonl")
        self.assertEqual([r["status"] for r in attempts], ["started", "failed"])
        self.assertEqual(attempts[-1]["exception_type"], "RuntimeError")
        self.assertFalse(runner.read(directory / "train-aborted.json")["retry_allowed"])
        self.assertTrue((directory / "train-partial-step.json").exists())
        with self.assertRaisesRegex(ValueError, "aborted"):
            runner.run_stage(directory, "train", self.factory)
        with self.assertRaisesRegex(ValueError, "Aborted"):
            runner.verify_run(directory, require_complete=False)

    def test_rehashed_proposal_grade_tamper_fails_replay(self):
        directory = self.clone()
        rows = runner.all_rows(directory)
        rows[0]["proposal_score"]["success"] = not rows[0]["proposal_score"]["success"]
        self.reseal_test(directory, rows)
        with self.assertRaisesRegex(ValueError, "does not replay"):
            runner._replay_phase(directory, runner.read(directory / "protocol.json"), "test")

    def test_rehashed_effect_tamper_fails_replay(self):
        directory = self.clone()
        rows = runner.all_rows(directory)
        rows[0]["executor_result"]["tool_result"]["amount_booked_cents"] = 1
        self.reseal_test(directory, rows)
        with self.assertRaisesRegex(ValueError, "does not replay"):
            runner._replay_phase(directory, runner.read(directory / "protocol.json"), "test")

    def test_rehashed_actor_request_tamper_fails_strict_event_replay(self):
        directory = self.clone()
        rows = runner.all_rows(directory)
        event = next(e for e in rows[0]["client_events"] if e["method"] == "complete")
        event["arguments"]["messages"][0]["content"] += " changed"
        self.reseal_test(directory, rows)
        with self.assertRaisesRegex(ValueError, "input/order drift"):
            runner._replay_phase(directory, runner.read(directory / "protocol.json"), "test")

    def test_selection_tamper_is_not_a_test_arm_substitution(self):
        selection = runner.read(self.fixture / "selection.json")
        self.assertEqual(len(selection["replicas"]), 5)
        self.assertTrue(all(row["deployment_arm"] == "fixed" for row in selection["replicas"]))
        self.assertEqual({row["arm"] for row in runner.all_rows(self.fixture)}, set(runner.ARMS))

    def test_unknown_artifact_cannot_hide_in_a_completed_campaign(self):
        directory = self.clone()
        self.rewrite(directory / "unlisted.json", {})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            runner.verify_run(directory)

    def test_report_writer_with_explicit_completed_fixture_receipt_is_exclusive(self):
        directory = self.clone()
        # The campaign fixture above has already passed exact full replay.
        # This unit isolates report creation/no-overwrite, not another replay.
        with patch.object(runner, "verify_run", return_value=self.receipt):
            result = runner.report(directory)
            before = runner._snapshot(directory)
            self.assertEqual(runner.report(directory), result)
            self.assertEqual(runner._snapshot(directory), before)
            (directory / "RESULTS.md").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Existing report differs"):
                runner.report(directory)

    def test_report_input_drift_during_analysis_is_rejected_before_writes(self):
        directory = self.clone()
        analyze = runner.analyze_r2

        def mutate(*args, **kwargs):
            result = analyze(*args, **kwargs)
            (directory / "test/public.json").write_bytes(b"{}\n")
            return result

        with patch.object(runner, "verify_run", return_value=self.receipt), \
                patch.object(runner, "analyze_r2", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during report recomputation"):
                runner.report(directory)
        self.assertFalse((directory / "report-started.json").exists())
        self.assertFalse((directory / "RESULTS.md").exists())

    def test_latin_schedule_balances_each_family_and_global_positions(self):
        from collections import Counter
        orders = [runner.schedule(r, "test") for r in runner.replicas()]
        for family in range(4):
            for position in range(3):
                counts = Counter(order[family][position] for order in orders)
                self.assertEqual(sorted(counts.values()), [1, 2, 2])
        for position in range(3):
            counts = Counter(arms[position] for order in orders for arms in order)
            self.assertEqual(sorted(counts.values()), [6, 7, 7])

    def test_fake_transport_cannot_be_labelled_real_model_evidence(self):
        directory = self.prepared()
        protocol = runner.read(directory / "protocol.json")
        protocol["evidence_kind"] = "real_local_model"
        with patch.object(runner, "validate_protocol", return_value=protocol):
            with self.assertRaisesRegex(ValueError, "R2 writes require|cannot produce real-model"):
                runner.run_stage(directory, "train", self.factory)

    def test_runtime_receipt_pins_are_strict_without_executing_a_helper(self):
        protocol = copy.deepcopy(runner.read(self.fixture / "protocol.json"))
        protocol["source_sha256"]["tools/rcwt_r2_runtime.ps1"] = "a" * 64
        runtime = protocol["runtime"]
        receipt = {"schema": "rcwt-r2-runtime-status/1", "pass": True, "status": "READY", "read_only": True,
                   "inference_calls": 0, "health_requests": 1, "pid": 123,
                   "process_start_time_utc": "EXPLICIT-SYNTHETIC-TIMESTAMP",
                   "executable_sha256": runtime["executable_sha256"], "model_sha256": runtime["model_sha256"],
                   "model_bytes": runtime["model_bytes"], "runtime_config_sha256": runner.digest(
                       runner._bytes(runner.ROOT / "docs/rcwt_agent_runtime.json")),
                   "helper_sha256": "a" * 64, "started_receipt_sha256": "b" * 64,
                   "intent_receipt_sha256": "c" * 64, "arguments_sha256": "d" * 64,
                   "health_status": "ok", "host": runtime["host"], "port": runtime["port"]}
        with patch.object(runner.subprocess, "run", side_effect=AssertionError("Offline validation never starts a helper")):
            self.assertEqual(runner._validate_runtime_status(receipt, protocol), receipt)
            for key, bad in (("status", "CLOSED"), ("model_sha256", "e" * 64), ("pid", True),
                             ("inference_calls", False), ("port", 18086), ("health_requests", 0)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner._validate_runtime_status({**receipt, key: bad}, protocol)


if __name__ == "__main__":
    unittest.main()
