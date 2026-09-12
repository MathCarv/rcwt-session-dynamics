"""Interrupted-prefix integrity tests using explicitly fake offline model replies.

No reserved confirmatory dataset is generated. Training examples with an
unregistered seed are relabeled as invented test fixtures. The frozen protocol
validator is mocked only in envelope tests; real transcript replay is exercised.
These fixtures never constitute model-quality, latency or gain evidence.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

import rcwt_online_v4 as runner
from rcwt_agent_env import generate_episodes
from rcwt_local_model import CallResult
from tools import verify_online_v4_partial as partial


class ExplicitFakeClient:
    model = "EXPLICIT-FAKE-PREFIX-NOT-MODEL-EVIDENCE"
    seed = runner.INFERENCE_SEED

    def tokenize(self, text):
        data = text.encode("utf-8")
        return [int.from_bytes(b"\x01" + data[i:i + 4], "big") for i in range(0, len(data), 4)]

    def detokenize(self, tokens):
        return b"".join(token.to_bytes((token.bit_length() + 7) // 8, "big")[1:]
                        for token in tokens).decode("utf-8", errors="replace")

    def complete(self, messages, max_tokens, schema=None, purpose=""):
        if purpose.startswith("draft:"):
            text = "EXPLICIT FAKE PLAN: ask for missing evidence."
        elif purpose.startswith("action:"):
            task = json.loads(messages[1]["content"])["current_step"]["task"]
            text = json.dumps({"tool": "record_decision", "arguments": {
                "case_id": task["case_id"], "decision": "ask_info",
                "amount_cents": 0, "reason_code": "missing_evidence"}})
        else:
            text = "EXPLICIT FAKE MEMORY: " + "x " * 512
        arguments = dict(messages=messages, max_tokens=max_tokens, schema=schema, purpose=purpose)
        return CallResult(text=text, prompt_tokens=13, completion_tokens=7, wall_seconds=.001,
                          model=self.model, purpose=purpose, finish_reason="stop", timings={"cache_n": 0},
                          request=runner._payload(self.model, self.seed, arguments),
                          response_id="explicit-fake-prefix-response")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
                    encoding="utf-8")


def _rechain(rows, summaries):
    chain, lasts = None, {}
    for row in rows:
        row["previous_sha256"] = chain
        row["sha256"] = runner.canonical_hash({key: value for key, value in row.items() if key != "sha256"})
        chain = row["sha256"]
        lasts[row["episode_id"], row["policy"]] = chain
    for item in summaries:
        item["last_trace_sha256"] = lasts.get((item["episode_id"], item["policy"]), item["last_trace_sha256"])


class OnlineV4PartialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        guard = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Real HTTP forbidden"))
        guard.start()
        cls.addClassCleanup(guard.stop)
        cls.project = partial.ROOT
        cls.client = ExplicitFakeClient()
        cls.protocol = {"schema": runner.SCHEMA, "mode": "confirmatory", "split": "test", "count": 32,
                        "schedule_seed": runner.SCHEDULE_SEED, "model": cls.client.model,
                        "inference_seed": cls.client.seed, "memory_budget": 256,
                        "source_sha256": runner.sources()}
        episodes = [replace(episode, episode_id=f"explicit-fake-prefix-{index}", split="test")
                    for index, episode in enumerate(generate_episodes("train", 32, 42))]
        cls.public = [episode.to_public_dict() for episode in episodes]
        cls.oracle = [episode.to_oracle_dict() for episode in episodes]
        cls.order = runner.schedule(32, runner.SCHEDULE_SEED)
        cls.traces, cls.summaries, chain = [], [], None
        for episode, policies in zip(episodes, cls.order):
            for policy in policies:
                simulator, memory, rows = runner.Simulator(episode), "", []
                for index in range(8):
                    if len(cls.traces) == 173:
                        break
                    client = runner.AuditedClient(cls.client)
                    core = runner.perform_step(client, simulator, memory, policy, index, 256)
                    row = {"schema": runner.SCHEMA, "episode_id": episode.episode_id,
                           "family": episode.family, "split": "test", "policy": policy,
                           "step_index": index, **core, "client_events": client.events,
                           "step_seconds": .1, "previous_sha256": chain}
                    row["sha256"] = runner.canonical_hash(row)
                    chain, memory = row["sha256"], core["memory_after"]
                    cls.traces.append(row)
                    rows.append(row)
                if len(rows) == 8:
                    cls.summaries.append(runner.summarize(rows, 1.0))

    def replay(self, rows=None, summaries=None, order=None):
        return partial._verify_prefix(runner, self.protocol, self.public, self.oracle,
                                      self.order if order is None else order,
                                      self.traces if rows is None else rows,
                                      self.summaries if summaries is None else summaries)

    def envelope(self):
        temporary = tempfile.TemporaryDirectory(prefix="explicit-fake-partial-v4-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        root_patch = patch.object(partial, "ROOT", root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        directory = root / ".runs/explicit-fake-prefix"
        _write(directory / "protocol.json", self.protocol)
        _write(directory / "started.json", {"protocol_sha256": partial._digest((directory / "protocol.json").read_bytes())})
        _write(directory / "freeze.json", {"fixture": "EXPLICIT FAKE ENVELOPE; validation mocked"})
        _write(directory / "schedule.json", self.order)
        _write(directory / "public.json", self.public)
        _write(directory / "oracle.json", self.oracle)
        _jsonl(directory / "traces.jsonl", self.traces)
        _jsonl(directory / "episodes.jsonl", self.summaries)
        for name in (*partial.INPUTS, "completion.json"):
            if name.endswith(".jsonl"):
                _jsonl(directory / "development" / name, [])
            else:
                _write(directory / "development" / name, {"fixture": "fake envelope"})
        for name in partial.NAMES:
            for base in (root / "src", directory / "sources", directory / "development/sources"):
                base.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.project / "src" / name, base / name)
        for name in partial.HISTORY:
            _write(root / name, {"fixture": "fake historical envelope"})
        validator = patch.object(runner, "validate_protocol", return_value=self.protocol)
        mocked = validator.start()
        self.addCleanup(validator.stop)
        return root, directory, mocked

    def test_exact_interrupted_prefix_and_all_completed_step_resources(self):
        result = self.replay()
        self.assertEqual((result["status"], result["confirmation_status"], result["gain"]),
                         ("PARTIAL_VERIFIED", "INCOMPLETE", "NOT_EVALUATED"))
        self.assertEqual((result["verified_steps"], result["missing_steps"], result["verified_episode_summaries"]),
                         (173, 339, 21))
        self.assertEqual(result["partial_last_trajectory"]["completed_steps"], 5)
        resources = result["recorded_completed_step_resources"]
        self.assertEqual(resources["generation_calls"], 421)
        self.assertEqual(resources["prompt_tokens"], 421 * 13)
        self.assertEqual(resources["completion_tokens"], 421 * 7)
        self.assertNotIn("successes", result)

    def test_empty_complete_extra_or_missing_summaries_are_rejected(self):
        for rows, summaries in (([], []), ((self.traces * 3)[:512], self.summaries),
                                (self.traces * 3, self.summaries),
                                (self.traces, self.summaries[:-1]),
                                (self.traces, self.summaries + [self.summaries[-1]])):
            with self.subTest(count=len(rows), summaries=len(summaries)), self.assertRaises(ValueError):
                self.replay(rows, summaries)
        result = self.replay(self.traces[:168], self.summaries)
        self.assertIsNone(result["partial_last_trajectory"])

    def test_rehashed_trace_actions_memory_state_grades_and_identities_fail(self):
        for field in ("action", "memory_after", "state_after", "score", "step_index", "episode_id", "sha256"):
            rows, summaries = copy.deepcopy((self.traces, self.summaries))
            if field in {"action", "memory_after", "episode_id"}:
                rows[0][field] += " changed"
            elif field == "state_after":
                rows[0][field]["unsafe_booked_cents"] += 1
            elif field == "score":
                rows[0][field]["success"] = not rows[0][field]["success"]
            elif field == "step_index":
                rows[0][field] = True
            _rechain(rows, summaries)
            if field == "sha256":
                rows[0][field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.replay(rows, summaries)

    def test_rehashed_unused_missing_reordered_events_and_requests_fail(self):
        for mutation in ("unused", "missing", "order", "request", "completion", "tokens", "metering"):
            rows, summaries = copy.deepcopy((self.traces, self.summaries))
            events = rows[0]["client_events"]
            completion = next(event for event in events if event["method"] == "complete")
            if mutation == "unused":
                events.append(copy.deepcopy(events[0]))
            elif mutation == "missing":
                events.pop()
            elif mutation == "order":
                events[0], events[1] = events[1], events[0]
            elif mutation == "request":
                completion["result"]["request"]["seed"] += 1
            elif mutation == "completion":
                completion["result"]["text"] += " changed"
            elif mutation == "tokens":
                events[0]["result"] = [True]
            else:
                completion["result"]["prompt_tokens"] = True
            _rechain(rows, summaries)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.replay(rows, summaries)

    def test_rehashed_schedule_trace_and_summary_order_fail(self):
        rows, summaries = copy.deepcopy((self.traces, self.summaries))
        rows[0], rows[1] = rows[1], rows[0]
        _rechain(rows, summaries)
        with self.assertRaisesRegex(ValueError, "order"):
            self.replay(rows, summaries)
        summaries = copy.deepcopy(self.summaries)
        summaries[0], summaries[1] = summaries[1], summaries[0]
        with self.assertRaisesRegex(ValueError, "order"):
            self.replay(summaries=summaries)
        order = copy.deepcopy(self.order)
        order[0].reverse()
        with self.assertRaisesRegex(ValueError, "schedule"):
            self.replay(order=order)

    def test_duration_and_episode_metering_mutations_fail(self):
        for field, value in (("step_seconds", -1), ("step_seconds", float("inf")),
                             ("step_seconds", .0001), ("episode_seconds", -.1),
                             ("prompt_tokens", 123456)):
            rows, summaries = copy.deepcopy((self.traces, self.summaries))
            if field == "step_seconds":
                rows[0][field] = value
            else:
                summaries[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                _rechain(rows, summaries)
                self.replay(rows, summaries)

    def test_envelope_checks_sources_and_inputs_without_writes_or_client(self):
        root, directory, validator = self.envelope()
        before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
        with patch.object(runner.LocalModelClient, "__init__", side_effect=AssertionError("No model constructor")):
            result = partial.verify_partial(directory)
        self.assertTrue(result["read_only"])
        self.assertEqual(result["inference_calls"], 0)
        self.assertFalse(result["retry_allowed"])
        self.assertTrue(result["unmetered_inflight_call_possible"])
        self.assertEqual(before, {path: path.read_bytes() for path in root.rglob("*") if path.is_file()})
        validator.assert_called_once_with(directory)

    def test_fabricated_completion_and_start_drift_fail_before_protocol_replay(self):
        _, directory, validator = self.envelope()
        _write(directory / "completion.json", {"status": "PASS", "verified_steps": 512})
        with self.assertRaisesRegex(ValueError, "completion manifest"):
            partial.verify_partial(directory)
        (directory / "completion.json").unlink()
        _write(directory / "started.json", {"protocol_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "Start marker"):
            partial.verify_partial(directory)
        validator.assert_not_called()

    def test_all_three_source_locations_are_hash_guarded(self):
        root, directory, validator = self.envelope()
        for base in (root / "src", directory / "sources", directory / "development/sources"):
            path = base / "rcwt_context_v4.py"
            original = path.read_bytes()
            path.write_bytes(original + b"\n# fake source mutation\n")
            with self.subTest(base=base), self.assertRaisesRegex(ValueError, "source hash mismatch"):
                partial.verify_partial(directory)
            path.write_bytes(original)
        validator.assert_not_called()

    def test_input_mutation_and_new_completion_during_validation_fail(self):
        _, directory, validator = self.envelope()
        original = (directory / "started.json").read_bytes()
        for kind in ("input", "completion"):
            def change(_):
                if kind == "input":
                    (directory / "started.json").write_bytes(original + b"\n")
                else:
                    _write(directory / "completion.json", {"fabricated": True})
                return self.protocol
            validator.side_effect = change
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "changed during"):
                partial.verify_partial(directory)
            (directory / "started.json").write_bytes(original)

    def test_interrupted_inflight_events_are_not_added_to_resource_totals(self):
        _, directory, _ = self.envelope()
        _write(directory / "partial-step.json", {"client_events": [{"method": "complete", "result": {
            "prompt_tokens": 999999, "completion_tokens": 999999}}]})
        result = partial.verify_partial(directory)
        self.assertEqual(result["recorded_completed_step_resources"]["generation_calls"], 421)

    def test_json_and_network_guards_and_stdout_only_cli(self):
        for text in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                partial._json(text)
        with partial._offline(), self.assertRaisesRegex(RuntimeError, "Network is forbidden"):
            socket.create_connection(("127.0.0.1", 1))
        with patch.object(partial, "verify_partial", return_value={"status": "PARTIAL_VERIFIED"}) as verify, \
                patch("sys.argv", ["verify_online_v4_partial.py", "--run-dir", "explicit-fake"]), \
                patch("builtins.print") as printed:
            partial.main()
        verify.assert_called_once_with(Path("explicit-fake"))
        self.assertEqual(json.loads(printed.call_args.args[0])["status"], "PARTIAL_VERIFIED")


if __name__ == "__main__":
    unittest.main()
