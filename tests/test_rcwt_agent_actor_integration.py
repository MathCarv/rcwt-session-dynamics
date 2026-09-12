"""Actor/runner/client contracts with fake outputs only; no inference or network.

Fixtures below are deliberately unsuitable as model-performance evidence. They
exercise the production extraction, bookkeeping and offline audit boundaries.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import rcwt_agent_run as runner
from rcwt_agent_actor import ACTOR_SCHEMA, TRUNCATED_ACTION, extract_action, extract_evidence_check
from rcwt_agent_env import generate_episodes, parse_action
from rcwt_local_model import LocalModelClient, LocalModelError, SAMPLING
from test_rcwt_agent_pipeline_smoke import OfflineTransport


class PendingButApproveTransport(OfflineTransport):
    """Contradict the self-check without calling an oracle or a real model."""

    def _request(self, path, payload=None):
        response = super()._request(path, payload)
        if path == "/v1/chat/completions" and "response_format" in payload:
            public = json.loads(payload["messages"][1]["content"])["current_step"]
            if public["step_index"] == 1:
                invoice = next(event["content"] for event in public["observations"]
                               if event.get("tool") == "read_invoice")
                value = {
                    "evidence_check": {"invoice_amount_cents": invoice["total_cents"],
                                       "account_status": "active", "ownership_match": "yes",
                                       "payment_status": "pending", "return_status": "accepted",
                                       "operation_already_booked": "no_record"},
                    "tool": "record_decision",
                    "arguments": {"case_id": public["task"]["case_id"], "decision": "approve",
                                  "amount_cents": invoice["total_cents"],
                                  "reason_code": "authorized_payout"},
                }
                response["choices"][0]["message"]["content"] = json.dumps(value)
        return response


class TruncatedActorTransport(OfflineTransport):
    def _request(self, path, payload=None):
        response = super()._request(path, payload)
        if path == "/v1/chat/completions" and "response_format" in payload:
            response["choices"][0]["finish_reason"] = "length"
        return response


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            for row in records), encoding="utf-8", newline="\n")


def fake_pipeline(directory: Path) -> None:
    args = SimpleNamespace(train_count=4, validation_count=4, test_count=4,
                           seed=83, memory_budget=256, model="rcwt-local-qwen35-4b",
                           runtime_receipt=runner.ROOT / "docs/rcwt_agent_runtime.json")
    protocol = runner.prepare(directory, args)
    client = OfflineTransport(seed=args.seed)
    runner.run_cohort(client, generate_episodes("train", 4, args.seed),
                      [runner.Policy("summary", "summary")], 256, directory / "train", args.seed)
    runner.validate_stage(client, directory, protocol)
    runner.test_stage(client, directory, protocol)


class ActorRunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Any accidental fallback to a network transport fails this test suite.
        network = patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Network forbidden in unit tests"))
        network.start(); self.addCleanup(network.stop)
        quiet = patch("builtins.print")
        quiet.start(); self.addCleanup(quiet.stop)

    def test_self_check_pending_does_not_repair_model_approval(self):
        client = PendingButApproveTransport(seed=83)
        episode = generate_episodes("train", 1, 83)[0]
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "fake-traces.jsonl"
            summary = runner.run_episode(client, episode, runner.Policy("summary", "summary"), 256, trace)
            rows = runner.read_jsonl(trace)
        row = rows[1]
        raw_call = row["model_calls"][0]
        self.assertEqual(row["evidence_check"]["payment_status"], "pending")
        self.assertEqual(parse_action(row["action"]).decision, "approve")
        self.assertEqual(row["action"], extract_action(raw_call["text"], raw_call["finish_reason"]))
        self.assertEqual(row["evidence_check"], extract_evidence_check(raw_call["text"], raw_call["finish_reason"]))
        self.assertEqual(row["score"]["failure_category"], "unsafe_execution")
        self.assertEqual(row["score"]["expected_action"]["arguments"]["decision"], "hold")
        self.assertTrue(row["tool_result"]["accepted"])
        self.assertGreater(row["tool_result"]["amount_booked_cents"], 0)
        self.assertEqual(summary["unsafe_actions"], 1)
        self.assertEqual(summary["unsafe_booked_cents"], row["tool_result"]["amount_booked_cents"])
        for record in rows:
            call = record["model_calls"][0]
            schema = call["request"]["response_format"]["json_schema"]["schema"]
            self.assertEqual(schema, ACTOR_SCHEMA)
            if len(record["model_calls"]) == 2:
                payload = json.loads(record["model_calls"][1]["request"]["messages"][1]["content"])
                information = json.loads(payload["new_information"])
                self.assertEqual(set(information), {"observations", "completed_request", "action", "tool_result"})
                self.assertNotIn("evidence_check", information)
                self.assertNotIn("expected_action", information)

    def test_truncated_envelope_cannot_execute_or_supply_an_evidence_check(self):
        episode = generate_episodes("train", 1, 84)[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fake-truncated.jsonl"
            summary = runner.run_episode(TruncatedActorTransport(seed=84), episode,
                                         runner.Policy("tail", "tail"), 256, path)
            rows = runner.read_jsonl(path)
        self.assertEqual(summary["valid_actions"], 0)
        self.assertEqual(summary["failures"], {"invalid_action": 8})
        self.assertEqual(summary["unsafe_booked_cents"], 0)
        for row in rows:
            self.assertEqual(row["action"], TRUNCATED_ACTION)
            self.assertIsNone(row["evidence_check"])
            self.assertFalse(row["tool_result"]["accepted"])
            self.assertEqual(row["tool_result"]["amount_booked_cents"], 0)

    def test_changed_self_check_fails_after_trace_chain_and_completion_reseal(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "fake-audit-only"
            fake_pipeline(directory)
            self.assertEqual(runner.verify_evidence(directory)["verified_steps"], 224)
            stage = directory / "test"
            traces = runner.read_jsonl(stage / "traces.jsonl")
            summaries = runner.read_jsonl(stage / "episodes.jsonl")
            target = traces[-1]
            pair = (target["episode_id"], target["policy"])
            original_raw = target["model_calls"][0]["text"]
            target["evidence_check"]["payment_status"] = "cleared"
            previous = None
            for row in traces:
                if (row["episode_id"], row["policy"]) == pair:
                    row["previous_sha256"] = previous
                    row["sha256"] = runner.canonical_hash({key: value for key, value in row.items() if key != "sha256"})
                    previous = row["sha256"]
            for summary in summaries:
                if (summary["episode_id"], summary["policy"]) == pair:
                    summary["last_trace_sha256"] = previous
            write_jsonl(stage / "traces.jsonl", traces)
            write_jsonl(stage / "episodes.jsonl", summaries)
            completion = runner.read(stage / "completion.json")
            completion.update(traces_sha256=runner.file_hash(stage / "traces.jsonl"),
                              episodes_sha256=runner.file_hash(stage / "episodes.jsonl"))
            runner.dump(stage / "completion.json", completion)
            self.assertEqual(target["model_calls"][0]["text"], original_raw)
            with self.assertRaisesRegex(ValueError, "Evidence check differs from the actor's uncorrected output"):
                runner.verify_evidence(directory)


class ActorClientProfileTests(unittest.TestCase):
    @staticmethod
    def response() -> dict:
        return {"model": "rcwt-local-qwen35-4b", "id": "unit-test-only",
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}, "timings": {"cache_n": 0}}

    def test_actions_memory_and_proposal_share_frozen_non_thinking_profile(self):
        expected_sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                             "min_p": 0.0, "presence_penalty": 1.5, "repeat_penalty": 1.0}
        self.assertEqual(SAMPLING, expected_sampling)
        client = LocalModelClient(seed=1987)
        purposes = (("action:summary", ACTOR_SCHEMA), ("action:learned", ACTOR_SCHEMA),
                    ("memory:summary", None), ("memory:learned", None),
                    ("policy:train-proposal", None))
        with patch.object(client, "_request", return_value=self.response()) as request:
            for purpose, schema in purposes:
                result = client.complete([{"role": "user", "content": "unit fixture"}], 512, schema, purpose)
                payload = request.call_args.args[1]
                self.assertEqual({key: payload[key] for key in expected_sampling}, expected_sampling)
                self.assertEqual(payload["seed"], 1987)
                self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                self.assertFalse(payload["cache_prompt"])
                self.assertFalse(payload["stream"])
                self.assertNotIn("reasoning_budget", payload)
                self.assertEqual("response_format" in payload, schema is not None)
                if schema is not None:
                    self.assertEqual(payload["response_format"]["json_schema"]["schema"], ACTOR_SCHEMA)
                self.assertEqual(result.reasoning_chars, 0)
                self.assertIsNone(result.reasoning_sha256)
                self.assertEqual(result.api_cost_usd, 0)

    def test_repeated_action_requests_keep_the_same_seed_and_repeat_profile(self):
        client = LocalModelClient(seed=907)
        messages = [{"role": "user", "content": "same fixture"}]
        with patch.object(client, "_request", return_value=self.response()) as request:
            client.complete(messages, 512, ACTOR_SCHEMA, "action:summary")
            first = copy.deepcopy(request.call_args.args[1])
            client.complete(messages, 512, ACTOR_SCHEMA, "action:summary")
            second = copy.deepcopy(request.call_args.args[1])
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], 907)
        self.assertEqual(first["repeat_penalty"], 1.0)
        self.assertEqual(first["presence_penalty"], 1.5)

    def test_unexpected_reasoning_or_cached_response_is_rejected(self):
        for kind in ("reasoning", "cache"):
            response = self.response()
            if kind == "reasoning":
                response["choices"][0]["message"]["reasoning_content"] = "unexpected hidden reasoning"
            else:
                response["timings"]["cache_n"] = 1
            client = LocalModelClient()
            with patch.object(client, "_request", return_value=response):
                with self.assertRaises(LocalModelError):
                    client.complete([], 512, ACTOR_SCHEMA, "action:summary")
            self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
