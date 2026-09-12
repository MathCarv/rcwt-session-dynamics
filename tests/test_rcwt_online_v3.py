"""Offline v3 integrity tests with an EXPLICITLY FAKE transport.

The real frozen actor, simulator, compactors and runner execute against synthetic
transport replies and a reversible four-byte fake codec. These fixtures are never model-quality,
latency, tokenizer-accuracy or memory-gain evidence. Real HTTP is forbidden.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import rcwt_online_v3 as runner
from rcwt_agent_actor import ACTOR_SCHEMA
from rcwt_agent_env import Simulator, generate_episodes
from rcwt_agent_memory import MemoryState, compact_memory
from rcwt_local_model import LocalModelClient
from rcwt_memory_v3 import compact_structured
from rcwt_review_v3 import PLANNING_INSTRUCTION, REVIEW_INSTRUCTION


def _fake_actor_text(public):
    return json.dumps({"evidence_check": {
        "invoice_amount_cents": None, "account_status": "unknown",
        "ownership_match": "unknown", "payment_status": "unknown",
        "return_status": "unknown", "operation_already_booked": "no_record",
    }, "tool": "record_decision", "arguments": {
        "case_id": public["task"]["case_id"], "decision": "ask_info",
        "amount_cents": 0, "reason_code": "missing_evidence",
    }})


class ExplicitFakeV3Transport(LocalModelClient):
    """Deterministic JSON replies/four-byte tokens; never delegates to HTTP."""

    @staticmethod
    def encode_tokens(text):
        encoded = text.encode("utf-8")
        # A leading sentinel preserves each chunk's exact byte length, including
        # zero bytes. This is a test codec, not the real model's tokenizer.
        return [int.from_bytes(b"\x01" + encoded[i:i + 4], "big")
                for i in range(0, len(encoded), 4)]

    @staticmethod
    def decode_tokens(tokens):
        return b"".join(token.to_bytes((token.bit_length() + 7) // 8, "big")[1:]
                        for token in tokens).decode("utf-8", errors="replace")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transport_requests = []

    def _request(self, path, payload=None):
        self.transport_requests.append((path, copy.deepcopy(payload)))
        if path == "/tokenize":
            return {"tokens": self.encode_tokens(payload["content"])}
        if path == "/detokenize":
            return {"content": self.decode_tokens(payload["tokens"])}
        if path != "/v1/chat/completions":
            raise AssertionError("Unexpected request in explicitly fake offline transport")
        if "response_format" in payload:
            public = json.loads(payload["messages"][1]["content"])["current_step"]
            text = _fake_actor_text(public)
        elif payload["messages"][0]["content"].endswith(PLANNING_INSTRUCTION):
            # Distinct transient free-text plan, never a tool action or a summary.
            text = "EXPLICIT FAKE TEST PLAN: inspect public evidence; propose ask_info for missing evidence, amount zero."
        else:
            # Deliberately exercise truncation and detokenization, not quality.
            text = "EXPLICIT FAKE TEST SUMMARY: " + "x " * 512
        return {"model": self.model,
                "id": "explicit-fake-v3-transport-not-model-evidence",
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 13, "completion_tokens": 7},
                "timings": {"cache_n": 0}}


def _args(mode="development", development_run=None):
    return SimpleNamespace(mode=mode, dataset_seed=runner.DATASET_SEEDS[mode],
                           inference_seed=runner.INFERENCE_SEED,
                           schedule_seed=runner.SCHEDULE_SEED,
                           development_run=development_run)


def _fake(protocol=None):
    return ExplicitFakeV3Transport(
        model=(protocol or {}).get("model", "rcwt-local-qwen35-4b"),
        seed=runner.INFERENCE_SEED,
    )


def _first_complete(events):
    return next(event for event in events if event["method"] == "complete")


def _actor_complete(events, stage):
    return next(event for event in events if event["method"] == "complete"
                and event["arguments"]["purpose"].startswith(stage + ":"))


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False) + "\n" for row in rows),
                    encoding="utf-8", newline="\n")


def _reseal(directory, traces=None, summaries=None):
    """Rebuild every trace/manifest hash, without repairing semantic evidence."""
    if traces is None:
        traces = runner.read_jsonl(directory / "traces.jsonl")
    if summaries is None:
        summaries = runner.read_jsonl(directory / "episodes.jsonl")
    chain, last = None, {}
    for row in traces:
        row["previous_sha256"] = chain
        row["sha256"] = runner.canonical_hash({k: v for k, v in row.items() if k != "sha256"})
        chain = row["sha256"]
        last[(row["episode_id"], row["policy"])] = chain
    for summary in summaries:
        pair = summary["episode_id"], summary["policy"]
        if pair in last:
            summary["last_trace_sha256"] = last[pair]
    _write_jsonl(directory / "traces.jsonl", traces)
    _write_jsonl(directory / "episodes.jsonl", summaries)
    completion = runner.read(directory / "completion.json")
    completion.update(traces_sha256=runner.file_hash(directory / "traces.jsonl"),
                      episodes_sha256=runner.file_hash(directory / "episodes.jsonl"),
                      last_trace_sha256=chain)
    runner.dump(directory / "completion.json", completion)


def _reseal_protocol(directory, protocol):
    runner.dump(directory / "protocol.json", protocol)
    digest = runner.file_hash(directory / "protocol.json")
    freeze = runner.read(directory / "freeze.json")
    freeze.update(protocol_sha256=digest,
                  schedule_sha256=runner.file_hash(directory / "schedule.json"))
    runner.dump(directory / "freeze.json", freeze)
    for name in ("started.json", "completion.json"):
        if (directory / name).exists():
            item = runner.read(directory / name)
            item["protocol_sha256"] = digest
            runner.dump(directory / name, item)


class OnlineV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network_guard = patch("urllib.request.OpenerDirector.open",
                                  side_effect=AssertionError("Real HTTP forbidden in fake v3 tests"))
        cls.network_guard.start()
        cls.addClassCleanup(cls.network_guard.stop)
        cls.temporary = tempfile.TemporaryDirectory(prefix="explicit-fake-rcwt-v3-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.fixture = cls.root / "completed-fake-development"
        cls.source_hashes = runner.sources()
        cls.protocol = runner.prepare(cls.fixture, _args())
        client = _fake(cls.protocol)
        with patch("builtins.print"):
            runner.run(cls.fixture, cls.protocol, client)
        cls.fake_calls = len(client.calls)
        cls.verification = runner.verify_run(cls.fixture)

    def clone(self, label):
        destination = self.root / (self._testMethodName + "-" + label)
        shutil.copytree(self.fixture, destination)
        self.addCleanup(shutil.rmtree, destination)
        return destination

    def prepared(self, label):
        directory = self.root / (self._testMethodName + "-" + label)
        protocol = runner.prepare(directory, _args())
        self.addCleanup(shutil.rmtree, directory)
        return directory, protocol

    def test_fake_development_end_to_end_replays_4_episodes_times_2(self):
        self.assertEqual(self.verification["status"], "PASS")
        self.assertEqual(self.verification["verified_steps"], 64)
        self.assertEqual(self.fake_calls, 156)
        self.assertEqual(runner.sources(), self.source_hashes)
        traces = runner.read_jsonl(self.fixture / "traces.jsonl")
        summaries = runner.read_jsonl(self.fixture / "episodes.jsonl")
        self.assertEqual(len(summaries), 8)
        self.assertEqual(Counter(row["policy"] for row in traces), {"summary": 32, "structured": 32})
        self.assertTrue(all(row["split"] == "train" for row in traces))
        calls = [event["result"] for row in traces for event in row["client_events"]
                 if event["method"] == "complete"]
        self.assertEqual(sum(c["prompt_tokens"] for c in calls), 2028)
        self.assertEqual(sum(c["completion_tokens"] for c in calls), 1092)
        self.assertEqual(sum(row["model_calls"] for row in summaries), 156)
        self.assertEqual(sum(row["prompt_tokens"] for row in summaries), 2028)
        self.assertEqual(sum(row["completion_tokens"] for row in summaries), 1092)
        self.assertEqual({policy: sum(row["model_calls"] for row in summaries
                                     if row["policy"] == policy) for policy in runner.POLICIES},
                         {"structured": 64, "summary": 92})
        self.assertTrue(all(c["api_cost_usd"] == 0 for c in calls))
        self.assertTrue(all("explicit-fake" in c["response_id"] for c in calls))
        for row in traces:
            events = row["client_events"]
            purposes = [event["arguments"]["purpose"] for event in events if event["method"] == "complete"]
            expected = ["draft:" + row["policy"], "action:" + row["policy"]]
            if row["policy"] == "summary" and row["step_index"] < 7:
                expected.append("memory:summary")
            self.assertEqual(purposes, expected)
            self.assertLessEqual(row["memory_tokens"], 256)
            self.assertLessEqual(row["actor_memory_tokens"], 256)
            self.assertEqual(row["actor_memory_tokens"],
                             len(ExplicitFakeV3Transport.encode_tokens(row["actor_memory_before"])))
            first = _first_complete(events)
            base_messages = runner.actor_messages(row["actor_memory_before"], row["public_step"])
            self.assertEqual(first["arguments"]["messages"],
                             runner.planning_messages(base_messages))
            final = _actor_complete(events, "action")
            self.assertEqual(final["arguments"]["messages"],
                             runner.review_messages(base_messages, first["result"]["text"]))
            for event in (first, final):
                self.assertEqual(event["arguments"]["max_tokens"], 512)
            self.assertIsNone(first["arguments"]["schema"])
            self.assertNotIn("response_format", first["result"]["request"])
            self.assertEqual(final["arguments"]["schema"], ACTOR_SCHEMA)
            self.assertTrue(first["result"]["text"].startswith("EXPLICIT FAKE TEST PLAN:"))
            self.assertEqual(row["draft_format"], "unconstrained_text_plan")
            self.assertIsNone(row["draft_action"])
            self.assertIsNone(row["draft_evidence_check"])
            self.assertEqual(row["action"], runner.extract_action(
                final["result"]["text"], final["result"]["finish_reason"]))
            self.assertTrue(all(event["method"] == "tokenize"
                                for event in events[:events.index(first)]))
            if row["policy"] == "summary":
                self.assertEqual(row["actor_memory_before"], row["memory_before"])
        for summary in summaries:
            matching = [row for row in traces if (row["episode_id"], row["policy"])
                        == (summary["episode_id"], summary["policy"])]
            self.assertEqual(summary["decision_seconds"], [sum(
                _actor_complete(row["client_events"], stage)["result"]["wall_seconds"]
                for stage in ("draft", "action")) for row in matching])
        report = runner.report(self.clone("report"))
        self.assertFalse(report["passed"])
        self.assertIn("development-only", report["scope"])

    def test_public_boundary_and_summary_are_exactly_the_v2_compactor(self):
        episode = generate_episodes("train", 4, runner.DATASET_SEEDS["development"])[0]

        class CanarySimulator(Simulator):
            def execute_action(self, index, action):
                result = super().execute_action(index, action)
                return replace(result, score=replace(result.score, detail="PRIVATE_GRADE_CANARY"),
                               state={**result.state, "private": "PRIVATE_STATE_CANARY"})

        client = runner.AuditedClient(_fake())
        simulator = CanarySimulator(episode)
        public = simulator.public_step(0)
        previous = "retained public memory"
        with patch.object(runner, "compact_memory", wraps=compact_memory) as compactor:
            core = runner.perform_step(client, simulator, previous, "summary", 0, 256)
        args = compactor.call_args.args
        self.assertEqual(args[1:3], ("summary", previous))
        self.assertEqual(args[4], 256)
        information = json.loads(args[3])
        self.assertEqual(information, {"observations": public["observations"],
                         "completed_request": public["task"], "action": core["action"],
                         "tool_result": core["tool_result"]})
        self.assertNotIn("PRIVATE_", args[3])
        self.assertNotIn("expected_action", args[3])
        actor_event = _first_complete(client.events)
        self.assertEqual(actor_event["arguments"], {
            "messages": runner.planning_messages(runner.actor_messages(previous, public)),
            "max_tokens": runner.ACTION_MAX_TOKENS, "schema": None,
            "purpose": "draft:summary",
        })
        final_event = _actor_complete(client.events, "action")
        self.assertEqual(final_event["arguments"]["messages"],
                         runner.review_messages(runner.actor_messages(previous, public), actor_event["result"]["text"]))
        self.assertNotIn("PRIVATE_", json.dumps(final_event["arguments"]))
        self.assertNotIn("expected_action", json.dumps(final_event["arguments"]))
        direct = runner.AuditedClient(_fake())
        reference = compact_memory(direct, "summary", previous, args[3], 256)
        self.assertEqual(core["memory_after"], reference.text)
        self.assertEqual(core["memory_truncated"], reference.truncated)
        self.assertTrue(core["memory_truncated"])
        self.assertEqual(core["memory_tokens"], 256)
        observed = [(e["method"], e["arguments"])
                    for e in client.events[client.events.index(final_event) + 1:-1]]
        self.assertEqual(observed, [(e["method"], e["arguments"]) for e in direct.events])

    def test_initial_actor_input_parity_and_view_budget_precede_completion(self):
        codec = ExplicitFakeV3Transport
        self.assertEqual(codec.decode_tokens(codec.encode_tokens("fake\x00ação🙂end")), "fake\x00ação🙂end")
        episodes = generate_episodes("train", 4, runner.DATASET_SEEDS["development"])
        for episode in episodes:
            actor_inputs = {}
            for policy in runner.POLICIES:
                with self.subTest(family=episode.family, policy=policy):
                    audited = runner.AuditedClient(_fake())
                    core = runner.perform_step(audited, Simulator(episode), "", policy, 0, 256)
                    actor = _first_complete(audited.events)
                    self.assertEqual(actor["arguments"]["messages"],
                                     runner.planning_messages(runner.actor_messages("", episode.public_step(0))))
                    self.assertEqual(actor["result"]["request"]["messages"], actor["arguments"]["messages"])
                    self.assertEqual(core["memory_before"], "")
                    self.assertEqual(core["actor_memory_before"], "")
                    self.assertEqual(core["actor_memory_tokens"], 0)
                    self.assertEqual(audited.events[0]["method"], "tokenize")
                    self.assertNotIn("expected_action", json.dumps(actor["arguments"]["messages"]))
                    final = _actor_complete(audited.events, "action")
                    final_messages = final["arguments"]["messages"]
                    self.assertEqual(final_messages[:2], runner.actor_messages("", episode.public_step(0)))
                    self.assertEqual(final_messages[2], {"role": "assistant", "content": actor["result"]["text"]})
                    self.assertEqual(final_messages[3], {"role": "user", "content": REVIEW_INSTRUCTION})
                    self.assertNotIn("expected_action", json.dumps(final_messages))
                    actor_inputs[policy] = [event["result"]["request"] for event in (actor, final)]
            self.assertEqual(actor_inputs["summary"], actor_inputs["structured"])
        for policy in runner.POLICIES:
            audited = runner.AuditedClient(_fake())
            simulator = Simulator(episodes[0])
            previous = "" if policy == "structured" else "x" * 1025
            with patch.object(runner, "read_memory", return_value="x" * 1025):
                with self.subTest(overflow_policy=policy), self.assertRaisesRegex(ValueError, "Actor memory view exceeds"):
                    runner.perform_step(audited, simulator, previous, policy, 0, 256)
            self.assertFalse(any(event["method"] == "complete" for event in audited.events))
            self.assertFalse(audited.client.calls)
            self.assertEqual(simulator.next_step, 0)
        audited = runner.AuditedClient(_fake())
        with patch.object(runner, "read_memory", return_value="x" * 1024):
            core = runner.perform_step(audited, Simulator(episodes[0]), "", "structured", 0, 256)
        self.assertEqual(core["actor_memory_tokens"], 256)
        self.assertEqual(core["memory_before"], "")

    def test_reader_gets_only_public_inputs_and_does_not_import_new_values(self):
        episode = generate_episodes("train", 4, runner.DATASET_SEEDS["development"])[0]
        case_id = episode.public_step(0)["task"]["case_id"]
        previous = compact_structured(ExplicitFakeV3Transport.encode_tokens, "", json.dumps({
            "observations": [{"source": "tool", "tool": "read_payment",
                              "content": {"invoice_id": case_id, "status": "cleared"}}],
        }), 256).text
        self.assertIn("cleared", previous)
        # Synthetic boundary fixture: the current public update carries new
        # values, while the read card may only invalidate corresponding old facts.
        observations = (
            {"source": "tool", "tool": "read_payment",
             "content": {"invoice_id": case_id, "status": "pending"}},
            {"source": "tool", "tool": "read_invoice", "content": {
                "invoice_id": case_id, "total_cents": 9876543, "currency": "BRL",
                "account_id": "NEW_ACCOUNT_VALUE_CANARY",
                "merchant_id": "NEW_MERCHANT_VALUE_CANARY", "order_id": "NEW_ORDER_VALUE_CANARY",
            }},
        )
        first = replace(episode.steps[0], observations=observations)
        episode = replace(episode, steps=(first, *episode.steps[1:]))

        class PrivateCanarySimulator(Simulator):
            def execute_action(self, index, action):
                result = super().execute_action(index, action)
                return replace(result, score=replace(result.score, detail="PRIVATE_GRADE_CANARY"),
                               state={**result.state, "private": "PRIVATE_STATE_CANARY"})

        audited = runner.AuditedClient(_fake())
        reader = runner.read_memory
        with patch.object(runner, "read_memory", wraps=reader) as read_spy, \
                patch.object(runner, "compact_structured", wraps=compact_structured) as write_spy:
            core = runner.perform_step(audited, PrivateCanarySimulator(episode), previous,
                                       "structured", 0, 256)
        self.assertEqual(read_spy.call_args.args, (previous, episode.public_step(0)))
        self.assertEqual(set(read_spy.call_args.kwargs), {"tokenize", "budget"})
        self.assertEqual(read_spy.call_args.kwargs["budget"], 256)
        self.assertEqual(write_spy.call_args.args[1], previous)
        self.assertEqual(core["memory_before"], previous)
        self.assertNotEqual(core["actor_memory_before"], previous)
        card = json.loads(core["actor_memory_before"])
        self.assertIn("payment_status", card["invalidated_fields"])
        self.assertNotIn("payment_status", card["retained_evidence"])
        self.assertNotIn("invoice_amount_cents", card["retained_evidence"])
        for value in ("pending", "cleared", "9876543", "NEW_ACCOUNT_VALUE_CANARY",
                      "NEW_MERCHANT_VALUE_CANARY", "NEW_ORDER_VALUE_CANARY", "PRIVATE_"):
            self.assertNotIn(value, core["actor_memory_before"])
        actor = _first_complete(audited.events)
        payload = json.loads(actor["arguments"]["messages"][1]["content"])
        self.assertEqual(payload["retained_memory"], core["actor_memory_before"])
        self.assertEqual(payload["current_step"], episode.public_step(0))
        self.assertIn("9876543", json.dumps(payload["current_step"]))
        self.assertNotIn("PRIVATE_", json.dumps(actor["arguments"]))
        self.assertNotIn("expected_action", json.dumps(actor["arguments"]))
        self.assertNotIn("PRIVATE_", write_spy.call_args.args[2])

    def test_only_final_model_action_executes_even_when_draft_is_invalid_or_truncated(self):
        # Explicit transport controls, not model-quality evidence: an unsafe
        # draft must never book money or become a fallback for a bad final.
        episode = generate_episodes("train", 4, runner.DATASET_SEEDS["development"])[1]
        for policy in runner.POLICIES:
            for draft_kind, final_kind in (("unsafe", "safe"), ("invalid", "safe"),
                                            ("truncated", "safe"), ("safe", "unsafe"),
                                            ("safe", "invalid"), ("safe", "truncated")):
                with self.subTest(policy=policy, draft=draft_kind, final=final_kind):
                    client, simulator, timeline = _fake(), Simulator(episode), []
                    original_complete, original_execute = client.complete, simulator.execute_action

                    def controlled_complete(*args, **kwargs):
                        stage = kwargs["purpose"].split(":")[0]
                        call = original_complete(*args, **kwargs)
                        if stage not in {"draft", "action"}:
                            return call
                        # Neither pass has been executed when the model is called.
                        self.assertEqual(simulator.state_dict(), {
                            "next_step": 0, "payout_cents": 0, "refund_cents": 0,
                            "unsafe_booked_cents": 0, "completed": [], "ledger": [],
                        })
                        timeline.append(stage)
                        kind = draft_kind if stage == "draft" else final_kind
                        if kind == "invalid":
                            return replace(call, text="EXPLICIT FAKE INVALID DRAFT OR FINAL")
                        if kind == "truncated":
                            return replace(call, finish_reason="length")
                        if kind == "unsafe":
                            public = json.loads(call.request["messages"][1]["content"])["current_step"]
                            envelope = json.loads(_fake_actor_text(public))
                            envelope["arguments"].update(decision="approve", amount_cents=123,
                                                         reason_code="authorized_payout")
                            return replace(call, text=json.dumps(envelope))
                        return call

                    def observed_execute(index, action):
                        self.assertEqual(timeline, ["draft", "action"])
                        timeline.append("execute")
                        return original_execute(index, action)

                    audited = runner.AuditedClient(client)
                    with patch.object(client, "complete", side_effect=controlled_complete), \
                            patch.object(simulator, "execute_action", side_effect=observed_execute) as execution, \
                            patch.object(runner, "extract_action", wraps=runner.extract_action) as extract_action, \
                            patch.object(runner, "extract_evidence_check", wraps=runner.extract_evidence_check) as extract_check:
                        core = runner.perform_step(audited, simulator, "", policy, 0, 256)
                    self.assertEqual(timeline, ["draft", "action", "execute"])
                    execution.assert_called_once_with(0, core["action"])
                    draft = _actor_complete(audited.events, "draft")
                    final = _actor_complete(audited.events, "action")
                    extract_action.assert_called_once_with(final["result"]["text"], final["result"]["finish_reason"])
                    extract_check.assert_called_once_with(final["result"]["text"], final["result"]["finish_reason"])
                    self.assertEqual(final["arguments"]["messages"][2]["content"], draft["result"]["text"])
                    self.assertEqual(core["draft_format"], "unconstrained_text_plan")
                    self.assertIsNone(core["draft_action"])
                    self.assertIsNone(core["draft_evidence_check"])
                    self.assertEqual(core["action"], runner.extract_action(
                        final["result"]["text"], final["result"]["finish_reason"]))
                    self.assertEqual(core["state_after"]["next_step"], 1)
                    booked = 123 if final_kind == "unsafe" else 0
                    self.assertEqual(core["state_after"]["payout_cents"], booked)
                    self.assertEqual(core["state_after"]["unsafe_booked_cents"], booked)
                    self.assertEqual(len(core["state_after"]["ledger"]),
                                     int(final_kind in {"safe", "unsafe"}))
                    if final_kind == "unsafe":
                        # The runner must not silently repair or veto the final
                        # proposal using its self-check or private evaluator.
                        self.assertEqual(core["evidence_check"]["payment_status"], "unknown")
                        self.assertEqual(core["tool_result"]["amount_booked_cents"], 123)
                    if policy == "summary":
                        compaction = next(e for e in audited.events if e["method"] == "complete"
                                          and e["arguments"]["purpose"] == "memory:summary")
                        text = json.dumps(compaction["arguments"]["messages"])
                        self.assertNotIn("draft_evidence_check", text)
                        self.assertNotIn("draft_action", text)
                        self.assertNotIn("EXPLICIT FAKE TEST PLAN:", text)

    def test_review_failure_preserves_metered_draft_but_executes_nothing(self):
        directory, protocol = self.prepared("review-failure")
        client = _fake(protocol)
        original = client.complete

        def fail_only_review(*args, **kwargs):
            if kwargs["purpose"].startswith("action:"):
                raise RuntimeError("explicit fake review failure")
            return original(*args, **kwargs)

        with patch.object(client, "complete", side_effect=fail_only_review), \
                patch.object(runner.Simulator, "execute_action", side_effect=AssertionError("Draft must not execute")) as execute:
            with self.assertRaisesRegex(RuntimeError, "fake review failure"):
                runner.run(directory, protocol, client)
        execute.assert_not_called()
        partial = runner.read(directory / "partial-step.json")
        events = [event for event in partial["client_events"] if event["method"] == "complete"]
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["arguments"]["purpose"].startswith("draft:"))
        self.assertEqual((events[0]["result"]["prompt_tokens"], events[0]["result"]["completion_tokens"]), (13, 7))
        self.assertTrue((directory / "aborted.json").is_file())
        self.assertFalse((directory / "completion.json").exists())

    def test_structured_uses_only_tokenization_and_no_llm_compactor(self):
        episode = generate_episodes("train", 4, runner.DATASET_SEEDS["development"])[0]
        client = runner.AuditedClient(_fake())
        with patch.object(runner, "compact_memory", side_effect=AssertionError("LLM compactor forbidden")), \
                patch.object(runner, "compact_structured", wraps=compact_structured) as reducer:
            core = runner.perform_step(client, Simulator(episode), "", "structured", 0, 256)
        self.assertEqual(sum(event["method"] == "complete" for event in client.events), 2)
        self.assertFalse(any(event["method"] == "detokenize" for event in client.events))
        self.assertEqual(set(json.loads(reducer.call_args.args[2])),
                         {"observations", "completed_request", "action", "tool_result"})
        self.assertEqual(len(ExplicitFakeV3Transport.encode_tokens(core["memory_after"])), core["memory_tokens"])
        for retained, message in ((MemoryState("{}", calls=[object()]), "hide model calls"),
                                  (MemoryState("x" * 1025), "exceeds")):
            with self.subTest(message=message), patch.object(runner, "compact_structured", return_value=retained):
                with self.assertRaisesRegex(ValueError, message):
                    runner.perform_step(runner.AuditedClient(_fake()), Simulator(episode), "", "structured", 0, 256)

    def test_final_step_neither_compacts_nor_changes_retained_memory(self):
        episode = generate_episodes("train", 4, runner.DATASET_SEEDS["development"])[0]
        for policy in runner.POLICIES:
            simulator, memory = Simulator(episode), ""
            for index in range(7):
                memory = runner.perform_step(runner.AuditedClient(_fake()), simulator, memory,
                                             policy, index, 256)["memory_after"]
            audited = runner.AuditedClient(_fake())
            with patch.object(runner, "compact_memory", side_effect=AssertionError("No final compaction")), \
                    patch.object(runner, "compact_structured", side_effect=AssertionError("No final compaction")):
                core = runner.perform_step(audited, simulator, memory, policy, 7, 256)
            self.assertEqual(core["memory_after"], memory)
            self.assertFalse(core["memory_truncated"])
            first = _first_complete(audited.events)
            self.assertTrue(all(event["method"] == "tokenize"
                                for event in audited.events[:audited.events.index(first)]))
            self.assertEqual([event["method"] for event in audited.events[audited.events.index(first):]],
                             ["complete", "complete", "tokenize"])
        client = _fake()
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            runner.perform_step(client, Simulator(episode), "", "unregistered", 0, 256)
        self.assertFalse(client.transport_requests)

    def test_prepare_uses_real_frozen_sources_and_rejects_existing_or_invalid_args(self):
        protocol = self.protocol
        self.assertEqual(protocol["schema"], "rcwt-online-memory/3.3")
        self.assertEqual(len(runner.SOURCE_NAMES), 11)
        self.assertIn("rcwt_retrieval_v3.py", runner.SOURCE_NAMES)
        self.assertIn("rcwt_review_v3.py", runner.SOURCE_NAMES)
        self.assertEqual(protocol["actor_passes"], 2)
        self.assertEqual(protocol["draft_format"], "unconstrained_text_plan")
        self.assertEqual(protocol["self_review"],
                         "Always, identical fixed instruction in both arms; only final model action executes")
        self.assertEqual((protocol["count"], protocol["split"], protocol["counts"]), (4, "train", {"train": 4}))
        self.assertEqual(protocol["source_sha256"], runner.sources())
        for name in runner.SOURCE_NAMES:
            self.assertEqual(runner.file_hash(self.fixture / "sources" / name),
                             runner.file_hash(runner.ROOT / "src" / name))
        with self.assertRaisesRegex(ValueError, "nonexistent"):
            runner.prepare(self.fixture, _args())
        for key, value in (("mode", "invalid"), ("dataset_seed", 17),
                           ("inference_seed", 18), ("schedule_seed", 19), ("dataset_seed", True)):
            args = _args()
            setattr(args, key, value)
            directory = self.root / (self._testMethodName + f"-{key}-{value}")
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runner.prepare(directory, args)
            self.assertFalse(directory.exists())

    def test_schedule_is_frozen_and_balanced_within_each_family(self):
        order = runner.schedule(32, runner.SCHEDULE_SEED)
        self.assertEqual(order, runner.schedule(32, runner.SCHEDULE_SEED))
        self.assertTrue(all(sorted(pair) == sorted(runner.POLICIES) for pair in order))
        for family_index in range(4):
            self.assertEqual(Counter(order[i][0] for i in range(family_index, 32, 4)),
                             {"summary": 4, "structured": 4})

    def test_run_refuses_existing_or_divergent_inputs_before_any_fake_request(self):
        for marker in ("started.json", "traces.jsonl", "episodes.jsonl", "completion.json", "aborted.json"):
            directory, protocol = self.prepared(marker)
            (directory / marker).write_text("{}", encoding="utf-8")
            client = _fake(protocol)
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, "already started"):
                runner.run(directory, protocol, client)
            self.assertFalse(client.transport_requests)
        for kind in ("protocol", "seed", "model"):
            directory, protocol = self.prepared(kind)
            client = _fake(protocol)
            if kind == "protocol":
                protocol["memory_budget"] = 255
            else:
                setattr(client, kind, "wrong" if kind == "model" else runner.INFERENCE_SEED + 1)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "diverges"):
                runner.run(directory, protocol, client)
            self.assertFalse(client.transport_requests)
            self.assertFalse((directory / "started.json").exists())

    def test_failed_first_call_is_aborted_and_cannot_be_retried(self):
        directory, protocol = self.prepared("abort")
        client = _fake(protocol)
        with patch.object(client, "_request", side_effect=RuntimeError("explicit fake failure")):
            with self.assertRaisesRegex(RuntimeError, "fake failure"):
                runner.run(directory, protocol, client)
        self.assertTrue((directory / "started.json").is_file())
        self.assertTrue((directory / "partial-step.json").is_file())
        aborted = runner.read(directory / "aborted.json")
        self.assertFalse(aborted["retry_allowed"])
        self.assertTrue(aborted["unmetered_inflight_call_possible"])
        with self.assertRaisesRegex(ValueError, "already started"):
            runner.run(directory, protocol, _fake(protocol))
        with self.assertRaisesRegex(ValueError, "aborted or partial"):
            runner.verify_run(directory)

    def test_invalid_but_metered_call_is_preserved_in_partial_receipt(self):
        directory, protocol = self.prepared("invalid-metered-call")
        client = _fake(protocol)
        original = client.complete

        def invalid_complete(*args, **kwargs):
            return replace(original(*args, **kwargs), api_cost_usd=1)

        with patch.object(client, "complete", side_effect=invalid_complete):
            with self.assertRaisesRegex(ValueError, "inference contract"):
                runner.run(directory, protocol, client)
        partial = runner.read(directory / "partial-step.json")
        self.assertEqual(sum(e["method"] == "complete" for e in partial["client_events"]), 1)
        result = _first_complete(partial["client_events"])["result"]
        self.assertEqual((result["prompt_tokens"], result["completion_tokens"]), (13, 7))
        self.assertEqual(result["api_cost_usd"], 1)
        self.assertTrue((directory / "aborted.json").is_file())
        self.assertFalse((directory / "completion.json").exists())

    def test_protocol_mutations_are_rejected_even_after_refreezing(self):
        mutations = {
            "schema": "rcwt-online-memory/3.2", "actor_passes": 1, "draft_format": "json_action",
            "self_review": "Conditional review only after an oracle failure",
            "memory_budget": 255, "count": 8, "counts": {"train": 8}, "split": "test",
            "dataset_seed": 17, "inference_seed": 18, "schedule_seed": 19,
            "analysis_seed": 20, "bootstrap_samples": 1, "accuracy_gate": "always pass",
            "model": "different-model", "api_cost_usd": 1,
            "total_monetary_cost_usd": 0, "development": {"gate": {"passed": True}},
            "policies": ["structured", "summary"], "steps_per_episode": 7,
            "v2_result_sha256": "0" * 64, "v2_diagnosis_sha256": "0" * 64,
        }
        for key, value in mutations.items():
            directory = self.clone(key)
            protocol = runner.read(directory / "protocol.json")
            protocol[key] = value
            _reseal_protocol(directory, protocol)
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_protocol(directory)
        directory = self.clone("runtime")
        protocol = runner.read(directory / "protocol.json")
        protocol["runtime"]["host"] = "203.0.113.1"
        _reseal_protocol(directory, protocol)
        with self.assertRaisesRegex(ValueError, "drift"):
            runner.validate_protocol(directory)

    def test_schedule_corpus_and_snapshot_tampering_cannot_be_rehashed_away(self):
        directory = self.clone("schedule")
        order = runner.read(directory / "schedule.json")
        order[0].reverse()
        runner.dump(directory / "schedule.json", order)
        _reseal_protocol(directory, runner.read(directory / "protocol.json"))
        with self.assertRaisesRegex(ValueError, "schedule drift"):
            runner.validate_protocol(directory)
        for kind in ("public", "oracle"):
            directory = self.clone(kind)
            corpus = runner.read(directory / (kind + ".json"))
            corpus[0]["episode_id"] += "-tampered"
            runner.dump(directory / (kind + ".json"), corpus)
            protocol = runner.read(directory / "protocol.json")
            protocol[kind + "_sha256"] = runner.canonical_hash(corpus)
            _reseal_protocol(directory, protocol)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Corpus or oracle"):
                runner.validate_protocol(directory)
        directory = self.clone("source")
        source = directory / "sources" / runner.SOURCE_NAMES[0]
        source.write_text(source.read_text(encoding="utf-8") + "\n# fake tamper\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Source snapshot"):
            runner.validate_protocol(directory)
        protocol = runner.read(directory / "protocol.json")
        protocol["source_sha256"]["src/" + runner.SOURCE_NAMES[0]] = runner.file_hash(source)
        _reseal_protocol(directory, protocol)
        with self.assertRaisesRegex(ValueError, "source or schedule drift"):
            runner.validate_protocol(directory)

    def test_rehashed_actions_checks_memory_and_tool_state_must_replay(self):
        for field in ("draft_format", "draft_action", "draft_evidence_check", "action", "evidence_check",
                      "memory_before", "actor_memory_before", "memory_after",
                      "memory_tokens", "actor_memory_tokens", "memory_truncated", "tool_result", "state_after", "score"):
            directory = self.clone(field)
            traces = runner.read_jsonl(directory / "traces.jsonl")
            row = traces[0]
            if field in {"draft_format", "action", "memory_before", "actor_memory_before", "memory_after"}:
                row[field] += " "
            elif field == "draft_action":
                row[field] = row["action"]
            elif field == "draft_evidence_check":
                row[field] = row["evidence_check"]
            elif field == "evidence_check":
                row[field]["account_status"] = "active"
            elif field in {"memory_tokens", "actor_memory_tokens"}:
                row[field] += 1
            elif field == "memory_truncated":
                row[field] = not row[field]
            elif field == "tool_result":
                row[field]["accepted"] = not row[field]["accepted"]
            elif field == "state_after":
                row[field]["unsafe_booked_cents"] += 1
            else:
                row[field]["success"] = not row[field]["success"]
            _reseal(directory, traces)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "does not replay"):
                runner.verify_run(directory)

    def test_rehashed_request_completion_and_call_metering_tampering_is_detected(self):
        mutations = ("messages", "schema", "model", "seed", "raw_text", "purpose",
                     "prompt_tokens", "completion_tokens", "cache", "reasoning", "cost")
        for stage, mutation in ((stage, mutation) for stage in ("draft", "action") for mutation in mutations):
            directory = self.clone(stage + "-" + mutation)
            traces = runner.read_jsonl(directory / "traces.jsonl")
            event = _actor_complete(traces[0]["client_events"], stage)
            call = event["result"]
            if mutation == "messages":
                call["request"]["messages"][1]["content"] += " "
            elif mutation == "schema":
                if stage == "draft":
                    call["request"]["response_format"] = {"type": "json_object"}
                else:
                    call["request"]["response_format"]["json_schema"]["strict"] = False
            elif mutation == "model":
                call["request"]["model"] = "wrong"
            elif mutation == "seed":
                call["request"]["seed"] += 1
            elif mutation == "raw_text":
                call["text"] = "INVALID FAKE ACTION"
            elif mutation == "purpose":
                call["purpose"] = "unregistered"
            elif mutation == "prompt_tokens":
                call["prompt_tokens"] = 0
            elif mutation == "completion_tokens":
                call["completion_tokens"] = runner.ACTION_MAX_TOKENS + 1
            elif mutation == "cache":
                call["timings"]["cache_n"] = 1
            elif mutation == "reasoning":
                call["reasoning_chars"] = 1
            else:
                call["api_cost_usd"] = 1
            _reseal(directory, traces)
            with self.subTest(stage=stage, mutation=mutation), self.assertRaises(ValueError):
                runner.verify_run(directory)

    def test_literal_draft_and_review_bindings_cannot_be_rehashed_away(self):
        for mutation in ("draft_whitespace", "assistant_draft", "review_instruction", "remove_review", "swap_passes"):
            directory = self.clone(mutation)
            traces = runner.read_jsonl(directory / "traces.jsonl")
            events = traces[0]["client_events"]
            draft, final = (_actor_complete(events, stage) for stage in ("draft", "action"))
            if mutation == "draft_whitespace":
                # No draft action is parsed, but the review must receive the
                # literal plan actually recorded as the first completion.
                draft["result"]["text"] += " "
            elif mutation in {"assistant_draft", "review_instruction"}:
                index = 2 if mutation == "assistant_draft" else 3
                final["arguments"]["messages"][index]["content"] += " "
                final["result"]["request"]["messages"] = copy.deepcopy(final["arguments"]["messages"])
            elif mutation == "remove_review":
                events.remove(final)
            else:
                draft_index, final_index = events.index(draft), events.index(final)
                events[draft_index], events[final_index] = events[final_index], events[draft_index]
            _reseal(directory, traces)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.verify_run(directory)

    def test_rehashed_tokenization_input_invalid_ids_and_operation_order_fail(self):
        for mutation in ("text", "negative_id", "boolean_id", "result_type", "count", "unused", "missing", "order"):
            directory = self.clone(mutation)
            traces = runner.read_jsonl(directory / "traces.jsonl")
            events = traces[0]["client_events"]
            token = next(event for event in events if event["method"] == "tokenize")
            if mutation == "text":
                token["arguments"]["text"] += "changed"
            elif mutation == "negative_id":
                token["result"] = [-1]
            elif mutation == "boolean_id":
                token["result"] = [True]
            elif mutation == "result_type":
                token["result"] = "not-token-ids"
            elif mutation == "count":
                self.assertEqual(events[-1]["method"], "tokenize")
                events[-1]["result"].append(0)
            elif mutation == "unused":
                events.append(copy.deepcopy(token))
            elif mutation == "missing":
                events.pop()
            else:
                events[0], events[1] = events[1], events[0]
            _reseal(directory, traces)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.verify_run(directory)

    def test_rehashed_extra_missing_reordered_records_and_counters_fail(self):
        for mutation in ("extra_trace", "missing_trace", "extra_summary", "trace_order",
                         "step_seconds", "episode_seconds", "decision_seconds", "prompt_tokens", "unsafe_actions"):
            directory = self.clone(mutation)
            traces = runner.read_jsonl(directory / "traces.jsonl")
            summaries = runner.read_jsonl(directory / "episodes.jsonl")
            if mutation == "extra_trace":
                traces.append(copy.deepcopy(traces[-1]))
            elif mutation == "missing_trace":
                traces.pop()
            elif mutation == "extra_summary":
                summaries.append(copy.deepcopy(summaries[-1]))
            elif mutation == "trace_order":
                traces[0], traces[1] = traces[1], traces[0]
            elif mutation == "step_seconds":
                traces[0]["step_seconds"] = -1
            elif mutation == "episode_seconds":
                summaries[0]["episode_seconds"] = -1
            elif mutation == "decision_seconds":
                summaries[0]["decision_seconds"][0] += 1
            else:
                summaries[0][mutation] += 1
            _reseal(directory, traces, summaries)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.verify_run(directory)

    def test_confirmatory_prepare_refuses_absent_or_failed_development_before_new_data(self):
        for label, development in (("absent", None), ("failed", self.fixture)):
            directory = self.root / (self._testMethodName + "-" + label)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "development|Development"):
                runner.prepare(directory, _args("confirmatory", development))
            self.assertFalse(directory.exists())

    def test_replay_missing_extra_and_invalid_detokenization_are_rejected(self):
        replay = runner.ReplayClient([], "fake", 1)
        with self.assertRaisesRegex(ValueError, "Missing"):
            replay.tokenize("x")
        event = {"method": "detokenize", "arguments": {"tokens": [1]}, "result": 5}
        replay = runner.ReplayClient([event], "fake", 1)
        with self.assertRaisesRegex(ValueError, "decoded text"):
            replay.detokenize([1])
        replay = runner.ReplayClient([event], "fake", 1)
        with self.assertRaisesRegex(ValueError, "Unused"):
            replay.finish()

    def test_development_gate_needs_accuracy_gain_without_worse_safety(self):
        base = {"policy": "summary", "successes": 5, "steps": 8,
                "unsafe_actions": 1, "unsafe_booked_cents": 100}
        candidate = {**base, "policy": "structured", "successes": 6}
        self.assertTrue(runner.development_gate([base, candidate])["passed"])
        for update in ({"successes": 5}, {"unsafe_actions": 2}, {"unsafe_booked_cents": 101}):
            with self.subTest(update=update):
                self.assertFalse(runner.development_gate([base, {**candidate, **update}])["passed"])
        with self.assertRaisesRegex(ValueError, "Empty"):
            runner.summarize([], 0)

    def test_episode_summary_recomputes_exact_accuracy_safety_and_all_call_costs(self):
        def action(decision, amount, reason):
            return {"tool": "record_decision", "arguments": {
                "case_id": "explicit-fake-case", "decision": decision,
                "amount_cents": amount, "reason_code": reason,
            }}

        hold = action("hold", 0, "account_restricted")
        rows = []
        for index, (actual, expected, success, category) in enumerate((
            (hold, hold, True, "correct"),
            (action("approve", 20, "authorized_payout"), hold, False, "unsafe_execution"),
            ("INVALID_FAKE_ACTION", action("ask_info", 0, "missing_evidence"), False, "invalid_action"),
        )):
            events = [{"method": "tokenize", "arguments": {"text": "fake actor view"}, "result": [1]},
                      {"method": "complete", "result": {
                "purpose": "draft:summary", "prompt_tokens": 4, "completion_tokens": 2, "wall_seconds": 0.25,
            }}, {"method": "complete", "result": {
                "purpose": "action:summary", "prompt_tokens": 10, "completion_tokens": 5, "wall_seconds": index + 1,
            }}]
            if index == 0:
                events.append({"method": "complete", "result": {
                    "purpose": "memory:summary", "prompt_tokens": 2, "completion_tokens": 3, "wall_seconds": 0.5,
                }})
            rows.append({"episode_id": "explicit-fake-unit-summary", "family": "fake",
                         "split": "train", "policy": "summary", "action": actual,
                         "score": {"success": success, "failure_category": category,
                                   "expected_action": expected}, "client_events": events,
                         "state_after": {"unsafe_booked_cents": 0 if index == 0 else 20},
                         "memory_truncated": index != 1, "step_seconds": [2, 3, 4][index],
                         "sha256": "fake-row-" + str(index)})
        self.assertEqual(runner.summarize(rows, 10), {
            "episode_id": "explicit-fake-unit-summary", "family": "fake", "split": "train",
            "policy": "summary", "steps": 3, "successes": 1,
            "failures": {"unsafe_execution": 1, "invalid_action": 1},
            "valid_actions": 2, "unsafe_actions": 1, "unsafe_booked_cents": 20,
            "prompt_tokens": 44, "completion_tokens": 24, "model_calls": 7,
            "inference_seconds": 7.25, "decision_seconds": [1.25, 2.25, 3.25],
            "step_seconds": [2, 3, 4], "episode_seconds": 10, "memory_truncations": 2,
            "last_trace_sha256": "fake-row-2",
        })


if __name__ == "__main__":
    unittest.main()
