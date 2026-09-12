"""Invented display fixtures with a FAKE verifier; never inference evidence.

These tests do not open a real run, generate a corpus, or contact a local model.
The actual archive gate is tested separately by its owner.
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rcwt_decision_v4 import task_schema

SPEC = importlib.util.spec_from_file_location(
    "online_v4_replay", Path(__file__).resolve().parents[1] / "tools/replay_online_v4.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def fake_artifacts():
    public, traces = [], []
    protocol = {"schema": "rcwt-online-memory/4", "mode": "development", "split": "train",
                "policies": ["summary", "structured"], "count": 2, "memory_budget": 256}
    for episode_index, episode_id in enumerate(("FAKE-FIRST", "FAKE-BEST")):
        steps = [{"step_index": i,
                  "task": {"case_id": f"FAKE-CASE-{episode_index}-{i}", "operation": "payout"},
                  "observations": [{"source": "message", "content": f"FAKE-OBS-{episode_index}-{i}"}]}
                 for i in range(2)]
        public.append({"episode_id": episode_id, "steps": steps})
        for policy in ("summary", "structured"):
            for i, step in enumerate(steps):
                check = {"invoice_amount_cents": None, "account_status": "unknown", "ownership_match": "unknown",
                         "payment_status": "unknown", "return_status": "unknown", "operation_already_booked": "no_record"}
                action = {"tool": "record_decision", "arguments": {
                    "case_id": step["task"]["case_id"], "decision": "ask_info", "amount_cents": 0,
                    "reason_code": "missing_evidence"}}
                draft = {"purpose": f"draft:{policy}", "text": "FAKE PLAN: not an executable action.",
                         "finish_reason": "stop", "request": {}, "prompt_tokens": 30,
                         "completion_tokens": 4, "wall_seconds": 0.5, "api_cost_usd": 0}
                final = {"purpose": f"action:{policy}", "text": json.dumps({"evidence_check": check, **action}),
                         "finish_reason": "stop", "request": {"response_format": {"type": "json_schema",
                         "json_schema": {"schema": task_schema(step)}}},
                         "prompt_tokens": 50, "completion_tokens": 6, "wall_seconds": 0.75, "api_cost_usd": 0}
                events = [{"method": "tokenize", "result": [1, 2, 3]},
                          {"method": "complete", "result": draft}, {"method": "complete", "result": final}]
                if policy == "summary" and i == 0:
                    events.append({"method": "complete", "result": {
                        "purpose": "memory:summary", "text": "FAKE SUMMARY", "prompt_tokens": 20,
                        "completion_tokens": 5, "wall_seconds": 0.25, "api_cost_usd": 0}})
                success = episode_index == 1 or (policy == "summary" and i == 1)
                traces.append({"episode_id": episode_id, "split": "train", "policy": policy, "step_index": i,
                               "public_step": copy.deepcopy(step), "memory_before": f"FAKE-STORED-{policy}-{i}",
                               "actor_memory_before": f"FAKE-CONTEXT-{policy}-{i}", "memory_after": f"FAKE-AFTER-{policy}-{i}",
                               "memory_tokens": 40, "actor_memory_tokens": 50, "memory_truncated": i == 0,
                               "action": replay.extract_action(final["text"], "stop"), "evidence_check": check,
                               "tool_result": {"accepted": True, "amount_booked_cents": 0, "simulated": True},
                               "score": {"success": success, "failure_category": "correct" if success else "unnecessary_deferral"},
                               "client_events": events, "step_seconds": 2.0, "sha256": f"FAKE-HASH-{episode_index}-{policy}-{i}"})
    return protocol, public, list(reversed(traces))


class OnlineV4ReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path("explicit-fake-v4-replay")
        self.protocol, self.public, self.traces = fake_artifacts()
        self.verified = False
        self.output = io.StringIO()

        def verify(directory):
            self.assertEqual(directory, self.directory)
            self.assertEqual(self.output.getvalue(), "")
            self.verified = True
            return {"status": "PASS", "inference_calls": 0, "verification": {"status": "PASS"},
                    "mode": "EXPLICITLY FAKE DISPLAY-TEST RECEIPT"}

        def read(path):
            self.assertTrue(self.verified, "Artifact opened before gate PASS")
            if path.name == "protocol.json":
                return copy.deepcopy(self.protocol)
            self.assertEqual(path.name, "public.json")
            return copy.deepcopy(self.public)

        def read_jsonl(path):
            self.assertTrue(self.verified, "Trace opened before gate PASS")
            self.assertEqual(path.name, "traces.jsonl")
            return copy.deepcopy(self.traces)

        self.mocks = {}
        for name, patcher in {
            "verify": patch.object(replay, "verify_archive", side_effect=verify),
            "read": patch.object(replay, "read", side_effect=read),
            "traces": patch.object(replay, "read_jsonl", side_effect=read_jsonl),
            "network": patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("No HTTP allowed")),
            "model": patch("rcwt_local_model.LocalModelClient.__init__", side_effect=AssertionError("No model client allowed")),
        }.items():
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def capture(self, **kwargs):
        self.output = io.StringIO()
        self.verified = False
        with redirect_stdout(self.output):
            replay.replay(self.directory, **kwargs)
        return self.output.getvalue()

    def first(self):
        return next(row for row in self.traces if row["episode_id"] == "FAKE-FIRST"
                    and row["policy"] == "structured" and row["step_index"] == 0)

    def test_default_first_manifest_episode_not_trace_order_or_best(self):
        text = self.capture()
        self.assertIn("episode index 0; FAKE-FIRST", text)
        self.assertNotIn("FAKE-BEST", text)
        self.assertIn("FIRST manifest episode, never the best-scoring episode", text)
        self.assertEqual(text.count("\nSTEP "), 2)
        self.assertEqual(text.count("\nSUMMARY\n"), 2)
        self.assertEqual(text.count("\nSTRUCTURED\n"), 2)
        self.assertIn('"success": false', text)
        self.assertIn("A single episode is not proof of general gain", text)
        self.mocks["verify"].assert_called_once_with(self.directory)
        self.mocks["network"].assert_not_called()
        self.mocks["model"].assert_not_called()

    def test_stored_memory_context_plan_final_receipt_and_grade_always_visible(self):
        text = self.capture()
        for label in ("STORED MEMORY BEFORE:", "EXACT CONTEXT SENT TO ACTOR:", "RAW PLAN (NOT EXECUTED; NO TOOL PARSING):",
                      "RAW FINAL RESPONSE:", "FINAL ACTOR SELF-REPORTED CHECK:", "ACTUAL ACTION (FINAL ONLY):",
                      "ACTUAL FICTIONAL TOOL RECEIPT:", "PRIVATE EVALUATION:", "RETAINED MEMORY AFTER:"):
            self.assertEqual(text.count(label), 4)
        self.assertIn("FAKE-STORED-structured-0", text)
        self.assertIn("FAKE-CONTEXT-structured-0", text)
        self.assertIn("FAKE-AFTER-summary-1", text)
        self.assertIn("TRACE: traces.jsonl line", text)
        self.assertIn("FAKE-HASH-0-structured-0", text)

    def test_all_calls_and_separate_whole_run_costs_include_summary(self):
        text = self.capture()
        self.assertEqual(text.count("RECORDED GENERATION CALL:"), 9)
        self.assertIn('"purpose": "memory:summary"', text)
        resource_lines = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("ALL-CALL STEP RESOURCES:")]
        resources = [json.loads(line) for line in resource_lines]
        self.assertEqual(resources[0]["generation_calls"], 3)
        self.assertEqual(resources[0]["model_tokens"], 115)
        self.assertEqual(resources[0]["inference_seconds"], 1.5)
        self.assertEqual(resources[1]["generation_calls"], 2)
        self.assertEqual(resources[1]["model_tokens"], 90)
        self.assertEqual(resources[1]["tokenizer_operations"], 1)
        self.assertIn("WHOLE-RUN RESOURCE TOTALS BY ARM (COST ONLY; NO CROSS-ATTEMPT POOLING)", text)
        self.assertIn('"generation_calls": 10', text)
        self.assertIn('"total_monetary_cost_usd": null', text)
        self.assertIn("zero electricity, hardware, or total monetary cost", text)
        self.assertIn("sum_step_seconds includes recorded per-step processing", text)

    def test_mode_and_split_are_explicit_not_confused_with_new_heldout(self):
        text = self.capture()
        self.assertIn("DEVELOPMENT / TRAIN: this is not a new held-out result", text)
        self.assertNotIn("CONFIRMATORY / TEST", text)
        # Only relabel this invented display fixture; no corpus is generated or read.
        self.protocol.update(mode="confirmatory", split="test")
        for row in self.traces:
            row["split"] = "test"
        text = self.capture()
        self.assertIn("CONFIRMATORY / TEST: this is a recorded episode, not a new run", text)
        self.assertNotIn("DEVELOPMENT / TRAIN", text)

    def test_task_schema_excerpt_and_shared_actor_limit_are_shown(self):
        text = self.capture()
        self.assertEqual(text.count("RECORDED TASK_SCHEMA ARGUMENT CONSTRAINTS"), 4)
        self.assertIn('"const": "FAKE-CASE-0-0"', text)
        self.assertIn('"enum": ["hold", "ask_info", "approve"]', text)
        self.assertIn("public task-bound schema; schema eligibility is NOT computed", text)
        self.assertIn("Private grades shown below were never fed", text)

    def test_invalid_final_is_not_replaced_by_json_shaped_plan(self):
        row = self.first()
        draft, final = [event["result"] for event in row["client_events"] if event["method"] == "complete"]
        draft["text"] = final["text"]
        final["text"] = "{invalid final"
        row["action"] = replay.extract_action(final["text"], "stop")
        row["evidence_check"] = None
        with patch.object(replay, "extract_action", wraps=replay.extract_action) as extract:
            text = self.capture()
        self.assertEqual(extract.call_count, 4, "Only four final completions may be parsed")
        self.assertIn('ACTUAL ACTION (FINAL ONLY): "INVALID_ACTOR_OUTPUT"', text)
        self.assertIn("RAW FINAL RESPONSE: {invalid final", text)
        self.assertNotIn("DRAFT PROPOSED ACTION", text)

    def test_truncated_or_empty_plan_is_displayed_verbatim_without_execution(self):
        row = self.first()
        draft = next(event["result"] for event in row["client_events"]
                     if event["method"] == "complete" and event["result"]["purpose"] == "draft:structured")
        for text in ("", "FAKE unfinished plan", "{FAKE not JSON}"):
            draft.update(text=text, finish_reason="length")
            output = self.capture()
            self.assertIn("RAW PLAN (NOT EXECUTED; NO TOOL PARSING): " + text, output)
            self.assertIn("PLAN FINISH REASON (NOT FINAL GRADE): length", output)
            self.assertNotIn("TRUNCATED_ACTION", output)

    def test_repeated_render_is_deterministic_and_explicit_index_is_honored(self):
        self.assertEqual(self.capture(), self.capture())
        text = self.capture(index=1)
        self.assertIn("episode index 1; FAKE-BEST", text)
        self.assertNotIn("FAKE-FIRST", text)

    def test_invalid_indices_fail_before_output(self):
        for index in (-1, 2, True, False, 0.0, "0", None):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.capture(index=index)
            self.assertEqual(self.output.getvalue(), "")

    def test_gate_failure_blocks_all_reads_and_output(self):
        for receipt in ({"status": "FAIL"}, {"status": "PASS", "inference_calls": 1},
                        {"status": "PASS"}, {}, None):
            self.mocks["verify"].side_effect = None
            self.mocks["verify"].return_value = receipt
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                self.capture()
            self.assertEqual(self.output.getvalue(), "")
        self.mocks["read"].assert_not_called()
        self.mocks["traces"].assert_not_called()

    def test_verifier_exception_blocks_all_reads_and_output(self):
        self.mocks["verify"].side_effect = ValueError("FAKE archive integrity failure")
        with self.assertRaisesRegex(ValueError, "FAKE archive integrity failure"):
            self.capture()
        self.assertEqual(self.output.getvalue(), "")
        self.mocks["read"].assert_not_called()
        self.mocks["traces"].assert_not_called()

    def test_missing_or_duplicate_pairs_or_changed_public_step_fail_closed(self):
        original = copy.deepcopy(self.traces)
        for mutation in ("missing", "duplicate", "public", "split"):
            self.traces = copy.deepcopy(original)
            row = self.first()
            if mutation == "missing":
                self.traces.remove(row)
            elif mutation == "duplicate":
                self.traces.append(copy.deepcopy(row))
            elif mutation == "public":
                row["public_step"]["task"]["case_id"] = "OTHER"
            else:
                row["split"] = "test"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.capture()
            self.assertEqual(self.output.getvalue(), "")

    def test_raw_binding_schema_and_generation_order_fail_before_output(self):
        original = copy.deepcopy(self.traces)
        for mutation in ("action", "check", "schema", "missing_plan", "duplicate_final", "plan_after", "schema_plan"):
            self.traces = copy.deepcopy(original)
            row = self.first()
            draft_event, final_event = [event for event in row["client_events"] if event["method"] == "complete"]
            if mutation == "action":
                row["action"] = "UNBOUND"
            elif mutation == "check":
                row["evidence_check"]["payment_status"] = "cleared"
            elif mutation == "schema":
                final_event["result"]["request"]["response_format"]["json_schema"]["schema"]["properties"]["arguments"]["properties"]["case_id"]["const"] = "OTHER"
            elif mutation == "missing_plan":
                row["client_events"].remove(draft_event)
            elif mutation == "duplicate_final":
                row["client_events"].append(copy.deepcopy(final_event))
            elif mutation == "plan_after":
                row["client_events"].remove(draft_event)
                row["client_events"].append(draft_event)
            else:
                draft_event["result"]["request"] = {"response_format": {}}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.capture()
            self.assertEqual(self.output.getvalue(), "")

    def test_nonlocal_cost_or_nonfinite_accounting_is_rejected(self):
        original = copy.deepcopy(self.traces)
        for key, value in (("api_cost_usd", 0.01), ("wall_seconds", float("nan")), ("prompt_tokens", True)):
            self.traces = copy.deepcopy(original)
            next(event["result"] for event in self.first()["client_events"] if event["method"] == "complete")[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.capture()
            self.assertEqual(self.output.getvalue(), "")

    def test_bad_protocol_version_or_mode_split_is_rejected(self):
        original = copy.deepcopy(self.protocol)
        for key, value in (("schema", "rcwt-online-memory/3"), ("split", "test"), ("count", 3)):
            self.protocol = {**original, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.capture()
            self.assertEqual(self.output.getvalue(), "")

    def test_cli_default_and_explicit_episode(self):
        with patch.object(replay, "replay") as render, patch("sys.argv", ["replay_online_v4.py"]):
            replay.main()
        render.assert_called_once_with(Path("results/agent_v4"), 0)
        with patch.object(replay, "replay") as render, patch("sys.argv", [
            "replay_online_v4.py", "--run-dir", "FAKE-CLI", "--episode-index", "1"]):
            replay.main()
        render.assert_called_once_with(Path("FAKE-CLI"), 1)


if __name__ == "__main__":
    unittest.main()
