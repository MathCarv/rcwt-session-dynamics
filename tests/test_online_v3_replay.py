"""Offline replay contracts with invented fixtures, never model evidence.

No real run, test corpus, local server, or model is read or invoked. The
verification receipt and artifact reads are patched solely for display tests.
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

import rcwt_agent_run as runner


SPEC = importlib.util.spec_from_file_location("online_v3_replay", runner.ROOT / "tools/replay_online_v3.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def _fake_artifacts():
    """The first invented episode is intentionally not the best-looking one."""
    public, traces = [], []
    for episode_index, episode_id in enumerate(("FAKE-FIRST-EPISODE", "FAKE-BEST-EPISODE")):
        steps = [
            {"step_index": step,
             "observations": [{"source": "text", "content": f"FAKE-OBSERVATION-{episode_index}-{step}"}],
             "task": {"case_id": f"FAKE-CASE-{episode_index}", "requested_operation": "payout"}}
            for step in range(8)
        ]
        public.append({"episode_id": episode_id, "steps": steps})
        for policy in ("summary", "structured"):
            for step, public_step in enumerate(steps):
                success = episode_index == 1 or (step % 3 != (0 if policy == "summary" else 1))
                check = {"invoice_amount_cents": 12000, "account_status": "unknown", "ownership_match": "unknown",
                         "payment_status": "unknown", "return_status": "unknown", "operation_already_booked": "no_record"}
                action = {"tool": "record_decision", "arguments": {
                    "case_id": f"FAKE-CASE-{episode_index}", "decision": "ask_info",
                    "amount_cents": 0, "reason_code": "missing_evidence"}}
                calls = [{"method": "complete", "result": {
                    "prompt_tokens": 100 + step, "completion_tokens": 10,
                    "wall_seconds": 0.875 + step / 10, "purpose": f"action:{policy}",
                    "text": json.dumps({"evidence_check": check, **action}), "finish_reason": "stop"}}]
                if policy == "summary" and step < 7:
                    calls.append({"method": "complete", "result": {
                        "prompt_tokens": 20, "completion_tokens": 5, "wall_seconds": 0.375, "purpose": "memory:summary"}})
                # Tokenizer operations must not be counted as generation calls/tokens.
                calls.append({"method": "tokenize", "result": [999, 998, 997]})
                traces.append({
                    "episode_id": episode_id, "policy": policy, "step_index": step,
                    "public_step": copy.deepcopy(public_step),
                    "memory_before": f"FAKE-MEMORY-BEFORE-{episode_index}-{policy}-{step}",
                    "actor_memory_before": f"FAKE-ACTOR-VIEW-{episode_index}-{policy}-{step}",
                    "memory_after": f"FAKE-MEMORY-AFTER-{episode_index}-{policy}-{step}",
                    "evidence_check": check,
                    "action": replay.extract_action(calls[0]["result"]["text"], "stop"),
                    "tool_result": {"tool": "record_decision", "accepted": True, "amount_booked_cents": 0},
                    "score": {"success": success, "failure_category": "correct" if success else "unnecessary_deferral"},
                    "client_events": calls,
                    "step_seconds": (2.125 if policy == "summary" else 1.375) + step / 10,
                    "memory_tokens": 43 if policy == "summary" else 37,
                })
    # Trace order must not override the explicit public-manifest index.
    return public, list(reversed(traces))


class OnlineV3ReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path("explicit-fake-v3-replay-fixture")
        self.public, self.traces = _fake_artifacts()
        self.verification_finished = False
        self.output = io.StringIO()

        def verify(directory):
            self.assertEqual(directory, self.directory)
            self.assertEqual(self.output.getvalue(), "", "Output occurred before verification")
            self.verification_finished = True
            return {"status": "PASS", "scope": "patched fake contract receipt; no inference evidence"}

        def read(path):
            self.assertTrue(self.verification_finished, "Artifacts read before verification")
            self.assertEqual(path, self.directory / "public.json")
            return copy.deepcopy(self.public)

        def read_jsonl(path):
            self.assertTrue(self.verification_finished, "Traces read before verification")
            self.assertEqual(path, self.directory / "traces.jsonl")
            return copy.deepcopy(self.traces)

        patches = {
            "verify": patch.object(replay, "verify_run", side_effect=verify),
            "read": patch.object(replay, "read", side_effect=read),
            "read_jsonl": patch.object(replay, "read_jsonl", side_effect=read_jsonl),
            "network": patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Replay attempted network access")),
            "model": patch("rcwt_local_model.LocalModelClient.__init__", side_effect=AssertionError("Replay instantiated a model client")),
        }
        self.mocks = {}
        for name, patcher in patches.items():
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def capture(self, *args, **kwargs):
        self.output = io.StringIO()
        self.verification_finished = False
        with redirect_stdout(self.output):
            replay.replay(self.directory, *args, **kwargs)
        return self.output.getvalue()

    def test_default_is_first_manifest_episode_after_verification_not_best_trace(self):
        text = self.capture()
        self.mocks["verify"].assert_called_once_with(self.directory)
        self.assertTrue(text.startswith("RECORDED PAIRED REPLAY / NO MODEL CALLS\n"))
        self.assertIn("episode index 0; FAKE-FIRST-EPISODE", text)
        self.assertNotIn("FAKE-BEST-EPISODE", text)
        self.assertEqual(text.count("STEP "), 8)
        self.assertEqual(text.count("\nSUMMARY\n"), 8)
        self.assertEqual(text.count("\nSTRUCTURED\n"), 8)
        self.assertIn("first manifest episode, not a best-case example", text.lower())
        self.assertIn("A single episode is not proof of general gain", text)
        self.mocks["network"].assert_not_called()
        self.mocks["model"].assert_not_called()

    def test_displays_recorded_failures_actions_receipts_and_exact_step_resources(self):
        text = self.capture()
        self.assertEqual(text.count("PUBLIC OBSERVATIONS AND REQUEST:"), 8)
        self.assertEqual(text.count("ACTOR SELF-REPORTED CHECK:"), 16)
        self.assertEqual(text.count("ACTUAL ACTION (FINAL):"), 16)
        self.assertEqual(text.count("ACTUAL FICTIONAL TOOL RECEIPT:"), 16)
        self.assertEqual(text.count("PRIVATE EVALUATION:"), 16)
        self.assertIn('"failure_category": "unnecessary_deferral"', text)
        self.assertIn('"success": false', text)
        self.assertIn("never provided to the actor or its memory reducer", text)
        self.assertIn("Fictional tools and money", text)
        self.assertIn("RECORDED: 2.125s full step; 135 model tokens; 2 generation calls; 43/256 memory tokens", text)
        self.assertIn("RECORDED: 1.375s full step; 110 model tokens; 1 generation calls; 37/256 memory tokens", text)
        self.assertIn("RECORDED: 2.825s full step; 117 model tokens; 1 generation calls; 43/256 memory tokens", text)

    def test_memory_is_hidden_by_default_and_present_only_when_requested(self):
        default = self.capture()
        explicit_false = self.capture(show_memory=False)
        self.assertEqual(default, explicit_false)
        self.assertNotIn("STORED MEMORY BEFORE:", default)
        self.assertNotIn("MEMORY VIEW SENT TO ACTOR:", default)
        self.assertNotIn("RETAINED MEMORY AFTER:", default)
        self.assertNotIn("FAKE-MEMORY-", default)
        self.assertNotIn("FAKE-ACTOR-VIEW-", default)
        enabled = self.capture(show_memory=True)
        self.assertEqual(enabled.count("STORED MEMORY BEFORE:"), 16)
        self.assertEqual(enabled.count("MEMORY VIEW SENT TO ACTOR:"), 16)
        self.assertEqual(enabled.count("RETAINED MEMORY AFTER:"), 16)
        self.assertIn("FAKE-MEMORY-BEFORE-0-summary-0", enabled)
        self.assertIn("FAKE-MEMORY-AFTER-0-structured-7", enabled)
        self.assertIn("FAKE-ACTOR-VIEW-0-structured-7", enabled)

    def test_legacy_missing_actor_view_falls_back_to_stored_memory(self):
        for row in self.traces:
            del row["actor_memory_before"]
        text = self.capture(show_memory=True)
        self.assertIn("MEMORY VIEW SENT TO ACTOR: FAKE-MEMORY-BEFORE-0-structured-0", text)

    def test_draft_is_displayed_as_unexecuted_and_final_call_is_selected_by_purpose(self):
        for row in self.traces:
            final = next(event["result"] for event in row["client_events"] if event["method"] == "complete")
            envelope = json.loads(final["text"])
            envelope["arguments"].update(case_id="FAKE-DRAFT-NOT-EXECUTED", decision="approve", amount_cents=999, reason_code="authorized_payout")
            draft = {**final, "purpose": f"draft:{row['policy']}", "text": json.dumps(envelope),
                     "request": {"response_format": {"type": "json_schema"}},
                     "prompt_tokens": 50, "completion_tokens": 5, "wall_seconds": 0.25}
            row["client_events"].insert(0, {"method": "complete", "result": draft})
            row["step_seconds"] += 0.25
        text = self.capture()
        self.assertEqual(text.count("DRAFT OUTPUT (NOT EXECUTED):"), 16)
        self.assertEqual(text.count("DRAFT PROPOSED ACTION (NOT EXECUTED):"), 16)
        final_lines = [line for line in text.splitlines() if line.startswith("ACTUAL ACTION (FINAL):")]
        self.assertEqual(len(final_lines), 16)
        self.assertTrue(all("FAKE-DRAFT-NOT-EXECUTED" not in line for line in final_lines))
        self.assertIn("RECORDED: 2.375s full step; 190 model tokens; 3 generation calls", text)

    def test_valid_draft_is_not_used_as_fallback_for_invalid_final(self):
        row = next(row for row in self.traces if row["episode_id"] == "FAKE-FIRST-EPISODE" and row["policy"] == "structured" and row["step_index"] == 0)
        final = next(event["result"] for event in row["client_events"] if event["method"] == "complete")
        row["client_events"].insert(0, {"method": "complete", "result": {
            **copy.deepcopy(final), "purpose": "draft:structured", "request": {"response_format": {"type": "json_schema"}}}})
        final["text"] = "{malformed final"
        row["action"] = replay.extract_action(final["text"], final["finish_reason"])
        row["evidence_check"] = None
        text = self.capture()
        self.assertIn('ACTUAL ACTION (FINAL): "INVALID_ACTOR_OUTPUT"', text)
        self.assertIn("DRAFT PROPOSED ACTION (NOT EXECUTED):", text)

    def test_schema_free_plan_is_displayed_without_tool_parsing_or_final_repair(self):
        original = copy.deepcopy(self.traces)
        for plan_text, finish in (("FAKE plaintext: approve an unsupported amount.", "stop"), ("", "stop"),
                                  ("FAKE unfinished plan", "length"), ("{FAKE not JSON}", "stop")):
            self.traces = copy.deepcopy(original)
            for row in self.traces:
                final = next(event["result"] for event in row["client_events"] if event["method"] == "complete")
                plan = {**copy.deepcopy(final), "purpose": f"draft:{row['policy']}", "text": plan_text,
                        "finish_reason": finish, "request": {}, "prompt_tokens": 50,
                        "completion_tokens": 5, "wall_seconds": 0.25}
                row["client_events"].insert(0, {"method": "complete", "result": plan})
                row["step_seconds"] += 0.25
            with self.subTest(plan_text=plan_text, finish=finish), patch.object(replay, "extract_action", wraps=replay.extract_action) as actions:
                text = self.capture()
                self.assertEqual(actions.call_count, 16, "Only the selected episode's final outputs may be parsed")
            self.assertEqual(text.count("PLAN OUTPUT (NOT EXECUTED; NO TOOL PARSING):"), 16)
            self.assertEqual(text.count(f"PLAN FINISH REASON (NOT A FINAL-ACTION GRADE): {finish}"), 16)
            self.assertNotIn("DRAFT PROPOSED ACTION", text)
            self.assertNotIn("INVALID_ACTOR_OUTPUT", text)
            self.assertNotIn("TRUNCATED_ACTION", text)
            self.assertIn('"failure_category": "unnecessary_deferral"', text)
            self.assertIn("RECORDED: 2.375s full step; 190 model tokens; 3 generation calls", text)

    def test_non_executable_plan_does_not_replace_an_invalid_final_action(self):
        row = next(row for row in self.traces if row["episode_id"] == "FAKE-FIRST-EPISODE" and row["policy"] == "structured" and row["step_index"] == 0)
        final = next(event["result"] for event in row["client_events"] if event["method"] == "complete")
        # Even a complete JSON-shaped plan is not an executable fallback without a schema.
        row["client_events"].insert(0, {"method": "complete", "result": {
            **copy.deepcopy(final), "purpose": "draft:structured", "request": {}}})
        final["text"] = "{malformed final"
        row["action"] = replay.extract_action(final["text"], final["finish_reason"])
        row["evidence_check"] = None
        text = self.capture()
        self.assertIn('ACTUAL ACTION (FINAL): "INVALID_ACTOR_OUTPUT"', text)
        self.assertIn("PLAN OUTPUT (NOT EXECUTED; NO TOOL PARSING):", text)
        self.assertNotIn("DRAFT PROPOSED ACTION", text)

    def test_missing_duplicate_or_mismatched_final_fails_before_output(self):
        original = copy.deepcopy(self.traces)
        for change in ("missing", "duplicate", "mismatched_action"):
            self.traces = copy.deepcopy(original)
            row = next(row for row in self.traces if row["episode_id"] == "FAKE-FIRST-EPISODE")
            final_event = next(event for event in row["client_events"] if event["method"] == "complete")
            if change == "missing":
                final_event["result"]["purpose"] = f"draft:{row['policy']}"
            elif change == "duplicate":
                row["client_events"].append(copy.deepcopy(final_event))
            else:
                row["action"] = "FAKE UNBOUND ACTION"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.capture()
            self.assertEqual(self.output.getvalue(), "")

    def test_repeated_render_is_deterministic_and_explicit_index_is_honored(self):
        self.assertEqual(self.capture(show_memory=True), self.capture(show_memory=True))
        text = self.capture(index=1)
        self.assertIn("episode index 1; FAKE-BEST-EPISODE", text)
        self.assertNotIn("FAKE-FIRST-EPISODE", text)

    def test_invalid_indices_fail_without_display_or_silent_substitution(self):
        for index in (-1, 2, True, False, 0.0, "0", None):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "index must exist"):
                self.capture(index=index)
            self.assertEqual(self.output.getvalue(), "")

    def test_verifier_exception_prevents_output_and_artifact_reads(self):
        self.mocks["verify"].side_effect = ValueError("Fake altered evidence")
        with self.assertRaisesRegex(ValueError, "Fake altered evidence"):
            self.capture()
        self.assertEqual(self.output.getvalue(), "")
        self.mocks["read"].assert_not_called()
        self.mocks["read_jsonl"].assert_not_called()

    def test_non_pass_receipt_prevents_output_and_artifact_reads(self):
        for receipt in ({"status": "FAIL"}, {"status": "WARN"}, {"status": "pass"}, {}):
            self.mocks["verify"].side_effect = None
            self.mocks["verify"].return_value = receipt
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                self.capture()
            self.assertEqual(self.output.getvalue(), "")
        self.mocks["read"].assert_not_called()
        self.mocks["read_jsonl"].assert_not_called()

    def test_cli_defaults_and_explicit_memory_flag(self):
        with patch.object(replay, "replay") as render, patch("sys.argv", ["replay_online_v3.py"]):
            replay.main()
        render.assert_called_once_with(Path("results/agent_v3"), 0, False)
        with patch.object(replay, "replay") as render, patch("sys.argv", [
            "replay_online_v3.py", "--run-dir", "explicit-fake-cli", "--episode-index", "1", "--show-memory"
        ]):
            replay.main()
        render.assert_called_once_with(Path("explicit-fake-cli"), 1, True)


if __name__ == "__main__":
    unittest.main()
