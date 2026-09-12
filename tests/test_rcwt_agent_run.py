"""Fast structural tests; simulated clients are not reported as LLM results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rcwt_agent_env import generate_episodes
from rcwt_agent_run import Policy, actor_messages, read_jsonl, run_episode, select_policy, training_failure_view
from rcwt_local_model import CallResult


class FakeClient:
    def __init__(self):
        self.requests = []

    def tokenize(self, text):
        return list(text.encode("utf-8"))

    def detokenize(self, tokens):
        return bytes(tokens).decode("utf-8", errors="replace")

    def complete(self, messages, max_tokens, schema=None, purpose=""):
        request = {"messages": messages}
        self.requests.append(request)
        if purpose.startswith("action:"):
            step = json.loads(messages[1]["content"])["current_step"]
            text = json.dumps({"evidence_check": {
                "invoice_amount_cents": None, "account_status": "unknown", "ownership_match": "unknown",
                "payment_status": "unknown", "return_status": "unknown", "operation_already_booked": "no_record"
            }, "tool": "record_decision", "arguments": {
                "case_id": step["task"]["case_id"], "decision": "ask_info",
                "amount_cents": 0, "reason_code": "missing_evidence"}})
        else:
            text = "retained-memory-only"
        return CallResult(text, 10, 2, 0, "test-local", purpose, "stop", {}, request, "test")


class RunnerTests(unittest.TestCase):
    def test_cumulative_memory_no_oracle_feedback(self):
        client = FakeClient()
        episode = generate_episodes("train", 1, 19)[0]
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            path = Path(directory) / "traces.jsonl"
            summary = run_episode(client, episode, Policy("summary", "summary"), 256, path)
            rows = read_jsonl(path)
        self.assertEqual(len(rows), 8)
        self.assertEqual(summary["model_calls"], 15)
        self.assertEqual(summary["prompt_tokens"], 150)
        for index, row in enumerate(rows):
            messages = row["model_calls"][0]["request"]["messages"]
            self.assertEqual(messages, actor_messages(row["memory_before"], episode.public_step(index)))
            if index:
                self.assertEqual(row["memory_before"], rows[index - 1]["memory_after"])
            for call in row["model_calls"]:
                payload = json.loads(call["request"]["messages"][1]["content"])
                self.assertNotIn("oracle", payload)
                self.assertNotIn("score", payload)
                self.assertNotIn("expected_action", json.dumps(payload))
                if call["purpose"].startswith("memory:"):
                    info = json.loads(payload["new_information"])
                    self.assertEqual(set(info), {"observations", "completed_request", "action", "tool_result"})
        self.assertEqual(sum(summary["failures"].values()) + summary["successes"], 8)

    def test_tail_never_calls_compressor(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            summary = run_episode(client, generate_episodes("train", 1, 19)[0], Policy("tail", "tail"),
                                  80, Path(directory) / "traces.jsonl")
        self.assertEqual(summary["model_calls"], 8)
        self.assertTrue(all(len(p["messages"]) == 2 for p in client.requests))

    def test_failure_view_refuses_heldout_and_excludes_answers(self):
        row = {"split": "train", "family": "a", "score": {
            "success": False, "failure_category": "wrong_amount", "expected_action": "SECRET"}}
        view = training_failure_view([row])
        self.assertNotIn("SECRET", json.dumps(view))
        row["split"] = "test"
        with self.assertRaises(ValueError):
            training_failure_view([row])

    def test_validation_selection_is_paired_and_predefined(self):
        policies = [Policy("summary", "summary"), Policy("repair", "learned", "facts")]
        rows = [{"split": "validation", "episode_id": "v1", "policy": p.name,
                 "steps": 8, "successes": 6, "unsafe_actions": i, "completion_tokens": 20}
                for i, p in enumerate(policies)]
        winner, _ = select_policy(rows, policies)
        self.assertEqual(winner.name, "summary")
        rows[1]["episode_id"] = "different"
        with self.assertRaises(ValueError):
            select_policy(rows, policies)
        rows[1]["split"] = "test"
        with self.assertRaises(ValueError):
            select_policy(rows, policies)


if __name__ == "__main__":
    unittest.main()
