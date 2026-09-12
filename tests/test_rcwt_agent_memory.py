"""Offline contract tests; fake clients do not constitute agent task evidence."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from rcwt_agent_memory import (
    CANDIDATE_INSTRUCTIONS,
    SUMMARY_INSTRUCTION,
    MemoryState,
    compact_memory,
    propose_policy,
)


@dataclass
class FakeResult:
    text: str
    prompt_tokens: int = 100
    completion_tokens: int = 10
    wall_seconds: float = 0.5


class FakeClient:
    """Character tokenizer makes every truncation boundary inspectable."""

    def __init__(self, outputs: list[str] | None = None):
        self.outputs = list(outputs or [])
        self.requests: list[dict] = []
        self.results: list[FakeResult] = []

    def tokenize(self, text):
        return [ord(char) for char in text]

    def detokenize(self, ids):
        return "".join(chr(token) for token in ids)

    def complete(self, messages, max_tokens, schema=None, purpose=""):
        self.requests.append(
            {"messages": messages, "max_tokens": max_tokens, "schema": schema, "purpose": purpose}
        )
        result = FakeResult(self.outputs.pop(0))
        self.results.append(result)
        return result


class MemoryPolicyTests(unittest.TestCase):
    def test_state_call_lists_are_not_shared(self):
        one, two = MemoryState(), MemoryState()
        one.calls.append("call")
        self.assertEqual(two.calls, [])

    def test_tail_retains_only_last_tokens_without_completion(self):
        client = FakeClient()
        state = compact_memory(client, "tail", "discarded", "new-data", 5)
        self.assertEqual(state.text, "-data")
        self.assertTrue(state.truncated)
        self.assertEqual(state.calls, [])
        self.assertEqual(client.requests, [])

    def test_tail_at_cap_does_not_report_truncation(self):
        client = FakeClient()
        state = compact_memory(client, "tail", "old", "new", 7)
        self.assertEqual(state.text, "old\nnew")
        self.assertFalse(state.truncated)

    def test_summary_reuses_only_retained_memory_and_current_input(self):
        client = FakeClient(["retained", "updated"])
        first = compact_memory(client, "summary", "", "previous observation", 40)
        second = compact_memory(client, "summary", first.text, "current observation", 40)
        request = client.requests[1]
        public_input = json.loads(request["messages"][1]["content"])
        self.assertEqual(
            public_input,
            {"previous_memory": "retained", "new_information": "current observation"},
        )
        self.assertNotIn("previous observation", request["messages"][1]["content"])
        self.assertIn(SUMMARY_INSTRUCTION, request["messages"][0]["content"])
        self.assertEqual(request["purpose"], "memory:summary")
        self.assertEqual(request["max_tokens"], 104)
        self.assertIsNone(request["schema"])
        self.assertIs(second.calls[0], client.results[1])

    def test_summary_caps_actual_output_and_marks_truncation(self):
        client = FakeClient(["abcdefghijk"])
        state = compact_memory(client, "summary", "", "input", 5)
        self.assertEqual(state.text, "abcde")
        self.assertTrue(state.truncated)
        self.assertEqual(len(client.tokenize(state.text)), 5)
        self.assertEqual(len(state.calls), 1)

    def test_cap_rechecks_after_decode_reencode_expansion(self):
        class ExpandingClient(FakeClient):
            def detokenize(self, ids):
                return super().detokenize(ids) + "!"

        client = ExpandingClient()
        state = compact_memory(client, "tail", "", "abcdef", 3)
        self.assertEqual(state.text, "ef!")
        self.assertLessEqual(len(client.tokenize(state.text)), 3)

    def test_unicode_is_counted_by_client_tokenizer(self):
        client = FakeClient(["ação 🙂 fim"])
        state = compact_memory(client, "summary", "", "olá", 6)
        self.assertEqual(state.text, "ação 🙂")
        self.assertEqual(len(client.tokenize(state.text)), 6)

    def test_learned_requires_frozen_instruction(self):
        client = FakeClient(["memory"])
        with self.assertRaises(ValueError):
            compact_memory(client, "learned", "old", "new", 20)
        state = compact_memory(client, "learned", "old", "new", 20, "Keep a ledger.")
        self.assertIn("Keep a ledger.", client.requests[0]["messages"][0]["content"])
        self.assertNotIn(SUMMARY_INSTRUCTION, client.requests[0]["messages"][0]["content"])
        self.assertEqual(client.requests[0]["purpose"], "memory:learned")
        self.assertEqual(state.text, "memory")

    def test_summary_cannot_silently_accept_learned_instruction(self):
        with self.assertRaises(ValueError):
            compact_memory(FakeClient(), "summary", "", "", 20, "different")

    def test_model_failure_and_empty_output_are_not_replaced(self):
        class BrokenClient(FakeClient):
            def complete(self, *args, **kwargs):
                raise RuntimeError("local inference failed")

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            compact_memory(BrokenClient(), "summary", "old", "new", 20)
        with self.assertRaisesRegex(ValueError, "no usable text"):
            compact_memory(FakeClient([" "]), "summary", "old", "new", 20)

    def test_policy_and_budget_validation(self):
        for budget in (0, -1, True, 3.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                compact_memory(FakeClient(), "tail", "", "", budget)
        with self.assertRaises(ValueError):
            compact_memory(FakeClient(), "unknown", "", "", 10)
        with self.assertRaises(TypeError):
            compact_memory(FakeClient(), "tail", None, "", 10)

    def test_candidates_are_distinct_nonempty_named_instructions(self):
        self.assertEqual(len(CANDIDATE_INSTRUCTIONS), 3)
        self.assertEqual(len(set(CANDIDATE_INSTRUCTIONS.values())), 3)
        self.assertTrue(all(name and instruction.strip() for name, instruction in CANDIDATE_INSTRUCTIONS.items()))


class PolicyProposalTests(unittest.TestCase):
    def test_train_only_proposal_and_call_accounting(self):
        client = FakeClient(["Track current evidence in a case ledger."])
        instruction, call = propose_policy(
            client, [{"split": "train", "failure_class": "stale_evidence", "count": 3}]
        )
        self.assertEqual(instruction, call.text)
        self.assertIs(call, client.results[0])
        self.assertEqual(client.requests[0]["purpose"], "policy:train-proposal")
        records = json.loads(client.requests[0]["messages"][1]["content"])
        self.assertEqual(records[0]["count"], 3)

    def test_heldout_or_unlabelled_records_block_before_call(self):
        for split in (None, "validation", "test"):
            client = FakeClient()
            with self.subTest(split=split), self.assertRaises(ValueError):
                propose_policy(client, [{"split": split, "failure_class": "missing_context"}])
            self.assertEqual(client.requests, [])

    def test_nested_oracle_and_future_fields_are_not_forwarded(self):
        client = FakeClient(["Retain current evidence."])
        propose_policy(
            client,
            [{
                "split": "train",
                "failure_class": "missing_evidence",
                "oracle": "PRIVATE-ORACLE",
                "future_tasks": "PRIVATE-FUTURE",
                "example": {
                    "observations": "Observed case C1.",
                    "memory": "Retained C1.",
                    "expected_action": "PRIVATE-ANSWER",
                    "action": {
                        "tool": "record_decision",
                        "arguments": {"case_id": "C1", "decision": "hold", "oracle": "PRIVATE-NESTED"},
                        "oracle": "PRIVATE-ACTION",
                    },
                    "feedback": "Required evidence was lost.",
                },
            }],
        )
        content = client.requests[0]["messages"][1]["content"]
        self.assertIn("Observed case C1.", content)
        self.assertIn("Required evidence was lost.", content)
        self.assertNotIn("PRIVATE-", content)

    def test_no_failure_no_fake_improvement(self):
        with self.assertRaises(ValueError):
            propose_policy(FakeClient(), [])
        with self.assertRaises(ValueError):
            propose_policy(FakeClient([""]), [{"split": "train", "failure_class": "stale"}])

    def test_all_records_checked_before_prompt_bound(self):
        records = [{"split": "train", "failure_class": "stale"} for _ in range(65)]
        records.append({"split": "test", "failure_class": "missing"})
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "split='train'"):
            propose_policy(client, records)
        self.assertEqual(client.requests, [])


if __name__ == "__main__":
    unittest.main()
