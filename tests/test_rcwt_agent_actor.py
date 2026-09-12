"""Actor-envelope validation must never become an oracle or automatic repair."""

from __future__ import annotations

import copy
import json
import unittest

from rcwt_agent_actor import (
    ACTOR_INSTRUCTION, ACTOR_SCHEMA, INVALID_ACTOR_OUTPUT, TRUNCATED_ACTION,
    extract_action, extract_evidence_check,
)
from rcwt_agent_env import ACTION_SCHEMA, parse_action


def envelope() -> dict:
    return {
        "evidence_check": {
            "invoice_amount_cents": 1200, "account_status": "active",
            "ownership_match": "yes", "payment_status": "cleared",
            "return_status": "unknown", "operation_already_booked": "no_record",
        },
        "tool": "record_decision",
        "arguments": {"case_id": "inv-fictional", "decision": "approve",
                      "amount_cents": 1200, "reason_code": "authorized_payout"},
    }


class AgentActorTests(unittest.TestCase):
    def test_schema_preserves_original_tool_and_argument_contract(self):
        self.assertEqual(list(ACTOR_SCHEMA["properties"]), ["evidence_check", "tool", "arguments"])
        self.assertEqual(ACTOR_SCHEMA["properties"]["tool"], ACTION_SCHEMA["properties"]["tool"])
        self.assertEqual(ACTOR_SCHEMA["properties"]["arguments"], ACTION_SCHEMA["properties"]["arguments"])
        self.assertIsNot(ACTOR_SCHEMA["properties"]["arguments"], ACTION_SCHEMA["properties"]["arguments"])
        self.assertIn("Current observations override", ACTOR_INSTRUCTION)
        self.assertIn("not proof of correct evidence", ACTOR_INSTRUCTION)
        self.assertIn("Subject to higher-priority rules", ACTOR_INSTRUCTION)

    def test_valid_envelope_extracts_canonical_unchanged_action(self):
        value = envelope()
        actual = extract_action(json.dumps(value), "stop")
        expected = json.dumps({"tool": value["tool"], "arguments": value["arguments"]},
                              sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(actual, expected)
        self.assertEqual(parse_action(actual).amount_cents, 1200)
        self.assertNotIn("evidence_check", json.loads(actual))
        self.assertEqual(extract_evidence_check(json.dumps(value), "stop"), value["evidence_check"])

    def test_missing_evidence_does_not_silently_repair_unsafe_action(self):
        value = envelope()
        value["evidence_check"].update(invoice_amount_cents=None, payment_status="unknown",
                                       account_status="unknown", ownership_match="unknown")
        # Structurally valid self-report; approving is potentially wrong, but
        # extraction must leave that decision available to independent scoring.
        action = parse_action(extract_action(json.dumps(value), "stop"))
        self.assertEqual(action.decision, "approve")
        self.assertEqual(action.amount_cents, 1200)
        self.assertIsNone(extract_evidence_check(json.dumps(value), "stop")["invoice_amount_cents"])

    def test_nonstop_valid_prefix_is_never_executable(self):
        for reason in ("length", "unknown", "content_filter", "", None):
            result = extract_action(json.dumps(envelope()), reason)
            self.assertEqual(result, TRUNCATED_ACTION)
            self.assertIsNone(extract_evidence_check(json.dumps(envelope()), reason))
            with self.assertRaises(ValueError):
                parse_action(result)

    def test_malformed_or_nonjson_output_is_not_repaired(self):
        for text in ("", "{", "[]", "null", "true", "{}", None,
                     "```json\n" + json.dumps(envelope()) + "\n```",
                     json.dumps(envelope()) + " trailing text"):
            self.assertEqual(extract_action(text, "stop"), INVALID_ACTOR_OUTPUT)
            self.assertIsNone(extract_evidence_check(text, "stop"))

    def test_missing_extra_and_reordered_envelope_fields_are_invalid(self):
        for field in ("evidence_check", "tool", "arguments"):
            value = envelope(); del value[field]
            self.assertEqual(extract_action(json.dumps(value), "stop"), INVALID_ACTOR_OUTPUT)
        value = envelope(); value["explanation"] = "approve"
        self.assertEqual(extract_action(json.dumps(value), "stop"), INVALID_ACTOR_OUTPUT)
        value = envelope()
        reordered = {"tool": value["tool"], "arguments": value["arguments"],
                     "evidence_check": value["evidence_check"]}
        self.assertEqual(extract_action(json.dumps(reordered), "stop"), INVALID_ACTOR_OUTPUT)

    def test_evidence_check_requires_exact_fields_and_types(self):
        original = envelope()
        for field in original["evidence_check"]:
            value = copy.deepcopy(original); del value["evidence_check"][field]
            self.assertEqual(extract_action(json.dumps(value), "stop"), INVALID_ACTOR_OUTPUT)
        for invalid in (True, False, -1, "1200", 1.5, {}, []):
            value = envelope(); value["evidence_check"]["invoice_amount_cents"] = invalid
            self.assertEqual(extract_action(json.dumps(value), "stop"), INVALID_ACTOR_OUTPUT)
        for field in ("account_status", "ownership_match", "payment_status", "return_status", "operation_already_booked"):
            for invalid in (True, None, 1, [], {}, "unexpected"):
                value = envelope(); value["evidence_check"][field] = invalid
                self.assertEqual(extract_action(json.dumps(value), "stop"), INVALID_ACTOR_OUTPUT)
        value = envelope(); value["evidence_check"]["oracle"] = "hidden"
        self.assertEqual(extract_action(json.dumps(value), "stop"), INVALID_ACTOR_OUTPUT)

    def test_action_arguments_remain_strictly_validated(self):
        for invalid in (True, -1, 0, "1200", 1.5):
            value = envelope(); value["arguments"]["amount_cents"] = invalid
            self.assertEqual(extract_action(json.dumps(value), "stop"), INVALID_ACTOR_OUTPUT)
        value = envelope(); value["arguments"]["extra"] = "field"
        self.assertEqual(extract_action(json.dumps(value), "stop"), INVALID_ACTOR_OUTPUT)
        value = envelope(); value["tool"] = "send_money"
        self.assertEqual(extract_action(json.dumps(value), "stop"), INVALID_ACTOR_OUTPUT)

    def test_duplicate_keys_and_nonfinite_numbers_are_invalid(self):
        text = json.dumps(envelope())
        duplicate = text.replace('"account_status": "active"',
                                 '"account_status": "unknown", "account_status": "active"')
        self.assertEqual(extract_action(duplicate, "stop"), INVALID_ACTOR_OUTPUT)
        duplicate = text.replace('"tool": "record_decision"',
                                 '"tool": "send_money", "tool": "record_decision"')
        self.assertEqual(extract_action(duplicate, "stop"), INVALID_ACTOR_OUTPUT)
        for nonfinite in ("NaN", "Infinity", "-Infinity"):
            changed = text.replace('"invoice_amount_cents": 1200', '"invoice_amount_cents": ' + nonfinite)
            self.assertEqual(extract_action(changed, "stop"), INVALID_ACTOR_OUTPUT)


if __name__ == "__main__":
    unittest.main()
