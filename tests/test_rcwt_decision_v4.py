"""Invented schema contracts; no generated corpus, model or performance proof."""

from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import patch

from rcwt_agent_actor import ACTOR_INSTRUCTION, ACTOR_SCHEMA, extract_action, extract_evidence_check
from rcwt_decision_v4 import ORDERED_RULE_GUIDE, ordered_messages, task_schema
from rcwt_review_v3 import ORIGINAL_JSON_REQUIREMENT, PLANNING_INSTRUCTION, planning_messages, review_messages


class _OnlyNamedKeys(dict):
    """Fail if production code inspects anything beyond the declared task keys."""
    def __init__(self, values, allowed):
        super().__init__(values)
        self.allowed = set(allowed)

    def get(self, key, default=None):
        if key not in self.allowed:
            raise AssertionError("Attempted evidence read: " + str(key))
        return super().get(key, default)

    def __getitem__(self, key):
        if key not in self.allowed:
            raise AssertionError("Attempted evidence read: " + str(key))
        return super().__getitem__(key)

    def __iter__(self):
        raise AssertionError("Attempted task/evidence enumeration")

    def items(self):
        raise AssertionError("Attempted task/evidence enumeration")


class DecisionV4Tests(unittest.TestCase):
    def test_literal_task_identity_and_exact_operation_vocabulary(self):
        reasons = ACTOR_SCHEMA["properties"]["arguments"]["properties"]["reason_code"]["enum"]
        for operation, monetary, excluded in (
            ("payout", "approve", {"eligible_refund", "return_rejected"}),
            ("refund", "refund", {"authorized_payout"}),
        ):
            identifier = 'FAKE case \"literal\" / á-42'
            result = task_schema({"task": {"case_id": identifier, "operation": operation}})
            arguments = result["properties"]["arguments"]["properties"]
            self.assertEqual(arguments["case_id"], {"type": "string", "minLength": 1, "const": identifier})
            self.assertEqual(arguments["decision"]["enum"], ["hold", "ask_info", monetary])
            self.assertEqual(arguments["reason_code"]["enum"], [reason for reason in reasons if reason not in excluded])
            self.assertIn("already_completed", arguments["reason_code"]["enum"])
            self.assertIn("missing_evidence", arguments["reason_code"]["enum"])
            json.dumps(result, allow_nan=False)

    def test_only_three_schema_leaves_change_and_each_call_is_independent(self):
        original = copy.deepcopy(ACTOR_SCHEMA)
        first = task_schema({"task": {"case_id": "FAKE-A", "operation": "payout"}})
        second = task_schema({"task": {"case_id": "FAKE-B", "operation": "refund"}})
        restored = copy.deepcopy(first)
        args = restored["properties"]["arguments"]["properties"]
        original_args = original["properties"]["arguments"]["properties"]
        for name in ("case_id", "decision", "reason_code"):
            args[name] = copy.deepcopy(original_args[name])
        self.assertEqual(restored, original)
        first["properties"]["evidence_check"]["required"].clear()
        first["properties"]["arguments"]["properties"]["amount_cents"]["minimum"] = 999999
        self.assertEqual(ACTOR_SCHEMA, original)
        self.assertEqual(second["properties"]["evidence_check"], original["properties"]["evidence_check"])
        self.assertEqual(second["properties"]["arguments"]["properties"]["amount_cents"], original_args["amount_cents"])

    def test_neither_evidence_nor_policy_can_affect_the_schema(self):
        task = _OnlyNamedKeys({"case_id": "FAKE", "operation": "payout", "instruction": object()}, {"case_id", "operation"})
        public_step = _OnlyNamedKeys({"task": task, "observations": object(), "memory": object(),
                                     "evidence_check": object(), "oracle": object(), "policy": object()}, {"task"})
        expected = task_schema({"task": {"case_id": "FAKE", "operation": "payout"}})
        self.assertEqual(task_schema(public_step), expected)
        for policy in ("summary", "structured"):
            supplied = {"task": {"case_id": "FAKE", "operation": "payout"}, "policy": policy,
                        "observations": [{"account_status": "revoked", "amount_cents": 1}],
                        "memory": "untrusted text", "evidence_check": {"payment_status": "cleared"}}
            before = copy.deepcopy(supplied)
            self.assertEqual(task_schema(supplied), expected)
            self.assertEqual(supplied, before)

    def test_semantically_unsafe_monetary_actions_and_wrong_amounts_remain_allowed(self):
        for operation, decision, reason in (("payout", "approve", "authorized_payout"),
                                             ("refund", "refund", "eligible_refund")):
            schema = task_schema({"task": {"case_id": "FAKE", "operation": operation}})
            args = schema["properties"]["arguments"]["properties"]
            check = {"invoice_amount_cents": 100, "account_status": "revoked", "ownership_match": "no",
                     "payment_status": "unknown", "return_status": "unknown", "operation_already_booked": "yes"}
            # Each value satisfies its independent schema leaf. No conditional
            # schema ties a monetary action to these clearly adverse facts.
            self.assertIn(decision, args["decision"]["enum"])
            self.assertIn(reason, args["reason_code"]["enum"])
            self.assertEqual(args["amount_cents"], {"type": "integer", "minimum": 0})
            self.assertEqual(schema["properties"]["evidence_check"], ACTOR_SCHEMA["properties"]["evidence_check"])
            envelope = {"evidence_check": check, "tool": "record_decision", "arguments": {
                "case_id": "FAKE", "decision": decision, "amount_cents": 999999, "reason_code": reason}}
            actual = json.loads(extract_action(json.dumps(envelope), "stop"))
            self.assertEqual(actual["arguments"], envelope["arguments"], "No veto, amount computation, or repair")
            self.assertEqual(extract_evidence_check(json.dumps(envelope), "stop"), check)

    def test_invalid_task_shapes_fail_without_coercion_or_identity_invention(self):
        for supplied in (None, [], "task", {}, {"task": None}, {"task": []}, {"task": {}},
                         {"task": {"case_id": "FAKE"}}, {"task": {"operation": "payout"}}):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                task_schema(supplied)
        for case_id in (None, "", True, 123, [], {}):
            with self.subTest(case_id=case_id), self.assertRaises(ValueError):
                task_schema({"task": {"case_id": case_id, "operation": "payout"}})
        for operation in (None, "", "PAYOUT", "approve", True, 1, [], {}):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                task_schema({"task": {"case_id": "FAKE", "operation": operation}})


class OrderedMessagesTests(unittest.TestCase):
    def messages(self, payload="FAKE opaque user text, not necessarily JSON"):
        return [{"role": "system", "content": ACTOR_INSTRUCTION, "metadata": {"nested": ["system"]}},
                {"role": "user", "content": payload, "metadata": {"nested": ["user"]}}]

    def test_only_system_suffix_changes_and_original_byte_text_is_preserved(self):
        original = self.messages('  FAKE á\n{"broken":\tYES\r\n')
        before = copy.deepcopy(original)
        result = ordered_messages(original)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["content"], before[0]["content"] + "\n\n" + ORDERED_RULE_GUIDE)
        self.assertEqual(result[1], before[1])
        self.assertEqual(result[0]["metadata"], before[0]["metadata"])
        self.assertEqual(original, before)
        self.assertEqual(result[0]["content"][:len(before[0]["content"])].encode("utf-8"), before[0]["content"].encode("utf-8"))
        self.assertEqual(result[1]["content"].encode("utf-8"), before[1]["content"].encode("utf-8"))

    def test_deep_copies_both_messages_including_nested_extra_fields(self):
        original = self.messages()
        result = ordered_messages(original)
        result[0]["metadata"]["nested"].append("changed")
        result[1]["metadata"]["nested"].append("changed")
        self.assertEqual(original[0]["metadata"]["nested"], ["system"])
        self.assertEqual(original[1]["metadata"]["nested"], ["user"])
        another = ordered_messages(original)
        self.assertEqual(another[0]["metadata"]["nested"], ["system"])
        self.assertEqual(another[1]["metadata"]["nested"], ["user"])

    def test_user_values_and_policy_never_change_the_fixed_guide(self):
        systems = []
        for payload in ('', 'not JSON', '{"policy":"summary","payment_status":"pending"}',
                        '{"policy":"structured","payment_status":"cleared","operation_already_booked":"yes"}',
                        '{"oracle":{"decision":"approve"},"future":true}',
                        '{"task":{"operation":"refund"},"invoice_amount_cents":999999}'):
            original = self.messages(payload)
            with patch("json.loads", side_effect=AssertionError("No JSON inspection")), \
                 patch("rcwt_decision_v4.task_schema", side_effect=AssertionError("No task schema or rule evaluation")), \
                 patch("rcwt_agent_env.expected_action", side_effect=AssertionError("No oracle evaluation")):
                result = ordered_messages(original)
            systems.append(result[0]["content"])
            self.assertEqual(result[1]["content"], payload)
        self.assertEqual(len(set(systems)), 1)

    def test_priority_guide_is_literal_ordered_text_not_case_specific(self):
        lines = ORDERED_RULE_GUIDE.splitlines()
        rules = [line for line in lines if len(line) > 2 and line[0].isdigit() and line[1:3] == ". "]
        self.assertEqual([line[:2] for line in rules], [f"{index}." for index in range(7)])
        for index, expected in enumerate(("operation_already_booked=yes", "account_status=blocked or revoked",
                                           "ownership_match=no", "payment_status=unknown", "payment_status=pending",
                                           "operation=refund AND return_status=rejected", "ONLY if every condition 0..5 is false")):
            self.assertIn(expected, rules[index])
        self.assertIn("A payout does not require return_status", rules[3])
        self.assertIn("operation=refund AND return_status=unknown", rules[3])
        for term in ("invoice_amount_cents=null", "account_status=unknown", "ownership_match=unknown"):
            self.assertIn(term, rules[3])
        self.assertNotIn("inv-train-", ORDERED_RULE_GUIDE)
        self.assertNotIn("rcwta-", ORDERED_RULE_GUIDE)

    def test_guide_preserves_zero_amounts_exact_monetary_amounts_and_booking_priority(self):
        rules = ORDERED_RULE_GUIDE.splitlines()[1:8]
        for rule in rules[:6]:
            self.assertIn("amount_cents=0", rule)
        self.assertIn("hold / already_completed", rules[0])
        self.assertIn("does not authorize another execution", rules[0])
        self.assertIn("exact positive invoice_amount_cents", rules[6])
        self.assertIn("approve / authorized_payout", rules[6])
        self.assertIn("refund / eligible_refund", rules[6])
        self.assertIn("regenerate decision, reason_code and amount_cents coherently", ORDERED_RULE_GUIDE)
        self.assertIn("Do not retain a zero amount", ORDERED_RULE_GUIDE)

    def test_no_update_or_other_id_does_not_instruct_erasure_of_retained_facts(self):
        self.assertIn("remains valid until replaced by an update for the same source and identifier", ORDERED_RULE_GUIDE)
        self.assertIn("a record for another identifier does not erase the retained fact", ORDERED_RULE_GUIDE)
        self.assertIn("Pending is an observed payment status, not unknown", ORDERED_RULE_GUIDE)
        self.assertIn("stopping at the first YES", ORDERED_RULE_GUIDE)
        self.assertIn("Use rule 6 only after six NO outcomes", ORDERED_RULE_GUIDE)

    def test_planning_wrapper_preserves_guide_and_removes_only_original_json_requirement(self):
        base = ordered_messages(self.messages())
        self.assertEqual(base[0]["content"].count(ORIGINAL_JSON_REQUIREMENT), 1)
        before = copy.deepcopy(base)
        plan = planning_messages(base)
        self.assertEqual(plan[0]["content"], base[0]["content"].replace(ORIGINAL_JSON_REQUIREMENT, "", 1)
                         + "\n\n" + PLANNING_INSTRUCTION)
        self.assertIn(ORDERED_RULE_GUIDE, plan[0]["content"])
        self.assertEqual(plan[1], base[1])
        self.assertEqual(base, before)

    def test_review_keeps_ordered_base_unchanged_and_passes_raw_plan_verbatim(self):
        base = ordered_messages(self.messages())
        raw = '{FAKE malformed or incomplete plan, amount=0'
        result = review_messages(base, raw)
        self.assertEqual(result[:2], base)
        self.assertEqual(result[2], {"role": "assistant", "content": raw})
        self.assertEqual(result[0]["content"].count(ORDERED_RULE_GUIDE), 1)

    def test_guide_does_not_mutate_or_replace_task_schema(self):
        public = {"task": {"case_id": "FAKE-ID", "operation": "refund"}}
        before = task_schema(public)
        original_actor = copy.deepcopy(ACTOR_SCHEMA)
        ordered_messages(self.messages(json.dumps(public)))
        self.assertEqual(task_schema(public), before)
        self.assertEqual(ACTOR_SCHEMA, original_actor)
        self.assertEqual(before["properties"]["arguments"]["properties"]["amount_cents"],
                         {"type": "integer", "minimum": 0})

    def test_invalid_message_interface_fails_closed_without_coercing_text(self):
        valid = self.messages()
        invalid = [None, {}, (), [], valid[:1], valid + [{"role": "assistant", "content": "x"}],
                   list(reversed(valid)), [None, valid[1]], [valid[0], None],
                   [{"role": "system", "content": 123}, valid[1]],
                   [valid[0], {"role": "user", "content": {"task": "no parsing"}}]]
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                ordered_messages(messages)


if __name__ == "__main__":
    unittest.main()
