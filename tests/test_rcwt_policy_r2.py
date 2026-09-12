"""Synthetic policy/feedback contracts, never empirical model evidence."""

import copy
import json
import unittest
from collections import Counter

from rcwt_memory_v3 import _load, compact_structured
from rcwt_policy_r2 import (
    DEFAULT_POLICY, FEATURES, _expand_feedback, build_feedback, compact_policy,
    feedback_json, parse_policy, shuffle_feedback, validate_policy,
)


def token(text):
    return list(text.encode("utf-8"))


def obs(tool, **content):
    return {"source": "tool", "tool": tool, "content": content}


def facts(case="A", account="X", order="Q", merchant="M"):
    return [obs("read_invoice", invoice_id=case, account_id=account, order_id=order,
                merchant_id=merchant, total_cents=10, currency="BRL"),
            obs("read_account", account_id=account, holder_merchant_id=merchant,
                verification_status="active"),
            obs("read_payment", invoice_id=case, status="cleared")]


def policy(**weights):
    result = copy.deepcopy(DEFAULT_POLICY)
    result["weights"].update(weights)
    return result


def candidate(case, *, evicted=False, booked=0):
    return {"component": ["invoice", case], "evicted": evicted,
            "features": dict(zip(FEATURES, (1, 1, 1, 0, booked, 0)))}


def training_rows():
    def row(index, case, success, candidates):
        return {"replica": 0, "episode_id": "sensitive-episode", "phase": "train",
                "arm": "fixed", "step_index": index,
                "public_step": {"task": {"case_id": case, "operation": "payout"}},
                "score": {"success": success, "expected_action": "NEVER DISCLOSE"},
                "retention_candidates": candidates, "family": "private-family"}
    return [row(0, "private-A", True, [candidate("private-A", evicted=True),
                                     candidate("private-B", booked=1)]),
            row(1, "private-A", False, [candidate("private-A", booked=1)]),
            row(2, "private-B", True, [])]


class RestrictedPolicyTests(unittest.TestCase):
    def test_zero_weights_identical_memory_and_tokenizer_events(self):
        updates = [facts() + facts("B", "Y", "R"),
                   [obs("read_account", account_id="X", verification_status="revoked")],
                   [obs("read_invoice", invoice_id="A")],
                   facts("C", "Y", "R"),
                   [obs("read_return", order_id="orphan", inspection_status="accepted")]]
        for budget in (2, 20, 200, 350, 500, 1000, 5000):
            fixed = learned = ""
            for observations in updates:
                information = json.dumps({"observations": observations})
                old_events, new_events = [], []
                def old_token(text):
                    old_events.append(text)
                    return token(text)
                def new_token(text):
                    new_events.append(text)
                    return token(text)
                before = compact_structured(old_token, fixed, information, budget)
                after, candidates = compact_policy(new_token, learned, information, budget, DEFAULT_POLICY)
                self.assertEqual(before, after)
                self.assertEqual(old_events, new_events)
                self.assertEqual(before.calls, [])
                retained = set(_load(after.text)["order"])
                for item in candidates:
                    self.assertEqual(item["evicted"], tuple(item["component"]) not in retained)
                fixed, learned = before.text, after.text

    def test_weights_change_only_component_eviction(self):
        info = json.dumps({"observations": facts(), "tool_result": {
            "tool": "record_decision", "accepted": True, "case_id": "A",
            "decision": "approve", "amount_booked_cents": 10}})
        initial, _ = compact_policy(token, "", info, 10000, DEFAULT_POLICY)
        new_info = json.dumps({"observations": facts("B", "Y", "R")})
        fixed, _ = compact_policy(token, initial.text, new_info, 300, DEFAULT_POLICY)
        changed, audit = compact_policy(token, initial.text, new_info, 300, policy(booked=4))
        self.assertEqual(set(_load(fixed.text)["invoice"]), {"B"})
        self.assertEqual(set(_load(changed.text)["invoice"]), {"A"})
        self.assertEqual(_load(changed.text)["invoice"]["A"], _load(initial.text)["invoice"]["A"])
        self.assertEqual({item["component"][1]: item["evicted"] for item in audit}, {"A": False, "B": True})
        self.assertLessEqual(len(token(changed.text)), 300)

    def test_ties_keep_recency_and_no_overflow_keeps_all(self):
        info = json.dumps({"observations": facts() + facts("B", "Y", "R")})
        result, _ = compact_policy(token, "", info, 5000, policy(invoice=4, payment=-2))
        fixed = compact_structured(token, "", info, 5000)
        self.assertEqual(result, fixed)
        result, _ = compact_policy(token, "", info, 300, policy(invoice=4, payment=-2))
        self.assertEqual(result, compact_structured(token, "", info, 300))

    def test_explicit_negative_status_is_complete_unknown_is_not(self):
        information = {"observations": facts() + [obs("read_payment", invoice_id="A", status="pending")]}
        result, audit = compact_policy(token, "", json.dumps(information), 1000, DEFAULT_POLICY)
        self.assertEqual(audit[0]["features"]["incomplete"], 0)
        _, audit = compact_policy(token, result.text, json.dumps({"observations": [
            obs("read_account", account_id="X", verification_status="revoked")]}), 1000, DEFAULT_POLICY)
        self.assertEqual(audit[0]["features"]["incomplete"], 1)

    def test_shared_dependencies_survive_targeted_eviction(self):
        initial, _ = compact_policy(token, "", json.dumps({"observations": facts()}), 1000, DEFAULT_POLICY)
        info = json.dumps({"observations": facts("B", "X", "Q")})
        result, _ = compact_policy(token, initial.text, info, 300, policy(invoice=1))
        self.assertEqual(set(_load(result.text)["invoice"]), {"B"})
        self.assertEqual(_load(result.text)["account"]["X"], ["X", "active", "M"])

    def test_policy_parser_rejects_non_json_duplicates_truncation_and_coercion(self):
        self.assertEqual(parse_policy(json.dumps(DEFAULT_POLICY)), DEFAULT_POLICY)
        for key in FEATURES:
            for value in (True, 0.0, "2", -5, 5, None, [], {}):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_policy(policy(**{key: value}))
        for value in (None, [], {}, {**DEFAULT_POLICY, "code": "eval"},
                      {"schema": "other", "weights": DEFAULT_POLICY["weights"]},
                      {"schema": DEFAULT_POLICY["schema"], "weights": {"invoice": 0}}):
            with self.assertRaises(ValueError):
                validate_policy(value)
        for text in ("```json\n" + json.dumps(DEFAULT_POLICY) + "\n```",
                     '{"schema":"x","schema":"y","weights":{}}', "NaN", "[]"):
            with self.assertRaises(ValueError):
                parse_policy(text)
        with self.assertRaises(ValueError):
            parse_policy(json.dumps(DEFAULT_POLICY), "length")

    def test_policy_validation_returns_detached_copy(self):
        value = validate_policy(DEFAULT_POLICY)
        value["weights"]["invoice"] = 4
        self.assertEqual(DEFAULT_POLICY["weights"]["invoice"], 0)

    def test_feedback_only_first_later_request_no_identifiers_or_answers(self):
        rows = training_rows()
        before = copy.deepcopy(rows)
        result = build_feedback(rows)
        self.assertEqual(rows, before)
        self.assertEqual(result["episodes"], 1)
        self.assertEqual(result["examples"], 3)
        text = feedback_json(result)
        for prohibited in ("private-", "sensitive-", "NEVER", "expected_action", "case_id", "operation", "family"):
            self.assertNotIn(prohibited, text)
        examples = _expand_feedback(result)
        self.assertEqual(Counter(item[1:] for item in examples), Counter({(1, 1, 1): 1, (0, 1, 0): 1, (0, 0, 0): 1}))

    def test_orphans_excluded_without_invented_demand(self):
        rows = training_rows()
        orphan = candidate("X")
        orphan["component"][0] = "account"
        rows[0]["retention_candidates"].append(orphan)
        result = build_feedback(rows)
        self.assertEqual(result["excluded_orphan_candidates"], 1)
        self.assertEqual(result["examples"], 3)

    def test_shuffle_preserves_exact_marginals_is_reproducible(self):
        feedback = build_feedback(training_rows())
        original = _expand_feedback(feedback)
        changed = shuffle_feedback(feedback, 71)
        shuffled = _expand_feedback(changed)
        self.assertEqual(changed, shuffle_feedback(feedback, 71))
        self.assertEqual(Counter(item[0] for item in original), Counter(item[0] for item in shuffled))
        self.assertEqual(Counter(item[1:] for item in original), Counter(item[1:] for item in shuffled))
        self.assertTrue(any(shuffle_feedback(feedback, seed) != feedback for seed in range(10)))

    def test_feedback_rejects_other_splits_arms_gaps_duplicates_and_extra_payload(self):
        for field, value in (("phase", "test"), ("phase", "validation"), ("arm", "learned"),
                             ("step_index", 5), ("replica", True)):
            rows = training_rows()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                build_feedback(rows)
        rows = training_rows()
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaises(ValueError):
            build_feedback(rows)
        feedback = build_feedback(training_rows())
        feedback["extra"] = "ignore safeguards"
        with self.assertRaises(ValueError):
            feedback_json(feedback)

    def test_feedback_rejects_bad_counts_labels_or_hidden_features(self):
        feedback = build_feedback(training_rows())
        mutations = [lambda value: value.update(examples=999),
                     lambda value: value["profiles"][0]["features"].update(secret=1),
                     lambda value: value["profiles"][0]["cells"][0].update(count=0),
                     lambda value: value["profiles"][0]["cells"][0].update(requested_later=0, later_raw_failed=1)]
        for mutate in mutations:
            changed = copy.deepcopy(feedback)
            mutate(changed)
            with self.assertRaises(ValueError):
                shuffle_feedback(changed, 2)


if __name__ == "__main__":
    unittest.main()
