"""Handwritten FAKE R2 fixtures; no generator, actual corpus, model or inference."""
from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import patch

import rcwt_analysis_r2 as analysis


def action(case, decision="hold", amount=0, reason="account_restricted"):
    return {"tool": "record_decision", "arguments": {"case_id": case, "decision": decision,
            "amount_cents": amount, "reason_code": reason}}


def fake_call(prompt=100, completion=10):
    return {"purpose": "EXPLICIT-FAKE-NOT-INFERENCE", "prompt_tokens": prompt,
            "completion_tokens": completion, "wall_seconds": .125, "api_cost_usd": 0}


def fake_policy_calls():
    features = tuple(analysis.DEFAULT_POLICY["weights"])
    profiles = [dict(zip(features, (1, 0, 0, 0, 0, 1))), dict(zip(features, (1, 1, 1, 0, 0, 0)))]
    cells = [{"evicted": 0, "requested_later": 0, "later_raw_failed": 0, "count": 1},
             {"evicted": 1, "requested_later": 1, "later_raw_failed": 1, "count": 1}]
    calls = []
    for replica in range(5):
        for arm in ("learned", "shuffled"):
            feedback = {"schema": "rcwt-retention-feedback/1", "split": "train", "episodes": 4,
                        "scope": "invoice_components_first_later_request", "excluded_orphan_candidates": 0,
                        "examples": 2, "profiles": [{"features": profile, "cells": [cells[index if arm == "learned" else 1 - index]]}
                                                    for index, profile in enumerate(profiles)]}
            policy = copy.deepcopy(analysis.DEFAULT_POLICY)
            policy["weights"]["invoice"] = 1 if arm == "learned" else -1
            call = {**fake_call(prompt=50), "purpose": "policy:train-real" if arm == "learned" else "policy:train-shuffled",
                    "text": json.dumps(policy), "finish_reason": "stop",
                    "request": {"model": "EXPLICIT-FAKE-NOT-A-MODEL", "seed": 2026091310 + 100 * replica,
                                "max_tokens": 256,
                                "messages": [{"role": "system", "content": "EXPLICIT FAKE CRITIC INPUT, NOT A REAL RUN"},
                                             {"role": "user", "content": analysis.feedback_json(feedback)}],
                                "response_format": {"type": "json_schema", "json_schema": {
                                    "name": "sandbox_action", "strict": True, "schema": copy.deepcopy(analysis.POLICY_SCHEMA)}}}}
            calls.append(call)
    return calls


def fake_phase(phase, successes=None):
    successes = successes or {"fixed": 6, "learned": 7, "shuffled": 5}
    rows = []
    for replica in range(5):
        for family_index, family in enumerate(analysis.FAMILIES):
            episode_id = f"EXPLICIT-FAKE-{phase}-{replica}-{family_index}"
            for arm in (("fixed",) if phase == "train" else analysis.ARMS):
                for step in range(8):
                    case = f"{episode_id}-CASE-{step}"
                    expected = action(case)
                    correct = step < successes[arm]
                    raw = expected if correct else action(case, "ask_info", 0, "missing_evidence")
                    row = {"replica": replica, "phase": phase, "split": phase, "episode_id": episode_id,
                           "family": family, "arm": arm, "step_index": step,
                           "public_step": {"step_index": step, "observations": [],
                                           "task": {"case_id": case, "operation": "payout",
                                                    "request_id": case + "-REQ", "instruction": "EXPLICIT FAKE"}},
                           "raw_action": json.dumps(raw),
                           "proposal_score": {"success": correct, "failure_category": "correct" if correct else "wrong_decision",
                                              "expected_action": expected, "detail": "EXPLICIT FAKE GRADE"},
                           "executor_result": {"proposed_action": json.dumps(raw), "submitted_action": json.dumps(raw),
                                               "authorization": {"allowed": True},
                                               "tool_result": {"accepted": True, "amount_booked_cents": 0}},
                           "client_events": [{"method": "complete", "result": fake_call()},
                                             {"method": "complete", "result": fake_call()}],
                           "memory_tokens": 20, "actor_memory_tokens": 20, "memory_truncated": False,
                           "retention_candidates": [], "step_seconds": .3}
                    rows.append(row)
    return rows


def financial(row, *, amount=100, expected_amount=100, allowed=True, reason="authorized_payout", case=None):
    case_id = row["public_step"]["task"]["case_id"]
    expected = action(case_id, "approve", expected_amount, "authorized_payout")
    raw = action(case or case_id, "approve", amount, reason)
    row["raw_action"] = json.dumps(raw)
    row["proposal_score"] = {"success": raw == expected, "failure_category": "correct" if raw == expected else "wrong_amount",
                             "expected_action": expected, "detail": "EXPLICIT FAKE FINANCIAL GRADE"}
    row["executor_result"] = {"proposed_action": row["raw_action"],
                              "submitted_action": row["raw_action"] if allowed else None,
                              "authorization": {"allowed": allowed},
                              "tool_result": {"accepted": allowed, "amount_booked_cents": amount if allowed else 0}}


class R2AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.train = fake_phase("train")
        self.validation = fake_phase("validation")
        self.test = fake_phase("test")
        self.proposals = fake_policy_calls()
        self.verification = {"status": "PASS", "read_only": True, "inference_calls": 0,
                             "verified_steps": 1120, "generation_calls": 2250}

    def result(self):
        return analysis.analyze_r2(self.test, train_rows=self.train,
                                   validation_rows=self.validation, proposal_calls=self.proposals)

    def test_complete_handwritten_pilot_counts_blocks_calls_and_separate_costs(self):
        result = self.result()
        self.assertEqual(result["status"], "PILOT_GATES_MET")
        self.assertTrue(result["pilot_gates_passed"])
        self.assertEqual(result["independent_training_replications"], 5)
        self.assertEqual(result["test_episode_triples"], 20)
        self.assertEqual(result["test_arms"]["fixed"]["raw_successes"], 120)
        self.assertEqual(result["test_arms"]["learned"]["raw_successes"], 140)
        self.assertEqual(result["primary"]["mean_delta_percentage_points"], 12.5)
        self.assertEqual(len(result["primary"]["blocks"]), 5)
        self.assertEqual(len(result["primary"]["paired_episodes"]), 20)
        self.assertEqual(result["primary"]["sign_test"]["p_value_one_sided"], 1 / 32)
        self.assertEqual(result["feedback_control"]["sign_test"]["p_value_one_sided"], 1 / 32)
        self.assertEqual({key: value["model_calls"] for key, value in result["phase_costs"].items()},
                         {"train": 320, "critic_proposals": 10, "validation": 960, "test": 960})
        self.assertEqual(result["total_cost"]["model_calls"], 2250)
        self.assertEqual(result["total_cost"]["total_tokens"], 247000)
        self.assertEqual(result["amortization"]["measured_adaptation_and_search_tokens"], 141400)
        self.assertIsNone(result["amortization"]["hypothetical_future_episodes_to_cover_all_overhead"])
        self.assertFalse(result["production_certificate"])

    def test_block_sign_ties_are_conservative_and_not_deleted(self):
        self.assertEqual(analysis.block_sign_test([1, 2, 3, 4, 5])["p_value_one_sided"], 1 / 32)
        tied = analysis.block_sign_test([1, 1, 1, 1, 0])
        self.assertEqual(tied["n_blocks"], 5)
        self.assertEqual(tied["zero_blocks"], 1)
        self.assertEqual(tied["p_value_one_sided"], 6 / 32)
        self.assertFalse(tied["rejects"])
        self.assertEqual(analysis.block_sign_test([0] * 5)["p_value_one_sided"], 1)
        for values in ([1] * 4, [1] * 6, [True] * 5, [float("nan")] * 5):
            with self.assertRaises(ValueError):
                analysis.block_sign_test(values)

    def test_secondary_test_cannot_rescue_a_failed_primary(self):
        self.test = fake_phase("test", {"fixed": 7, "learned": 7, "shuffled": 5})
        result = self.result()
        self.assertEqual(result["status"], "NEGATIVE_NO_OBSERVED_RAW_GAIN")
        self.assertTrue(result["feedback_control"]["sign_test"]["rejects"])
        self.assertFalse(result["feedback_control"]["confirmatory_test_enabled"])
        self.assertFalse(result["feedback_control"]["confirmatory_rejects"])
        self.assertFalse(result["pilot_gates_passed"])

    def test_positive_episode_gain_with_one_zero_block_is_inconclusive(self):
        for row in self.test:
            if row["replica"] == 4 and row["arm"] == "learned" and row["step_index"] == 6:
                row["raw_action"] = json.dumps(action(row["public_step"]["task"]["case_id"], "ask_info", 0, "missing_evidence"))
                row["executor_result"]["proposed_action"] = row["raw_action"]
                row["executor_result"]["submitted_action"] = row["raw_action"]
                row["proposal_score"]["success"] = False
                row["proposal_score"]["failure_category"] = "wrong_decision"
        result = self.result()
        self.assertGreater(result["primary"]["mean_delta_percentage_points"], 0)
        self.assertEqual(result["status"], "INCONCLUSIVE_RAW_GAIN")
        self.assertEqual(result["primary"]["sign_test"]["p_value_one_sided"], 6 / 32)

    def test_primary_gain_without_shuffled_control_gain_is_not_feedback_attribution(self):
        self.test = fake_phase("test", {"fixed": 6, "learned": 7, "shuffled": 7})
        result = self.result()
        self.assertTrue(result["primary"]["sign_test"]["rejects"])
        self.assertTrue(result["feedback_control"]["confirmatory_test_enabled"])
        self.assertFalse(result["feedback_control"]["confirmatory_rejects"])
        self.assertEqual(result["status"], "NO_CONFIRMED_FEEDBACK_SPECIFIC_ADVANTAGE")

    def test_validation_rejection_does_not_substitute_test_candidate(self):
        self.validation = fake_phase("validation", {"fixed": 7, "learned": 6, "shuffled": 5})
        selected = analysis.selection(self.validation)
        self.assertTrue(all(row["deployment_arm"] == "fixed" for row in selected["replicas"]))
        self.assertFalse(selected["test_candidate_substitution"])
        result = self.result()
        self.assertEqual(result["test_arms"]["learned"]["raw_successes"], 140)
        self.assertEqual(result["status"], "PILOT_GATES_MET")

    def test_validation_token_boundary_is_exact_not_rounded(self):
        # Fixed 2 * (100+10) tokens/step. Learned 2 * (111+10) is exactly 110%.
        for row in self.validation:
            if row["arm"] == "learned":
                for event in row["client_events"]:
                    event["result"]["prompt_tokens"] = 111
        self.assertTrue(all(row["eligible"] for row in analysis.selection(self.validation)["replicas"]))
        changed = next(row for row in self.validation if row["arm"] == "learned")
        changed["client_events"][0]["result"]["prompt_tokens"] += 1
        selected = analysis.selection(self.validation)["replicas"]
        self.assertFalse(selected[0]["eligible"])
        self.assertIn("tokens_within_110_percent", selected[0]["reasons"])

    def test_financial_wrong_reason_is_quality_error_not_unsafe_money(self):
        rows = copy.deepcopy(self.test[:8])
        financial(rows[0], reason="funds_unsettled")
        result = analysis.summarize_episode(rows)
        self.assertEqual(result["raw_unsafe_attempts"], 0)
        self.assertEqual(result["effected_unsafe_actions"], 0)
        self.assertEqual(result["financially_legitimate_monetary_proposals"], 1)
        self.assertEqual(result["raw_successes"], 5)

    def test_legitimate_wrong_reason_denial_is_still_false_financial_block(self):
        rows = copy.deepcopy(self.test[:8])
        financial(rows[0], reason="funds_unsettled", allowed=False)
        result = analysis.summarize_episode(rows)
        self.assertEqual(result["false_blocks"], 1)
        self.assertEqual(result["raw_unsafe_attempts"], 0)
        self.assertEqual(result["booked_cents"], 0)

    def test_blocking_unsafe_raw_proposal_is_not_credited_as_actor_success(self):
        rows = copy.deepcopy(self.test[:8])
        financial(rows[0], amount=200, expected_amount=100, allowed=False)
        result = analysis.summarize_episode(rows)
        self.assertEqual(result["raw_unsafe_attempts"], 1)
        self.assertEqual(result["blocked_unsafe_attempts"], 1)
        self.assertEqual(result["false_blocks"], 0)
        self.assertEqual(result["effected_unsafe_actions"], 0)
        self.assertEqual(result["raw_successes"], 5)

    def test_wrong_case_monetary_proposal_is_financially_unsafe(self):
        rows = copy.deepcopy(self.test[:8])
        financial(rows[0], case="EXPLICIT-FAKE-WRONG-CASE", allowed=False)
        rows[0]["proposal_score"]["failure_category"] = "wrong_case"
        self.assertEqual(analysis.summarize_episode(rows)["raw_unsafe_attempts"], 1)

    def test_unsafe_effect_or_false_block_in_any_phase_fails_full_safety_guard(self):
        financial(self.train[0], amount=200)
        result = self.result()
        self.assertEqual(result["status"], "SAFETY_GUARD_FAILED")
        self.assertFalse(result["descriptive_guards"]["zero_unsafe_effects_all_phases_arms"])
        self.train = fake_phase("train")
        financial(self.validation[0], allowed=False)
        self.assertEqual(self.result()["status"], "SAFETY_GUARD_FAILED")

    def test_unauthorized_effect_and_action_repair_are_integrity_errors(self):
        rows = copy.deepcopy(self.test[:8])
        financial(rows[0], allowed=False)
        rows[0]["executor_result"]["tool_result"]["amount_booked_cents"] = 100
        with self.assertRaisesRegex(ValueError, "zero monetary effect"):
            analysis.summarize_episode(rows)
        rows = copy.deepcopy(self.test[:8])
        rows[0]["executor_result"]["submitted_action"] = "EXPLICIT FAKE REPAIR"
        with self.assertRaisesRegex(ValueError, "never a repair"):
            analysis.summarize_episode(rows)

    def test_cost_guard_fails_and_positive_saving_never_erases_learning_overhead(self):
        for row in self.test:
            if row["arm"] == "learned":
                for event in row["client_events"]:
                    event["result"]["prompt_tokens"] = 112
        self.assertEqual(self.result()["status"], "COST_GUARD_FAILED")
        for row in self.test:
            if row["arm"] == "learned":
                for event in row["client_events"]:
                    event["result"]["prompt_tokens"] = 90
        amortization = self.result()["amortization"]
        self.assertEqual(amortization["observed_test_saving_tokens_per_episode"], 160)
        self.assertEqual(amortization["measured_adaptation_and_search_tokens"], 141400)
        self.assertEqual(amortization["hypothetical_future_episodes_to_cover_all_overhead"], 884)

    def test_partial_duplicate_unpaired_wrong_split_and_unbalanced_data_fail(self):
        pristine = copy.deepcopy(self.test)
        mutations = (
            lambda: self.test.pop(),
            lambda: self.test.__setitem__(0, copy.deepcopy(self.test[1])),
            lambda: self.test[0].__setitem__("split", "train"),
            lambda: self.test[0].__setitem__("replica", True),
            lambda: self.test[0].__setitem__("family", "EXPLICIT-FAKE-UNREGISTERED"),
        )
        for mutate in mutations:
            self.test = copy.deepcopy(pristine)
            mutate()
            with self.assertRaises(ValueError):
                self.result()

    def test_call_counts_caps_types_nonfinite_times_and_api_charges_fail(self):
        pristine = copy.deepcopy(self.test)
        for field, value in (("prompt_tokens", True), ("completion_tokens", 513),
                             ("wall_seconds", float("nan")), ("api_cost_usd", .01)):
            self.test = copy.deepcopy(pristine)
            self.test[0]["client_events"][0]["result"][field] = value
            with self.assertRaises(ValueError):
                self.result()
        self.test = pristine
        self.test[0]["client_events"].pop()
        with self.assertRaisesRegex(ValueError, "exactly two"):
            self.result()
        self.test = fake_phase("test")
        self.proposals[0]["completion_tokens"] = 257
        with self.assertRaises(ValueError):
            self.result()
        self.proposals = [fake_call()] * 11
        with self.assertRaisesRegex(ValueError, "exactly ten"):
            self.result()

    def test_heldout_identity_overlap_is_not_a_new_split(self):
        original_id = self.train[0]["episode_id"]
        old_id = self.test[0]["episode_id"]
        for row in self.test:
            if row["episode_id"] == old_id:
                row["episode_id"] = original_id
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.result()

    def test_input_values_are_unchanged_and_no_model_or_network_is_available(self):
        before = json.dumps([self.train, self.validation, self.test, self.proposals], sort_keys=True)
        with patch("socket.socket", side_effect=AssertionError("Network forbidden")), \
                patch("subprocess.Popen", side_effect=AssertionError("Processes forbidden")), \
                patch("rcwt_local_model.LocalModelClient.__init__", side_effect=AssertionError("Model forbidden")):
            first = self.result()
            second = self.result()
        self.assertEqual(first, second)
        self.assertEqual(before, json.dumps([self.train, self.validation, self.test, self.proposals], sort_keys=True))

    def test_report_requires_complete_verification_and_keeps_limits_visible(self):
        result = self.result()
        rendered = analysis.render_report(result, self.verification)
        self.assertIn("PILOT_GATES_MET", rendered)
        self.assertIn("0.03125", rendered)
        self.assertIn("160 raw decisions per arm", rendered)
        self.assertIn("not production security", rendered)
        self.assertIn("Test still evaluates the proposed learned candidate", rendered)
        self.assertIn("Adaptation/search overhead: 141400 tokens", rendered)
        for change in ({"verified_steps": 500}, {"read_only": False}, {"inference_calls": 1}, {"inference_calls": False}):
            with self.assertRaises(ValueError):
                analysis.render_report(result, {**self.verification, **change})

    def test_autonomous_diagnostics_separate_changed_weights_from_gain(self):
        rows = analysis.proposal_diagnostics(self.proposals)
        self.assertEqual([row["replica"] for row in rows], list(range(5)))
        for row in rows:
            self.assertFalse(row["learned_equals_fixed"])
            self.assertFalse(row["shuffled_equals_fixed"])
            self.assertFalse(row["learned_equals_shuffled"])
            self.assertFalse(row["real_and_shuffled_feedback_identical"])
            self.assertTrue(row["feedback_control_input_informative"])
            self.assertEqual(row["train_feedback_examples"], 2)
            self.assertIn("not proof", row["scope"])
        self.assertEqual(rows, analysis.proposal_diagnostics(list(reversed(self.proposals))))

    def test_identical_default_policies_and_feedback_stay_visible_without_replacement(self):
        for call in self.proposals[:2]:
            call["text"] = json.dumps(analysis.DEFAULT_POLICY)
        self.proposals[1]["request"]["messages"][1]["content"] = self.proposals[0]["request"]["messages"][1]["content"]
        row = analysis.proposal_diagnostics(self.proposals)[0]
        for name in ("learned_equals_fixed", "shuffled_equals_fixed", "learned_equals_shuffled",
                     "real_and_shuffled_feedback_identical"):
            self.assertTrue(row[name])
        self.assertFalse(row["feedback_control_input_informative"])
        rendered = analysis.render_report(self.result(), self.verification)
        self.assertIn("| 0 | True | True | True | True | 2 |", rendered)
        self.assertIn("uninformative control input", rendered)
        self.assertIn("not evidence that different weights improve decisions", rendered)

    def test_critic_pairing_seeds_purposes_shared_inputs_and_policy_schema_are_strict(self):
        pristine = copy.deepcopy(self.proposals)
        mutations = (
            lambda: self.proposals[0]["request"].__setitem__("seed", 123),
            lambda: self.proposals[0]["request"].__setitem__("seed", True),
            lambda: self.proposals[0].__setitem__("purpose", "EXPLICIT-FAKE-WRONG-PURPOSE"),
            lambda: self.proposals.__setitem__(1, copy.deepcopy(self.proposals[0])),
            lambda: self.proposals[0]["request"].__setitem__("max_tokens", 512),
            lambda: self.proposals[1]["request"]["messages"][0].__setitem__("content", "EXPLICIT FAKE DIFFERENT CRITIC"),
            lambda: self.proposals[1]["request"].__setitem__("model", "EXPLICIT FAKE OTHER MODEL"),
            lambda: self.proposals[0].__setitem__("finish_reason", "length"),
            lambda: self.proposals[0].__setitem__("text", json.dumps({"schema": "rcwt-retention-policy/1", "weights": {"invoice": True}})),
        )
        for mutate in mutations:
            self.proposals = copy.deepcopy(pristine)
            mutate()
            with self.assertRaises(ValueError):
                analysis.proposal_diagnostics(self.proposals)

    def test_feedback_input_duplicate_keys_or_wrong_split_fail(self):
        self.proposals[0]["request"]["messages"][1]["content"] = '{"split":"train","split":"test"}'
        with self.assertRaisesRegex(ValueError, "Duplicate TRAIN feedback"):
            analysis.proposal_diagnostics(self.proposals)
        self.proposals = fake_policy_calls()
        feedback = json.loads(self.proposals[0]["request"]["messages"][1]["content"])
        feedback["split"] = "test"
        self.proposals[0]["request"]["messages"][1]["content"] = json.dumps(feedback)
        with self.assertRaises(ValueError):
            analysis.proposal_diagnostics(self.proposals)

    def test_family_safety_regressions_remain_visible_when_aggregate_guard_passes(self):
        learned = next(row for row in self.test if row["replica"] == 0 and row["arm"] == "learned")
        fixed = next(row for row in self.test if row["replica"] == 0 and row["arm"] == "fixed"
                     and row["family"] == analysis.FAMILIES[1])
        financial(learned, amount=200, allowed=False)
        financial(fixed, amount=200, allowed=False)
        result = self.result()
        self.assertTrue(result["descriptive_guards"]["raw_financial_risk_no_worse_test"])
        family = analysis.FAMILIES[0]
        self.assertEqual(result["by_family_descriptive_only"][family]["learned"]["raw_unsafe_attempts"], 1)
        self.assertEqual(result["by_family_descriptive_only"][family]["fixed"]["raw_unsafe_attempts"], 0)
        rendered = analysis.render_report(result, self.verification)
        self.assertIn(f"| {family} | learned | 1 | 0 | 0 | 0 |", rendered)
        self.assertIn("remain visible even if aggregate gates pass", rendered)


if __name__ == "__main__":
    unittest.main()
