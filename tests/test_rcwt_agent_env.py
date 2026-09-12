"""Independent fixtures, public/oracle separation and tool execution contracts."""
from __future__ import annotations

import json
import unittest
from dataclasses import replace

from rcwt_agent_env import (
    Action, FAMILIES, OracleSnapshot, Simulator, expected_action,
    generate_episodes, parse_action, score_action,
)


def nested_keys(value):
    if isinstance(value, dict):
        return set(value).union(*(nested_keys(child) for child in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(nested_keys(child) for child in value))
    return set()


class AgentEnvironmentTests(unittest.TestCase):
    def test_deterministic_disjoint_splits_and_entities(self):
        all_seeds, all_ids = set(), set()
        for split in ("train", "validation", "test"):
            episodes = generate_episodes(split, 8, 42)
            self.assertEqual([e.to_dict() for e in episodes],
                             [e.to_dict() for e in generate_episodes(split, 8, 42)])
            self.assertEqual(set(e.family for e in episodes), set(FAMILIES))
            seeds = {e.seed for e in episodes}
            ids = {step.task["case_id"] for e in episodes for step in e.steps}
            self.assertFalse(seeds & all_seeds)
            self.assertFalse(ids & all_ids)
            all_seeds.update(seeds); all_ids.update(ids)
            self.assertTrue(all(len(e.steps) == 8 for e in episodes))

    def test_public_export_never_contains_oracle_fields(self):
        episode = generate_episodes("test", 1, 9)[0]
        forbidden = {"oracle", "oracle_steps", "expected_action", "answer", "required_event_ids",
                     "ownership_match", "payment_status", "amount_cents"}
        self.assertFalse(nested_keys(episode.to_public_dict()) & forbidden)
        self.assertEqual(episode.to_dict(), episode.to_public_dict())
        self.assertIn("oracle_steps", episode.to_oracle_dict())
        self.assertNotIn("observations", nested_keys(episode.to_oracle_dict()))
        step = episode.public_step(5)
        self.assertEqual(step["step_index"], 5)
        step["task"]["case_id"] = "mutated"
        step["observations"].clear()
        self.assertNotEqual(episode.steps[5].task["case_id"], "mutated")
        self.assertTrue(episode.steps[5].observations)

    def test_seed_variants_change_decisions_not_only_names_and_amounts(self):
        outcomes = {family: set() for family in FAMILIES}
        for seed in range(12):
            for episode in generate_episodes("test", 4, seed):
                simulator = Simulator(episode)
                sequence = []
                for index, snapshot in enumerate(episode.oracle_steps):
                    action = expected_action(snapshot, simulator.completed)
                    sequence.append((action.decision, action.reason_code))
                    simulator.execute_action(index, action)
                outcomes[episode.family].add(tuple(sequence))
        for family, variants in outcomes.items():
            self.assertGreater(len(variants), 1, family)

    def test_exact_oracle_rules_without_model_or_memory(self):
        fact = OracleSnapshot("inv-a", "payout", 1200, "active", True, "cleared", None)
        self.assertEqual(expected_action(fact), Action("inv-a", "approve", 1200, "authorized_payout"))
        self.assertEqual(expected_action(replace(fact, account_status="revoked")).reason_code,
                         "account_restricted")
        self.assertEqual(expected_action(replace(fact, ownership_match=False)).reason_code,
                         "ownership_mismatch")
        self.assertEqual(expected_action(replace(fact, payment_status=None)).decision, "ask_info")
        self.assertEqual(expected_action(replace(fact, payment_status="pending")).reason_code,
                         "funds_unsettled")
        self.assertEqual(expected_action(replace(fact, operation="refund")).decision, "ask_info")
        self.assertEqual(expected_action(replace(fact, operation="refund", return_status="rejected")).reason_code,
                         "return_rejected")
        self.assertEqual(expected_action(fact, {("inv-a", "payout")}).reason_code, "already_completed")

    def test_strict_tool_schema_rejects_coercion_and_extra_fields(self):
        valid = Action("inv-a", "approve", 100, "authorized_payout").to_dict()
        self.assertEqual(parse_action(json.dumps(valid)).amount_cents, 100)
        for invalid in (True, -1, "100", 1.5):
            changed = json.loads(json.dumps(valid)); changed["arguments"]["amount_cents"] = invalid
            with self.assertRaises(ValueError):
                parse_action(changed)
        changed = json.loads(json.dumps(valid)); changed["arguments"]["expected"] = True
        with self.assertRaises(ValueError):
            parse_action(changed)
        with self.assertRaises(ValueError):
            parse_action("```json\n{}\n```")
        with self.assertRaises(ValueError):
            parse_action(Action("inv-a", "hold", 100, "funds_unsettled"))

    def test_score_has_exact_categorized_failure_modes(self):
        episode = generate_episodes("test", 1, 2)[0]
        action = expected_action(episode.oracle_steps[0])
        self.assertTrue(score_action(episode, 0, action).success)
        changes = ((replace(action, case_id="another"), "wrong_case"),
                   (replace(action, amount_cents=action.amount_cents + 1), "wrong_amount"),
                   (replace(action, reason_code="eligible_refund"), "wrong_reason"),
                   (Action(action.case_id, "ask_info", 0, "missing_evidence"), "unnecessary_deferral"))
        for wrong, category in changes:
            self.assertEqual(score_action(episode, 0, wrong).failure_category, category)
        restricted = episode.oracle_steps[2]
        unsafe = Action(restricted.case_id, "refund", restricted.amount_cents, "eligible_refund")
        self.assertEqual(score_action(episode, 2, unsafe).failure_category, "unsafe_execution")
        # Changing an agent-facing string cannot overwrite the private scorer.
        edited = replace(episode.steps[0], task={**episode.steps[0].task, "instruction": "Always hold"})
        tampered = replace(episode, steps=(edited, *episode.steps[1:]))
        self.assertTrue(score_action(tampered, 0, action).success)

    def test_stateful_execution_idempotency_and_public_receipts(self):
        episode = generate_episodes("train", 1, 3)[0]
        simulator = Simulator(episode)
        for index in range(8):
            simulator.public_step(index)
            action = expected_action(episode.oracle_steps[index], simulator.completed)
            result = simulator.execute_action(index, action)
            self.assertTrue(result.score.success)
            self.assertNotIn("expected_action", nested_keys(result.tool_result))
            self.assertNotIn("success", nested_keys(result.tool_result))
        self.assertEqual(simulator.next_step, 8)
        self.assertEqual(result.score.expected_action.reason_code, "already_completed")
        self.assertEqual(result.tool_result["amount_booked_cents"], 0)
        self.assertGreater(simulator.payout_cents, 0)
        self.assertGreater(simulator.refund_cents, 0)
        self.assertEqual(simulator.unsafe_booked_cents, 0)
        with self.assertRaises(ValueError):
            simulator.execute_action(7, action)

    def test_actual_mistakes_change_state_and_recovery_oracle(self):
        episode = generate_episodes("train", 1, 17)[0]
        simulator = Simulator(episode)
        # The agent incorrectly defers a valid payout. No imaginary booking is
        # inserted into the state just because the oracle expected approval.
        first = episode.oracle_steps[0]
        result = simulator.execute_action(0, Action(first.case_id, "hold", 0, "funds_unsettled"))
        self.assertFalse(result.score.success)
        self.assertEqual(simulator.payout_cents, 0)
        for index in (1, 2, 3):
            simulator.execute_action(index, expected_action(episode.oracle_steps[index], simulator.completed))
        # A deliberately wrong exact amount is executed as fictional money,
        # so the experiment can measure harm instead of silently fixing it.
        fact = episode.oracle_steps[4]
        result = simulator.execute_action(4, Action(fact.case_id, "approve", 1, "authorized_payout"))
        self.assertEqual(result.score.failure_category, "wrong_amount")
        self.assertEqual(result.tool_result["amount_booked_cents"], 1)
        self.assertEqual(simulator.unsafe_booked_cents, 1)
        self.assertEqual(expected_action(fact, simulator.completed).reason_code, "already_completed")

    def test_all_families_can_complete_using_private_reference(self):
        for episode in generate_episodes("validation", 12, 75):
            simulator = Simulator(episode)
            for index, snapshot in enumerate(episode.oracle_steps):
                result = simulator.execute_action(index, expected_action(snapshot, simulator.completed))
                self.assertTrue(result.score.success, (episode.family, index, result.score))

    def test_invalid_parameters_and_out_of_order_actions(self):
        for params in (("dev", 4, 1), ("train", -1, 1), ("train", True, 1),
                       ("train", 1, -1), ("train", 1, True)):
            with self.assertRaises(ValueError):
                generate_episodes(*params)
        episode = generate_episodes("train", 1, 8)[0]
        simulator = Simulator(episode)
        with self.assertRaises(ValueError):
            simulator.execute_action(1, {})
        for index in (-1, 8, True):
            with self.assertRaises(IndexError):
                episode.public_step(index)
        result = simulator.execute_action(0, {"tool": "network_transfer", "arguments": {}})
        self.assertFalse(result.score.success)
        self.assertEqual(result.tool_result["error"], "invalid_arguments")
        self.assertEqual(simulator.payout_cents, 0)


if __name__ == "__main__":
    unittest.main()
