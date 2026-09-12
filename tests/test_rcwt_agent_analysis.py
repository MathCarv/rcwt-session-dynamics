"""Statistical and validation contracts; fixtures are not model-run evidence."""

from __future__ import annotations

import copy
import unittest

from rcwt_agent_analysis import analyze, render_report


def _record(episode_id, policy, successes, *, steps=4, family="updates"):
    return {
        "episode_id": episode_id,
        "family": family,
        "split": "test",
        "policy": policy,
        "steps": steps,
        "successes": successes,
        "failures": {"missing_context": steps - successes} if successes < steps else {},
        "valid_actions": steps,
        "unsafe_actions": 0,
        "prompt_tokens": 100 * steps,
        "completion_tokens": 20 * steps,
        "decision_seconds": [float(index + 1) for index in range(steps)],
        "step_seconds": [float(index + 2) for index in range(steps)],
        "episode_seconds": sum(index + 2 for index in range(steps)) + 0.25,
        "model_calls": 2 * steps,
        "memory_truncations": 1,
    }


def _cohort(scores=None):
    if scores is None:
        scores = [(1, 2, 4), (2, 2, 4), (1, 3, 4), (3, 3, 4)]
    return [
        _record(f"E{index}", policy, score, family="updates" if index % 2 == 0 else "dependencies")
        for index, episode_scores in enumerate(scores)
        for policy, score in zip(("tail", "summary", "learned"), episode_scores)
    ]


class AgentAnalysisTests(unittest.TestCase):
    def test_policy_metrics_and_positive_gate(self):
        result = analyze(_cohort(), bootstrap_samples=300)
        learned = result["policies"]["learned"]
        self.assertEqual(result["n_episodes"], 4)
        self.assertEqual(learned["macro_episode_success_rate"], 1)
        self.assertEqual(learned["full_success_rate"], 1)
        self.assertEqual(learned["total_tokens"], 1920)
        self.assertEqual(learned["tokens_per_step"], 120)
        self.assertEqual(learned["model_calls"], 32)
        self.assertEqual(learned["api_cost_usd"], 0)
        self.assertIsNone(learned["total_monetary_cost_usd"])
        self.assertEqual(learned["decision_latency_seconds"]["p50"], 2.5)
        self.assertEqual(learned["decision_latency_seconds"]["p95"], 4)
        self.assertEqual(learned["step_latency_seconds"]["p50"], 3.5)
        self.assertEqual(learned["by_family"]["updates"]["n_episodes"], 2)
        self.assertTrue(result["improvement_gate"]["passed"])
        comparison = result["comparisons"]["learned_vs_summary"]
        self.assertEqual(comparison["delta_percentage_points"], 37.5)
        self.assertEqual((comparison["wins"], comparison["ties"], comparison["losses"]), (4, 0, 0))
        self.assertEqual(comparison["bootstrap_unit"], "paired_episode")

    def test_macro_rate_is_not_micro_rate_for_unequal_episode_sizes(self):
        records = []
        for policy in ("tail", "summary", "learned"):
            records += [_record("short", policy, 1, steps=1), _record("long", policy, 0, steps=9)]
        result = analyze(records, bootstrap_samples=50)
        self.assertEqual(result["policies"]["learned"]["macro_episode_success_rate"], 0.5)
        self.assertEqual(result["policies"]["learned"]["micro_step_success_rate"], 0.1)

    def test_paired_bootstrap_preserves_identical_arm_differences(self):
        result = analyze(_cohort([(0, 1, 1), (0, 3, 3), (1, 4, 4)]), bootstrap_samples=400)
        comparison = result["comparisons"]["learned_vs_summary"]
        self.assertEqual(comparison["ci95_percentage_points"], [0, 0])
        self.assertEqual(comparison["ties"], 3)
        self.assertFalse(result["improvement_gate"]["passed"])
        self.assertGreater(result["comparisons"]["learned_vs_tail"]["ci95_percentage_points"][0], 0)

    def test_single_episode_ci_does_not_invent_step_sample_size(self):
        result = analyze(_cohort([(0, 1, 3)]), bootstrap_samples=100)
        comparison = result["comparisons"]["learned_vs_summary"]
        self.assertEqual(comparison["n_episodes"], 1)
        self.assertEqual(comparison["ci95_percentage_points"], [50, 50])

    def test_whole_episode_bootstrap_uncertainty_with_correlated_steps(self):
        result = analyze(_cohort([(0, 0, 4), (0, 4, 0)]), bootstrap_samples=2000)
        comparison = result["comparisons"]["learned_vs_summary"]
        self.assertEqual(comparison["delta_percentage_points"], 0)
        self.assertEqual(comparison["ci95_percentage_points"], [-100, 100])
        self.assertEqual((comparison["wins"], comparison["ties"], comparison["losses"]), (1, 0, 1))

    def test_order_invariance_determinism_and_input_immutability(self):
        records = _cohort()
        before = copy.deepcopy(records)
        first = analyze(records, bootstrap_samples=100, seed=7)
        second = analyze(list(reversed(records)), bootstrap_samples=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(records, before)

    def test_failure_counts_and_family_comparisons(self):
        result = analyze(_cohort(), bootstrap_samples=100)
        self.assertEqual(result["policies"]["summary"]["failure_counts"], {"missing_context": 6})
        self.assertEqual(set(result["comparisons"]["learned_vs_summary"]["by_family"]), {"updates", "dependencies"})

    def test_missing_pair_policy_duplicate_and_mismatch_are_rejected(self):
        bad_inputs = []
        records = _cohort()
        bad_inputs.append(records[:-1])
        bad_inputs.append([record for record in records if record["policy"] != "tail"])
        bad_inputs.append(records + [copy.deepcopy(records[0])])
        mismatch = copy.deepcopy(records)
        mismatch[2]["family"] = "wrong"
        bad_inputs.append(mismatch)
        mismatch_steps = copy.deepcopy(records)
        mismatch_steps[2] = _record("E0", "learned", 4, steps=5)
        bad_inputs.append(mismatch_steps)
        for records in bad_inputs:
            with self.subTest(records=records), self.assertRaises(ValueError):
                analyze(records, bootstrap_samples=10)

    def test_split_mixing_is_rejected(self):
        records = _cohort()
        records[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "one split"):
            analyze(records, bootstrap_samples=10)

    def test_invalid_counts_and_nonfinite_values_fail_closed(self):
        bad_values = {
            "steps": [0, -1, True, 1.5],
            "successes": [-1, 5, True],
            "valid_actions": [-1, 5, True],
            "unsafe_actions": [-1, 5, True],
            "prompt_tokens": [-1, True, 2.5],
            "completion_tokens": [-1, True, float("inf")],
            "model_calls": [-1, True],
            "memory_truncations": [-1, 5],
            "episode_seconds": [-1, float("nan"), float("inf"), True, 0],
            "decision_seconds": [[], [1, 2, 3, float("nan")], [1, 2, 3, -1]],
            "step_seconds": [[], [1, 1, 1, 1]],
            "failures": [{"missing_context": 1}, {"missing_context": -1}, {"missing_context": True}],
            "family": ["", None],
        }
        for field, values in bad_values.items():
            for value in values:
                records = _cohort()
                records[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    analyze(records, bootstrap_samples=10)

    def test_parameter_validation(self):
        for samples in (0, -1, True, 2.5):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                analyze(_cohort(), bootstrap_samples=samples)
        with self.assertRaises(ValueError):
            analyze([])
        with self.assertRaises(ValueError):
            analyze(_cohort(), candidate="summary")
        with self.assertRaises(ValueError):
            analyze(_cohort(), seed=True)

    def test_report_exposes_limits_and_unknown_total_cost(self):
        report = render_report(analyze(_cohort(), bootstrap_samples=100))
        for phrase in ("synthetic", "quantized", "no weight updates", "hardware", "electricity", "not zero", "other domains", "paired episode", "training and policy-search"):
            self.assertIn(phrase, report.lower())
        self.assertIn("learned", report)
        self.assertIn("summary", report)
        self.assertIn("tail", report)
        self.assertIn("+37.50 pp", report)
        self.assertIn("**PASS**", report)

    def test_report_does_not_claim_gain_when_only_tail_is_beaten(self):
        report = render_report(analyze(_cohort([(0, 3, 3), (0, 2, 2)]), bootstrap_samples=100))
        self.assertIn("**NOT DEMONSTRATED**", report)

    def test_report_includes_search_cost_selection_and_verification(self):
        records = _cohort()
        records[0]["unsafe_booked_cents"] = 1250
        result = analyze(records, bootstrap_samples=100)
        result["selected_policy"] = {"name": "repair_proposed", "kind": "learned", "instruction": "Keep evidence."}
        result["verification"] = {"status": "PASS", "verified_steps": 128, "scope": "paired action replay"}
        result["training_and_search_cost"] = {"model_calls": 12, "prompt_tokens": 1000, "completion_tokens": 200, "wall_seconds": 90.5, "api_cost_usd": 0.0, "total_monetary_cost": None}
        report = render_report(result)
        self.assertEqual(result["policies"]["tail"]["unsafe_booked_cents"], 1250)
        for phrase in ("repair_proposed", "verified steps: 128", "paired action replay", "12 model calls", "1200 total", "90.500", "1250 fictional cents", "include rejected calls", "Combined training/search and evaluation", "including candidates not selected"):
            self.assertIn(phrase, report)

    def test_unchanged_summary_choice_cannot_be_presented_as_learned_gain(self):
        result = analyze(_cohort(), bootstrap_samples=100)
        result["selected_policy"] = {"name": "summary", "kind": "summary", "instruction": None}
        report = render_report(result)
        self.assertIn("**NOT DEMONSTRATED**", report)
        self.assertIn("no memory-policy change was selected", report)
        self.assertIn("unchanged baseline", report)


if __name__ == "__main__":
    unittest.main()
