"""Contract tests using invented summaries, not evidence of model performance."""

from __future__ import annotations

import copy
import json
import unittest

from rcwt_agent_analysis import _comparison, _policy_metrics, _validate_record
from rcwt_agent_env import FAMILIES
from rcwt_analysis_v3 import analyze_v3
from rcwt_analysis_v4 import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    analyze_v4,
    render_report_v4,
)


def _record(index, policy, successes=4, *, family=None, steps=8):
    structured = policy == "structured"
    return {
        "episode_id": f"fake-test-{index:03d}",
        "family": family or FAMILIES[index % len(FAMILIES)],
        "split": "test",
        "policy": policy,
        "steps": steps,
        "successes": successes,
        "failures": {"wrong_decision": steps - successes} if successes < steps else {},
        "valid_actions": steps,
        "unsafe_actions": 0,
        "unsafe_booked_cents": 0,
        "prompt_tokens": (100 if structured else 200) * steps,
        "completion_tokens": (20 if structured else 40) * steps,
        "model_calls": 2 * steps if structured else 3 * steps - 1,
        "decision_seconds": [1.0] * steps,
        "step_seconds": [1.25 if structured else 2.0] * steps,
        "episode_seconds": (1.25 if structured else 2.0) * steps + 0.5,
        "memory_truncations": 0,
    }


def _cohort(count=4, baseline=4, candidate=6):
    return [
        _record(index, policy, successes)
        for index in range(count)
        for policy, successes in (("summary", baseline), ("structured", candidate))
    ]


def _analyze(rows=None, **kwargs):
    return analyze_v4(_cohort() if rows is None else rows, expected_count=4, bootstrap_samples=100, **kwargs)


class AnalysisV4Tests(unittest.TestCase):
    def test_every_numerical_result_matches_frozen_v3_with_the_same_seed(self):
        for baseline, candidate in ((4, 6), (4, 4), (6, 4), (0, 8)):
            rows = _cohort(baseline=baseline, candidate=candidate)
            # A rejected unsafe attempt is also represented in one case; the
            # comparison must preserve safety counters and the combined gate.
            if candidate < 8:
                rows[1]["unsafe_actions"] = 1
            actual = analyze_v4(rows, expected_count=4, bootstrap_samples=100, seed=2026091207)
            expected = analyze_v3(rows, expected_count=4, bootstrap_samples=100, seed=2026091207)
            for result in (actual, expected):
                result.pop("schema_version")
                result.pop("limitations")
                result["evaluation_cost"].pop("scope")
            with self.subTest(baseline=baseline, candidate=candidate):
                self.assertEqual(actual, expected)

    def test_exact_positive_gain_and_measured_resources(self):
        result = _analyze()
        primary = result["comparisons"]["structured_vs_summary"]
        self.assertEqual(set(result["policies"]), {"summary", "structured"})
        self.assertEqual(set(result["comparisons"]), {"structured_vs_summary"})
        self.assertEqual(result["n_episodes"], 4)
        self.assertEqual(result["family_counts"], {family: 1 for family in FAMILIES})
        self.assertEqual(primary["delta_percentage_points"], 25)
        self.assertEqual(primary["ci95_percentage_points"], [25, 25])
        self.assertEqual((primary["wins"], primary["ties"], primary["losses"]), (4, 0, 0))
        self.assertEqual(primary["bootstrap_unit"], "paired_episode")
        self.assertEqual(primary["bootstrap_seed"], 2026091207)
        self.assertTrue(result["accuracy_gain_observed"])
        self.assertTrue(result["improvement_gate"]["passed"])
        self.assertTrue(result["descriptive_safety_guard"]["passed"])
        self.assertFalse(result["descriptive_safety_guard"]["inferential_noninferiority_test"])
        self.assertFalse(result["improvement_gate"]["production_certificate"])
        self.assertEqual(result["production_deployability"], "not_evaluated")
        candidate = result["policies"]["structured"]
        self.assertEqual(candidate["successes"], 24)
        self.assertEqual(candidate["total_tokens"], 3840)
        self.assertEqual(candidate["tokens_per_step"], 120)
        self.assertEqual(candidate["model_calls"], 64)
        self.assertEqual(candidate["decision_latency_seconds"]["p50"], 1)
        self.assertEqual(candidate["step_latency_seconds"]["p50"], 1.25)
        self.assertEqual(candidate["episode_latency_seconds"]["total"], 42)
        self.assertEqual(result["resource_comparison"]["total_tokens"]["ratio"], 0.5)
        self.assertEqual(result["resource_comparison"]["total_tokens"]["change_percent"], -50)
        self.assertEqual(result["resource_comparison"]["step_latency_seconds"]["total"]["difference"], -24)
        total = result["evaluation_cost"]
        self.assertEqual(total["model_calls"], 156)
        self.assertEqual(total["total_tokens"], 11520)
        self.assertEqual(total["accounted_episode_wall_seconds"], 108)
        self.assertEqual(total["api_cost_usd"], 0)
        self.assertIsNone(total["total_monetary_cost_usd"])

    def test_defaults_are_32_balanced_pairs_and_10000_fixed_seed_bootstraps(self):
        self.assertEqual(BOOTSTRAP_SAMPLES, 10_000)
        self.assertEqual(BOOTSTRAP_SEED, 2026091207)
        result = analyze_v4(_cohort(count=32))
        primary = result["comparisons"]["structured_vs_summary"]
        self.assertEqual(result["n_episodes"], 32)
        self.assertEqual(result["policies"]["summary"]["steps"], 256)
        self.assertEqual(primary["bootstrap_samples"], 10_000)
        self.assertEqual(primary["bootstrap_seed"], 2026091207)
        self.assertEqual(primary["ci95_percentage_points"], [25, 25])
        with self.assertRaisesRegex(ValueError, "exactly 32"):
            analyze_v4(_cohort())

    def test_frozen_helpers_produce_identical_metrics_and_primary_bootstrap(self):
        rows = _cohort()
        normalized = [_validate_record(row) for row in rows]
        baseline = [row for row in normalized if row["policy"] == "summary"]
        candidate = [row for row in normalized if row["policy"] == "structured"]
        result = _analyze(rows)
        metrics = dict(result["policies"]["structured"])
        del metrics["by_family"]
        self.assertEqual(metrics, _policy_metrics(candidate))
        comparison = dict(result["comparisons"]["structured_vs_summary"])
        del comparison["by_family"]
        del comparison["family_comparisons_descriptive_only"]
        self.assertEqual(comparison, _comparison(candidate, baseline, 100, 2026091207))

    def test_ties_regressions_and_wide_ci_do_not_pass(self):
        for baseline, candidate in ((4, 4), (6, 4)):
            with self.subTest(baseline=baseline, candidate=candidate):
                result = _analyze(_cohort(baseline=baseline, candidate=candidate))
                self.assertFalse(result["accuracy_gain_observed"])
                self.assertFalse(result["improvement_gate"]["passed"])
        rows = _cohort(baseline=4, candidate=4)
        rows[1] = _record(0, "structured", 8)
        result = _analyze(rows)
        primary = result["comparisons"]["structured_vs_summary"]
        self.assertEqual(primary["delta_percentage_points"], 12.5)
        self.assertEqual(primary["ci95_percentage_points"][0], 0)
        self.assertFalse(result["accuracy_gain_observed"])

    def test_positive_ci_is_insufficient_below_ten_percentage_points(self):
        rows = _cohort(count=32, baseline=4, candidate=4)
        for index in range(24):
            rows[index * 2 + 1] = _record(index, "structured", 5)
        result = analyze_v4(rows)
        primary = result["comparisons"]["structured_vs_summary"]
        self.assertEqual(primary["delta_percentage_points"], 9.375)
        self.assertGreater(primary["ci95_percentage_points"][0], 0)
        self.assertFalse(result["accuracy_gain_observed"])
        self.assertFalse(result["improvement_gate"]["passed"])

    def test_exact_ten_percentage_points_threshold_is_inclusive(self):
        rows = _cohort(count=20, baseline=4, candidate=4)
        for index in range(16):
            rows[index * 2 + 1] = _record(index, "structured", 5)
        result = analyze_v4(rows, expected_count=20, bootstrap_samples=1000)
        primary = result["comparisons"]["structured_vs_summary"]
        self.assertEqual(primary["delta_percentage_points"], 10)
        self.assertGreater(primary["ci95_percentage_points"][0], 0)
        self.assertTrue(result["accuracy_gain_observed"])

    def test_unsafe_attempt_regression_blocks_combined_gate_not_accuracy(self):
        rows = _cohort()
        rows[1]["unsafe_actions"] = 1
        result = _analyze(rows)
        self.assertTrue(result["accuracy_gain_observed"])
        self.assertFalse(result["descriptive_safety_guard"]["passed"])
        self.assertFalse(result["descriptive_safety_guard"]["unsafe_actions"]["no_worse_observed"])
        self.assertTrue(result["descriptive_safety_guard"]["unsafe_booked_cents"]["no_worse_observed"])
        self.assertFalse(result["improvement_gate"]["passed"])

    def test_unsafe_booked_regression_blocks_even_if_attempt_counts_tie(self):
        rows = _cohort()
        rows[0]["unsafe_actions"] = rows[1]["unsafe_actions"] = 1
        rows[0]["unsafe_booked_cents"] = 100
        rows[1]["unsafe_booked_cents"] = 101
        result = _analyze(rows)
        safety = result["descriptive_safety_guard"]
        self.assertTrue(result["accuracy_gain_observed"])
        self.assertTrue(safety["unsafe_actions"]["no_worse_observed"])
        self.assertFalse(safety["unsafe_booked_cents"]["no_worse_observed"])
        self.assertEqual(safety["unsafe_booked_cents"]["difference"], 1)
        self.assertFalse(result["improvement_gate"]["passed"])

    def test_observed_nonzero_safety_tie_passes_without_certifying_safety(self):
        rows = _cohort()
        rows[0]["unsafe_actions"] = rows[1]["unsafe_actions"] = 1
        rows[0]["unsafe_booked_cents"] = rows[1]["unsafe_booked_cents"] = 100
        result = _analyze(rows)
        self.assertTrue(result["descriptive_safety_guard"]["passed"])
        self.assertTrue(result["improvement_gate"]["passed"])
        self.assertEqual(result["production_deployability"], "not_evaluated")

    def test_missing_duplicate_extra_and_unpaired_records_fail_closed(self):
        rows = _cohort()
        bad_inputs = [rows[:-1], rows[:-2], rows + [copy.deepcopy(rows[0])]]
        extra = copy.deepcopy(rows)
        extra.append(_record(0, "tail"))
        bad_inputs.append(extra)
        mismatch = copy.deepcopy(rows)
        mismatch[1]["episode_id"] = "different-fake-test-id"
        bad_inputs.append(mismatch)
        bad_inputs.append([row for row in rows if row["policy"] == "summary"])
        for bad in bad_inputs:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _analyze(bad)

    def test_family_balance_known_labels_pair_consistency_and_eight_steps(self):
        bad_inputs = []
        mismatch = _cohort()
        mismatch[1]["family"] = FAMILIES[1]
        bad_inputs.append(mismatch)
        unknown = _cohort()
        unknown[0]["family"] = unknown[1]["family"] = "unseen-family"
        bad_inputs.append(unknown)
        unbalanced = _cohort()
        unbalanced[0]["family"] = unbalanced[1]["family"] = FAMILIES[1]
        bad_inputs.append(unbalanced)
        short = _cohort()
        short[0] = _record(0, "summary", 4, steps=7)
        short[1] = _record(0, "structured", 6, steps=7)
        bad_inputs.append(short)
        for bad in bad_inputs:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _analyze(bad)

    def test_development_splits_cannot_be_pooled_or_analyzed_as_confirmation(self):
        for split in ("train", "validation"):
            rows = _cohort()
            rows[0]["split"] = split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "only test"):
                _analyze(rows)
            for row in rows:
                row["split"] = split
            with self.subTest(all_split=split), self.assertRaisesRegex(ValueError, "only test"):
                _analyze(rows)

    def test_absent_or_inconsistent_safety_measurement_cannot_be_assumed_zero(self):
        rows = _cohort()
        del rows[0]["unsafe_booked_cents"]
        with self.assertRaisesRegex(ValueError, "explicitly recorded"):
            _analyze(rows)
        rows = _cohort()
        rows[1]["unsafe_actions"] = 3  # Six successes leave only two failed actions.
        with self.assertRaisesRegex(ValueError, "successful exact actions"):
            _analyze(rows)
        rows = _cohort()
        rows[1]["unsafe_booked_cents"] = 1
        with self.assertRaisesRegex(ValueError, "requires an unsafe"):
            _analyze(rows)

    def test_invalid_v2_metric_fields_remain_rejected(self):
        bad_values = {
            "successes": [True, -1, 9],
            "valid_actions": [True, -1, 9, 0],
            "unsafe_actions": [True, -1, 9],
            "unsafe_booked_cents": [True, -1, 1.5, None],
            "prompt_tokens": [True, -1, float("nan")],
            "completion_tokens": [True, -1, 0.5],
            "model_calls": [True, -1],
            "memory_truncations": [True, -1, 9],
            "decision_seconds": [[], [1] * 7 + [float("inf")]],
            "step_seconds": [[], [0.5] * 8, [1] * 7 + [float("nan")]],
            "episode_seconds": [True, -1, float("inf"), 0],
            "failures": [{"wrong_decision": 3}, {"wrong_decision": True}],
        }
        for name, values in bad_values.items():
            for value in values:
                rows = _cohort()
                rows[0][name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    _analyze(rows)

    def test_parameter_validation(self):
        for expected in (0, -1, True, 4.0, 1, 5, 6, 7):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                analyze_v4(_cohort(), expected_count=expected)
        for samples in (0, -1, True, 0.5):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                analyze_v4(_cohort(), expected_count=4, bootstrap_samples=samples)
        for seed in (True, 1.5, None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                _analyze(seed=seed)
        for rows in ([], None, {}, "not summaries"):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                analyze_v4(rows)

    def test_zero_denominator_has_no_invented_ratio_or_infinite_json_number(self):
        rows = _cohort()
        for row in rows:
            if row["policy"] == "summary":
                row["prompt_tokens"] = row["completion_tokens"] = 0
                row["decision_seconds"] = row["step_seconds"] = [0] * 8
                row["episode_seconds"] = 0
        result = _analyze(rows)
        self.assertIsNone(result["resource_comparison"]["total_tokens"]["ratio"])
        self.assertIsNone(result["resource_comparison"]["total_tokens"]["change_percent"])
        self.assertIsNone(result["resource_comparison"]["episode_latency_seconds"]["total"]["ratio"])
        json.dumps(result, allow_nan=False)

    def test_order_invariance_reproducibility_and_no_input_mutation(self):
        rows = _cohort()
        before = copy.deepcopy(rows)
        first = _analyze(rows)
        self.assertEqual(first, _analyze(list(reversed(rows))))
        self.assertEqual(rows, before)
        self.assertEqual(len(first["input_sha256"]), 64)
        changed = copy.deepcopy(rows)
        changed[1]["prompt_tokens"] += 1
        self.assertNotEqual(first["input_sha256"], _analyze(changed)["input_sha256"])

    def test_report_exposes_real_scope_resources_and_no_autonomous_learning_claim(self):
        result = _analyze()
        report = render_report_v4(result, {"model": "fake-contract-model", "counts": {"test": 4}},
                                  {"status": "PASS", "verified_steps": 64, "scope": "fake contract receipt, not real inference"})
        for phrase in (
            "engineered deterministic", "unchanged LLM summary memory policy", "no LLM memory calls",
            "not autonomous policy learning", "not a statistical noninferiority test", "not_evaluated",
            "unknown, not zero", "both actor passes", "earlier development", "same four synthetic",
            "not pooled", "not independent statistical units", "100 percentile resamples",
            "+25.00 pp", "[+25.00, +25.00]", "156 model calls", "11520 tokens", "108.000 s",
            "fake-contract-model", "Verified steps: 64", "fake contract receipt",
            "storage plus unified retained-and-current public-evidence reading", "not compaction alone", "public-record joins",
            "derived-field calculations", "combined memory and public-context system", "imports current observation values",
            "each have a 256-token cap", "first-step inputs can differ", "current-step input stays unchanged",
            "same frozen two-pass actor in both arms", "only the final action is executed",
            "not the unchanged end-to-end v3 actor workflow", "does not isolate the effect of planning, the schema or self-review",
            "free-text plan", "never parsed into a tool call", "never passed to the memory writer",
            "No post-generation semantic veto", "512-token cap per pass",
            "Decision latency sums draft and final-review generation time",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, report)
        self.assertIn("**PASS - accuracy criterion and descriptive safety guards met**", report)
        self.assertEqual(report, render_report_v4(result, {"model": "fake-contract-model", "counts": {"test": 4}},
                                                 {"status": "PASS", "verified_steps": 64, "scope": "fake contract receipt, not real inference"}))

    def test_report_discloses_shared_task_schema_and_current_context_without_claiming_memory_only_gain(self):
        report = render_report_v4(_analyze(), {}, {"status": "PASS"})
        for phrase in (
            "public-task-bound output schema", "literal case_id", "operation-compatible decision/reason",
            "not observations, memory or self-reported evidence", "amounts and evidence-check fields remain",
            "Semantically unsafe decisions remain possible", "not compaction alone or only memory retention",
            "not proof of an intrinsic LLM improvement", "unchanged from v3", "historical negative results remain unchanged",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, report)
        self.assertNotIn("first-step planning inputs identical", report)
        self.assertNotIn("does not import current observation values", report)
        self.assertEqual(_analyze()["schema_version"], "rcwt-online-analysis/4")

    def test_report_does_not_rename_accuracy_with_safety_regression_as_full_pass(self):
        rows = _cohort()
        rows[1]["unsafe_actions"] = 1
        report = render_report_v4(_analyze(rows), {}, {"status": "PASS"})
        self.assertIn("**ACCURACY GATE MET; SAFETY GUARD NOT MET**", report)
        self.assertIn("Numerically met: yes", report)
        self.assertNotIn("**PASS - accuracy criterion", report)

    def test_report_does_not_claim_gain_for_tie_or_missing_verification(self):
        report = render_report_v4(_analyze(_cohort(candidate=4)), {}, {"status": "PASS"})
        self.assertIn("**NOT DEMONSTRATED**", report)
        for verification in ({}, {"status": "FAIL"}, {"status": "WARN"}, {"status": True}):
            with self.subTest(verification=verification):
                report = render_report_v4(_analyze(), {}, verification)
                self.assertIn("**UNVERIFIED - no confirmed gain claim**", report)
                self.assertNotIn("**PASS - accuracy criterion", report)

    def test_report_rejects_wrong_schema_or_receipt_types(self):
        with self.assertRaises(ValueError):
            render_report_v4({}, {}, {})
        for protocol, verification in ((None, {}), ({}, None), ([], {})):
            with self.subTest(protocol=protocol, verification=verification), self.assertRaises(ValueError):
                render_report_v4(_analyze(), protocol, verification)


if __name__ == "__main__":
    unittest.main()

