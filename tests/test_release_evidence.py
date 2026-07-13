"""Evidence-level regression tests for the values reported in the paper."""

from __future__ import annotations

import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

from analyze_new_results import fit_logistic, load_trials, pooled_by_target
from rcwt_intact_scoring import EXPECTED, score_response

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class MainEvidenceTests(unittest.TestCase):
    """Lock the observed scores and corrected derived midpoints."""

    def test_q90_observations_are_unchanged(self) -> None:
        rows = load_trials(RESULTS / "rcwt_controlled.csv")
        q90 = [row for row in rows if float(row["proportion"]) == 0.90]
        self.assertEqual({int(row["coordination_tokens"]) for row in q90}, {3383})
        self.assertEqual({int(row["reasoning_tokens"]) for row in q90}, {376})

        scores: dict[str, list[float]] = defaultdict(list)
        for row in q90:
            scores[row["model"]].append(float(row["mean_score_effective"]))
        expected = {
            "gemini-2.0-flash": 0.659375,
            "claude-haiku-4-5-20251001": 0.665625,
            "gpt-4.1-mini": 0.853125,
        }
        for model, expected_mean in expected.items():
            self.assertAlmostEqual(sum(scores[model]) / len(scores[model]), expected_mean)

    def test_corrected_full_budget_midpoints(self) -> None:
        rows = load_trials(RESULTS / "rcwt_controlled.csv")
        expected = {
            "gemini-2.0-flash": 0.837563,
            "claude-haiku-4-5-20251001": 0.841269,
            "gpt-4.1-mini": 0.861254,
        }
        for model, expected_p0 in expected.items():
            fit = fit_logistic(pooled_by_target(rows, model, 4096))
            self.assertAlmostEqual(fit["p0"], expected_p0, places=6)


class AblationEvidenceTests(unittest.TestCase):
    """Verify the intact-task result under strict per-field scoring."""

    def test_all_saved_calls_pass_strict_scoring(self) -> None:
        path = RESULTS / "intact_ablation" / "rcwt_intact_ablation_responses.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        scores = [score_response(str(record["response"])) for record in records]
        self.assertEqual(len(scores), 150)
        self.assertEqual(sum(sum(score.values()) for score in scores), 1200)
        self.assertTrue(all(set(score) == set(EXPECTED) for score in scores))


class PackEvidenceTests(unittest.TestCase):
    """Keep the reported untruncated Haiku/GSM8K exception visible."""

    def test_haiku_gsm8k_exception(self) -> None:
        path = RESULTS / "cross_benchmark_pack_summary_with_drop.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        selected = {
            float(row["proportion"]): row
            for row in rows
            if row["benchmark"] == "gsm8k_pack"
            and row["model"] == "claude-haiku-4-5-20251001"
        }
        self.assertEqual(float(selected[0.0]["accuracy"]), 0.60)
        self.assertEqual(float(selected[0.25]["accuracy"]), 0.40)
        self.assertEqual(float(selected[0.50]["accuracy"]), 0.38)
        self.assertEqual(float(selected[0.25]["task_truncated_rate"]), 0.0)
        self.assertEqual(float(selected[0.50]["task_truncated_rate"]), 0.0)

    def test_invalid_distraction_probe_is_not_released(self) -> None:
        self.assertFalse((ROOT / "src" / "rcwt_task3.py").exists())
        self.assertFalse((RESULTS / "task3").exists())


if __name__ == "__main__":
    unittest.main()
