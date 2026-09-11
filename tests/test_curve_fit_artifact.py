"""Check that numerical portability does not permit substantive result drift."""
from __future__ import annotations

import copy
import json
import unittest
from decimal import Decimal
from pathlib import Path

from verify_curve_fit_artifact import compare_curve_fit_artifacts

ROOT = Path(__file__).resolve().parents[1]


class CurveFitArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = json.loads(
            (ROOT / "results/rcwt_curve_fits.json").read_text(encoding="utf-8")
        )
        self.model = next(iter(self.expected))

    def test_unchanged_artifact_passes(self) -> None:
        compare_curve_fit_artifacts(self.expected, copy.deepcopy(self.expected))

    def test_one_published_unit_passes_but_two_fail(self) -> None:
        paths = [
            (("params", "R₀"), "0.00001"),
            (("predicted", 0), "0.00001"),
            (("r_squared",), "0.00001"),
            (("rmse",), "0.00001"),
            (("aic",), "0.0001"),
            (("c_star",), "0.0001"),
        ]
        for path, unit in paths:
            for units in (1, 2):
                with self.subTest(path=path, units=units):
                    actual = copy.deepcopy(self.expected)
                    value = actual[self.model]["fits"]["power"]
                    for key in path[:-1]:
                        value = value[key]
                    key = path[-1]
                    value[key] = float(Decimal(str(value[key])) + Decimal(unit) * units)
                    if units == 1:
                        compare_curve_fit_artifacts(self.expected, actual)
                    else:
                        with self.assertRaisesRegex(ValueError, "exceeds"):
                            compare_curve_fit_artifacts(self.expected, actual)

    def test_empirical_inputs_have_no_numerical_tolerance(self) -> None:
        actual = copy.deepcopy(self.expected)
        actual[self.model]["empirical"]["scores"][0] += 0.000001
        with self.assertRaisesRegex(ValueError, "exceeds 0"):
            compare_curve_fit_artifacts(self.expected, actual)

    def test_schema_change_fails(self) -> None:
        actual = copy.deepcopy(self.expected)
        actual[self.model]["fits"]["power"]["unexpected"] = 0
        with self.assertRaisesRegex(ValueError, "keys changed"):
            compare_curve_fit_artifacts(self.expected, actual)

    def test_missing_prediction_fails(self) -> None:
        actual = copy.deepcopy(self.expected)
        actual[self.model]["fits"]["power"]["predicted"].pop()
        with self.assertRaisesRegex(ValueError, "length changed"):
            compare_curve_fit_artifacts(self.expected, actual)

    def test_provider_identity_change_fails(self) -> None:
        actual = copy.deepcopy(self.expected)
        actual[self.model]["provider"] = "different provider"
        with self.assertRaisesRegex(ValueError, "value changed"):
            compare_curve_fit_artifacts(self.expected, actual)

    def test_large_piecewise_location_drift_fails(self) -> None:
        actual = copy.deepcopy(self.expected)
        actual[self.model]["fits"]["piecewise"]["c_star"] += 0.07
        with self.assertRaisesRegex(ValueError, "exceeds"):
            compare_curve_fit_artifacts(self.expected, actual)

    def test_nonfinite_and_boolean_fit_outputs_fail(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(value=value):
                actual = copy.deepcopy(self.expected)
                actual[self.model]["fits"]["power"]["rmse"] = value
                with self.assertRaises(ValueError):
                    compare_curve_fit_artifacts(self.expected, actual)

    def test_aic_winner_change_fails_even_inside_rounding_tolerance(self) -> None:
        expected = copy.deepcopy(self.expected)
        expected[self.model]["fits"]["logistic"]["aic"] = -100.0
        expected[self.model]["fits"]["piecewise"]["aic"] = -99.9999
        actual = copy.deepcopy(expected)
        actual[self.model]["fits"]["logistic"]["aic"] = -99.9999
        actual[self.model]["fits"]["piecewise"]["aic"] = -100.0
        with self.assertRaisesRegex(ValueError, "best AIC model changed"):
            compare_curve_fit_artifacts(expected, actual)


if __name__ == "__main__":
    unittest.main()
