"""Verify legacy numerical fits at their serialized precision, failing closed."""
from __future__ import annotations

import json
import math
import subprocess
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = "results/rcwt_curve_fits.json"


def _tolerance(path: tuple[object, ...]) -> Decimal:
    if len(path) >= 4 and path[1] == "fits":
        field = path[3]
        if len(path) == 4 and field in ("aic", "c_star"):
            return Decimal("0.0001")
        if len(path) == 4 and field in ("r_squared", "rmse"):
            return Decimal("0.00001")
        if len(path) == 5 and field in ("params", "predicted"):
            return Decimal("0.00001")
    return Decimal(0)


def _compare(expected: object, actual: object, path: tuple[object, ...]) -> None:
    where = ".".join(map(str, path)) or "root"
    if type(expected) is not type(actual):
        raise ValueError(f"{where}: JSON type changed")
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f"{where}: object keys changed")
        for key in expected:
            _compare(expected[key], actual[key], (*path, key))
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError(f"{where}: list length changed")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, (*path, index))
    elif type(expected) in (int, float):
        if not math.isfinite(expected) or not math.isfinite(actual):
            raise ValueError(f"{where}: non-finite number")
        tolerance = _tolerance(path)
        if abs(Decimal(str(expected)) - Decimal(str(actual))) > tolerance:
            raise ValueError(f"{where}: difference exceeds {tolerance}")
    elif expected != actual:
        raise ValueError(f"{where}: value changed")


def compare_curve_fit_artifacts(expected: dict, actual: dict) -> None:
    """Require exact structure/inputs, bounded fit rounding, and stable AIC winners."""
    _compare(expected, actual, ())
    for model in expected:
        old_fits, new_fits = expected[model]["fits"], actual[model]["fits"]
        old_winner = min(old_fits, key=lambda name: old_fits[name]["aic"])
        new_winner = min(new_fits, key=lambda name: new_fits[name]["aic"])
        if old_winner != new_winner:
            raise ValueError(f"{model}: best AIC model changed")


def main() -> int:
    try:
        baseline = subprocess.run(
            ["git", "show", f"HEAD:{ARTIFACT}"], cwd=ROOT,
            check=True, capture_output=True, text=True, encoding="utf-8",
        ).stdout
        expected = json.loads(baseline)
        actual = json.loads((ROOT / ARTIFACT).read_text(encoding="utf-8"))
        compare_curve_fit_artifacts(expected, actual)
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: legacy curve-fit artifact: {exc}")
        return 1
    print("PASS: legacy fit structure, inputs, finite values, published precision and AIC winners")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
