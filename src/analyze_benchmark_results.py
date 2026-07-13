"""Create pooled cross-benchmark RCWT summary tables from aggregate artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger("analyze_benchmark_results")


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score interval for a binomial hit rate."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def logistic(proportion: np.ndarray, r0: float, k: float, midpoint: float) -> np.ndarray:
    """Three-parameter logistic decay used by the RCWT paper."""
    return r0 / (1.0 + np.exp(k * (np.asarray(proportion) - midpoint)))


def fit_logistic(points: list[dict[str, Any]]) -> dict[str, float | str]:
    """Fit the logistic decay curve to pooled proportion-level accuracy."""
    proportions = np.array([float(point["proportion"]) for point in points])
    scores = np.array([float(point["accuracy"]) for point in points])
    try:
        params, _ = curve_fit(
            logistic,
            proportions,
            scores,
            p0=[max(scores), 12.0, 0.85],
            bounds=([0.0, 0.0, 0.0], [1.2, 200.0, 1.0]),
            maxfev=20_000,
            method="trf",
        )
        predicted = logistic(proportions, *params)
        ss_res = float(np.sum((scores - predicted) ** 2))
        ss_tot = float(np.sum((scores - np.mean(scores)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "r0": round(float(params[0]), 5),
            "k": round(float(params[1]), 5),
            "midpoint": round(float(params[2]), 5),
            "r_squared": round(r2, 5),
            "rmse": round(float(np.sqrt(np.mean((scores - predicted) ** 2))), 5),
        }
    except Exception as exc:
        return {"error": str(exc)}


def load_aggregates(path: Path) -> dict[str, Any]:
    """Load a benchmark aggregate JSON artifact."""
    return json.loads(path.read_text())


def pool_by_proportion(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool coord_first and reason_first cells per proportion."""
    grouped: dict[float, list[dict[str, Any]]] = {}
    for aggregate in aggregates:
        grouped.setdefault(float(aggregate["proportion"]), []).append(aggregate)

    pooled: list[dict[str, Any]] = []
    for proportion, cells in sorted(grouped.items()):
        n = sum(int(cell["n"]) for cell in cells)
        hits = sum(round(float(cell["accuracy"]) * int(cell["n"])) for cell in cells)
        accuracy = hits / n if n else 0.0
        ci_low, ci_high = wilson_ci(hits, n)
        pooled.append(
            {
                "proportion": proportion,
                "n": n,
                "hits": hits,
                "accuracy": round(accuracy, 5),
                "ci95": [round(ci_low, 5), round(ci_high, 5)],
                "task_truncated_rate": round(
                    sum(float(cell["task_truncated_rate"]) * int(cell["n"]) for cell in cells)
                    / n,
                    5,
                )
                if n
                else 0.0,
            }
        )
    return pooled


def summarize(paths: list[Path]) -> list[dict[str, Any]]:
    """Summarize each aggregate artifact into pooled points plus logistic fit."""
    summaries: list[dict[str, Any]] = []
    for path in paths:
        raw = load_aggregates(path)
        aggregates = raw["aggregates"]
        model = str(aggregates[0]["model"]) if aggregates else "unknown"
        provider = str(aggregates[0]["provider"]) if aggregates else "unknown"
        pooled = pool_by_proportion(aggregates)
        summaries.append(
            {
                "benchmark": raw["benchmark"],
                "model": model,
                "provider": provider,
                "records": raw["records"],
                "total_metering_redacted": raw["total_metering_redacted"],
                "pooled": pooled,
                "logistic_fit": fit_logistic(pooled),
            }
        )
    return summaries


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    """Write a flat proportion-level CSV table."""
    fieldnames = [
        "benchmark",
        "model",
        "provider",
        "proportion",
        "n",
        "accuracy",
        "ci95_low",
        "ci95_high",
        "task_truncated_rate",
        "logistic_midpoint",
        "logistic_k",
        "logistic_r_squared",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            fit = summary["logistic_fit"]
            for point in summary["pooled"]:
                writer.writerow(
                    {
                        "benchmark": summary["benchmark"],
                        "model": summary["model"],
                        "provider": summary["provider"],
                        "proportion": point["proportion"],
                        "n": point["n"],
                        "accuracy": point["accuracy"],
                        "ci95_low": point["ci95"][0],
                        "ci95_high": point["ci95"][1],
                        "task_truncated_rate": point["task_truncated_rate"],
                        "logistic_midpoint": fit.get("midpoint"),
                        "logistic_k": fit.get("k"),
                        "logistic_r_squared": fit.get("r_squared"),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    summaries = summarize(args.aggregate)
    args.output_json.write_text(json.dumps(summaries, indent=2))
    write_csv(args.output_csv, summaries)
    logger.info(
        "analysis_saved summaries=%d output_json=%s output_csv=%s",
        len(summaries),
        args.output_json,
        args.output_csv,
    )


if __name__ == "__main__":
    main()
