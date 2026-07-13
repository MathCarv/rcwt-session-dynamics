"""Summarize RCWT high-overhead and window-scaling release trials."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from rcwt_statistics import stable_seed, stratified_bootstrap_mean_ci

LOGGER = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODEL_ORDER = [
    "gemini-2.0-flash",
    "claude-haiku-4-5-20251001",
    "gpt-4.1-mini",
]
MODEL_LABELS = {
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gpt-4.1-mini": "GPT-4.1-mini",
}
WINDOW_TRIALS = {
    4096: RESULTS_DIR / "rcwt_controlled.csv",
    8192: RESULTS_DIR / "w8192" / "rcwt_controlled.csv",
    16384: RESULTS_DIR / "w16384" / "rcwt_controlled.csv",
}


@dataclass(frozen=True)
class PooledCell:
    """Call-level summary for one model and runner target ratio."""

    target_ratio: float
    coordination_share: float
    mean: float
    n_calls: int
    ci95: tuple[float, float]


def load_trials(path: Path) -> list[dict[str, str]]:
    """Load one released trial CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pooled_by_target(
    rows: list[dict[str, str]],
    model: str,
    total_budget: int,
) -> dict[float, PooledCell]:
    """Pool prompt orders with a stratified call-level bootstrap interval."""
    grouped: dict[float, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["model"] == model:
            grouped[float(row["proportion"])][row["order"]].append(row)

    result: dict[float, PooledCell] = {}
    for target_ratio, by_order in sorted(grouped.items()):
        all_rows = [row for order_rows in by_order.values() for row in order_rows]
        shares = {
            int(row["coordination_tokens"]) / int(row["total_context_tokens"])
            for row in all_rows
        }
        if len(shares) != 1:
            raise ValueError(f"inconsistent coordination shares for {model} at q={target_ratio}")
        strata = {
            order: [float(row["mean_score_effective"]) for row in order_rows]
            for order, order_rows in by_order.items()
        }
        mean, low, high = stratified_bootstrap_mean_ci(
            strata,
            seed=stable_seed("window-summary", total_budget, model, target_ratio),
        )
        result[target_ratio] = PooledCell(
            target_ratio=target_ratio,
            coordination_share=shares.pop(),
            mean=mean,
            n_calls=len(all_rows),
            ci95=(low, high),
        )
    return result


def logistic(p: np.ndarray, r0: float, k: float, p0: float) -> np.ndarray:
    """Logistic decay curve over the realized share ``p=c/W``."""
    return r0 / (1 + np.exp(k * (np.asarray(p) - p0)))


def fit_logistic(cells: dict[float, PooledCell]) -> dict[str, float]:
    """Fit a logistic curve to pooled per-target cells."""
    ordered = [cells[target] for target in sorted(cells)]
    shares = np.array([cell.coordination_share for cell in ordered], dtype=float)
    scores = np.array([cell.mean for cell in ordered], dtype=float)
    popt, _ = curve_fit(
        logistic,
        shares,
        scores,
        p0=[1.0, 12.0, 0.85],
        bounds=([0.5, 0.0, 0.4], [1.2, 300.0, 1.0]),
        maxfev=50_000,
        method="trf",
    )
    predicted = logistic(shares, *popt)
    ss_res = float(np.sum((scores - predicted) ** 2))
    ss_tot = float(np.sum((scores - np.mean(scores)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"r0": float(popt[0]), "k": float(popt[1]), "p0": float(popt[2]), "r2": r2}


def analyze_cliff() -> None:
    """Log the canonical W=4096 table and fitted transition."""
    rows = load_trials(WINDOW_TRIALS[4096])
    for model in MODEL_ORDER:
        cells = pooled_by_target(rows, model, 4096)
        fit = fit_logistic(cells)
        LOGGER.info(
            "cliff_fit model=%s p0=%.6f theta_tokens=%.2f r2=%.4f",
            model,
            fit["p0"],
            4096 * (1 - fit["p0"]),
            fit["r2"],
        )
        for target_ratio, cell in cells.items():
            LOGGER.info(
                "cliff_cell model=%s target_ratio=%.3f coordination_share=%.6f "
                "mean=%.6f ci95_low=%.6f ci95_high=%.6f n_calls=%d",
                model,
                target_ratio,
                cell.coordination_share,
                cell.mean,
                cell.ci95[0],
                cell.ci95[1],
                cell.n_calls,
            )


def analyze_window_scaling() -> None:
    """Log fitted full-budget midpoints by model and window size."""
    trials_by_window = {window: load_trials(path) for window, path in WINDOW_TRIALS.items()}
    for model in MODEL_ORDER:
        p0_by_window: dict[int, float] = {}
        for window, rows in trials_by_window.items():
            fit = fit_logistic(pooled_by_target(rows, model, window))
            p0_by_window[window] = fit["p0"]
        LOGGER.info(
            "window_scaling model=%s p0_4096=%.6f p0_8192=%.6f p0_16384=%.6f",
            model,
            p0_by_window[4096],
            p0_by_window[8192],
            p0_by_window[16384],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="event=%(message)s")
    analyze_cliff()
    analyze_window_scaling()
