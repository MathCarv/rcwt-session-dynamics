"""Replace legacy field-level intervals with call-level bootstrap intervals."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from rcwt_statistics import stable_seed, stratified_bootstrap_mean_ci

LOGGER = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REQUIRED_COLUMNS = {
    "model",
    "proportion",
    "order",
    "mean_score_raw",
    "mean_score_effective",
}


def aggregate_path_for(csv_path: Path) -> Path:
    """Return the conventional aggregate path for one trial CSV."""
    return csv_path.with_name(f"{csv_path.stem}_aggregates.json")


def update_pair(csv_path: Path, aggregate_path: Path) -> bool:
    """Update one compatible aggregate file and return whether it was handled."""
    try:
        seed_path = csv_path.resolve().relative_to(RESULTS_DIR.resolve()).as_posix()
    except ValueError:
        seed_path = csv_path.name

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            return False
        rows = list(reader)

    grouped: dict[tuple[str, float, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], float(row["proportion"]), row["order"])].append(row)

    payload: list[dict[str, Any]] = json.loads(aggregate_path.read_text(encoding="utf-8"))
    for experiment in payload:
        model = str(experiment["model"])
        for cell in experiment["aggregates"]:
            target_ratio = float(cell["proportion"])
            order = str(cell["order"])
            trial_rows = grouped[(model, target_ratio, order)]
            if not trial_rows:
                raise ValueError(
                    f"missing trials for {model}, q={target_ratio}, order={order} in {csv_path}"
                )
            raw_scores = [float(row["mean_score_raw"]) for row in trial_rows]
            effective_scores = [float(row["mean_score_effective"]) for row in trial_rows]
            _, raw_low, raw_high = stratified_bootstrap_mean_ci(
                {"calls": raw_scores},
                seed=stable_seed(seed_path, model, target_ratio, order, "raw"),
            )
            _, effective_low, effective_high = stratified_bootstrap_mean_ci(
                {"calls": effective_scores},
                seed=stable_seed(seed_path, model, target_ratio, order, "effective"),
            )
            cell["ci95_raw"] = [raw_low, raw_high]
            cell["ci95_effective"] = [effective_low, effective_high]

        experiment["ci_method"] = "percentile bootstrap over call-level mean scores"
        experiment["ci_resamples"] = 10_000
        experiment["ci_unit"] = "call"

    aggregate_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("intervals_rebuilt csv=%s aggregates=%s", csv_path, aggregate_path)
    return True


def rebuild_all(results_dir: Path = RESULTS_DIR) -> int:
    """Rebuild every compatible CSV/aggregate pair below ``results_dir``."""
    updated = 0
    for csv_path in sorted(results_dir.rglob("*.csv")):
        aggregate_path = aggregate_path_for(csv_path)
        if aggregate_path.exists() and update_pair(csv_path, aggregate_path):
            updated += 1
    return updated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="event=%(message)s")
    LOGGER.info("interval_rebuild_complete files=%d", rebuild_all())
