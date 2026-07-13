"""Merge sharded RCWT controlled trials and rebuild standard artifacts."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from rcwt_controlled import (
    MODELS,
    ExperimentResult,
    TrialResult,
    aggregate_trials,
    plot_results,
    save_aggregates,
    save_csv,
)
from rcwt_controlled_resume import trial_key

logger = logging.getLogger("merge_controlled_shards")


def load_records(paths: list[Path]) -> dict[tuple[str, float, str, int], dict[str, Any]]:
    records: dict[tuple[str, float, str, int], dict[str, Any]] = {}
    duplicates = 0
    for path in paths:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = trial_key(record)
                if key in records:
                    duplicates += 1
                records[key] = record
    if duplicates:
        logger.warning("duplicate_records_overwritten duplicates=%d", duplicates)
    return records


def to_trial(record: dict[str, Any]) -> TrialResult:
    return TrialResult(
        model=str(record["model"]),
        provider=str(record["provider"]),
        proportion=float(record["proportion"]),
        order=str(record["order"]),
        trial_index=int(record["trial_index"]),
        coordination_tokens=int(record["coordination_tokens"]),
        reasoning_tokens=int(record["reasoning_tokens"]),
        total_context_tokens=int(record["total_context_tokens"]),
        response=str(record["response"]),
        scores={str(k): int(v) for k, v in record["scores"].items()},
        mean_score_raw=float(record["mean_score_raw"]),
        mean_score_effective=float(record["mean_score_effective"]),
        input_tokens_used=int(record["input_tokens_used"]),
        output_tokens_used=int(record["output_tokens_used"]),
        metering_redacted=float(record["metering_redacted"]),
        elapsed_ms=float(record["elapsed_ms"]),
    )


def merge(source_root: Path, output_dir: Path, model: str, n_trials: int) -> None:
    paths = sorted(source_root.glob("shard_*/rcwt_controlled_trials.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No shard trial files found under {source_root}")
    records = load_records(paths)
    trials = [to_trial(records[key]) for key in sorted(records)]
    proportions = sorted({trial.proportion for trial in trials})
    exp = ExperimentResult(
        model=model,
        provider=MODELS[model]["provider"],
        total_budget=trials[0].total_context_tokens if trials else 0,
        n_trials_per_cell=n_trials,
        trials=trials,
        total_metering_redacted=sum(trial.metering_redacted for trial in trials),
        elapsed_seconds=sum(trial.elapsed_ms for trial in trials) / 1000.0,
    )
    for proportion in proportions:
        for order in ["coord_first", "reason_first"]:
            exp.aggregates.append(aggregate_trials(trials, proportion, order))
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "rcwt_controlled_trials.jsonl"
    with full_path.open("w") as handle:
        for trial in trials:
            json.dump(trial.__dict__, handle)
            handle.write("\n")
    save_csv([exp], output_dir / "rcwt_controlled.csv")
    save_aggregates([exp], output_dir / "rcwt_controlled_aggregates.json")
    try:
        plot_results([exp], output_dir / "rcwt_controlled.png")
    except ImportError:
        logger.warning("matplotlib_unavailable")
    logger.info("merge_complete records=%d output_dir=%s", len(trials), output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--n-trials", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    started = time.monotonic()
    args = parse_args()
    merge(args.source_root, args.output_dir, args.model, args.n_trials)
    logger.info("elapsed_seconds=%.1f", time.monotonic() - started)


if __name__ == "__main__":
    main()
