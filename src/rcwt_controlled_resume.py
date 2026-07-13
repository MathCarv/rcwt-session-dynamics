"""Resumable/sharded runner for the RCWT controlled experiment."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from rcwt_controlled import MODELS, TrialResult, run_trial

logger = logging.getLogger("rcwt_controlled_resume")


def trial_key(record: dict[str, Any]) -> tuple[str, float, str, int]:
    return (
        str(record["model"]),
        float(record["proportion"]),
        str(record["order"]),
        int(record["trial_index"]),
    )


def load_existing(path: Path) -> dict[tuple[str, float, str, int], dict[str, Any]]:
    records: dict[tuple[str, float, str, int], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                records[trial_key(record)] = record
    return records


def append_trial(path: Path, trial: TrialResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        json.dump(asdict(trial), handle)
        handle.write("\n")


def build_schedule(
    model: str,
    proportions: list[float],
    n_trials: int,
    seed: int,
) -> list[tuple[str, float, str, int]]:
    schedule = [
        (model, proportion, order, trial_index)
        for proportion in proportions
        for order in ["coord_first", "reason_first"]
        for trial_index in range(n_trials)
    ]
    random.Random(seed).shuffle(schedule)
    return schedule


def run_trial_with_retry(
    model: str,
    proportion: float,
    order: str,
    trial_index: int,
    budget: int,
    max_retries: int,
    retry_sleep_seconds: float,
) -> TrialResult:
    for attempt in range(max_retries + 1):
        try:
            return run_trial(model, proportion, order, trial_index, budget)
        except Exception as exc:
            if attempt >= max_retries:
                logger.error(
                    "trial_failed model=%s prop=%.3f order=%s trial=%d error_type=%s error=%s",
                    model,
                    proportion,
                    order,
                    trial_index,
                    type(exc).__name__,
                    exc,
                )
                raise
            sleep_for = retry_sleep_seconds * (2 ** attempt)
            logger.warning(
                "trial_retry model=%s prop=%.3f order=%s trial=%d attempt=%d sleep=%.1f "
                "error_type=%s error=%s",
                model,
                proportion,
                order,
                trial_index,
                attempt + 1,
                sleep_for,
                type(exc).__name__,
                exc,
            )
            time.sleep(sleep_for)
    raise RuntimeError("unreachable retry state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--proportions", default="0,0.25,0.50,0.75,0.90")
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-new-calls", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    proportions = [float(part) for part in args.proportions.split(",")]
    schedule = build_schedule(args.model, proportions, args.n_trials, args.seed)
    schedule = [
        entry
        for index, entry in enumerate(schedule)
        if index % args.num_shards == args.shard_index
    ]
    jsonl_path = args.output_dir / "rcwt_controlled_trials.jsonl"
    existing = load_existing(jsonl_path)
    logger.info(
        "run_config model=%s budget=%d calls=%d existing=%d dry_run=%s",
        args.model,
        args.budget,
        len(schedule),
        len(existing),
        args.dry_run,
    )
    if args.dry_run:
        return

    new_calls = 0
    for model, proportion, order, trial_index in schedule:
        key = (model, proportion, order, trial_index)
        if key in existing:
            continue
        if args.max_new_calls is not None and new_calls >= args.max_new_calls:
            logger.info("max_new_calls_reached max_new_calls=%d", args.max_new_calls)
            break
        trial = run_trial_with_retry(
            model,
            proportion,
            order,
            trial_index,
            args.budget,
            args.max_retries,
            args.retry_sleep_seconds,
        )
        append_trial(jsonl_path, trial)
        existing[trial_key(asdict(trial))] = asdict(trial)
        new_calls += 1
        if new_calls % 10 == 0:
            logger.info("progress model=%s new_calls=%d total_done=%d", model, new_calls, len(existing))
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
