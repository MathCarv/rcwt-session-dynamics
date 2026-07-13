"""Re-score saved RCWT intact-ablation responses deterministically."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rcwt_intact_ablation import Trial, save_outputs
from rcwt_intact_scoring import EXPECTED, score_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s level=%(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rescore_intact_ablation")


def load_trials(path: Path) -> list[Trial]:
    trials: list[Trial] = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        scores = score_response(record["response"])
        record["item_scores"] = scores
        record["score"] = sum(scores.values()) / len(EXPECTED)
        trials.append(Trial(**record))
    return trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score intact ablation JSONL.")
    parser.add_argument(
        "--responses",
        default="results/intact_ablation/rcwt_intact_ablation_responses.jsonl",
    )
    parser.add_argument("--output-dir", default="results/intact_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = load_trials(Path(args.responses))
    save_outputs(trials, Path(args.output_dir))
    logger.info("rescored_trials=%d output_dir=%s", len(trials), args.output_dir)


if __name__ == "__main__":
    main()
