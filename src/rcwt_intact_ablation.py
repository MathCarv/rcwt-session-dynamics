"""RCWT intact-task ablation with deterministic scoring."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rcwt_controlled import (
    MODELS,
    REASONING_CONTEXT_TEMPLATE,
    build_coordination_context,
    call_model,
    count_tokens,
    redacted_metering,
)
from rcwt_intact_scoring import (
    EXPECTED,
    QA_TASK,
    coordination_tokens_for_ratio,
    score_response,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s level=%(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rcwt_intact_ablation")


@dataclass(frozen=True)
class Trial:
    model: str
    provider: str
    target_ratio: float
    order: str
    trial_index: int
    task_tokens: int
    coordination_tokens: int
    prompt_tokens_estimated: int
    input_tokens_used: int
    output_tokens_used: int
    score: float
    item_scores: dict[str, int]
    elapsed_ms: float
    response: str


def assemble_prompt(coordination_tokens: int, order: str) -> tuple[str, str]:
    reference = f"## Technical Reference\n{REASONING_CONTEXT_TEMPLATE}"
    coordination = build_coordination_context(coordination_tokens)
    if order == "coord_first" and coordination:
        system = f"{coordination}\n\n{reference}"
    elif coordination:
        system = f"{reference}\n\n{coordination}"
    else:
        system = reference
    return system, QA_TASK


def run_trial(model: str, ratio: float, order: str, trial_index: int) -> Trial:
    provider = MODELS[model]["provider"]
    task_tokens = count_tokens(REASONING_CONTEXT_TEMPLATE + "\n\n" + QA_TASK)
    coord_tokens = coordination_tokens_for_ratio(task_tokens, ratio)
    system, user = assemble_prompt(coord_tokens, order)
    t0 = time.monotonic()
    response, input_tokens, output_tokens = call_model(
        model=model,
        system=system,
        user=user,
        max_tokens=700,
        temperature=0.0,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000
    item_scores = score_response(response)
    score = sum(item_scores.values()) / len(EXPECTED)
    prompt_tokens = count_tokens(system + "\n\n" + user)
    logger.info(
        "trial_done model=%s ratio=%.2f order=%s trial=%d score=%.3f input_tokens=%d",
        model,
        ratio,
        order,
        trial_index,
        score,
        input_tokens,
    )
    return Trial(
        model=model,
        provider=provider,
        target_ratio=ratio,
        order=order,
        trial_index=trial_index,
        task_tokens=task_tokens,
        coordination_tokens=coord_tokens,
        prompt_tokens_estimated=prompt_tokens,
        input_tokens_used=input_tokens,
        output_tokens_used=output_tokens,
        score=score,
        item_scores=item_scores,
        elapsed_ms=elapsed_ms,
        response=response,
    )


def save_outputs(trials: list[Trial], output_dir: Path) -> None:
    if not trials:
        logger.warning("event=no_trials_to_save dir=%s", output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rcwt_intact_ablation.csv"
    responses_path = output_dir / "rcwt_intact_ablation_responses.jsonl"
    fieldnames = [k for k in asdict(trials[0]).keys() if k not in {"item_scores", "response"}]
    fieldnames.extend(EXPECTED.keys())
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for trial in trials:
            row = {k: v for k, v in asdict(trial).items() if k not in {"item_scores", "response"}}
            row.update(trial.item_scores)
            writer.writerow(row)
    with responses_path.open("w") as handle:
        for trial in trials:
            handle.write(json.dumps(asdict(trial)) + "\n")
    save_aggregates(trials, output_dir / "rcwt_intact_ablation_aggregates.json")
    logger.info("outputs_saved dir=%s", output_dir)


def save_aggregates(trials: list[Trial], output_path: Path) -> None:
    data: list[dict[str, object]] = []
    groups = sorted({(t.model, t.target_ratio, t.order) for t in trials})
    for model, ratio, order in groups:
        cell = [t for t in trials if (t.model, t.target_ratio, t.order) == (model, ratio, order)]
        data.append(
            {
                "model": model,
                "provider": cell[0].provider,
                "target_ratio": ratio,
                "order": order,
                "n": len(cell),
                "mean_score": round(sum(t.score for t in cell) / len(cell), 4),
                "task_tokens": cell[0].task_tokens,
                "coordination_tokens": cell[0].coordination_tokens,
                "mean_input_tokens": round(sum(t.input_tokens_used for t in cell) / len(cell), 1),
            }
        )
    output_path.write_text(json.dumps(data, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RCWT intact-task ablation.")
    parser.add_argument("--models", default="gpt-4.1-mini,claude-haiku-4-5-20251001,gemini-2.5-flash")
    parser.add_argument("--ratios", default="0,0.5,0.75,0.9,0.95")
    parser.add_argument("--orders", default="coord_first,reason_first")
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--output-dir", default="results/intact_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    ratios = [float(value) for value in args.ratios.split(",")]
    orders = [order.strip() for order in args.orders.split(",") if order.strip()]
    trials: list[Trial] = []
    output_dir = Path(args.output_dir)
    for model in models:
        for ratio in ratios:
            for order in orders:
                for trial_index in range(args.n_trials):
                    try:
                        trials.append(run_trial(model, ratio, order, trial_index))
                        save_outputs(trials, output_dir)
                    except Exception as exc:
                        logger.exception(
                            "trial_failed model=%s ratio=%.2f order=%s trial=%d error=%s",
                            model,
                            ratio,
                            order,
                            trial_index,
                            exc,
                        )
                        save_outputs(trials, output_dir)
                        raise


if __name__ == "__main__":
    main()
