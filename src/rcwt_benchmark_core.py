"""Shared RCWT benchmark runner for exact-match external benchmarks."""
from __future__ import annotations

import logging
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent))
from rcwt_benchmark_cli import available_providers, parse_common_args, select_models
from rcwt_benchmark_io import append_record, finalize_outputs, load_existing, trial_key
from rcwt_benchmark_types import (
    DEFAULT_PROPORTIONS,
    ORDERS,
    BenchmarkConfig,
    BenchmarkItem,
    ScoreResult,
    TrialRecord,
)
from rcwt_controlled import (
    MODELS,
    build_coordination_context,
    count_tokens,
    redacted_metering,
)
from rcwt_provider_retry import call_model_with_retry

logger = logging.getLogger("rcwt_benchmark")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )



def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def build_system_prompt(task_block: str, coord_tokens: int, order: str) -> str:
    coord_block = build_coordination_context(coord_tokens) if coord_tokens > 10 else ""
    parts = [coord_block, task_block] if order == "coord_first" else [task_block, coord_block]
    return "\n\n".join(part for part in parts if part)



def run_one_trial(
    config: BenchmarkConfig,
    model: str,
    item: BenchmarkItem,
    proportion: float,
    order: str,
    trial_index: int,
    budget: int,
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
    retry_sleep_seconds: float,
) -> TrialRecord:
    provider = MODELS[model]["provider"]
    coord_tokens = int(budget * proportion)
    reasoning_tokens = max(0, budget - coord_tokens)
    full_task = config.build_task(item)
    task_block = truncate_to_tokens(full_task, reasoning_tokens)
    system_prompt = build_system_prompt(task_block, coord_tokens, order)
    started = time.monotonic()
    response, input_tokens, output_tokens = call_model_with_retry(
        model,
        system_prompt,
        "Answer now. Follow the output format exactly.",
        max_output_tokens,
        temperature,
        max_retries,
        retry_sleep_seconds,
    )
    score = config.score_response(response, item)
    if isinstance(score, ScoreResult):
        correct = score.correct
        prediction = score.prediction
        score_hits = score.score_hits
        score_total = score.score_total
    else:
        correct, prediction = score
        score_hits = 1 if correct else 0
        score_total = 1
    metering_value = redacted_metering(model, input_tokens, output_tokens)
    return TrialRecord(
        benchmark=config.name,
        model=model,
        provider=provider,
        item_id=item.item_id,
        category=item.category,
        proportion=proportion,
        order=order,
        trial_index=trial_index,
        coordination_tokens=coord_tokens,
        reasoning_tokens=reasoning_tokens,
        task_tokens_full=count_tokens(full_task),
        task_tokens_used=count_tokens(task_block),
        task_truncated=count_tokens(full_task) > reasoning_tokens,
        response=response,
        prediction=prediction,
        answer=item.answer,
        correct=correct,
        score_hits=score_hits,
        score_total=score_total,
        score_fraction=score_hits / score_total if score_total else 0.0,
        input_tokens_used=input_tokens,
        output_tokens_used=output_tokens,
        metering_redacted=metering_value,
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )


def build_schedule(
    models: Iterable[str],
    items: list[BenchmarkItem],
    proportions: list[float],
    n_trials: int,
    seed: int,
) -> list[tuple[str, BenchmarkItem, float, str, int]]:
    schedule = [
        (model, item, proportion, order, trial_index)
        for model in models
        for item in items
        for proportion in proportions
        for order in ORDERS
        for trial_index in range(n_trials)
    ]
    random.Random(seed).shuffle(schedule)
    return schedule


def run_cli(config: BenchmarkConfig) -> None:
    configure_logging()
    args = parse_common_args(config)
    output_dir = Path(args.output_dir) if args.output_dir else config.default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    providers = available_providers()
    models = select_models(args, providers)
    if not args.dry_run and not providers:
        raise SystemExit("No API keys set. Load provider keys before running benchmark calls.")
    items = config.load_items(args.n_items, args.seed)
    proportions = [float(part) for part in args.proportions.split(",")]
    schedule = build_schedule(models, items, proportions, args.n_trials, args.seed)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    schedule = [
        entry
        for index, entry in enumerate(schedule)
        if index % args.num_shards == args.shard_index
    ]
    jsonl_path = output_dir / f"rcwt_{config.name}_responses.jsonl"
    existing = load_existing(jsonl_path)
    logger.info(
        "run_config benchmark=%s models=%s calls=%d existing=%d dry_run=%s",
        config.name,
        models,
        len(schedule),
        len(existing),
        args.dry_run,
    )
    if args.dry_run:
        return
    new_calls = 0
    for model, item, proportion, order, trial_index in schedule:
        key = (config.name, model, item.item_id, proportion, order, trial_index)
        if key in existing:
            continue
        record = run_one_trial(
            config,
            model,
            item,
            proportion,
            order,
            trial_index,
            args.budget,
            args.max_output_tokens,
            args.temperature,
            args.max_retries,
            args.retry_sleep_seconds,
        )
        append_record(jsonl_path, record)
        existing[trial_key(asdict(record))] = asdict(record)
        new_calls += 1
        if new_calls % 25 == 0:
            logger.info("progress benchmark=%s new_calls=%d total_done=%d", config.name, new_calls, len(existing))
        if args.max_calls is not None and new_calls >= args.max_calls:
            logger.info("max_calls_reached benchmark=%s max_calls=%d", config.name, args.max_calls)
            break
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    finalize_outputs(output_dir, config)
