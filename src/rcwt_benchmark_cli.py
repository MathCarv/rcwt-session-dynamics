"""CLI argument parsing for RCWT benchmark runners."""
from __future__ import annotations

import argparse
import os

from rcwt_benchmark_types import DEFAULT_PROPORTIONS, BenchmarkConfig
from rcwt_controlled import MODELS


def parse_common_args(config: BenchmarkConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"RCWT benchmark runner: {config.name}")
    parser.add_argument("--model", default=None, help="Single model to test")
    parser.add_argument("--models", default=None, help="Comma-separated models")
    parser.add_argument("--n-items", type=int, default=50, help="Benchmark subset size")
    parser.add_argument("--seed", type=int, default=20260429, help="Subset/schedule seed")
    parser.add_argument("--n-trials", type=int, default=10, help="Trials per cell")
    parser.add_argument("--budget", type=int, default=4096, help="Total input context budget")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--max-output-tokens", type=int, default=config.default_max_output_tokens)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature; default matches the original RCWT runner.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--retry-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--proportions",
        default=",".join(str(p) for p in DEFAULT_PROPORTIONS),
        help="Comma-separated coordination proportions",
    )
    return parser.parse_args()


def available_providers() -> set[str]:
    providers: set[str] = set()
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.add("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        providers.add("openai")
    if os.environ.get("GEMINI_API_KEY"):
        providers.add("google")
    return providers


def select_models(args: argparse.Namespace, providers: set[str]) -> list[str]:
    if args.model:
        return [args.model]
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    return [
        model
        for model, info in MODELS.items()
        if info["provider"] in providers and info["tier"] == "cheap"
    ]
