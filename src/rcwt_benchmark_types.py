"""Dataclasses and constants for RCWT benchmark runners."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeAlias

ORDERS = ["coord_first", "reason_first"]
DEFAULT_PROPORTIONS = [0.0, 0.25, 0.50, 0.75, 0.90]


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    question: str
    answer: str
    choices: list[str] | None = None
    category: str | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    default_output_dir: Path
    default_max_output_tokens: int
    load_items: Callable[[int, int], list[BenchmarkItem]]
    build_task: Callable[[BenchmarkItem], str]
    score_response: Callable[[str, BenchmarkItem], "ScoreReturn"]


@dataclass(frozen=True)
class ScoreResult:
    correct: bool
    prediction: str
    score_hits: int
    score_total: int


ScoreReturn: TypeAlias = tuple[bool, str] | ScoreResult


@dataclass
class TrialRecord:
    benchmark: str
    model: str
    provider: str
    item_id: str
    category: str | None
    proportion: float
    order: str
    trial_index: int
    coordination_tokens: int
    reasoning_tokens: int
    task_tokens_full: int
    task_tokens_used: int
    task_truncated: bool
    response: str
    prediction: str
    answer: str
    correct: bool
    score_hits: int
    score_total: int
    score_fraction: float
    input_tokens_used: int
    output_tokens_used: int
    metering_redacted: float
    elapsed_ms: int
