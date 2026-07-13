"""RCWT-style cross-validation on GSM8K exact numeric answers."""
from __future__ import annotations

import re
from pathlib import Path

from datasets import load_dataset

from rcwt_answer_parsing import extract_last_number, normalize_decimal
from rcwt_benchmark_core import run_cli
from rcwt_benchmark_types import BenchmarkConfig, BenchmarkItem


def load_items(n_items: int, seed: int) -> list[BenchmarkItem]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    shuffled = dataset.shuffle(seed=seed).select(range(n_items))
    items: list[BenchmarkItem] = []
    for index, row in enumerate(shuffled):
        answer_text = str(row["answer"])
        match = re.search(r"####\s*([^\n]+)", answer_text)
        answer = match.group(1).strip() if match else answer_text.strip()
        items.append(
            BenchmarkItem(
                item_id=f"gsm8k-{seed}-{index}",
                question=str(row["question"]),
                answer=answer,
                category="gsm8k",
            )
        )
    return items


def build_task(item: BenchmarkItem) -> str:
    return f"""You are solving a GSM8K grade-school math word problem.
Use only the problem statement below. Compute the numeric answer.
End your response with exactly one line: ANSWER: <number>

Problem:
{item.question}
"""


def score_response(response: str, item: BenchmarkItem) -> tuple[bool, str]:
    prediction = extract_last_number(response)
    pred_value = normalize_decimal(prediction)
    answer_value = normalize_decimal(item.answer)
    return (pred_value is not None and pred_value == answer_value), prediction


CONFIG = BenchmarkConfig(
    name="gsm8k",
    default_output_dir=Path(__file__).parent.parent / "results" / "gsm8k",
    default_max_output_tokens=512,
    load_items=load_items,
    build_task=build_task,
    score_response=score_response,
)


if __name__ == "__main__":
    run_cli(CONFIG)
