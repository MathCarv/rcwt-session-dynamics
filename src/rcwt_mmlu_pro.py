"""RCWT-style cross-validation on MMLU-Pro exact multiple-choice answers."""
from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from rcwt_answer_parsing import extract_choice
from rcwt_benchmark_core import run_cli
from rcwt_benchmark_types import BenchmarkConfig, BenchmarkItem

LETTERS = "ABCDEFGHIJ"


def load_items(n_items: int, seed: int) -> list[BenchmarkItem]:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    shuffled = dataset.shuffle(seed=seed).select(range(n_items))
    items: list[BenchmarkItem] = []
    for row in shuffled:
        options = [str(option) for option in row["options"]]
        answer = str(row["answer"]).strip().upper()
        items.append(
            BenchmarkItem(
                item_id=f"mmlu-pro-{row['question_id']}",
                question=str(row["question"]),
                answer=answer,
                choices=options,
                category=str(row["category"]),
            )
        )
    return items


def build_task(item: BenchmarkItem) -> str:
    choices = item.choices or []
    rendered_choices = "\n".join(
        f"{LETTERS[index]}. {choice}" for index, choice in enumerate(choices)
    )
    return f"""You are answering a MMLU-Pro multiple-choice question.
Choose exactly one option.
Start your response with exactly one line: ANSWER: <letter>
Keep any explanation brief.

Question:
{item.question}

Options:
{rendered_choices}
"""


def score_response(response: str, item: BenchmarkItem) -> tuple[bool, str]:
    valid_letters = set(LETTERS[: len(item.choices or [])])
    prediction = extract_choice(response, valid_letters)
    return prediction == item.answer, prediction


CONFIG = BenchmarkConfig(
    name="mmlu_pro",
    default_output_dir=Path(__file__).parent.parent / "results" / "mmlu_pro",
    default_max_output_tokens=256,
    load_items=load_items,
    build_task=build_task,
    score_response=score_response,
)


if __name__ == "__main__":
    run_cli(CONFIG)
