"""RCWT-style MMLU-Pro benchmark using long packs to make truncation measurable."""
from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset

from rcwt_benchmark_core import run_cli
from rcwt_benchmark_types import BenchmarkConfig, BenchmarkItem, ScoreResult

LETTERS = "ABCDEFGHIJ"
PACK_SIZE = 10


def _pack_answer(answers: list[str]) -> str:
    return json.dumps(answers)


def _unpack_answer(answer: str) -> list[str]:
    raw = json.loads(answer)
    return [str(value).upper() for value in raw]


def load_items(n_items: int, seed: int) -> list[BenchmarkItem]:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = list(dataset.shuffle(seed=seed).select(range(n_items)))
    items: list[BenchmarkItem] = []
    for pack_index, start in enumerate(range(0, len(rows), PACK_SIZE)):
        pack = rows[start : start + PACK_SIZE]
        rendered_questions: list[str] = []
        answers: list[str] = []
        categories: list[str] = []
        for question_index, row in enumerate(pack, start=1):
            options = [str(option) for option in row["options"]]
            rendered_options = "\n".join(
                f"{LETTERS[index]}. {option}" for index, option in enumerate(options)
            )
            rendered_questions.append(
                f"Question {question_index} [{row['category']}]:\n"
                f"{row['question']}\nOptions:\n{rendered_options}"
            )
            answers.append(str(row["answer"]).strip().upper())
            categories.append(str(row["category"]))
        items.append(
            BenchmarkItem(
                item_id=f"mmlu-pro-pack-{seed}-{pack_index}",
                question="\n\n".join(rendered_questions),
                answer=_pack_answer(answers),
                category=",".join(sorted(set(categories))),
            )
        )
    return items


def build_task(item: BenchmarkItem) -> str:
    answer_count = len(_unpack_answer(item.answer))
    lines = "\n".join(f"ANSWER {index}: <letter>" for index in range(1, answer_count + 1))
    return f"""You are answering a pack of MMLU-Pro multiple-choice questions.
Use only the questions and options below. Choose exactly one option per question.
Return exactly {answer_count} answer lines and no prose:
{lines}

{item.question}
"""


def _parse_numbered_answers(response: str, expected: int) -> list[str]:
    predictions = [""] * expected
    pattern = re.compile(r"ANSWER\s*(\d+)\s*[:.)-]\s*([A-J])\b", re.IGNORECASE)
    for match in pattern.finditer(response):
        index = int(match.group(1)) - 1
        if 0 <= index < expected:
            predictions[index] = match.group(2).upper()
    return predictions


def score_response(response: str, item: BenchmarkItem) -> ScoreResult:
    answers = _unpack_answer(item.answer)
    predictions = _parse_numbered_answers(response, len(answers))
    hits = sum(1 for prediction, answer in zip(predictions, answers) if prediction == answer)
    return ScoreResult(
        correct=hits == len(answers),
        prediction=json.dumps(predictions),
        score_hits=hits,
        score_total=len(answers),
    )


CONFIG = BenchmarkConfig(
    name="mmlu_pro_pack",
    default_output_dir=Path(__file__).parent.parent / "results" / "mmlu_pro_pack",
    default_max_output_tokens=512,
    load_items=load_items,
    build_task=build_task,
    score_response=score_response,
)


if __name__ == "__main__":
    run_cli(CONFIG)
