"""RCWT-style GSM8K benchmark using long packs to make truncation measurable."""
from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

from datasets import load_dataset

from rcwt_answer_parsing import normalize_decimal
from rcwt_benchmark_core import run_cli
from rcwt_benchmark_types import BenchmarkConfig, BenchmarkItem, ScoreResult

PACK_SIZE = 10


def _extract_gsm_answer(answer_text: str) -> str:
    match = re.search(r"####\s*([^\n]+)", answer_text)
    return match.group(1).strip() if match else answer_text.strip()


def _pack_answer(answers: list[str]) -> str:
    return json.dumps(answers)


def _unpack_answer(answer: str) -> list[str]:
    raw = json.loads(answer)
    return [str(value) for value in raw]


def load_items(n_items: int, seed: int) -> list[BenchmarkItem]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(dataset.shuffle(seed=seed).select(range(n_items)))
    items: list[BenchmarkItem] = []
    for pack_index, start in enumerate(range(0, len(rows), PACK_SIZE)):
        pack = rows[start : start + PACK_SIZE]
        questions = [
            f"Problem {index + 1}:\n{row['question']}"
            for index, row in enumerate(pack)
        ]
        answers = [_extract_gsm_answer(str(row["answer"])) for row in pack]
        items.append(
            BenchmarkItem(
                item_id=f"gsm8k-pack-{seed}-{pack_index}",
                question="\n\n".join(questions),
                answer=_pack_answer(answers),
                category="gsm8k_pack",
            )
        )
    return items


def build_task(item: BenchmarkItem) -> str:
    answer_count = len(_unpack_answer(item.answer))
    lines = "\n".join(f"ANSWER {index}: <number>" for index in range(1, answer_count + 1))
    return f"""You are solving a pack of GSM8K grade-school math word problems.
Use only the problem statements below. Compute every numeric answer.
Return exactly {answer_count} answer lines and no prose:
{lines}

{item.question}
"""


def _parse_numbered_answers(response: str, expected: int) -> list[str]:
    predictions = [""] * expected
    pattern = re.compile(r"ANSWER\s*(\d+)\s*[:.)-]\s*([^\n]+)", re.IGNORECASE)
    for match in pattern.finditer(response):
        index = int(match.group(1)) - 1
        if 0 <= index < expected:
            numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", match.group(2))
            predictions[index] = numbers[-1] if numbers else ""
    return predictions


def _same_decimal(left: str, right: str) -> bool:
    left_value: Decimal | None = normalize_decimal(left)
    right_value: Decimal | None = normalize_decimal(right)
    return left_value is not None and left_value == right_value


def score_response(response: str, item: BenchmarkItem) -> ScoreResult:
    answers = _unpack_answer(item.answer)
    predictions = _parse_numbered_answers(response, len(answers))
    hits = sum(1 for prediction, answer in zip(predictions, answers) if _same_decimal(prediction, answer))
    return ScoreResult(
        correct=hits == len(answers),
        prediction=json.dumps(predictions),
        score_hits=hits,
        score_total=len(answers),
    )


CONFIG = BenchmarkConfig(
    name="gsm8k_pack",
    default_output_dir=Path(__file__).parent.parent / "results" / "gsm8k_pack",
    default_max_output_tokens=1024,
    load_items=load_items,
    build_task=build_task,
    score_response=score_response,
)


if __name__ == "__main__":
    run_cli(CONFIG)
