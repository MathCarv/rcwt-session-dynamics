"""RCWT-style DROP benchmark using long packs to make truncation measurable."""
from __future__ import annotations

import json
import re
import string
from decimal import Decimal
from pathlib import Path
from typing import Any

from datasets import load_dataset

from rcwt_answer_parsing import normalize_decimal
from rcwt_benchmark_core import run_cli
from rcwt_benchmark_types import BenchmarkConfig, BenchmarkItem, ScoreResult

PACK_SIZE = 10


def _pack_answer(answer_aliases: list[list[str]]) -> str:
    return json.dumps(answer_aliases)


def _unpack_answer(answer: str) -> list[list[str]]:
    raw = json.loads(answer)
    return [[str(alias) for alias in aliases] for aliases in raw]


def _clean_alias(alias: object) -> str:
    return str(alias).strip()


def _render_drop_answer(answer: dict[str, Any]) -> tuple[str, str] | None:
    number = _clean_alias(answer.get("number", ""))
    if number:
        return number, "number"

    date = answer.get("date") or {}
    date_parts = [
        _clean_alias(date.get("month", "")),
        _clean_alias(date.get("day", "")),
        _clean_alias(date.get("year", "")),
    ]
    rendered_date = " ".join(part for part in date_parts if part)
    if rendered_date:
        return rendered_date, "date"

    spans = [_clean_alias(span) for span in answer.get("spans", []) if _clean_alias(span)]
    if len(spans) == 1:
        return spans[0], "span"
    return None


def _validated_aliases(row: dict[str, Any]) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    validated = row.get("validated_answers") or {}

    for number in validated.get("number") or []:
        rendered = _clean_alias(number)
        if rendered:
            aliases.append((rendered, "number"))

    for date in validated.get("date") or []:
        date_parts = [
            _clean_alias(date.get("month", "")),
            _clean_alias(date.get("day", "")),
            _clean_alias(date.get("year", "")),
        ]
        rendered_date = " ".join(part for part in date_parts if part)
        if rendered_date:
            aliases.append((rendered_date, "date"))

    for spans in validated.get("spans") or []:
        rendered_spans = [_clean_alias(span) for span in spans if _clean_alias(span)]
        if len(rendered_spans) == 1:
            aliases.append((rendered_spans[0], "span"))

    return aliases


def _answer_aliases(row: dict[str, Any]) -> tuple[list[str], str] | None:
    aliases_with_type: list[tuple[str, str]] = []
    primary = _render_drop_answer(row["answer"])
    if primary is not None:
        aliases_with_type.append(primary)
    aliases_with_type.extend(_validated_aliases(row))

    aliases: list[str] = []
    answer_types: list[str] = []
    for alias, answer_type in aliases_with_type:
        if alias and alias not in aliases:
            aliases.append(alias)
        if answer_type not in answer_types:
            answer_types.append(answer_type)

    if not aliases:
        return None
    return aliases, "+".join(answer_types)


def load_items(n_items: int, seed: int) -> list[BenchmarkItem]:
    dataset = load_dataset("EleutherAI/drop", split="validation")
    rows: list[tuple[dict[str, Any], list[str], str]] = []
    for row in dataset.shuffle(seed=seed):
        row_dict = dict(row)
        aliases_and_type = _answer_aliases(row_dict)
        if aliases_and_type is None:
            continue
        aliases, answer_type = aliases_and_type
        rows.append((row_dict, aliases, answer_type))
        if len(rows) >= n_items:
            break

    items: list[BenchmarkItem] = []
    for pack_index, start in enumerate(range(0, len(rows), PACK_SIZE)):
        pack = rows[start : start + PACK_SIZE]
        rendered_questions: list[str] = []
        answer_aliases: list[list[str]] = []
        answer_types: list[str] = []
        for question_index, (row, aliases, answer_type) in enumerate(pack, start=1):
            rendered_questions.append(
                f"Item {question_index}:\n"
                f"Passage:\n{row['passage']}\n"
                f"Question: {row['question']}"
            )
            answer_aliases.append(aliases)
            answer_types.append(answer_type)
        items.append(
            BenchmarkItem(
                item_id=f"drop-pack-{seed}-{pack_index}",
                question="\n\n".join(rendered_questions),
                answer=_pack_answer(answer_aliases),
                category=",".join(sorted(set(answer_types))),
            )
        )
    return items


def build_task(item: BenchmarkItem) -> str:
    answer_count = len(_unpack_answer(item.answer))
    lines = "\n".join(f"ANSWER {index}: <short answer>" for index in range(1, answer_count + 1))
    return f"""You are answering a pack of DROP reading-comprehension questions.
Use only each item's passage. Answers are short spans, numbers, or dates.
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
            predictions[index] = match.group(2).strip()
    return predictions


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    without_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in without_punctuation.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def _extract_numbers(text: str) -> list[Decimal]:
    values: list[Decimal] = []
    for match in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text):
        value = normalize_decimal(match)
        if value is not None:
            values.append(value)
    return values


def _matches_alias(prediction: str, aliases: list[str]) -> bool:
    prediction_numbers = _extract_numbers(prediction)
    for alias in aliases:
        alias_value = normalize_decimal(alias)
        if alias_value is not None and alias_value in prediction_numbers:
            return True

        normalized_prediction = _normalize_text(prediction)
        normalized_alias = _normalize_text(alias)
        if not normalized_alias:
            continue
        if normalized_prediction == normalized_alias:
            return True
        if len(normalized_alias) > 2 and normalized_alias in normalized_prediction:
            return True
    return False


def score_response(response: str, item: BenchmarkItem) -> ScoreResult:
    answer_aliases = _unpack_answer(item.answer)
    predictions = _parse_numbered_answers(response, len(answer_aliases))
    hits = sum(
        1
        for prediction, aliases in zip(predictions, answer_aliases)
        if _matches_alias(prediction, aliases)
    )
    return ScoreResult(
        correct=hits == len(answer_aliases),
        prediction=json.dumps(predictions),
        score_hits=hits,
        score_total=len(answer_aliases),
    )


CONFIG = BenchmarkConfig(
    name="drop_pack",
    default_output_dir=Path(__file__).parent.parent / "results" / "drop_pack",
    default_max_output_tokens=768,
    load_items=load_items,
    build_task=build_task,
    score_response=score_response,
)


if __name__ == "__main__":
    run_cli(CONFIG)
