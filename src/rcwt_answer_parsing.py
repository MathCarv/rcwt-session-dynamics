"""Exact-answer parsing helpers for RCWT external benchmarks."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_decimal(text: str) -> Decimal | None:
    candidate = text.strip().replace(",", "").replace(chr(36), "")
    try:
        return Decimal(candidate)
    except InvalidOperation:
        return None


def extract_last_number(text: str) -> str:
    answer_match = re.search(r"ANSWER\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    target = answer_match.group(1) if answer_match else text
    matches = _NUMBER_RE.findall(target)
    if not matches and target is not text:
        matches = _NUMBER_RE.findall(text)
    return matches[-1].strip() if matches else ""


def extract_choice(text: str, valid_letters: set[str]) -> str:
    answer_match = re.search(r"ANSWER\s*:\s*([A-J])\b", text, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    explicit_match = re.search(
        r"\b(?:answer|option|choice|letter)\s*(?:is|:)?\s*([A-J])\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit_match and explicit_match.group(1).upper() in valid_letters:
        return explicit_match.group(1).upper()
    final_line = text.strip().splitlines()[-1].strip().upper() if text.strip() else ""
    final_match = re.fullmatch(r"(?:ANSWER\s*:\s*)?([A-J])[\).]?", final_line)
    if final_match and final_match.group(1) in valid_letters:
        return final_match.group(1)
    return ""
