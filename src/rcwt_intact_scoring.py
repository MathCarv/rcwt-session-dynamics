"""Deterministic scoring for the RCWT intact-task ablation."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("rcwt_intact_scoring")

QA_TASK = """Using ONLY the technical reference in context, return one JSON object.
No prose. Use exactly these keys:

{
  "pg_limit": "...",
  "redis_pubsub_rate": "...",
  "redis_streams_latency": "...",
  "redis_pubsub_latency": "...",
  "recommended_encryption": "...",
  "prosemirror_role": "...",
  "backend_rampup": "...",
  "infra_migration": "..."
}
"""

EXPECTED: dict[str, tuple[str, ...]] = {
    "pg_limit": ("8kb", "8 kb"),
    "redis_pubsub_rate": ("100k", "100,000"),
    "redis_streams_latency": ("5ms", "5 ms"),
    "redis_pubsub_latency": ("1ms", "1 ms"),
    "recommended_encryption": (
        "per-document symmetric key",
        "per document symmetric key",
        "symmetric key per document",
    ),
    "prosemirror_role": ("rich text",),
    "backend_rampup": ("1 week", "one week"),
    "infra_migration": ("2-3", "2–3", "2 to 3", "two to three", "3 weeks"),
}


def coordination_tokens_for_ratio(task_tokens: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    if ratio >= 1:
        raise ValueError("ratio must be less than 1.0")
    return round((ratio * task_tokens) / (1.0 - ratio))


def extract_json_object(text: str) -> dict[str, object]:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            logger.warning("event=json_parse_failed chars=%d", len(text))
    parsed: dict[str, object] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().strip('"').strip("'")
        if key in EXPECTED:
            parsed[key] = value.strip().strip('",')
    return parsed


def normalize(value: object) -> str:
    return str(value).lower().replace("_", " ").strip()


def score_response(response: str) -> dict[str, int]:
    parsed = extract_json_object(response)
    scores: dict[str, int] = {}
    for key, accepted in EXPECTED.items():
        value = normalize(parsed.get(key, ""))
        scores[key] = int(any(option in value for option in accepted))
    return scores
