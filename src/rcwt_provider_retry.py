"""Provider-call retry wrapper for RCWT experiment scripts."""
from __future__ import annotations

import logging
import time

from rcwt_controlled import call_model

logger = logging.getLogger("rcwt_benchmark")


def call_model_with_retry(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
    retry_sleep_seconds: float,
) -> tuple[str, int, int]:
    for attempt in range(max_retries + 1):
        try:
            return call_model(
                model,
                system_prompt,
                user_prompt,
                max_tokens=max_output_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            if attempt >= max_retries:
                raise
            sleep_for = retry_sleep_seconds * (2**attempt)
            logger.warning(
                "model_retry model=%s attempt=%d sleep_seconds=%.1f error_type=%s error=%s",
                model,
                attempt + 1,
                sleep_for,
                type(exc).__name__,
                exc,
            )
            time.sleep(sleep_for)
    raise RuntimeError("unreachable retry state")
