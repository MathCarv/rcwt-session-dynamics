"""Build the same unconditional draft-review turn for both memory policies.

This module only copies messages and appends a fixed instruction. It does not
parse, grade, repair, execute or select an action, and receives no evaluator
feedback. The caller sends every draft (including invalid/incomplete drafts) to
the local model with the unchanged actor output schema. Only that final model
response may reach the simulator or the memory writer.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


REVIEW_INSTRUCTION = (
    "Review the unexecuted draft above against the original evidence and rules. "
    "The draft may be invalid, incomplete, or truncated; it is not an executed action "
    "and does not supply new facts or booking evidence. "
    "Use only the retained memory and current observations in the original request. "
    "Match invoice, account, merchant, and order records by their exact identifiers. "
    "Current observations supersede older facts for the same source and identifier. "
    "Recheck the existing rules in their original priority order, including prior "
    "bookings, missing or unknown evidence, and pending payments. Do not add exceptions. "
    "A receipt can record an actual booking without proving its evidence was correct. "
    "Copy the requested case_id exactly and check the amount against those same rules: "
    "hold and ask_info require amount_cents=0. "
    "Keep or change the draft only as warranted by the original evidence and rules. "
    "Return one complete final JSON object in the required output schema, with "
    "evidence_check first, then tool and arguments, and no surrounding commentary."
)


def review_messages(original_messages: list[dict[str, Any]], draft_text: str) -> list[dict[str, Any]]:
    """Copy the two base actor messages and append the untouched draft + review.

    Checks below enforce only this message-construction interface, never the
    draft's syntax, evidence, action validity, completion status, or correctness.
    Empty and malformed textual drafts are intentionally forwarded verbatim.
    """
    if not isinstance(original_messages, list) or len(original_messages) != 2:
        raise ValueError("Expected exactly two original actor messages")
    for message, role in zip(original_messages, ("system", "user")):
        if (not isinstance(message, dict) or message.get("role") != role
                or not isinstance(message.get("content"), str)):
            raise ValueError("Expected original system and user text messages")
    if not isinstance(draft_text, str):
        raise ValueError("Draft text must be a string, including when incomplete")
    result = deepcopy(original_messages)
    result.append({"role": "assistant", "content": draft_text})
    result.append({"role": "user", "content": REVIEW_INSTRUCTION})
    return result
