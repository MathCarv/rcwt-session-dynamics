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

ORIGINAL_JSON_REQUIREMENT = (
    "Return one JSON object with evidence_check first, then tool and arguments. "
)
PLANNING_INSTRUCTION = (
    "This is an unexecuted planning pass, not the final tool response. "
    "Instead of tool JSON, write a short plain-text decision plan of at most 180 words. "
    "Identify the exact requested case and operation. List the relevant known and unknown "
    "facts, keeping each observation attached to its actual identifier. "
    "State the first applicable rule in the original priority order, checking an existing "
    "booking before the numbered rules. Give a concise reason that rule applies and "
    "the proposed decision, reason code and exact amount. Do not skip an applicable "
    "higher-priority rule. Distinguish retained facts from current updates and absent "
    "evidence; do not invent missing values. Nothing in this plan has been executed. "
    "The final pass will independently return the required tool JSON."
)


def planning_messages(original_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the same public rules, replacing only the first-pass output format.

    The requirement is matched exactly so a drifted base cannot silently lose
    unrelated text. No facts, proposed action, or rule selection are computed.
    The resulting plan is transient and never passed to the memory writer.
    """
    result = _copy_original(original_messages)
    instruction = result[0]["content"]
    if instruction.count(ORIGINAL_JSON_REQUIREMENT) != 1:
        raise ValueError("Expected the exact frozen first-pass JSON requirement")
    result[0]["content"] = instruction.replace(ORIGINAL_JSON_REQUIREMENT, "", 1) + "\n\n" + PLANNING_INSTRUCTION
    return result


def review_messages(original_messages: list[dict[str, Any]], draft_text: str) -> list[dict[str, Any]]:
    """Copy the two base actor messages and append the untouched draft + review.

    Checks below enforce only this message-construction interface, never the
    draft's syntax, evidence, action validity, completion status, or correctness.
    Empty and malformed textual drafts are intentionally forwarded verbatim.
    """
    result = _copy_original(original_messages)
    if not isinstance(draft_text, str):
        raise ValueError("Draft text must be a string, including when incomplete")
    result.append({"role": "assistant", "content": draft_text})
    result.append({"role": "user", "content": REVIEW_INSTRUCTION})
    return result


def _copy_original(original_messages):
    if not isinstance(original_messages, list) or len(original_messages) != 2:
        raise ValueError("Expected exactly two original actor messages")
    for message, role in zip(original_messages, ("system", "user")):
        if (not isinstance(message, dict) or message.get("role") != role
                or not isinstance(message.get("content"), str)):
            raise ValueError("Expected original system and user text messages")
    return deepcopy(original_messages)
