"""Evidence-first actor envelope for the local decision experiment.

The check is the model's own report, never an oracle or a second decision
engine. Extraction validates shape and types, then forwards the model's action
unchanged. It does not repair decisions that contradict the reported evidence.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from rcwt_agent_env import ACTION_SCHEMA, PUBLIC_RULES, parse_action

INVALID_ACTOR_OUTPUT = "INVALID_ACTOR_OUTPUT"
TRUNCATED_ACTION = "TRUNCATED_ACTION"

_EVIDENCE_ENUMS = {
    "account_status": ("unknown", "active", "blocked", "revoked"),
    "ownership_match": ("unknown", "yes", "no"),
    "payment_status": ("unknown", "pending", "cleared"),
    "return_status": ("unknown", "accepted", "rejected"),
    "operation_already_booked": ("no_record", "yes"),
}

EVIDENCE_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "invoice_amount_cents": {"type": ["integer", "null"], "minimum": 0},
        **{name: {"enum": list(values)} for name, values in _EVIDENCE_ENUMS.items()},
    },
    "required": ["invoice_amount_cents", *_EVIDENCE_ENUMS],
}

ACTOR_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["evidence_check", "tool", "arguments"],
    "properties": {"evidence_check": copy.deepcopy(EVIDENCE_CHECK_SCHEMA),
                   **copy.deepcopy(ACTION_SCHEMA["properties"])},
}

# Preserve the development pilot's evidence-first instruction, making the rule
# precedence and online-memory boundary explicit. No environment rule changes.
_ACTOR_RULES = PUBLIC_RULES.replace(
    "An invoice, account or report about another identifier does not supply missing\n"
    "evidence for this request.",
    "Records for other identifiers cannot satisfy missing evidence for this request.",
)

ACTOR_INSTRUCTION = (
    "Inspect the evidence BEFORE choosing the decision. Return one JSON object with evidence_check first, then tool and arguments. "
    "For each evidence_check field, extract the current value from the supplied observations or memory. "
    "If a required fact was not observed, mark unknown (amount null). Never substitute another case's record. "
    "Do not infer payment_status from an invoice, an active account, a request or a past proposed action. "
    "After the check, apply the following rules to choose the action. Subject to higher-priority rules below, "
    "missing payment status requires ask_info even if other records look valid. "
    "If required evidence is otherwise present, a pending payment requires hold. "
    "The check is your evidence record, not an answer key. "
    "Read ONLY the retained memory and this step's new observations. Earlier facts may have been discarded; "
    "never invent them. Current observations override conflicting older memory. "
    "A previously accepted tool call is a booking record, not proof of correct evidence.\n\n"
    + _ACTOR_RULES
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _parse_envelope(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(text, str):
        raise ValueError("Actor output must be JSON text")
    envelope = json.loads(text, object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    # JSON field order is normally immaterial. Here generation order is an
    # explicit part of the evidence-before-action protocol and is audited.
    if not isinstance(envelope, dict) or list(envelope) != ["evidence_check", "tool", "arguments"]:
        raise ValueError("Expected evidence_check, tool, arguments in that order")
    check = envelope["evidence_check"]
    if not isinstance(check, dict) or set(check) != {"invoice_amount_cents", *_EVIDENCE_ENUMS}:
        raise ValueError("Missing or extra evidence_check fields")
    amount = check["invoice_amount_cents"]
    if amount is not None and (type(amount) is not int or amount < 0):
        raise ValueError("Evidence amount must be null or a nonnegative integer")
    for name, allowed in _EVIDENCE_ENUMS.items():
        if not isinstance(check[name], str) or check[name] not in allowed:
            raise ValueError("Invalid evidence_check value: " + name)
    action = parse_action({"tool": envelope["tool"], "arguments": envelope["arguments"]})
    return check, action.to_dict()


def extract_action(text: str, finish_reason: str) -> str:
    """Return the original validated action as canonical JSON, or a sentinel.

    Only a normal ``stop`` completion can become an executable tool call. A
    valid-looking prefix of a truncated response remains non-executable. No
    reasoning/evidence value is compared with hidden facts or used to change
    the action chosen by the model.
    """
    if finish_reason != "stop":
        return TRUNCATED_ACTION
    try:
        _, action = _parse_envelope(text)
    except (ValueError, TypeError, KeyError, RecursionError):
        return INVALID_ACTOR_OUTPUT
    return json.dumps(action, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def extract_evidence_check(text: str, finish_reason: str) -> dict[str, Any] | None:
    """Return the self-reported check for a valid envelope, or None.

    A returned check passed structural validation only; its facts may be wrong.
    It must not be treated as reference truth when scoring a decision.
    """
    if finish_reason != "stop":
        return None
    try:
        check, _ = _parse_envelope(text)
    except (ValueError, TypeError, KeyError, RecursionError):
        return None
    return dict(check)
