"""Shared task-bound schema and a fixed textual public-rule guide.

Only the literal public task identity and requested operation constrain the
output vocabulary. No observations, memory, self-check, or private reference
are inspected. This is not a semantic safety guard: unsafe monetary decisions
and wrong amounts remain generatable. The unchanged actor extractor still
validates the final response; this module never parses or repairs that response.
The optional message wrapper only deep-copies the two base messages and appends
the same constant guide to the system text. It never reads factual values or
evaluates the conditions described in that guide.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from rcwt_agent_actor import ACTOR_SCHEMA


ORDERED_RULE_GUIDE = (
    "Apply the existing public rules by FIRST TRUE CONDITION, in this exact order. "
    "A later condition never overrides an earlier true condition. Use evidence only "
    "for the exact requested invoice and operation, joined by explicit source IDs. "
    "A retained fact remains valid until replaced by an update for the same source "
    "and identifier. No update, a message without that fact, or a record for another "
    "identifier does not erase the retained fact. Do not invent missing evidence.\n"
    "0. operation_already_booked=yes for this invoice and operation: "
    "hold / already_completed / amount_cents=0. A booking means this operation was "
    "already executed; it does not authorize another execution.\n"
    "1. account_status=blocked or revoked: hold / account_restricted / amount_cents=0.\n"
    "2. ownership_match=no: hold / ownership_mismatch / amount_cents=0.\n"
    "3. invoice_amount_cents=null, account_status=unknown, ownership_match=unknown, "
    "payment_status=unknown, or (operation=refund AND return_status=unknown): "
    "ask_info / missing_evidence / amount_cents=0. A payout does not require return_status.\n"
    "4. payment_status=pending: hold / funds_unsettled / amount_cents=0. "
    "Pending is an observed payment status, not unknown.\n"
    "5. operation=refund AND return_status=rejected: hold / return_rejected / amount_cents=0.\n"
    "6. ONLY if every condition 0..5 is false: for payout use approve / authorized_payout; "
    "for refund use refund / eligible_refund. In either monetary case, amount_cents "
    "must equal the exact positive invoice_amount_cents for this invoice.\n"
    "In the short planning pass, state the relevant evidence and a compact YES/NO "
    "list for predicates 0..5 in order, stopping at the first YES; mark later "
    "predicates not evaluated rather than using them to override that result. "
    "Use rule 6 only after six NO outcomes. The list and plan are not executed. "
    "In the final pass, follow that same first-true procedure and return only the "
    "required JSON envelope, not the checklist. If changing the proposed decision "
    "between plan and final, regenerate decision, reason_code and amount_cents "
    "coherently under the same rules: hold/ask_info always use 0; approve/refund "
    "always use the exact positive invoice total. Do not retain a zero amount from "
    "an earlier nonmonetary proposal after switching to a monetary decision."
)


def ordered_messages(original_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append an identical static guide; never parse the opaque user payload.

    Only the two-message interface is validated. Conditions are English text
    for the model, not executable tests or a post-generation semantic guard.
    Original system/user text and all extra message fields are preserved.
    """
    if not isinstance(original_messages, list) or len(original_messages) != 2:
        raise ValueError("Expected exactly two original actor messages")
    for message, role in zip(original_messages, ("system", "user")):
        if (not isinstance(message, dict) or message.get("role") != role
                or not isinstance(message.get("content"), str)):
            raise ValueError("Expected original system and user text messages")
    result = deepcopy(original_messages)
    result[0]["content"] = result[0]["content"] + "\n\n" + ORDERED_RULE_GUIDE
    return result


def task_schema(public_step: dict[str, Any]) -> dict[str, Any]:
    """Copy the frozen envelope and bind only case ID / operation vocabulary.

    Returned dictionaries are independent on every call. Amount constraints,
    evidence-check fields, required properties, and the tool remain unchanged.
    The schema does not connect a particular reason to a decision or force the
    first applicable business rule; those remain the model's responsibility.
    """
    if not isinstance(public_step, dict):
        raise ValueError("Public step must be an object")
    task = public_step.get("task")
    if not isinstance(task, dict):
        raise ValueError("Public task must be an object")
    case_id, operation = task.get("case_id"), task.get("operation")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("Public task case_id must be a nonempty string")
    if not isinstance(operation, str) or operation not in ("payout", "refund"):
        raise ValueError("Public task operation must be payout or refund")

    result = deepcopy(ACTOR_SCHEMA)
    arguments = result["properties"]["arguments"]["properties"]
    arguments["case_id"]["const"] = case_id
    arguments["decision"]["enum"] = ["hold", "ask_info", "approve" if operation == "payout" else "refund"]
    excluded = {"eligible_refund", "return_rejected"} if operation == "payout" else {"authorized_payout"}
    arguments["reason_code"]["enum"] = [reason for reason in arguments["reason_code"]["enum"] if reason not in excluded]
    return result
