"""Task-bound generation schema shared by both v4 memory arms.

Only the literal public task identity and requested operation constrain the
output vocabulary. No observations, memory, self-check, or private reference
are inspected. This is not a semantic safety guard: unsafe monetary decisions
and wrong amounts remain generatable. The unchanged actor extractor still
validates the final response; this module never parses or repairs that response.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from rcwt_agent_actor import ACTOR_SCHEMA


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
