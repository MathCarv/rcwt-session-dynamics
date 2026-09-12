"""Public-fact context over bounded retained memory and the current step.

This adapter joins structured evidence, not business rules. It decodes only the
supplied persistent text, overlays the four known public read tools in observed
order, and selects the requested invoice by its literal ID. The last record for
a source/ID replaces that entire source record; omitted fields erase old facts.
Joins happen only after every current observation has been processed.

Unlike the v3 reader, current values are included, including on an empty-memory
first step. The actor also receives its unchanged raw current step. This is an
engineered evidence-normalization intervention, not compaction alone or learned
policy improvement. No action, recommendation, rule, reason, or private feedback
is computed. Only persisted prior receipts can establish an existing booking;
``no_record`` means no retained record, not proof that no booking ever occurred.

Both the supplied persistent text and the returned card must fit ``budget`` real
model tokens. Current observations are already-visible transient input, not an
additional persistent store. No sources are dropped before joining. Overflow
fails closed rather than truncating JSON. This transient card MUST NOT replace
the writer's previous_memory; all cross-step persistence remains in that text.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from rcwt_memory_v3 import _TOOLS, _identifier, _key, _load, _row


def _token_count(text: str, tokenize: Callable[[str], list[int]]) -> int:
    tokens = tokenize(text)
    if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Tokenizer must return a list of nonnegative integer token IDs")
    return len(tokens)


def _overlay_current(state: dict, observations: Any) -> None:
    """Overlay only structured read records, never receipts or proposed actions."""
    if not isinstance(observations, list):
        raise ValueError("Public observations must be a list")
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("Public observation must be an object")
        if observation.get("source") != "tool":
            continue
        tool = observation.get("tool")
        if not isinstance(tool, str) or tool not in _TOOLS:
            continue
        content = observation.get("content")
        if not isinstance(content, dict):
            raise ValueError("Known public tool content must be an object")
        source = _TOOLS[tool]
        row = _row(source, content)
        # This is an ephemeral copy decoded from text. Do not update writer
        # recency, ingest a receipt, run eviction, or serialize a new memory.
        state[source][_key(source, row)] = row


def build_context(previous_memory: str, current_public_step: dict, *,
                  tokenize: Callable[[str], list[int]], budget: int = 256) -> str:
    """Return named public facts and literal join IDs within the token cap.

    The caller meters every tokenizer invocation and must use the actor's real
    tokenizer in a model experiment. This function makes no completion calls.
    An absent case still yields explicit unknowns, with no cross-case guessing.
    """
    if type(budget) is not int or budget <= 0:
        raise ValueError("Context budget must be a positive integer")
    if not callable(tokenize):
        raise ValueError("A tokenizer callable is required")
    if not isinstance(previous_memory, str):
        raise ValueError("Stored memory must be a string")
    if not isinstance(current_public_step, dict):
        raise ValueError("Current public step must be an object")
    task = current_public_step.get("task")
    if not isinstance(task, dict):
        raise ValueError("Current public task must be an object")
    case_id = _identifier(task.get("case_id"))
    operation = task.get("operation")
    if not isinstance(operation, str) or operation not in {"payout", "refund"}:
        raise ValueError("Unsupported public operation")

    if previous_memory and _token_count(previous_memory, tokenize) > budget:
        raise ValueError("Stored memory exceeds token budget")
    state = _load(previous_memory)
    _overlay_current(state, current_public_step.get("observations", []))

    invoice = state["invoice"].get(case_id)
    account_id = invoice[1] if invoice is not None else None
    merchant_id = invoice[2] if invoice is not None else None
    order_id = invoice[3] if invoice is not None else None
    account = state["account"].get(account_id)
    holder_id = account[2] if account is not None else None
    ownership = "unknown"
    if merchant_id is not None and holder_id is not None:
        ownership = "yes" if merchant_id == holder_id else "no"
    facts = {
        "invoice_amount_cents": invoice[4] if invoice is not None else None,
        "account_status": account[1] if account is not None else "unknown",
        "ownership_match": ownership,
        "payment_status": state["payment"].get(case_id, [case_id, "unknown"])[1],
        "return_status": state["return"].get(order_id, [order_id, "unknown"])[1],
        "operation_already_booked": "yes" if (case_id, operation) in state["booked"] else "no_record",
    }
    ids = {key: value for key, value in (
        ("account_id", account_id), ("merchant_id", merchant_id),
        ("holder_merchant_id", holder_id), ("order_id", order_id)) if value is not None}
    card = {"case_id": case_id, "operation": operation,
            "available_evidence": facts, "linked_ids": ids}
    text = json.dumps(card, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    size = _token_count(text, tokenize)
    if size > budget:
        raise ValueError(f"Context card exceeds token budget: {size} > {budget}")
    return text
