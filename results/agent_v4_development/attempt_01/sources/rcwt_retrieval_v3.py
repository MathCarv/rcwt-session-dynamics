"""Stateless read policy over the already retained structured memory.

This is a public-schema adapter, not a model, oracle, or decision engine. It
selects one invoice by literal task identity and projects old public facts into
a named card. Current observations supply ONLY source/identity invalidation
signals: their amounts, statuses, holders and new links are never copied or
recomputed here. The actor still receives the original current step separately.

The returned card is transient. It must never replace the original stored text
or become the writer's previous_memory. Empty memory or an absent case returns
the identical empty string, without a hint or tokenization call. Nonempty cards
are counted with the supplied tokenizer and rejected on overflow, never cut.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from rcwt_memory_v3 import _load


_FACT_FIELDS = (
    "invoice_amount_cents", "account_status", "ownership_match", "payment_status",
    "return_status", "operation_already_booked",
)
_ID_FIELDS = ("account_id", "merchant_id", "holder_merchant_id", "order_id")
_PRIMARY_IDS = {"read_invoice": "invoice_id", "read_account": "account_id",
                "read_payment": "invoice_id", "read_return": "order_id"}


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Public identities must be nonempty strings")
    return value


def _updates(public_step: dict) -> set[tuple[str, str]]:
    """Read source and primary ID only; deliberately ignore all current values."""
    observations = public_step.get("observations", [])
    if not isinstance(observations, list):
        raise ValueError("Public observations must be a list")
    result = set()
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("Public observation must be an object")
        if observation.get("source") != "tool":
            continue
        tool = observation.get("tool")
        if not isinstance(tool, str) or tool not in _PRIMARY_IDS:
            continue
        content = observation.get("content")
        if not isinstance(content, dict):
            raise ValueError("Known public tool content must be an object")
        identifier = _identifier(content.get(_PRIMARY_IDS[tool]))
        result.add((tool, identifier))
    return result


def read_memory(previous_memory: str, current_public_step: dict, *,
                tokenize: Callable[[str], list[int]], budget: int = 256) -> str:
    """Return a token-capped old-fact card; never ingest current source values.

    ``budget`` is an exact model-token limit, not a character estimate. The
    experiment caller owns its frozen value (256) and meters these tokenizer
    calls before invoking the actor. Overflow raises ValueError before inference.
    """
    if type(budget) is not int or budget <= 0:
        raise ValueError("Read-memory budget must be a positive integer")
    if not callable(tokenize):
        raise ValueError("A tokenizer callable is required")
    if not isinstance(previous_memory, str):
        raise ValueError("Stored memory must be a string")
    if previous_memory == "":
        return ""
    if not isinstance(current_public_step, dict):
        raise ValueError("Current public step must be an object")
    task = current_public_step.get("task")
    if not isinstance(task, dict):
        raise ValueError("Current public task must be an object")
    case_id = _identifier(task.get("case_id"))
    operation = task.get("operation")
    if not isinstance(operation, str) or operation not in {"payout", "refund"}:
        raise ValueError("Unsupported public operation")

    state = _load(previous_memory)
    if (case_id not in state["invoice"] and case_id not in state["payment"]
            and not any(key[0] == case_id for key in state["booked"])):
        return ""

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
    identities = {key: value for key, value in zip(
        _ID_FIELDS, (account_id, merchant_id, holder_id, order_id)) if value is not None}

    updates = _updates(current_public_step)
    masked: set[str] = set()
    if ("read_invoice", case_id) in updates:
        # An invoice update can replace every dependency link. Do not inspect
        # the new account/order/merchant to salvage or construct a new join.
        masked.update(("invoice_amount_cents", "account_status", "ownership_match", "return_status"))
        masked.update(_ID_FIELDS)
    if account_id is not None and ("read_account", account_id) in updates:
        masked.update(("account_status", "ownership_match", "holder_merchant_id"))
    if ("read_payment", case_id) in updates:
        masked.add("payment_status")
    if order_id is not None and ("read_return", order_id) in updates:
        masked.add("return_status")
    # Payment is keyed by invoice, bookings by invoice+operation: replacing an
    # invoice's account/order links does not erase those independent old records.
    card = {
        "case_id": case_id,
        "operation": operation,
        "retained_evidence": {key: value for key, value in facts.items() if key not in masked},
        "retained_ids": {key: value for key, value in identities.items() if key not in masked},
        "invalidated_fields": [key for key in (*_FACT_FIELDS, *_ID_FIELDS) if key in masked],
    }
    text = json.dumps(card, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    tokens = tokenize(text)
    if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Tokenizer must return a list of nonnegative integer token IDs")
    if len(tokens) > budget:
        raise ValueError(f"Read-memory card exceeds token budget: {len(tokens)} > {budget}")
    return text
