"""Deterministic, public-tool-only rolling memory for the v3 experiment.

This is a schema adapter, not a learned model or a decision engine. Its entire
persistent state is the returned JSON text. Each record puts a textual invoice
identifier next to its facts and explicitly linked source records. There are no
numeric identifier references. An optional, explicitly labelled substring
replacement compresses identifiers *lexically*, never by guessing a relationship
from their names. All legends count toward the budget. Public ownership equality
is materialized, but no business rule, decision or action reason is computed.

Only known structured tool observations and accepted monetary booking receipts
are ingested. Free text, requests, proposed actions and actor self-reports are
not evidence. Missing source fields replace previous values with unknown/null.
The caller must invoke this reducer AFTER a decision, as in the v2 protocol.

If necessary, the oldest explicit invoice/dependency component is evicted as a
whole (shared dependencies remain for other invoices). Unlinked source records
are components too. Recency is represented by record order, not a hidden
history: least-recently-updated first. Forgetting can include an earlier booking;
``truncated`` reports any eviction in this call. Invalid serialized state fails closed, and a budget too
small even for ``{}`` is rejected. No inference, oracle, or evaluation imports.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Callable

from rcwt_agent_memory import MemoryState


_SOURCES = ("invoice", "account", "payment", "return", "booked")
_REFS = {"invoice": (0, 1, 2, 3), "account": (0, 2),
         "payment": (0,), "return": (0,), "booked": (0,)}
_STATUSES = {"account": {"unknown", "active", "blocked", "revoked"},
             "payment": {"unknown", "pending", "cleared"},
             "return": {"unknown", "accepted", "rejected"}}
_TOOLS = {"read_invoice": "invoice", "read_account": "account",
          "read_payment": "payment", "read_return": "return"}


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("Memory and information must be JSON strings")
    try:
        value = json.loads(text, object_pairs_hook=_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               ValueError("Non-finite JSON value")))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Malformed memory information") from exc
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _identifier(value: Any, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Identifiers must be nonempty strings")
    return value


def _cents(value: Any, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("Amounts must be nonnegative integer cents")
    return value


def _status(source: str, value: Any) -> str:
    if value is None:
        return "unknown"
    if not isinstance(value, str) or value not in _STATUSES[source]:
        raise ValueError(f"Invalid {source} status")
    return value


def _empty() -> dict:
    return {"order": [], **{source: {} for source in _SOURCES}}


def _row(source: str, content: dict) -> list:
    """Read exactly one source record; omitted fields erase stale facts."""
    if source == "invoice":
        return [_identifier(content.get("invoice_id")),
                _identifier(content.get("account_id"), optional=True),
                _identifier(content.get("merchant_id"), optional=True),
                _identifier(content.get("order_id"), optional=True),
                _cents(content.get("total_cents"), optional=True),
                _identifier(content.get("currency"), optional=True)]
    if source == "account":
        return [_identifier(content.get("account_id")),
                _status(source, content.get("verification_status")),
                _identifier(content.get("holder_merchant_id"), optional=True)]
    if source == "payment":
        return [_identifier(content.get("invoice_id")),
                _status(source, content.get("status"))]
    if source == "return":
        return [_identifier(content.get("order_id")),
                _status(source, content.get("inspection_status"))]
    raise ValueError("Unknown source")


def _validate_row(source: str, row: list) -> None:
    lengths = {"invoice": 6, "account": 3, "payment": 2,
               "return": 2, "booked": 3}
    if not isinstance(row, list) or len(row) != lengths[source]:
        raise ValueError("Invalid source row width")
    for index in _REFS[source]:
        _identifier(row[index], optional=index != 0)
    if source == "invoice":
        _cents(row[4], optional=True)
        _identifier(row[5], optional=True)
    elif source in _STATUSES:
        if _status(source, row[1]) != row[1]:
            raise ValueError("Stored status must be explicit")
    elif not isinstance(row[1], str) or row[1] not in {"payout", "refund"} or _cents(row[2]) == 0:
        raise ValueError("Invalid monetary booking row")


def _key(source: str, row: list) -> str | tuple:
    return (row[0], row[1]) if source == "booked" else row[0]


def _used_ids(state: dict) -> set[str]:
    return {row[index] for source in _SOURCES for row in state[source].values()
            for index in _REFS[source] if row[index] is not None}


def _groups(state: dict) -> set[tuple[str, str]]:
    anchors = set(state["invoice"]) | set(state["payment"]) | {
        key[0] for key in state["booked"]}
    accounts = {row[1] for row in state["invoice"].values()}
    orders = {row[3] for row in state["invoice"].values()}
    return ({("invoice", identifier) for identifier in anchors}
            | {("account", identifier) for identifier in state["account"] if identifier not in accounts}
            | {("return", identifier) for identifier in state["return"] if identifier not in orders})


def _sync_order(state: dict) -> None:
    groups = _groups(state)
    retained = [key for key in state["order"] if key in groups]
    # A dependency detached by a new invoice record is not newly observed.
    # Keep it as an oldest orphan unless an actual observation touches it.
    state["order"] = sorted(groups - set(retained)) + retained


def _records(state: dict) -> list[dict]:
    """Materialize only explicit joins; shared source facts remain consistent."""
    result = []
    for source, identifier in state["order"]:
        if source == "account":
            row = state[source][identifier]
            result.append({"account": row[0], "account_status": row[1], "holder": row[2]})
            continue
        if source == "return":
            row = state[source][identifier]
            result.append({"order": row[0], "return": row[1]})
            continue
        record = {"invoice": identifier}
        invoice = state["invoice"].get(identifier)
        if invoice is not None:
            record.update(cents=invoice[4], currency=invoice[5], account=invoice[1],
                          merchant=invoice[2], order=invoice[3])
            account = state["account"].get(invoice[1])
            if account is not None:
                record.update(account_status=account[1], holder=account[2])
            ownership = "unknown"
            if invoice[2] is not None and account is not None and account[2] is not None:
                ownership = "yes" if invoice[2] == account[2] else "no"
            record["ownership_match"] = ownership
            returned = state["return"].get(invoice[3])
            if returned is not None:
                record["return"] = returned[1]
        payment = state["payment"].get(identifier)
        if payment is not None:
            record["payment"] = payment[1]
        for operation in ("payout", "refund"):
            booked = state["booked"].get((identifier, operation))
            if booked is not None:
                record["booked_" + operation] = booked[2]
        result.append(record)
    return result


def _load(text: str) -> dict:
    """Decode and validate all persisted state; useful for offline inspection."""
    if text == "":
        return _empty()
    raw = _json(text)
    if not raw:
        return _empty()
    if set(raw) - {"records", "replace_in_ids"}:
        raise ValueError("Unknown serialized memory field")
    records = raw.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("A nonempty ledger requires complete records")
    marker = fragment = None
    if "replace_in_ids" in raw:
        replacement = raw["replace_in_ids"]
        if not isinstance(replacement, dict) or len(replacement) != 1:
            raise ValueError("Invalid identifier replacement legend")
        marker, fragment = next(iter(replacement.items()))
        if len(marker) != 1 or not isinstance(fragment, str) or not fragment or marker in fragment:
            raise ValueError("Invalid identifier replacement")
    state = _empty()
    decoded = []
    replacement_used = False

    def insert(source: str, row: list) -> None:
        _validate_row(source, row)
        key = _key(source, row)
        if key in state[source] and state[source][key] != row:
            raise ValueError("Conflicting copies of a shared source")
        state[source][key] = row

    for encoded in records:
        if not isinstance(encoded, dict):
            raise ValueError("Each memory record must be an object")
        record = dict(encoded)
        for field in ("invoice", "account", "merchant", "order", "holder"):
            if field not in record:
                continue
            value = _identifier(record[field], optional=field != "invoice")
            if value is not None and marker is not None:
                replacement_used |= marker in value
                value = value.replace(marker, fragment)
            record[field] = value
        if "invoice" in record:
            identifier = record["invoice"]
            allowed = {"invoice", "cents", "currency", "account", "merchant", "order",
                       "account_status", "holder", "ownership_match", "payment", "return",
                       "booked_payout", "booked_refund"}
            if set(record) - allowed:
                raise ValueError("Unknown invoice fact")
            if "cents" in record:
                required = {"account", "merchant", "order", "currency", "ownership_match"}
                if not required <= set(record):
                    raise ValueError("Incomplete invoice source")
                insert("invoice", [identifier, record["account"], record["merchant"],
                                   record["order"], record["cents"], record["currency"]])
            if "account_status" in record or "holder" in record:
                if ("cents" not in record or record.get("account") is None
                        or not {"account_status", "holder"} <= set(record)):
                    raise ValueError("Account facts lack their explicit invoice link")
                insert("account", [record["account"], record["account_status"], record["holder"]])
            if "return" in record:
                if "cents" not in record or record.get("order") is None:
                    raise ValueError("Return facts lack their explicit invoice link")
                insert("return", [record["order"], record["return"]])
            if "payment" in record:
                insert("payment", [identifier, record["payment"]])
            for operation in ("payout", "refund"):
                if "booked_" + operation in record:
                    insert("booked", [identifier, operation, record["booked_" + operation]])
            group = ("invoice", identifier)
        elif set(record) == {"account", "account_status", "holder"}:
            insert("account", [record["account"], record["account_status"], record["holder"]])
            group = ("account", record["account"])
        elif set(record) == {"order", "return"}:
            insert("return", [record["order"], record["return"]])
            group = ("return", record["order"])
        else:
            raise ValueError("Unrecognized or incomplete source record")
        if group in state["order"]:
            raise ValueError("Duplicate memory component")
        state["order"].append(group)
        decoded.append(record)
    if marker is not None and not replacement_used:
        raise ValueError("Unused identifier replacement")
    if set(state["order"]) != _groups(state) or _records(state) != decoded:
        # Re-materialization checks ownership equality, source completeness and
        # consistent copies of shared facts. It never applies decision rules.
        raise ValueError("Memory records disagree with their explicit source facts")
    return state


def _put(state: dict, source: str, row: list) -> None:
    state[source][_key(source, row)] = row
    _sync_order(state)
    if source in ("invoice", "payment", "booked"):
        touched = {("invoice", row[0])}
    else:
        position = 1 if source == "account" else 3
        touched = {("invoice", invoice[0]) for invoice in state["invoice"].values()
                   if invoice[position] == row[0]}
        if not touched:
            touched = {(source, row[0])}
    # Component order is the only recency metadata; it is in the counted text.
    state["order"] = ([key for key in state["order"] if key not in touched]
                      + [key for key in state["order"] if key in touched])


def _ingest(state: dict, information: dict) -> None:
    observations = information.get("observations", [])
    if not isinstance(observations, list):
        raise ValueError("Observations must be a list")
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("Observation must be an object")
        if observation.get("source") != "tool":
            continue
        tool = observation.get("tool")
        if not isinstance(tool, str) or tool not in _TOOLS:
            continue
        content = observation.get("content")
        if not isinstance(content, dict):
            raise ValueError("Structured tool content must be an object")
        source = _TOOLS[tool]
        _put(state, source, _row(source, content))
    # A proposed action (even an apparently valid one) is never a receipt.
    receipt = information.get("tool_result")
    if not isinstance(receipt, dict) or receipt.get("tool") != "record_decision":
        return
    if receipt.get("accepted") is not True:
        return
    decision = receipt.get("decision")
    amount = receipt.get("amount_booked_cents")
    if decision not in ("approve", "refund") or type(amount) is not int or amount <= 0:
        return
    case_id = _identifier(receipt.get("case_id"))
    _put(state, "booked", [case_id, "payout" if decision == "approve" else "refund", amount])


def _dump(state: dict, replacement: tuple[str, str] | None = None) -> str:
    records = _records(state)
    if not records:
        return "{}"
    raw: dict = {}
    if replacement is not None:
        marker, fragment = replacement
        raw["replace_in_ids"] = {marker: fragment}
        for record in records:
            for field in ("invoice", "account", "merchant", "order", "holder"):
                if record.get(field) is not None:
                    record[field] = record[field].replace(fragment, marker)
    raw["records"] = records
    return json.dumps(raw, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _represent(state: dict, tokenize: Callable[[str], list[int]]) -> tuple[str, int]:
    """Choose a lossless spelling by actual tokens, never by character length."""
    plain = _dump(state)
    best = (plain, len(tokenize(plain)))
    ids = sorted(_used_ids(state))
    marker = next((char for char in "~^§" if all(char not in value for value in ids)), None)
    if marker is None or len(ids) < 2:
        return best
    # Generic lexical matches: no namespace, family, split or ID pattern rules.
    fragments: set[str] = set()
    for left, right in combinations(ids, 2):
        match = SequenceMatcher(None, left, right, autojunk=False).find_longest_match()
        if match.size >= 4:
            fragments.add(left[match.a:match.a + match.size])
    ranked = sorted(fragments, key=lambda fragment: (
        -(sum(value.count(fragment) for value in ids) - 1) * (len(fragment) - 1), fragment))
    for fragment in ranked[:3]:
        text = _dump(state, (marker, fragment))
        size = len(tokenize(text))
        if (size, len(text), text) < (best[1], len(best[0]), best[0]):
            best = (text, size)
    return best


def _evict_oldest(state: dict) -> None:
    """Drop one whole logical component, retaining still-shared source rows."""
    if not state["order"]:
        raise ValueError("Cannot evict an empty memory")
    source, identifier = state["order"][0]
    if source != "invoice":
        del state[source][identifier]
    else:
        invoice = state["invoice"].pop(identifier, None)
        state["payment"].pop(identifier, None)
        for operation in ("payout", "refund"):
            state["booked"].pop((identifier, operation), None)
        if invoice is not None:
            for dependency, position in (("account", 1), ("return", 3)):
                linked = invoice[position]
                if not any(row[position] == linked for row in state["invoice"].values()):
                    state[dependency].pop(linked, None)
    _sync_order(state)


def compact_structured(tokenize: Callable[[str], list[int]], previous_memory: str,
                       new_information: str, budget: int) -> MemoryState:
    """Reduce public records after an action, with zero completion calls.

    The tokenizer must be the actor's real tokenizer for a real experiment.
    No detokenizer is needed: eviction always retains complete JSON records.
    """
    if type(budget) is not int or budget <= 0:
        raise ValueError("Memory budget must be a positive integer")
    if not callable(tokenize):
        raise ValueError("A tokenizer callable is required")
    if len(tokenize("{}")) > budget:
        raise ValueError("Budget cannot represent an empty valid memory")
    state = _load(previous_memory)
    _ingest(state, _json(new_information))
    truncated = False
    while True:
        text, size = _represent(state, tokenize)
        if size <= budget:
            # The decode check also guards accidental non-reversible compaction.
            if _load(text) != state:
                raise ValueError("Serialized memory failed lossless round-trip")
            return MemoryState(text=text, calls=[], truncated=truncated)
        _evict_oldest(state)
        truncated = True
