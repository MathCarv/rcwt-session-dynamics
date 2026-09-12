"""Deterministic, public-tool-only rolling memory for the v3 experiment.

This is a schema adapter, not a learned model or a decision engine. Its entire
persistent state is the returned JSON text. Rows preserve separate source facts;
numeric identifier references index ``ids`` from zero. An optional, explicitly
labelled substring replacement compresses identifiers *lexically*, never by
guessing a relationship from their names. All legends count toward the budget.

Only known structured tool observations and accepted monetary booking receipts
are ingested. Free text, requests, proposed actions and actor self-reports are
not evidence. Missing source fields replace previous values with unknown/null.
The caller must invoke this reducer AFTER a decision, as in the v2 protocol.

If necessary, the oldest explicit invoice/dependency component is evicted as a
whole (shared dependencies remain for other invoices). Unlinked source records
are components too. Recency is represented by identifier order, not a hidden
history. Forgetting can include an earlier booking; ``truncated`` reports any
eviction in this call. Invalid serialized state fails closed, and a budget too
small even for ``{}`` is rejected. No inference, oracle, or evaluation imports.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Callable

from rcwt_agent_memory import MemoryState


_HEADERS = {
    "invoice": "invoice[id,account,merchant,order,total_cents,currency]",
    "account": "account[id,verification_status,holder_merchant]",
    "payment": "payment[invoice,status]",
    "return": "return[order,inspection_status]",
    "booked": "booked[invoice,operation,cents]",
}
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
    return {"ids": [], **{source: {} for source in _HEADERS}}


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
    return {row[index] for source in _HEADERS for row in state[source].values()
            for index in _REFS[source] if row[index] is not None}


def _load(text: str) -> dict:
    """Decode and validate all persisted state; useful for offline inspection."""
    if text == "":
        return _empty()
    raw = _json(text)
    if not raw:
        return _empty()
    if set(raw) - {"ids (0-based)", "replace_in_ids", *_HEADERS.values()}:
        raise ValueError("Unknown serialized memory field")
    ids = raw.get("ids (0-based)")
    if not isinstance(ids, list) or not ids:
        raise ValueError("A nonempty ledger requires its identifier dictionary")
    ids = [_identifier(value) for value in ids]
    if "replace_in_ids" in raw:
        replacement = raw["replace_in_ids"]
        if not isinstance(replacement, dict) or len(replacement) != 1:
            raise ValueError("Invalid identifier replacement legend")
        marker, fragment = next(iter(replacement.items()))
        if len(marker) != 1 or not isinstance(fragment, str) or not fragment or marker in fragment:
            raise ValueError("Invalid identifier replacement")
        if not any(marker in value for value in ids):
            raise ValueError("Unused identifier replacement")
        ids = [value.replace(marker, fragment) for value in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate identifiers after decoding")
    state = _empty()
    state["ids"] = ids
    for source, header in _HEADERS.items():
        records = raw.get(header, [])
        if not isinstance(records, list) or (header in raw and not records):
            raise ValueError("Invalid source table")
        for encoded in records:
            if not isinstance(encoded, list):
                raise ValueError("Invalid source row")
            row = list(encoded)
            for index in _REFS[source]:
                if index >= len(row):
                    raise ValueError("Incomplete source row")
                reference = row[index]
                if reference is None and index != 0:
                    continue
                if type(reference) is not int or not 0 <= reference < len(ids):
                    raise ValueError("Invalid identifier reference")
                row[index] = ids[reference]
            _validate_row(source, row)
            key = _key(source, row)
            if key in state[source]:
                raise ValueError("Duplicate source identity")
            state[source][key] = row
    if set(ids) != _used_ids(state):
        raise ValueError("Unused identifiers or missing source records")
    return state


def _put(state: dict, source: str, row: list) -> None:
    state[source][_key(source, row)] = row
    for index in _REFS[source]:
        identifier = row[index]
        if identifier is not None and identifier not in state["ids"]:
            state["ids"].append(identifier)
    # Ordering is the only recency metadata and is inside the counted text.
    state["ids"].remove(row[0])
    state["ids"].append(row[0])


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
    used = _used_ids(state)
    ids = [identifier for identifier in state["ids"] if identifier in used]
    if not ids:
        return "{}"
    index = {identifier: number for number, identifier in enumerate(ids)}
    raw: dict = {}
    if replacement is not None:
        marker, fragment = replacement
        raw["replace_in_ids"] = {marker: fragment}
        raw["ids (0-based)"] = [identifier.replace(fragment, marker) for identifier in ids]
    else:
        raw["ids (0-based)"] = ids
    for source, header in _HEADERS.items():
        if not state[source]:
            continue
        rows = []
        for original in sorted(state[source].values(),
                               key=lambda row: (index[row[0]], str(_key(source, row)))):
            row = list(original)
            for position in _REFS[source]:
                row[position] = None if row[position] is None else index[row[position]]
            rows.append(row)
        raw[header] = rows
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
    ranks = {identifier: rank for rank, identifier in enumerate(state["ids"])}
    anchors = set(state["invoice"]) | set(state["payment"]) | {
        key[0] for key in state["booked"]}
    components: list[tuple[int, int, str, set[tuple[str, Any]]]] = []
    referenced: set[tuple[str, Any]] = set()
    for identifier in anchors:
        records = {(source, identifier) for source in ("invoice", "payment")
                   if identifier in state[source]}
        records.update(("booked", key) for key in state["booked"] if key[0] == identifier)
        invoice = state["invoice"].get(identifier)
        if invoice:
            for source, linked in (("account", invoice[1]), ("return", invoice[3])):
                if linked in state[source]:
                    records.add((source, linked))
                    referenced.add((source, linked))
        recency = max(ranks[state[source][key][0]] for source, key in records)
        components.append((recency, len(records), identifier, records))
    for source in ("account", "return"):
        for identifier in state[source]:
            if (source, identifier) not in referenced:
                components.append((ranks[identifier], 1, identifier, {(source, identifier)}))
    if not components:
        raise ValueError("Cannot evict an empty memory")
    records = min(components, key=lambda component: component[:3])[3]
    for source, key in records:
        if source in ("account", "return"):
            position = 1 if source == "account" else 3
            if any(invoice[position] == key and ("invoice", invoice[0]) not in records
                   for invoice in state["invoice"].values()):
                continue
        del state[source][key]
    used = _used_ids(state)
    state["ids"] = [identifier for identifier in state["ids"] if identifier in used]


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
            if _load(text) != {**state, "ids": [value for value in state["ids"]
                                                if value in _used_ids(state)]}:
                raise ValueError("Serialized memory failed lossless round-trip")
            return MemoryState(text=text, calls=[], truncated=truncated)
        _evict_oldest(state)
        truncated = True
