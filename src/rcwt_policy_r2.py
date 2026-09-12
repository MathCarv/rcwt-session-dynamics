"""Restricted R2 retention parameters and identity-free TRAIN feedback.

Weights choose which whole public-memory component to evict, and nothing else.
They cannot write facts, inspect tasks, authorize an action or change an actor's
prompt. Zero weights exactly reproduce v4's oldest-component reducer, including
tokenizer calls. The feedback builder is an offline TRAIN-only critic adapter;
it is never called by the online memory writer. A later error is an association,
not proof that eviction caused that error.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import random
from typing import Any, Callable

from rcwt_agent_memory import MemoryState
from rcwt_memory_v3 import _evict_oldest, _ingest, _json, _load, _represent


FEATURES = ("invoice", "account", "payment", "return", "booked", "incomplete")
POLICY_VERSION = "rcwt-retention-policy/1"
FEEDBACK_VERSION = "rcwt-retention-feedback/1"
DEFAULT_POLICY = {"schema": POLICY_VERSION, "weights": {key: 0 for key in FEATURES}}
POLICY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["schema", "weights"],
    "properties": {
        "schema": {"type": "string", "const": POLICY_VERSION},
        "weights": {
            "type": "object", "additionalProperties": False,
            "required": list(FEATURES),
            "properties": {key: {"type": "integer", "minimum": -4, "maximum": 4}
                           for key in FEATURES},
        },
    },
}


def validate_policy(value: Any) -> dict:
    """Return a fresh canonical value; reject coercion and executable content."""
    if (not isinstance(value, dict) or set(value) != {"schema", "weights"}
            or value["schema"] != POLICY_VERSION):
        raise ValueError("Invalid retention policy schema")
    weights = value["weights"]
    if (not isinstance(weights, dict) or set(weights) != set(FEATURES)
            or any(type(weights[key]) is not int or not -4 <= weights[key] <= 4
                   for key in FEATURES)):
        raise ValueError("Policy needs exactly six integer weights in [-4, 4]")
    return {"schema": POLICY_VERSION, "weights": {key: weights[key] for key in FEATURES}}


def parse_policy(text: str, finish_reason: str = "stop") -> dict:
    if finish_reason != "stop":
        raise ValueError("Truncated/non-final policy proposal")
    return validate_policy(_json(text))


def _features(state: dict, component: tuple[str, str]) -> dict[str, int]:
    kind, identifier = component
    invoice = state["invoice"].get(identifier) if kind == "invoice" else None
    account = (state["account"].get(invoice[1]) if invoice is not None else
               state["account"].get(identifier) if kind == "account" else None)
    payment = state["payment"].get(identifier) if kind == "invoice" else None
    returned = (state["return"].get(invoice[3]) if invoice is not None else
                state["return"].get(identifier) if kind == "return" else None)
    booked = kind == "invoice" and any(
        (identifier, operation) in state["booked"] for operation in ("payout", "refund"))
    if kind == "invoice":
        incomplete = (invoice is None or any(value is None for value in invoice[1:])
                      or account is None or account[1] == "unknown" or account[2] is None
                      or payment is None or payment[1] == "unknown")
    elif kind == "account":
        incomplete = account[1] == "unknown" or account[2] is None
    else:
        incomplete = returned[1] == "unknown"
    return dict(zip(FEATURES, map(int, (invoice is not None, account is not None,
                                      payment is not None, returned is not None,
                                      booked, incomplete))))


def compact_policy(tokenize: Callable[[str], list[int]], previous_memory: str,
                   new_information: str, budget: int, policy: dict
                   ) -> tuple[MemoryState, list[dict]]:
    """Same counted state/representation as v4; only eviction ranking differs.

    Candidate snapshots are audit output, not additional actor memory. Score is
    sum(weight * public presence/unknown feature); smallest is evicted first.
    Original recency breaks ties and surviving relative order is unchanged.
    """
    policy = validate_policy(policy)
    if type(budget) is not int or budget <= 0:
        raise ValueError("Memory budget must be a positive integer")
    if not callable(tokenize):
        raise ValueError("A tokenizer callable is required")
    if len(tokenize("{}")) > budget:
        raise ValueError("Budget cannot represent an empty valid memory")
    state = _load(previous_memory)
    _ingest(state, _json(new_information))
    candidates = [{"component": list(component), "features": _features(state, component),
                   "evicted": False} for component in state["order"]]
    truncated = False
    while True:
        text, size = _represent(state, tokenize)
        if size <= budget:
            if _load(text) != state:
                raise ValueError("Serialized memory failed lossless round-trip")
            retained = set(state["order"])
            for candidate in candidates:
                candidate["evicted"] = tuple(candidate["component"]) not in retained
            return MemoryState(text=text, calls=[], truncated=truncated), candidates
        weights = policy["weights"]
        victim = min(state["order"], key=lambda component: sum(
            weights[key] * value for key, value in _features(state, component).items()))
        state["order"] = [victim] + [key for key in state["order"] if key != victim]
        _evict_oldest(state)
        truncated = True


def _profile(features: Any) -> tuple[int, ...]:
    if (not isinstance(features, dict) or set(features) != set(FEATURES)
            or any(type(features[key]) is not int or features[key] not in (0, 1)
                   for key in FEATURES)):
        raise ValueError("Invalid public retention features")
    return tuple(features[key] for key in FEATURES)


def _aggregate(examples: list[tuple], *, episodes: int, excluded: int) -> dict:
    counts = Counter(examples)
    profiles = []
    for profile in sorted({item[0] for item in counts}):
        cells = [{"evicted": item[1], "requested_later": item[2],
                  "later_raw_failed": item[3], "count": count}
                 for item, count in sorted(counts.items()) if item[0] == profile]
        profiles.append({"features": dict(zip(FEATURES, profile)), "cells": cells})
    return {"schema": FEEDBACK_VERSION, "split": "train", "episodes": episodes,
            "scope": "invoice_components_first_later_request",
            "excluded_orphan_candidates": excluded, "examples": len(examples),
            "profiles": profiles}


def build_feedback(train_rows: list[dict]) -> dict:
    """Use completed TRAIN trajectories only; emit no identifiers or answers.

    Each invoice candidate after a TRAIN step is paired with the first later
    request for that literal invoice in the same episode. The label records
    whether that later raw decision failed. Account/return orphans have no
    literal task-case mapping and are excluded rather than labelled unneeded.
    Caller must bind these rows to a sealed TRAIN corpus before using them.
    """
    if not isinstance(train_rows, list) or not train_rows:
        raise ValueError("Nonempty TRAIN rows required")
    groups: dict[tuple, list] = defaultdict(list)
    for row in train_rows:
        if (not isinstance(row, dict) or row.get("phase") != "train"
                or row.get("arm") != "fixed"):
            raise ValueError("Feedback accepts fixed TRAIN rows only")
        replica, episode = row.get("replica"), row.get("episode_id")
        if type(replica) is not int or replica < 0 or not isinstance(episode, str) or not episode:
            raise ValueError("TRAIN trajectory identity missing")
        if type(row.get("step_index")) is not int or row["step_index"] < 0:
            raise ValueError("TRAIN step index invalid")
        task = row.get("public_step", {}).get("task", {})
        if not isinstance(task.get("case_id"), str) or not task["case_id"]:
            raise ValueError("TRAIN public task case missing")
        if type(row.get("score", {}).get("success")) is not bool:
            raise ValueError("TRAIN raw success label missing")
        groups[(replica, episode)].append(row)
    examples, excluded = [], 0
    for rows in groups.values():
        rows.sort(key=lambda row: row["step_index"])
        if [row["step_index"] for row in rows] != list(range(len(rows))):
            raise ValueError("TRAIN trajectory must be contiguous and unique")
        for index, row in enumerate(rows):
            candidates = row.get("retention_candidates")
            if not isinstance(candidates, list):
                raise ValueError("TRAIN retention candidate audit missing")
            seen = set()
            for candidate in candidates:
                if not isinstance(candidate, dict) or set(candidate) != {"component", "features", "evicted"}:
                    raise ValueError("Invalid retention candidate")
                component = candidate["component"]
                if (not isinstance(component, list) or len(component) != 2
                        or component[0] not in {"invoice", "account", "return"}
                        or not isinstance(component[1], str) or not component[1]
                        or type(candidate["evicted"]) is not bool):
                    raise ValueError("Invalid retention component")
                key = tuple(component)
                if key in seen:
                    raise ValueError("Duplicate retention component")
                seen.add(key)
                profile = _profile(candidate["features"])
                if component[0] != "invoice":
                    excluded += 1
                    continue
                later = next((item for item in rows[index + 1:]
                              if item["public_step"]["task"]["case_id"] == component[1]), None)
                examples.append((profile, int(candidate["evicted"]), int(later is not None),
                                 int(later is not None and not later["score"]["success"])))
    return _aggregate(examples, episodes=len(groups), excluded=excluded)


def _expand_feedback(feedback: dict) -> list[tuple]:
    required = {"schema", "split", "episodes", "scope", "excluded_orphan_candidates", "examples", "profiles"}
    if (not isinstance(feedback, dict) or set(feedback) != required
            or feedback["schema"] != FEEDBACK_VERSION or feedback["split"] != "train"
            or feedback["scope"] != "invoice_components_first_later_request"):
        raise ValueError("Invalid TRAIN feedback schema")
    if any(type(feedback[key]) is not int or feedback[key] < 0
           for key in ("episodes", "excluded_orphan_candidates", "examples")):
        raise ValueError("Invalid feedback count")
    if feedback["episodes"] < 1 or not isinstance(feedback["profiles"], list):
        raise ValueError("Invalid feedback profiles")
    examples = []
    for item in feedback["profiles"]:
        if not isinstance(item, dict) or set(item) != {"features", "cells"} or not isinstance(item["cells"], list):
            raise ValueError("Invalid feedback profile")
        profile = _profile(item["features"])
        for cell in item["cells"]:
            if (not isinstance(cell, dict)
                    or set(cell) != {"evicted", "requested_later", "later_raw_failed", "count"}
                    or any(type(cell[key]) is not int or cell[key] not in (0, 1)
                           for key in ("evicted", "requested_later", "later_raw_failed"))
                    or type(cell["count"]) is not int or not 1 <= cell["count"] <= 100000
                    or cell["later_raw_failed"] > cell["requested_later"]):
                raise ValueError("Invalid feedback cell")
            if len(examples) + cell["count"] > 100000:
                raise ValueError("Feedback exceeds bounded critic input")
            examples.extend([(profile, cell["evicted"], cell["requested_later"],
                              cell["later_raw_failed"])] * cell["count"])
    if _aggregate(examples, episodes=feedback["episodes"],
                  excluded=feedback["excluded_orphan_candidates"]) != feedback:
        raise ValueError("Feedback must be canonical with consistent counts")
    return examples


def shuffle_feedback(feedback: dict, seed: int) -> dict:
    """Permute profile/outcome associations, preserving both exact marginals."""
    if type(seed) is not int:
        raise ValueError("Shuffle seed must be an integer")
    examples = _expand_feedback(feedback)
    profiles = [item[0] for item in examples]
    random.Random(seed).shuffle(profiles)
    shuffled = [(profile, *item[1:]) for profile, item in zip(profiles, examples)]
    return _aggregate(shuffled, episodes=feedback["episodes"],
                      excluded=feedback["excluded_orphan_candidates"])


def feedback_json(feedback: dict) -> str:
    """Validate the exact allowlist before crossing into a model prompt."""
    _expand_feedback(feedback)
    return json.dumps(feedback, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
