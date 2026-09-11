"""Typed, deterministic data contracts for the RCWT session benchmark (RCWT-S).

The public corpus, private oracle, and treatment outputs deliberately use
separate records.  In particular, :class:`PublicCase` never embeds an answer
key and :class:`ContextRecord` can be reproduced without loading an oracle.
All hashes in this module use compact, key-sorted UTF-8 JSON without a trailing
newline unless a function explicitly documents file-level hashing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence, TypeAlias

Family: TypeAlias = Literal[
    "update-heavy",
    "dependency-heavy",
    "chronology-sensitive",
    "insufficient-evidence",
]
Treatment: TypeAlias = Literal["tail", "state_latest", "state_first"]
Disposition: TypeAlias = Literal["ready", "insufficient"]
JsonObject: TypeAlias = dict[str, Any]

FAMILIES: tuple[Family, ...] = (
    "update-heavy",
    "dependency-heavy",
    "chronology-sensitive",
    "insufficient-evidence",
)
TREATMENTS: tuple[Treatment, ...] = ("tail", "state_latest", "state_first")
DEFAULT_CHECKPOINTS: tuple[int, ...] = (8, 16, 32, 64)
DEFAULT_BUDGETS: tuple[int, ...] = (256, 512, 1024)
SCHEMA_VERSION = "rcwt-s/1"


def canonical_json(value: object) -> str:
    """Return the canonical JSON representation used by all RCWT-S hashes."""

    if hasattr(value, "to_dict"):
        value = value.to_dict()  # type: ignore[union-attr]
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    """Hash a value after canonical JSON serialization."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``payload``."""

    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """Return a file-level SHA-256 digest, including JSONL newlines."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_bytes(records: Iterable[object]) -> bytes:
    """Serialize records as deterministic JSONL with one final LF per record."""

    return b"".join(
        canonical_json(record).encode("utf-8") + b"\n" for record in records
    )


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_str(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    items = tuple(_require_str(item, f"{label}[]") for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{label} must not contain duplicates")
    return items


@dataclass(frozen=True, slots=True)
class Event:
    """One immutable event in a public synthetic session."""

    event_id: str
    turn: int
    actor: str
    kind: str
    entity: str
    field: str
    value: str
    supersedes: str | None
    depends_on: tuple[str, ...]
    rendered: str

    def __post_init__(self) -> None:
        _require_str(self.event_id, "event.event_id")
        _require_int(self.turn, "event.turn", minimum=1)
        _require_str(self.actor, "event.actor")
        _require_str(self.kind, "event.kind")
        _require_str(self.entity, "event.entity")
        _require_str(self.field, "event.field")
        _require_str(self.value, "event.value", allow_empty=True)
        if self.supersedes is not None:
            _require_str(self.supersedes, "event.supersedes")
            if self.supersedes == self.event_id:
                raise ValueError("event.supersedes cannot reference itself")
        if not isinstance(self.depends_on, tuple):
            object.__setattr__(self, "depends_on", tuple(self.depends_on))
        for dependency in self.depends_on:
            _require_str(dependency, "event.depends_on[]")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("event.depends_on must not contain duplicates")
        if self.event_id in self.depends_on:
            raise ValueError("event.depends_on cannot reference itself")
        _require_str(self.rendered, "event.rendered")

    def to_dict(self) -> JsonObject:
        """Convert this event to its exact public JSON shape."""

        return {
            "event_id": self.event_id,
            "turn": self.turn,
            "actor": self.actor,
            "kind": self.kind,
            "entity": self.entity,
            "field": self.field,
            "value": self.value,
            "supersedes": self.supersedes,
            "depends_on": list(self.depends_on),
            "rendered": self.rendered,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Event":
        """Validate and deserialize one event from JSON-compatible data."""

        data = _require_mapping(value, "event")
        supersedes_raw = data.get("supersedes")
        if supersedes_raw is not None:
            supersedes_raw = _require_str(supersedes_raw, "event.supersedes")
        return cls(
            event_id=_require_str(data.get("event_id"), "event.event_id"),
            turn=_require_int(data.get("turn"), "event.turn", minimum=1),
            actor=_require_str(data.get("actor"), "event.actor"),
            kind=_require_str(data.get("kind"), "event.kind"),
            entity=_require_str(data.get("entity"), "event.entity"),
            field=_require_str(data.get("field"), "event.field"),
            value=_require_str(data.get("value"), "event.value", allow_empty=True),
            supersedes=supersedes_raw,
            depends_on=_require_string_list(data.get("depends_on"), "event.depends_on"),
            rendered=_require_str(data.get("rendered"), "event.rendered"),
        )


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """A public query issued after a particular session turn."""

    checkpoint_id: str
    turn: int
    query: str
    query_entities: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_str(self.checkpoint_id, "checkpoint.checkpoint_id")
        _require_int(self.turn, "checkpoint.turn", minimum=1)
        _require_str(self.query, "checkpoint.query")
        if not isinstance(self.query_entities, tuple):
            object.__setattr__(self, "query_entities", tuple(self.query_entities))
        if not self.query_entities:
            raise ValueError("checkpoint.query_entities must not be empty")
        for entity in self.query_entities:
            _require_str(entity, "checkpoint.query_entities[]")
        if len(self.query_entities) != len(set(self.query_entities)):
            raise ValueError("checkpoint.query_entities must not contain duplicates")

    def to_dict(self) -> JsonObject:
        """Convert this checkpoint to its exact public JSON shape."""

        return {
            "checkpoint_id": self.checkpoint_id,
            "turn": self.turn,
            "query": self.query,
            "query_entities": list(self.query_entities),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Checkpoint":
        """Validate and deserialize one checkpoint."""

        data = _require_mapping(value, "checkpoint")
        return cls(
            checkpoint_id=_require_str(
                data.get("checkpoint_id"), "checkpoint.checkpoint_id"
            ),
            turn=_require_int(data.get("turn"), "checkpoint.turn", minimum=1),
            query=_require_str(data.get("query"), "checkpoint.query"),
            query_entities=_require_string_list(
                data.get("query_entities"), "checkpoint.query_entities"
            ),
        )


@dataclass(frozen=True, slots=True)
class PublicCase:
    """A public RCWT-S session containing events and answer-free queries."""

    case_id: str
    family: Family
    seed: int
    events: tuple[Event, ...]
    checkpoints: tuple[Checkpoint, ...]

    def __post_init__(self) -> None:
        _require_str(self.case_id, "case.case_id")
        if self.family not in FAMILIES:
            raise ValueError(f"unsupported family: {self.family!r}")
        _require_int(self.seed, "case.seed")
        if not isinstance(self.events, tuple):
            object.__setattr__(self, "events", tuple(self.events))
        if not isinstance(self.checkpoints, tuple):
            object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        if not self.events:
            raise ValueError("case.events must not be empty")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("case.events contains duplicate event_id values")
        turns = [event.turn for event in self.events]
        if turns != sorted(turns) or len(turns) != len(set(turns)):
            raise ValueError("case.events must have unique, increasing turns")
        known_ids = set(event_ids)
        for event in self.events:
            references = (*event.depends_on,)
            if any(reference not in known_ids for reference in references):
                raise ValueError(f"event {event.event_id} has an unknown dependency")
            if event.supersedes is not None and event.supersedes not in known_ids:
                raise ValueError(f"event {event.event_id} supersedes an unknown event")
            for reference in (*references, *((event.supersedes,) if event.supersedes else ())):
                referenced = self.events[event_ids.index(reference)]
                if referenced.turn >= event.turn:
                    raise ValueError(
                        f"event {event.event_id} references non-prior event {reference}"
                    )
        checkpoint_ids = [checkpoint.checkpoint_id for checkpoint in self.checkpoints]
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ValueError("case.checkpoints contains duplicate checkpoint_id values")
        checkpoint_turns = [checkpoint.turn for checkpoint in self.checkpoints]
        if checkpoint_turns != sorted(checkpoint_turns):
            raise ValueError("case.checkpoints must have increasing turns")
        if any(turn > turns[-1] for turn in checkpoint_turns):
            raise ValueError("checkpoint turn exceeds the final event turn")

    def to_dict(self) -> JsonObject:
        """Convert this case to its exact public JSON shape."""

        return {
            "case_id": self.case_id,
            "family": self.family,
            "seed": self.seed,
            "events": [event.to_dict() for event in self.events],
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
        }

    @property
    def sha256(self) -> str:
        """Return the canonical source-case digest used in context records."""

        return canonical_sha256(self.to_dict())

    def checkpoint(self, checkpoint_id: str) -> Checkpoint:
        """Return a checkpoint by ID or raise a descriptive error."""

        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        raise KeyError(f"unknown checkpoint {checkpoint_id!r} in {self.case_id}")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PublicCase":
        """Validate and deserialize one public case."""

        data = _require_mapping(value, "case")
        family = _require_str(data.get("family"), "case.family")
        if family not in FAMILIES:
            raise ValueError(f"unsupported family: {family!r}")
        raw_events = data.get("events")
        raw_checkpoints = data.get("checkpoints")
        if not isinstance(raw_events, list):
            raise ValueError("case.events must be a JSON array")
        if not isinstance(raw_checkpoints, list):
            raise ValueError("case.checkpoints must be a JSON array")
        return cls(
            case_id=_require_str(data.get("case_id"), "case.case_id"),
            family=family,  # type: ignore[arg-type]
            seed=_require_int(data.get("seed"), "case.seed"),
            events=tuple(Event.from_dict(_require_mapping(item, "event")) for item in raw_events),
            checkpoints=tuple(
                Checkpoint.from_dict(_require_mapping(item, "checkpoint"))
                for item in raw_checkpoints
            ),
        )


@dataclass(frozen=True, slots=True)
class Oracle:
    """A private answer key for one case/checkpoint pair."""

    case_id: str
    checkpoint_id: str
    required_current_event_ids: tuple[str, ...]
    stale_conflict_event_ids: tuple[str, ...]
    required_order_edges: tuple[tuple[str, str], ...]
    expected_disposition: Disposition

    def __post_init__(self) -> None:
        _require_str(self.case_id, "oracle.case_id")
        _require_str(self.checkpoint_id, "oracle.checkpoint_id")
        if not isinstance(self.required_current_event_ids, tuple):
            object.__setattr__(
                self,
                "required_current_event_ids",
                tuple(self.required_current_event_ids),
            )
        if not isinstance(self.stale_conflict_event_ids, tuple):
            object.__setattr__(
                self,
                "stale_conflict_event_ids",
                tuple(self.stale_conflict_event_ids),
            )
        if not isinstance(self.required_order_edges, tuple):
            object.__setattr__(
                self,
                "required_order_edges",
                tuple(tuple(edge) for edge in self.required_order_edges),
            )
        for event_id in self.required_current_event_ids:
            _require_str(event_id, "oracle.required_current_event_ids[]")
        for event_id in self.stale_conflict_event_ids:
            _require_str(event_id, "oracle.stale_conflict_event_ids[]")
        if len(self.required_current_event_ids) != len(set(self.required_current_event_ids)):
            raise ValueError("oracle.required_current_event_ids must not contain duplicates")
        if len(self.stale_conflict_event_ids) != len(set(self.stale_conflict_event_ids)):
            raise ValueError("oracle.stale_conflict_event_ids must not contain duplicates")
        if set(self.required_current_event_ids) & set(self.stale_conflict_event_ids):
            raise ValueError("oracle current and stale event IDs must be disjoint")
        for edge in self.required_order_edges:
            if len(edge) != 2 or edge[0] == edge[1]:
                raise ValueError("oracle.required_order_edges entries must be [before, after]")
            _require_str(edge[0], "oracle.required_order_edges[].before")
            _require_str(edge[1], "oracle.required_order_edges[].after")
        if self.expected_disposition not in ("ready", "insufficient"):
            raise ValueError("oracle.expected_disposition is invalid")

    def to_dict(self) -> JsonObject:
        """Convert this oracle to its exact private JSON shape."""

        return {
            "case_id": self.case_id,
            "checkpoint_id": self.checkpoint_id,
            "required_current_event_ids": list(self.required_current_event_ids),
            "stale_conflict_event_ids": list(self.stale_conflict_event_ids),
            "required_order_edges": [list(edge) for edge in self.required_order_edges],
            "expected_disposition": self.expected_disposition,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Oracle":
        """Validate and deserialize one private oracle record."""

        data = _require_mapping(value, "oracle")
        raw_edges = data.get("required_order_edges")
        if not isinstance(raw_edges, list):
            raise ValueError("oracle.required_order_edges must be a JSON array")
        edges: list[tuple[str, str]] = []
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, list) or len(raw_edge) != 2:
                raise ValueError("each required_order_edges entry must have two IDs")
            edges.append(
                (
                    _require_str(raw_edge[0], "order_edge.before"),
                    _require_str(raw_edge[1], "order_edge.after"),
                )
            )
        disposition = _require_str(
            data.get("expected_disposition"), "oracle.expected_disposition"
        )
        if disposition not in ("ready", "insufficient"):
            raise ValueError(f"unsupported disposition: {disposition!r}")
        return cls(
            case_id=_require_str(data.get("case_id"), "oracle.case_id"),
            checkpoint_id=_require_str(
                data.get("checkpoint_id"), "oracle.checkpoint_id"
            ),
            required_current_event_ids=_require_string_list(
                data.get("required_current_event_ids"),
                "oracle.required_current_event_ids",
            ),
            stale_conflict_event_ids=_require_string_list(
                data.get("stale_conflict_event_ids"),
                "oracle.stale_conflict_event_ids",
            ),
            required_order_edges=tuple(edges),
            expected_disposition=disposition,  # type: ignore[arg-type]
        )


# The longer name makes cross-module intent obvious while ``Oracle`` preserves
# the concise name used by the RCWT-S specification.
OracleCase = Oracle


@dataclass(frozen=True, slots=True)
class ContextRecord:
    """One deterministic treatment output for a case/checkpoint/budget cell."""

    case_id: str
    checkpoint_id: str
    family: Family
    seed: int
    treatment: Treatment
    budget_tokens: int
    included_event_ids: tuple[str, ...]
    rendered_context: str
    token_count: int
    source_case_sha256: str
    context_sha256: str

    def __post_init__(self) -> None:
        _require_str(self.case_id, "context.case_id")
        _require_str(self.checkpoint_id, "context.checkpoint_id")
        if self.family not in FAMILIES:
            raise ValueError(f"unsupported family: {self.family!r}")
        _require_int(self.seed, "context.seed")
        if self.treatment not in TREATMENTS:
            raise ValueError(f"unsupported treatment: {self.treatment!r}")
        _require_int(self.budget_tokens, "context.budget_tokens", minimum=1)
        if not isinstance(self.included_event_ids, tuple):
            object.__setattr__(
                self, "included_event_ids", tuple(self.included_event_ids)
            )
        for event_id in self.included_event_ids:
            _require_str(event_id, "context.included_event_ids[]")
        if len(self.included_event_ids) != len(set(self.included_event_ids)):
            raise ValueError("context.included_event_ids must not contain duplicates")
        _require_str(self.rendered_context, "context.rendered_context", allow_empty=True)
        _require_int(self.token_count, "context.token_count")
        if self.token_count > self.budget_tokens:
            raise ValueError("context.token_count exceeds context.budget_tokens")
        for label, digest in (
            ("context.source_case_sha256", self.source_case_sha256),
            ("context.context_sha256", self.context_sha256),
        ):
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    def to_dict(self, *, include_hash: bool = True) -> JsonObject:
        """Convert this context to JSON, optionally omitting its self-hash."""

        result: JsonObject = {
            "case_id": self.case_id,
            "checkpoint_id": self.checkpoint_id,
            "family": self.family,
            "seed": self.seed,
            "treatment": self.treatment,
            "budget_tokens": self.budget_tokens,
            "included_event_ids": list(self.included_event_ids),
            "rendered_context": self.rendered_context,
            "token_count": self.token_count,
            "source_case_sha256": self.source_case_sha256,
        }
        if include_hash:
            result["context_sha256"] = self.context_sha256
        return result

    @property
    def derived_sha256(self) -> str:
        """Recompute the tamper-evident hash over every field but the hash itself."""

        return canonical_sha256(self.to_dict(include_hash=False))

    def verify_hash(self) -> bool:
        """Return whether ``context_sha256`` matches the canonical record body."""

        return self.context_sha256 == self.derived_sha256

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        checkpoint_id: str,
        family: Family,
        seed: int,
        treatment: Treatment,
        budget_tokens: int,
        included_event_ids: Sequence[str],
        rendered_context: str,
        token_count: int,
        source_case_sha256: str,
    ) -> "ContextRecord":
        """Construct a context and derive its self-hash in one operation."""

        body: JsonObject = {
            "case_id": case_id,
            "checkpoint_id": checkpoint_id,
            "family": family,
            "seed": seed,
            "treatment": treatment,
            "budget_tokens": budget_tokens,
            "included_event_ids": list(included_event_ids),
            "rendered_context": rendered_context,
            "token_count": token_count,
            "source_case_sha256": source_case_sha256,
        }
        return cls(
            case_id=case_id,
            checkpoint_id=checkpoint_id,
            family=family,
            seed=seed,
            treatment=treatment,
            budget_tokens=budget_tokens,
            included_event_ids=tuple(included_event_ids),
            rendered_context=rendered_context,
            token_count=token_count,
            source_case_sha256=source_case_sha256,
            context_sha256=canonical_sha256(body),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ContextRecord":
        """Validate and deserialize one treatment output."""

        data = _require_mapping(value, "context")
        family = _require_str(data.get("family"), "context.family")
        treatment = _require_str(data.get("treatment"), "context.treatment")
        if family not in FAMILIES:
            raise ValueError(f"unsupported family: {family!r}")
        if treatment not in TREATMENTS:
            raise ValueError(f"unsupported treatment: {treatment!r}")
        return cls(
            case_id=_require_str(data.get("case_id"), "context.case_id"),
            checkpoint_id=_require_str(
                data.get("checkpoint_id"), "context.checkpoint_id"
            ),
            family=family,  # type: ignore[arg-type]
            seed=_require_int(data.get("seed"), "context.seed"),
            treatment=treatment,  # type: ignore[arg-type]
            budget_tokens=_require_int(
                data.get("budget_tokens"), "context.budget_tokens", minimum=1
            ),
            included_event_ids=_require_string_list(
                data.get("included_event_ids"), "context.included_event_ids"
            ),
            rendered_context=_require_str(
                data.get("rendered_context"),
                "context.rendered_context",
                allow_empty=True,
            ),
            token_count=_require_int(data.get("token_count"), "context.token_count"),
            source_case_sha256=_require_str(
                data.get("source_case_sha256"), "context.source_case_sha256"
            ),
            context_sha256=_require_str(
                data.get("context_sha256"), "context.context_sha256"
            ),
        )


SessionEvent = Event
SessionCheckpoint = Checkpoint


def read_jsonl(path: Path) -> list[JsonObject]:
    """Read a UTF-8 JSONL file and reject non-object or malformed rows."""

    records: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def load_public_cases(path: Path) -> list[PublicCase]:
    """Load and validate public cases, including cross-record ID uniqueness."""

    cases = [PublicCase.from_dict(record) for record in read_jsonl(path)]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"duplicate case_id in {path}")
    return cases


def load_oracles(path: Path) -> list[Oracle]:
    """Load private oracle rows without coupling them to public case loading."""

    return [Oracle.from_dict(record) for record in read_jsonl(path)]


def load_context_records(path: Path, *, verify_hashes: bool = True) -> list[ContextRecord]:
    """Load contexts and optionally reject any self-hash mismatch."""

    records = [ContextRecord.from_dict(record) for record in read_jsonl(path)]
    if verify_hashes:
        for record in records:
            if not record.verify_hash():
                raise ValueError(
                    f"context_sha256 mismatch for {record.case_id}/{record.checkpoint_id}/"
                    f"{record.treatment}/{record.budget_tokens}"
                )
    return records
