"""Deterministic scorer for the RCWT-S session-memory experiment.

The scorer deliberately receives generated contexts, public source cases, the
private scoring oracle, and the locked manifest as separate artifacts.  It
re-derives every score from those inputs, checks source and artifact hashes plus
token counts, and refuses to aggregate an incomplete or inconsistent treatment
matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
try:
    import tiktoken
except ImportError:  # pragma: no cover - exercised only before dependencies install.
    tiktoken = None  # type: ignore[assignment]

try:
    from rcwt_statistics import stable_seed
    from rcwt_session_types import PublicCase
except ImportError:  # pragma: no cover - permits ``python -m src...`` locally.
    from .rcwt_statistics import stable_seed
    from .rcwt_session_types import PublicCase


PROTOCOL_VERSION = "session-v1"
EXPECTED_TREATMENTS = ("tail", "state_latest", "state_first")
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE = 0.95
_Z_95 = 1.959963984540054
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MISSING_RE = re.compile(
    r"(?im)^\s*(?:DISPOSITION\s*[:=]?\s*)?MISSING(?:\b|\s*$)"
)
_READY_RE = re.compile(r"(?im)^\s*DISPOSITION\s*[:=]?\s*READY\b")
_ORACLE_ONLY_FIELDS = frozenset(
    {
        "required_current_event_ids",
        "stale_conflict_event_ids",
        "required_order_edges",
        "expected_disposition",
        "stale_current_pairs",
        "oracle",
    }
)

JsonObject: TypeAlias = dict[str, Any]
Cluster: TypeAlias = tuple[str, int]


class ScoreValidationError(ValueError):
    """Raised when a preregistered validity check fails."""


def _as_mapping(value: Any, where: str) -> Mapping[str, Any]:
    """Accept JSON mappings and the typed records exposed by session_types."""

    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, Mapping):
            return mapped
    raise ScoreValidationError(f"{where} must be a JSON mapping or typed RCWT-S record")


@dataclass(frozen=True)
class OracleRecord:
    """Private facts needed to score one case/checkpoint pair."""

    case_id: str
    checkpoint_id: str
    required_current_event_ids: tuple[str, ...]
    stale_conflict_event_ids: tuple[str, ...]
    required_order_edges: tuple[tuple[str, str], ...]
    expected_disposition: str
    stale_current_pairs: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, where: str) -> "OracleRecord":
        raw = _as_mapping(raw, where)
        case_id = _require_string(raw, "case_id", where)
        checkpoint_id = _require_string(raw, "checkpoint_id", where)
        required = _string_tuple(raw, "required_current_event_ids", where)
        stale = _string_tuple(raw, "stale_conflict_event_ids", where)
        edges = _edge_tuple(raw.get("required_order_edges"), f"{where}.required_order_edges")
        disposition = _require_string(raw, "expected_disposition", where)
        if disposition not in {"ready", "insufficient"}:
            raise ScoreValidationError(
                f"{where}.expected_disposition must be 'ready' or 'insufficient'"
            )
        if len(set(required)) != len(required):
            raise ScoreValidationError(f"{where} repeats a required current event id")
        if len(set(stale)) != len(stale):
            raise ScoreValidationError(f"{where} repeats a stale conflict event id")
        if set(required) & set(stale):
            raise ScoreValidationError(f"{where} marks an event both current and stale")

        pairs_raw = raw.get("stale_current_pairs", ())
        pairs = _edge_tuple(pairs_raw, f"{where}.stale_current_pairs")
        stale_set = set(stale)
        required_set = set(required)
        for stale_id, current_id in pairs:
            if stale_id not in stale_set or current_id not in required_set:
                raise ScoreValidationError(
                    f"{where}.stale_current_pairs must map declared stale ids to current ids"
                )
        return cls(
            case_id=case_id,
            checkpoint_id=checkpoint_id,
            required_current_event_ids=required,
            stale_conflict_event_ids=stale,
            required_order_edges=edges,
            expected_disposition=disposition,
            stale_current_pairs=pairs,
        )


@dataclass(frozen=True)
class ScoredContext:
    """All independently re-derived outcomes for one context."""

    case_id: str
    checkpoint_id: str
    checkpoint_turn: int
    family: str
    seed: int
    treatment: str
    budget_tokens: int
    realized_tokens: int
    token_utilization: float
    expected_disposition: str
    decision_ready: bool
    required_present: int
    required_total: int
    required_fact_recall: float | None
    stale_present: int
    stale_total: int
    stale_exposure: bool
    stale_conflict_fraction: float
    contradiction: bool
    causal_edges_preserved: int
    causal_edges_total: int
    explicit_missing: bool
    fabricated_sufficiency: bool
    source_case_sha256: str
    context_sha256: str

    @property
    def cluster(self) -> Cluster:
        return (self.case_id, self.seed)

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.case_id, self.checkpoint_id, self.budget_tokens, self.treatment)

    def to_dict(self) -> JsonObject:
        return asdict(self)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value using the hash format locked by RCWT-S."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_sha256(record: Mapping[str, Any]) -> str:
    """Re-hash a persisted ContextRecord, excluding only its hash field."""

    payload = dict(record)
    payload.pop("context_sha256", None)
    return canonical_sha256(payload)


def count_construction_tokens(text: str) -> int:
    """Count the locked ``cl100k_base`` construction tokens."""

    if tiktoken is None:
        raise ScoreValidationError(
            "tiktoken is required to re-derive cl100k_base token counts; "
            "install the pinned requirements before scoring"
        )
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = CONFIDENCE,
) -> tuple[float | None, float | None]:
    """Return a two-sided Wilson score interval for a binomial rate."""

    if successes < 0 or total < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if total == 0:
        return None, None
    if confidence == CONFIDENCE:
        z = _Z_95
    else:
        # statistics.NormalDist is deterministic and avoids a scipy dependency.
        z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
    observed = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (observed + z2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(observed * (1.0 - observed) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def clustered_bootstrap_mean_ci(
    values: Sequence[tuple[Cluster, float]],
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
) -> tuple[float, float, float]:
    """Bootstrap the mean with case/seed as the independent sampling unit.

    Observations inside a case remain together.  A cluster contributes its mean
    to each resample, which gives every independent case equal weight and avoids
    treating its checkpoints and budgets as independent replicates.
    """

    if not values:
        raise ValueError("at least one clustered value is required")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    grouped: dict[Cluster, list[float]] = defaultdict(list)
    for cluster, value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("bootstrap values must be finite")
        grouped[cluster].append(number)
    cluster_means = np.asarray(
        [statistics.fmean(grouped[key]) for key in sorted(grouped)], dtype=float
    )
    point = float(np.mean(cluster_means))
    if cluster_means.size == 1 or np.all(cluster_means == cluster_means[0]):
        return point, point, point

    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0,
        cluster_means.size,
        size=(n_resamples, cluster_means.size),
    )
    estimates = np.mean(cluster_means[sampled_indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
    return point, float(low), float(high)


def _require_string(raw: Mapping[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ScoreValidationError(f"{where}.{key} must be a non-empty string")
    return value


def _require_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScoreValidationError(f"{where}.{key} must be an integer")
    return value


def _string_tuple(raw: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ScoreValidationError(f"{where}.{key} must be a list of non-empty strings")
    return tuple(value)


def _edge_tuple(value: Any, where: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (list, tuple)):
        raise ScoreValidationError(f"{where} must be a list of [before, after] pairs")
    edges: list[tuple[str, str]] = []
    for index, edge in enumerate(value):
        if (
            not isinstance(edge, (list, tuple))
            or len(edge) != 2
            or any(not isinstance(item, str) or not item for item in edge)
        ):
            raise ScoreValidationError(f"{where}[{index}] must be [before, after]")
        before, after = edge
        if before == after:
            raise ScoreValidationError(f"{where}[{index}] is a self-edge")
        edges.append((before, after))
    if len(set(edges)) != len(edges):
        raise ScoreValidationError(f"{where} contains duplicate edges")
    return tuple(edges)


def load_jsonl(path: Path, *, label: str) -> list[JsonObject]:
    """Load an object-only UTF-8 JSONL artifact with useful line errors."""

    records: list[JsonObject] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ScoreValidationError(
                        f"{label}:{line_number} is not valid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ScoreValidationError(f"{label}:{line_number} must contain a JSON object")
                records.append(value)
    except OSError as exc:
        raise ScoreValidationError(f"cannot read {label} at {path}: {exc}") from exc
    if not records:
        raise ScoreValidationError(f"{label} is empty")
    return records


def _event_id_from_line(line: str) -> str | None:
    match = re.match(r"^\s*(?:[-*]\s*)?EVENT\b(.*)$", line, flags=re.IGNORECASE)
    if match is None:
        return None
    rest = match.group(1).strip().lstrip(":=").strip()
    if not rest:
        raise ScoreValidationError("rendered EVENT line has no event id")
    if rest.startswith("{"):
        try:
            value = json.loads(rest)
        except json.JSONDecodeError as exc:
            raise ScoreValidationError("rendered EVENT JSON is malformed") from exc
        if not isinstance(value, dict) or not isinstance(value.get("event_id"), str):
            raise ScoreValidationError("rendered EVENT JSON has no event_id")
        return str(value["event_id"])
    field_match = re.match(r"event_id\s*=\s*([^\s|,;]+)", rest, flags=re.IGNORECASE)
    if field_match is not None:
        return field_match.group(1).strip("[](){}<>\"'")
    bracket_match = re.match(r"[\[<]([^\]>]+)[\]>]", rest)
    if bracket_match is not None:
        return bracket_match.group(1).strip()
    return rest.split(maxsplit=1)[0].rstrip(":|,;").strip("[](){}<>\"'")


def extract_rendered_event_ids(rendered_context: str) -> tuple[str, ...]:
    """Re-derive the ordered event-id list from canonical EVENT lines."""

    event_ids: list[str] = []
    for line in rendered_context.splitlines():
        event_id = _event_id_from_line(line)
        if event_id is not None:
            if not event_id:
                raise ScoreValidationError("rendered EVENT line has an empty event id")
            event_ids.append(event_id)
    return tuple(event_ids)


def _rendered_event_lines(rendered_context: str) -> tuple[tuple[str, str], ...]:
    lines: list[tuple[str, str]] = []
    for line in rendered_context.splitlines():
        event_id = _event_id_from_line(line)
        if event_id is not None:
            lines.append((event_id, line))
    return tuple(lines)


def _rendered_checkpoint(rendered_context: str) -> tuple[str, int | None] | None:
    match = re.search(
        r"(?im)^\s*CHECKPOINT\s+([^\s]+)(?:\s+TURN\s+(\d+))?\s*$",
        rendered_context,
    )
    if match is None:
        return None
    turn = int(match.group(2)) if match.group(2) is not None else None
    return match.group(1), turn


def _rendered_query(rendered_context: str) -> str | None:
    match = re.search(r"(?im)^\s*QUERY\s+(.+?)\s*$", rendered_context)
    return match.group(1) if match is not None else None


def _public_case(value: Any, *, where: str) -> PublicCase:
    if isinstance(value, PublicCase):
        return value
    try:
        return PublicCase.from_dict(_as_mapping(value, where))
    except (TypeError, ValueError) as exc:
        raise ScoreValidationError(f"invalid {where}: {exc}") from exc


def _load_public_cases(records: Sequence[Any]) -> dict[str, PublicCase]:
    cases: dict[str, PublicCase] = {}
    for index, value in enumerate(records, start=1):
        case = _public_case(value, where=f"public_cases:{index}")
        if case.case_id in cases:
            raise ScoreValidationError(f"duplicate public case id: {case.case_id!r}")
        cases[case.case_id] = case
    if not cases:
        raise ScoreValidationError("public case artifact is empty")
    return cases


def _validate_context_source(
    context: Mapping[str, Any],
    case: PublicCase,
    *,
    where: str,
) -> None:
    """Bind redundant ContextRecord fields and rendered lines to public source."""

    case_id = _require_string(context, "case_id", where)
    checkpoint_id = _require_string(context, "checkpoint_id", where)
    family = _require_string(context, "family", where)
    seed = _require_int(context, "seed", where)
    source_hash = _require_string(context, "source_case_sha256", where)
    rendered = _require_string(context, "rendered_context", where)
    included_ids = _string_tuple(context, "included_event_ids", where)

    if case_id != case.case_id:
        raise ScoreValidationError(f"{where} is bound to the wrong public case")
    if family != case.family or seed != case.seed:
        raise ScoreValidationError(f"{where} family/seed differs from its public case")
    if not hmac.compare_digest(source_hash, case.sha256):
        raise ScoreValidationError(f"{where}.source_case_sha256 differs from public case")

    try:
        checkpoint = case.checkpoint(checkpoint_id)
    except KeyError as exc:
        raise ScoreValidationError(
            f"{where} references unknown public checkpoint {checkpoint_id!r}"
        ) from exc
    expected_header = f"CHECKPOINT {checkpoint.checkpoint_id} TURN {checkpoint.turn}"
    expected_query = f"QUERY {checkpoint.query}"
    rendered_lines = rendered.splitlines()
    if not rendered.endswith("\n") or "\r" in rendered:
        raise ScoreValidationError(f"{where} does not use canonical LF-terminated framing")
    if len(rendered_lines) < 3 or rendered_lines[0] != expected_header:
        raise ScoreValidationError(f"{where} does not render the exact public checkpoint header")
    if rendered_lines[1] != expected_query:
        raise ScoreValidationError(f"{where} does not render the exact public checkpoint query")
    if re.fullmatch(
        r"DISPOSITION (?:ready|MISSING \([a-z0-9_]+\))",
        rendered_lines[-1],
    ) is None:
        raise ScoreValidationError(f"{where} has a non-canonical disposition line")
    if any(_event_id_from_line(line) is None for line in rendered_lines[2:-1]):
        raise ScoreValidationError(f"{where} has a non-EVENT line inside the event block")

    eligible = {
        event.event_id: event for event in case.events if event.turn <= checkpoint.turn
    }
    actual_event_lines = _rendered_event_lines(rendered)
    if len(actual_event_lines) != len(rendered_lines) - 3:
        raise ScoreValidationError(f"{where} has EVENT lines outside canonical framing")
    if tuple(event_id for event_id, _ in actual_event_lines) != included_ids:
        raise ScoreValidationError(f"{where} EVENT order differs from included_event_ids")
    for event_id, line in actual_event_lines:
        event = eligible.get(event_id)
        if event is None:
            raise ScoreValidationError(
                f"{where} includes unknown or future event {event_id!r}"
            )
        if line != event.rendered:
            raise ScoreValidationError(
                f"{where} rendered payload differs from public event {event_id!r}"
            )


def _checkpoint_turn(record: Mapping[str, Any], checkpoint_id: str, rendered: str) -> int:
    for key in ("checkpoint_turn", "turn"):
        value = record.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ScoreValidationError(f"context.{key} must be a positive integer")
            return value
    header_patterns = (
        r"(?im)^\s*CHECKPOINT(?:_TURN|\s+TURN)\s*[:=]?\s*(\d+)\b",
        r"(?im)^\s*CHECKPOINT\b[^\n]*\bTURN\s*[:=]?\s*(\d+)\b",
    )
    for pattern in header_patterns:
        match = re.search(pattern, rendered)
        if match is not None:
            return int(match.group(1))
    numbers = re.findall(r"\d+", checkpoint_id)
    if numbers:
        turn = int(numbers[-1])
        if turn > 0:
            return turn
    raise ScoreValidationError(
        f"cannot derive checkpoint turn for {checkpoint_id!r}; persist checkpoint_turn"
    )


def _ordered_edge_count(
    edges: Sequence[tuple[str, str]],
    included_ids: Sequence[str],
    rendered_ids: Sequence[str],
) -> int:
    included_position = {event_id: index for index, event_id in enumerate(included_ids)}
    rendered_position = {event_id: index for index, event_id in enumerate(rendered_ids)}
    preserved = 0
    for before, after in edges:
        if (
            before in included_position
            and after in included_position
            and before in rendered_position
            and after in rendered_position
            and included_position[before] < included_position[after]
            and rendered_position[before] < rendered_position[after]
        ):
            preserved += 1
    return preserved


def score_context(
    context: Mapping[str, Any],
    oracle: OracleRecord | Mapping[str, Any],
    public_case: PublicCase | Mapping[str, Any],
    *,
    where: str = "context",
) -> ScoredContext:
    """Validate and score one context against separate source and oracle data."""

    context = _as_mapping(context, where)
    source_case = _public_case(public_case, where=f"{where}.public_case")
    if not isinstance(oracle, OracleRecord):
        oracle = OracleRecord.from_mapping(_as_mapping(oracle, "oracle"), where="oracle")
    leaked = sorted(_ORACLE_ONLY_FIELDS & set(context))
    if leaked:
        raise ScoreValidationError(f"{where} leaks oracle-only fields: {', '.join(leaked)}")

    case_id = _require_string(context, "case_id", where)
    checkpoint_id = _require_string(context, "checkpoint_id", where)
    family = _require_string(context, "family", where)
    treatment = _require_string(context, "treatment", where)
    seed = _require_int(context, "seed", where)
    budget = _require_int(context, "budget_tokens", where)
    persisted_tokens = _require_int(context, "token_count", where)
    rendered = _require_string(context, "rendered_context", where)
    source_hash = _require_string(context, "source_case_sha256", where).lower()
    persisted_hash = _require_string(context, "context_sha256", where).lower()
    included_ids = _string_tuple(context, "included_event_ids", where)

    if (case_id, checkpoint_id) != (oracle.case_id, oracle.checkpoint_id):
        raise ScoreValidationError(
            f"{where} key {(case_id, checkpoint_id)!r} does not match its oracle"
        )
    if treatment not in EXPECTED_TREATMENTS:
        raise ScoreValidationError(f"{where}.treatment is not a registered RCWT-S arm")
    if budget <= 0:
        raise ScoreValidationError(f"{where}.budget_tokens must be positive")
    if persisted_tokens < 0:
        raise ScoreValidationError(f"{where}.token_count cannot be negative")
    if len(set(included_ids)) != len(included_ids):
        raise ScoreValidationError(f"{where}.included_event_ids contains duplicates")
    if _SHA256_RE.fullmatch(source_hash) is None:
        raise ScoreValidationError(f"{where}.source_case_sha256 is not a SHA-256 digest")
    if _SHA256_RE.fullmatch(persisted_hash) is None:
        raise ScoreValidationError(f"{where}.context_sha256 is not a SHA-256 digest")

    derived_hash = context_sha256(context)
    if not _constant_time_equal(persisted_hash, derived_hash):
        raise ScoreValidationError(
            f"{where}.context_sha256 mismatch: persisted={persisted_hash} derived={derived_hash}"
        )
    derived_tokens = count_construction_tokens(rendered)
    if persisted_tokens != derived_tokens:
        raise ScoreValidationError(
            f"{where}.token_count mismatch: persisted={persisted_tokens} derived={derived_tokens}"
        )
    if derived_tokens > budget:
        raise ScoreValidationError(
            f"{where} exceeds token budget: token_count={derived_tokens} budget={budget}"
        )

    # Validate the self-authenticating record before binding its redundant
    # framing and EVENT payload to the independently loaded public case.  This
    # keeps ordinary byte tampering distinguishable from a fully rehashed but
    # source-inconsistent forgery; both paths still fail closed.
    _validate_context_source(context, source_case, where=where)

    rendered_ids = extract_rendered_event_ids(rendered)
    if rendered_ids != included_ids:
        raise ScoreValidationError(
            f"{where}.included_event_ids does not equal the ordered EVENT lines in rendered_context"
        )
    checkpoint_turn = _checkpoint_turn(context, checkpoint_id, rendered)
    rendered_checkpoint = _rendered_checkpoint(rendered)
    if rendered_checkpoint is not None:
        rendered_checkpoint_id, rendered_turn = rendered_checkpoint
        if rendered_checkpoint_id != checkpoint_id:
            raise ScoreValidationError(
                f"{where} renders checkpoint {rendered_checkpoint_id!r}, not {checkpoint_id!r}"
            )
        if rendered_turn is not None and rendered_turn != checkpoint_turn:
            raise ScoreValidationError(
                f"{where} renders turn {rendered_turn}, not {checkpoint_turn}"
            )

    included_set = set(included_ids)
    required_set = set(oracle.required_current_event_ids)
    stale_set = set(oracle.stale_conflict_event_ids)
    required_present = len(required_set & included_set)
    stale_present = len(stale_set & included_set)
    required_total = len(required_set)
    stale_total = len(stale_set)
    recall = required_present / required_total if required_total else None
    stale_fraction = stale_present / stale_total if stale_total else 0.0
    stale_exposure = stale_present > 0
    edge_count = _ordered_edge_count(oracle.required_order_edges, included_ids, rendered_ids)
    edge_total = len(oracle.required_order_edges)
    explicit_missing = _MISSING_RE.search(rendered) is not None
    declares_ready = _READY_RE.search(rendered) is not None

    if oracle.stale_current_pairs:
        contradiction = any(
            stale_id in included_set and current_id in included_set
            for stale_id, current_id in oracle.stale_current_pairs
        )
    else:
        # The locked oracle intentionally carries no stale-to-current mapping.
        # Counting every stale exposure as a contradiction is conservative and
        # makes the approximation explicit in the output metadata.
        contradiction = stale_exposure

    base_ready = (
        required_present == required_total
        and not stale_exposure
        and edge_count == edge_total
    )
    fabricated_sufficiency = oracle.expected_disposition == "insufficient" and declares_ready
    if oracle.expected_disposition == "ready":
        decision_ready = base_ready
    else:
        decision_ready = base_ready and explicit_missing and not fabricated_sufficiency

    return ScoredContext(
        case_id=case_id,
        checkpoint_id=checkpoint_id,
        checkpoint_turn=checkpoint_turn,
        family=family,
        seed=seed,
        treatment=treatment,
        budget_tokens=budget,
        realized_tokens=derived_tokens,
        token_utilization=derived_tokens / budget,
        expected_disposition=oracle.expected_disposition,
        decision_ready=decision_ready,
        required_present=required_present,
        required_total=required_total,
        required_fact_recall=recall,
        stale_present=stale_present,
        stale_total=stale_total,
        stale_exposure=stale_exposure,
        stale_conflict_fraction=stale_fraction,
        contradiction=contradiction,
        causal_edges_preserved=edge_count,
        causal_edges_total=edge_total,
        explicit_missing=explicit_missing,
        fabricated_sufficiency=fabricated_sufficiency,
        source_case_sha256=source_hash,
        context_sha256=persisted_hash,
    )


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def _load_oracles(records: Sequence[Any]) -> dict[tuple[str, str], OracleRecord]:
    oracles: dict[tuple[str, str], OracleRecord] = {}
    for index, raw in enumerate(records, start=1):
        oracle = OracleRecord.from_mapping(_as_mapping(raw, f"oracle:{index}"), where=f"oracle:{index}")
        key = (oracle.case_id, oracle.checkpoint_id)
        if key in oracles:
            raise ScoreValidationError(f"duplicate oracle key: {key!r}")
        oracles[key] = oracle
    return oracles


def _nested_values(root: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(root, Mapping):
        for child_key, child in root.items():
            if child_key == key:
                found.append(child)
            found.extend(_nested_values(child, key))
    elif isinstance(root, list):
        for child in root:
            found.extend(_nested_values(child, key))
    return found


def _manifest_sequence(manifest: Mapping[str, Any], key: str) -> tuple[Any, ...] | None:
    candidates = [value for value in _nested_values(manifest, key) if isinstance(value, list)]
    if not candidates:
        return None
    normalized = {canonical_json_bytes(value) for value in candidates}
    if len(normalized) > 1:
        raise ScoreValidationError(f"manifest has conflicting {key!r} declarations")
    return tuple(candidates[0])


def _digest_value(value: Any) -> str | None:
    if isinstance(value, str) and _SHA256_RE.fullmatch(value.lower()):
        return value.lower()
    if isinstance(value, Mapping):
        for key in ("sha256", "digest", "hash"):
            candidate = value.get(key)
            if isinstance(candidate, str) and _SHA256_RE.fullmatch(candidate.lower()):
                return candidate.lower()
    return None


def _declared_artifact_digest(
    manifest: Mapping[str, Any],
    path: Path,
    logical_names: Sequence[str],
) -> str | None:
    aliases = {path.name, path.as_posix(), str(path), *logical_names}
    aliases_lower = {alias.lower() for alias in aliases}
    candidates: list[str] = []
    for key in logical_names:
        for suffix in ("_sha256", "_digest", "_hash"):
            digest = _digest_value(manifest.get(f"{key}{suffix}"))
            if digest is not None:
                candidates.append(digest)
    for container_name in ("artifacts", "files", "digests", "hashes"):
        for container in _nested_values(manifest, container_name):
            if not isinstance(container, Mapping):
                continue
            for name, value in container.items():
                if str(name).lower() in aliases_lower or Path(str(name)).name.lower() in aliases_lower:
                    digest = _digest_value(value)
                    if digest is not None:
                        candidates.append(digest)
                if isinstance(value, Mapping):
                    declared_path = value.get("path") or value.get("file") or value.get("name")
                    if isinstance(declared_path, str) and (
                        declared_path.lower() in aliases_lower
                        or Path(declared_path).name.lower() in aliases_lower
                    ):
                        digest = _digest_value(value)
                        if digest is not None:
                            candidates.append(digest)
    unique = sorted(set(candidates))
    if len(unique) > 1:
        raise ScoreValidationError(f"manifest has conflicting digests for {path.name}")
    return unique[0] if unique else None


def _manifest_case_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("case_sha256", "case_hashes", "source_case_sha256"):
        for value in _nested_values(manifest, key):
            if isinstance(value, Mapping):
                for case_id, digest_value in value.items():
                    digest = _digest_value(digest_value)
                    if digest is not None:
                        previous = result.setdefault(str(case_id), digest)
                        if previous != digest:
                            raise ScoreValidationError(
                                f"manifest has conflicting source hashes for case {case_id}"
                            )
    for value in _nested_values(manifest, "cases"):
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, Mapping) or not isinstance(item.get("case_id"), str):
                continue
            digest = _digest_value(
                item.get("source_case_sha256") or item.get("case_sha256") or item.get("sha256")
            )
            if digest is not None:
                case_id = str(item["case_id"])
                previous = result.setdefault(case_id, digest)
                if previous != digest:
                    raise ScoreValidationError(
                        f"manifest has conflicting source hashes for case {case_id}"
                    )
    return result


def _declared_named_digest(manifest: Mapping[str, Any], name: str) -> str | None:
    """Read one logical digest from top-level and standard hash containers."""

    candidates: list[str] = []
    for key in (name, f"{name}_sha256", f"{name}_digest", f"{name}_hash"):
        digest = _digest_value(manifest.get(key))
        if digest is not None:
            candidates.append(digest)
    for container_name in ("hashes", "digests"):
        container = manifest.get(container_name)
        if isinstance(container, Mapping):
            digest = _digest_value(container.get(name))
            if digest is not None:
                candidates.append(digest)
    unique = sorted(set(candidates))
    if len(unique) > 1:
        raise ScoreValidationError(f"manifest has conflicting {name} digests")
    return unique[0] if unique else None


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    contexts_path: Path,
    oracle_path: Path,
    cases_path: Path,
) -> list[str]:
    checks: list[str] = []
    protocol_values = [
        value
        for key in ("protocol", "protocol_version")
        for value in _nested_values(manifest, key)
        if isinstance(value, str) and value.startswith("session-")
    ]
    if not protocol_values:
        raise ScoreValidationError("manifest does not declare the RCWT-S protocol version")
    if any(value != PROTOCOL_VERSION for value in protocol_values):
        raise ScoreValidationError(
            f"manifest protocol mismatch; expected {PROTOCOL_VERSION!r}"
        )
    checks.append("protocol_version")

    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ScoreValidationError("manifest.config must be a JSON object")
    config_digest = _declared_named_digest(manifest, "config")
    if config_digest is None:
        raise ScoreValidationError("manifest does not declare config_sha256")
    actual_config_digest = canonical_sha256(config)
    if not _constant_time_equal(config_digest, actual_config_digest):
        raise ScoreValidationError(
            f"manifest config hash mismatch: declared={config_digest} actual={actual_config_digest}"
        )
    checks.append("config_sha256")

    for path, names, label in (
        (contexts_path, ("contexts", "context"), "contexts_sha256"),
        (oracle_path, ("oracle_cases", "oracle"), "oracle_sha256"),
        (cases_path, ("public_cases", "cases", "public"), "public_cases_sha256"),
    ):
        declared = _declared_artifact_digest(manifest, path, names)
        if declared is None:
            raise ScoreValidationError(
                f"manifest does not declare a SHA-256 digest for {path.name}"
            )
        actual = file_sha256(path)
        if not _constant_time_equal(declared, actual):
            raise ScoreValidationError(
                f"manifest digest mismatch for {path.name}: declared={declared} actual={actual}"
            )
        checks.append(label)

    self_digest = _digest_value(manifest.get("manifest_sha256"))
    if self_digest is not None:
        without_self = dict(manifest)
        without_self.pop("manifest_sha256", None)
        actual_self = canonical_sha256(without_self)
        if not _constant_time_equal(self_digest, actual_self):
            raise ScoreValidationError("manifest_sha256 does not match canonical manifest payload")
        checks.append("manifest_sha256")

    preregistration_digest = _declared_named_digest(manifest, "preregistration")
    preregistration_name = manifest.get("preregistration_path")
    if preregistration_digest is not None and isinstance(preregistration_name, str):
        repository_root = Path(__file__).resolve().parents[1]
        preregistration_path = (repository_root / preregistration_name).resolve()
        try:
            preregistration_path.relative_to(repository_root.resolve())
        except ValueError as exc:
            raise ScoreValidationError("manifest preregistration_path escapes repository") from exc
        if not preregistration_path.is_file():
            raise ScoreValidationError(
                f"manifest preregistration file is missing: {preregistration_name}"
            )
        actual_preregistration_digest = file_sha256(preregistration_path)
        if not _constant_time_equal(preregistration_digest, actual_preregistration_digest):
            raise ScoreValidationError("manifest preregistration digest mismatch")
        checks.append("preregistration_sha256")

    if not manifest_path.is_file():  # Defensive after an already successful load.
        raise ScoreValidationError(f"manifest disappeared during scoring: {manifest_path}")
    return checks


def _validate_matrix(
    contexts: Sequence[Mapping[str, Any]],
    oracles: Mapping[tuple[str, str], OracleRecord],
    public_cases: Mapping[str, PublicCase],
    manifest: Mapping[str, Any],
) -> tuple[list[ScoredContext], JsonObject]:
    treatments_declared = _manifest_sequence(manifest, "treatments")
    if treatments_declared is not None:
        if any(not isinstance(item, str) for item in treatments_declared):
            raise ScoreValidationError("manifest treatments must be strings")
        expected_treatments = tuple(str(item) for item in treatments_declared)
    else:
        expected_treatments = EXPECTED_TREATMENTS
    if set(expected_treatments) != set(EXPECTED_TREATMENTS) or len(expected_treatments) != 3:
        raise ScoreValidationError(
            "manifest treatment arms must be tail,state_latest,state_first exactly once"
        )

    budgets_declared = _manifest_sequence(manifest, "budgets")
    if budgets_declared is None:
        budgets_declared = _manifest_sequence(manifest, "budget_tokens")
    if budgets_declared is not None:
        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in budgets_declared):
            raise ScoreValidationError("manifest budgets must be positive integers")
        expected_budgets = tuple(sorted(set(int(item) for item in budgets_declared)))
        if len(expected_budgets) != len(budgets_declared):
            raise ScoreValidationError("manifest budgets contain duplicates")
    else:
        expected_budgets = tuple(
            sorted(
                {
                    int(record["budget_tokens"])
                    for record in contexts
                    if isinstance(record.get("budget_tokens"), int)
                    and not isinstance(record.get("budget_tokens"), bool)
                }
            )
        )
    if not expected_budgets:
        raise ScoreValidationError("no token budgets are declared or observed")

    context_keys: set[tuple[str, str, int, str]] = set()
    raw_by_key: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    observed_checkpoint_keys: set[tuple[str, str]] = set()
    for index, raw in enumerate(contexts, start=1):
        case_id = _require_string(raw, "case_id", f"contexts:{index}")
        checkpoint_id = _require_string(raw, "checkpoint_id", f"contexts:{index}")
        budget = _require_int(raw, "budget_tokens", f"contexts:{index}")
        treatment = _require_string(raw, "treatment", f"contexts:{index}")
        key = (case_id, checkpoint_id, budget, treatment)
        if key in context_keys:
            raise ScoreValidationError(f"duplicate context key: {key!r}")
        context_keys.add(key)
        raw_by_key[key] = raw
        observed_checkpoint_keys.add((case_id, checkpoint_id))

    oracle_keys = set(oracles)
    oracle_case_ids = {case_id for case_id, _ in oracle_keys}
    if oracle_case_ids != set(public_cases):
        missing = sorted(oracle_case_ids - set(public_cases))
        extra = sorted(set(public_cases) - oracle_case_ids)
        raise ScoreValidationError(
            f"public case/oracle mismatch: missing={missing!r} extra={extra!r}"
        )
    expected_public_checkpoints = {
        (case.case_id, checkpoint.checkpoint_id)
        for case in public_cases.values()
        for checkpoint in case.checkpoints
    }
    if oracle_keys != expected_public_checkpoints:
        missing = sorted(expected_public_checkpoints - oracle_keys)
        extra = sorted(oracle_keys - expected_public_checkpoints)
        raise ScoreValidationError(
            f"public checkpoint/oracle mismatch: missing={missing!r} extra={extra!r}"
        )
    for key, oracle in sorted(oracles.items()):
        case = public_cases[key[0]]
        checkpoint = case.checkpoint(key[1])
        eligible_ids = {
            event.event_id for event in case.events if event.turn <= checkpoint.turn
        }
        referenced_ids = {
            *oracle.required_current_event_ids,
            *oracle.stale_conflict_event_ids,
            *(event_id for edge in oracle.required_order_edges for event_id in edge),
        }
        unknown = sorted(referenced_ids - eligible_ids)
        if unknown:
            raise ScoreValidationError(
                f"oracle {key!r} references unknown or future public events: {unknown!r}"
            )
    if observed_checkpoint_keys != oracle_keys:
        missing_oracles = sorted(observed_checkpoint_keys - oracle_keys)
        missing_contexts = sorted(oracle_keys - observed_checkpoint_keys)
        raise ScoreValidationError(
            "context/oracle checkpoint mismatch: "
            f"missing_oracles={missing_oracles!r} missing_contexts={missing_contexts!r}"
        )

    expected_keys = {
        (case_id, checkpoint_id, budget, treatment)
        for case_id, checkpoint_id in oracle_keys
        for budget in expected_budgets
        for treatment in expected_treatments
    }
    if context_keys != expected_keys:
        missing = sorted(expected_keys - context_keys)
        extra = sorted(context_keys - expected_keys)
        raise ScoreValidationError(
            f"treatment matrix mismatch: missing={missing[:10]!r} extra={extra[:10]!r}"
        )

    scored: list[ScoredContext] = []
    for key in sorted(context_keys):
        raw = raw_by_key[key]
        oracle = oracles[(key[0], key[1])]
        scored.append(
            score_context(
                raw,
                oracle,
                public_cases[key[0]],
                where=f"context{key!r}",
            )
        )

    per_case: dict[str, list[ScoredContext]] = defaultdict(list)
    per_checkpoint_budget: dict[tuple[str, str, int], list[ScoredContext]] = defaultdict(list)
    for record in scored:
        per_case[record.case_id].append(record)
        per_checkpoint_budget[
            (record.case_id, record.checkpoint_id, record.budget_tokens)
        ].append(record)

    for record in scored:
        oracle = oracles[(record.case_id, record.checkpoint_id)]
        family_is_insufficient = record.family == "insufficient-evidence"
        oracle_is_insufficient = oracle.expected_disposition == "insufficient"
        if family_is_insufficient != oracle_is_insufficient:
            raise ScoreValidationError(
                f"family/disposition mismatch for {record.case_id!r}/{record.checkpoint_id!r}"
            )

    manifest_case_hashes = _manifest_case_hashes(manifest)
    if manifest_case_hashes and set(manifest_case_hashes) != set(per_case):
        missing = sorted(set(per_case) - set(manifest_case_hashes))
        extra = sorted(set(manifest_case_hashes) - set(per_case))
        raise ScoreValidationError(
            f"manifest/source case map mismatch: missing={missing!r} extra={extra!r}"
        )
    for case_id, rows in sorted(per_case.items()):
        families = {row.family for row in rows}
        seeds = {row.seed for row in rows}
        source_hashes = {row.source_case_sha256 for row in rows}
        if len(families) != 1 or len(seeds) != 1 or len(source_hashes) != 1:
            raise ScoreValidationError(
                f"case {case_id!r} changes family, seed, or source hash across arms"
            )
        declared_source = manifest_case_hashes.get(case_id)
        actual_source = next(iter(source_hashes))
        if declared_source is not None and not _constant_time_equal(declared_source, actual_source):
            raise ScoreValidationError(f"source case hash mismatch for {case_id!r}")

    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ScoreValidationError("manifest.config must be a JSON object")
    declared_case_count = config.get("cases")
    if declared_case_count is not None and (
        isinstance(declared_case_count, bool)
        or not isinstance(declared_case_count, int)
        or declared_case_count != len(per_case)
    ):
        raise ScoreValidationError(
            f"manifest case count mismatch: declared={declared_case_count!r} "
            f"observed={len(per_case)}"
        )
    declared_encoding = config.get("encoding")
    if declared_encoding is not None and declared_encoding != "cl100k_base":
        raise ScoreValidationError(
            f"manifest encoding mismatch: expected 'cl100k_base', got {declared_encoding!r}"
        )
    declared_families = config.get("families")
    observed_families = sorted({rows[0].family for rows in per_case.values()})
    if declared_families is not None:
        if not isinstance(declared_families, list) or any(
            not isinstance(item, str) for item in declared_families
        ):
            raise ScoreValidationError("manifest config.families must be a string list")
        if not set(observed_families).issubset(set(declared_families)):
            raise ScoreValidationError(
                "observed source case family is absent from manifest"
            )
    declared_family_counts = config.get("family_counts")
    if declared_family_counts is not None:
        if not isinstance(declared_family_counts, Mapping):
            raise ScoreValidationError("manifest config.family_counts must be an object")
        normalized_declared = {
            str(family): count
            for family, count in declared_family_counts.items()
            if isinstance(count, int) and not isinstance(count, bool)
        }
        if len(normalized_declared) != len(declared_family_counts):
            raise ScoreValidationError(
                "manifest config.family_counts values must be integers"
            )
        observed_family_counts = {
            family: sum(rows[0].family == family for rows in per_case.values())
            for family in normalized_declared
        }
        for family in observed_families:
            if family not in observed_family_counts:
                observed_family_counts[family] = sum(
                    rows[0].family == family for rows in per_case.values()
                )
        if normalized_declared != observed_family_counts:
            raise ScoreValidationError(
                "manifest family_counts differs from observed source cases"
            )

    counts = manifest.get("counts")
    if isinstance(counts, Mapping):
        declared_oracles = counts.get("oracle_records")
        if declared_oracles is not None and declared_oracles != len(oracles):
            raise ScoreValidationError(
                f"manifest oracle count mismatch: declared={declared_oracles!r} "
                f"observed={len(oracles)}"
            )
        declared_checkpoints = counts.get("checkpoints")
        if declared_checkpoints is not None and declared_checkpoints != len(oracles):
            raise ScoreValidationError(
                f"manifest checkpoint count mismatch: declared={declared_checkpoints!r} "
                f"observed={len(oracles)}"
            )

    comparable_extra_fields = (
        "source_checkpoint_sha256",
        "configuration_sha256",
        "config_sha256",
        "manifest_sha256",
    )
    for cell, rows in sorted(per_checkpoint_budget.items()):
        if {row.treatment for row in rows} != set(expected_treatments):
            raise ScoreValidationError(f"arm mismatch in cell {cell!r}")
        if len({row.family for row in rows}) != 1 or len({row.seed for row in rows}) != 1:
            raise ScoreValidationError(f"source metadata differs across arms in cell {cell!r}")
        if len({row.source_case_sha256 for row in rows}) != 1:
            raise ScoreValidationError(f"source hash differs across arms in cell {cell!r}")
        if len({row.checkpoint_turn for row in rows}) != 1:
            raise ScoreValidationError(f"checkpoint turn differs across arms in cell {cell!r}")
        raw_rows = [raw_by_key[row.key] for row in rows]
        rendered_queries = {
            query
            for raw in raw_rows
            if (query := _rendered_query(str(raw["rendered_context"]))) is not None
        }
        if len(rendered_queries) > 1:
            raise ScoreValidationError(f"checkpoint query differs across arms in cell {cell!r}")
        rendered_checkpoints = {
            checkpoint
            for raw in raw_rows
            if (
                checkpoint := _rendered_checkpoint(str(raw["rendered_context"]))
            )
            is not None
        }
        if len(rendered_checkpoints) > 1:
            raise ScoreValidationError(f"checkpoint header differs across arms in cell {cell!r}")
        for field in comparable_extra_fields:
            values = {canonical_json_bytes(raw[field]) for raw in raw_rows if field in raw}
            present = sum(field in raw for raw in raw_rows)
            if present not in {0, len(raw_rows)} or len(values) > 1:
                raise ScoreValidationError(f"{field} differs across arms in cell {cell!r}")

    observed_turns = tuple(sorted({row.checkpoint_turn for row in scored}))
    turns_declared = _manifest_sequence(manifest, "checkpoints")
    if turns_declared is not None and all(isinstance(value, int) for value in turns_declared):
        if tuple(sorted(set(int(value) for value in turns_declared))) != observed_turns:
            raise ScoreValidationError(
                f"checkpoint turns differ from manifest: observed={observed_turns!r}"
            )
        expected_turns = set(int(value) for value in turns_declared)
        for case_id, rows in sorted(per_case.items()):
            case_turns = {row.checkpoint_turn for row in rows}
            if case_turns != expected_turns:
                raise ScoreValidationError(
                    f"case {case_id!r} checkpoint turns differ from manifest"
                )

    validation = {
        "status": "PASS",
        "valid": True,
        "context_records": len(scored),
        "oracle_records": len(oracles),
        "cases": len(per_case),
        "checkpoint_cells": len(oracles),
        "expected_records": len(expected_keys),
        "context_hashes_rederived": len(scored),
        "token_counts_rederived": len(scored),
        "rendered_event_lists_rederived": len(scored),
        "source_hashes_checked": len(per_case),
        "public_source_records_checked": len(scored),
        "public_source_validation": True,
        "treatment_matrix_complete": True,
    }
    return scored, validation


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 10)


def _rate_summary(
    records: Sequence[ScoredContext],
    *,
    metric: str,
    accessor: Callable[[ScoredContext], bool],
    scope: str,
    scope_values: Mapping[str, Any],
    n_resamples: int,
) -> JsonObject:
    values = [bool(accessor(record)) for record in records]
    total = len(values)
    hits = sum(values)
    naive_low, naive_high = wilson_interval(hits, total)
    seed_parts: list[object] = [PROTOCOL_VERSION, "rate", metric, scope]
    for key, value in sorted(scope_values.items()):
        seed_parts.extend((key, value))
    point, low, high = clustered_bootstrap_mean_ci(
        [(record.cluster, float(value)) for record, value in zip(records, values)],
        seed=stable_seed(*seed_parts),
        n_resamples=n_resamples,
    )
    return {
        "hits": hits,
        "n": total,
        "n_clusters": len({record.cluster for record in records}),
        "rate": _round(point),
        "cluster_bootstrap95": [_round(low), _round(high)],
        "naive_wilson95": [_round(naive_low), _round(naive_high)],
    }


def _numeric_summary(values: Sequence[float]) -> JsonObject:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "p95": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": _round(float(np.mean(array))),
        "median": _round(float(np.median(array))),
        "min": _round(float(np.min(array))),
        "max": _round(float(np.max(array))),
        "p95": _round(float(np.quantile(array, 0.95, method="linear"))),
    }


def summarize_cell(
    records: Sequence[ScoredContext],
    *,
    scope: str = "custom",
    scope_values: Mapping[str, Any] | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> JsonObject:
    """Summarize one pre-specified analysis cell."""

    labels = dict(scope_values or {})
    recall_values = [
        record.required_fact_recall
        for record in records
        if record.required_fact_recall is not None
    ]
    summary = {
        "n": len(records),
        "decision_ready": _rate_summary(
            records,
            metric="decision_ready",
            accessor=lambda record: record.decision_ready,
            scope=scope,
            scope_values=labels,
            n_resamples=n_resamples,
        ),
        "required_fact_recall": _numeric_summary(recall_values),
        "stale_exposure": _rate_summary(
            records,
            metric="stale_exposure",
            accessor=lambda record: record.stale_exposure,
            scope=scope,
            scope_values=labels,
            n_resamples=n_resamples,
        ),
        "stale_conflict_fraction": _numeric_summary(
            [record.stale_conflict_fraction for record in records]
        ),
        "contradiction": _rate_summary(
            records,
            metric="contradiction",
            accessor=lambda record: record.contradiction,
            scope=scope,
            scope_values=labels,
            n_resamples=n_resamples,
        ),
        "fabricated_sufficiency": _rate_summary(
            records,
            metric="fabricated_sufficiency",
            accessor=lambda record: record.fabricated_sufficiency,
            scope=scope,
            scope_values=labels,
            n_resamples=n_resamples,
        ),
        "realized_tokens": _numeric_summary([float(record.realized_tokens) for record in records]),
        "token_utilization": _numeric_summary([record.token_utilization for record in records]),
    }
    # Flat aliases make the aggregate straightforward to consume in a CSV or a
    # small plotting script without weakening the structured metric metadata.
    summary.update(
        {
            "decision_ready_rate": summary["decision_ready"]["rate"],
            "decision_ready_ci95": summary["decision_ready"]["cluster_bootstrap95"],
            "decision_ready_cluster_bootstrap95": summary["decision_ready"][
                "cluster_bootstrap95"
            ],
            "decision_ready_naive_wilson95": summary["decision_ready"][
                "naive_wilson95"
            ],
            "required_fact_recall_mean": summary["required_fact_recall"]["mean"],
            "stale_exposure_rate": summary["stale_exposure"]["rate"],
            "contradiction_rate": summary["contradiction"]["rate"],
            "mean_realized_tokens": summary["realized_tokens"]["mean"],
        }
    )
    return summary


def _breakdown(
    records: Sequence[ScoredContext],
    dimensions: Sequence[tuple[str, Callable[[ScoredContext], Any]]],
    *,
    scope: str,
    n_resamples: int,
) -> list[JsonObject]:
    groups: dict[tuple[Any, ...], list[ScoredContext]] = defaultdict(list)
    for record in records:
        groups[tuple(accessor(record) for _, accessor in dimensions)].append(record)
    result: list[JsonObject] = []
    for key in sorted(groups):
        labels = {name: value for (name, _), value in zip(dimensions, key)}
        result.append(
            {
                **labels,
                **summarize_cell(
                    groups[key],
                    scope=scope,
                    scope_values=labels,
                    n_resamples=n_resamples,
                ),
            }
        )
    return result


_METRIC_ACCESSORS: tuple[tuple[str, Callable[[ScoredContext], float | None]], ...] = (
    ("decision_ready", lambda row: float(row.decision_ready)),
    ("required_fact_recall", lambda row: row.required_fact_recall),
    ("stale_exposure", lambda row: float(row.stale_exposure)),
    ("contradiction", lambda row: float(row.contradiction)),
    ("realized_tokens", lambda row: float(row.realized_tokens)),
)


def _paired_delta(
    records: Sequence[ScoredContext],
    *,
    left: str,
    right: str,
    metric: str,
    accessor: Callable[[ScoredContext], float | None],
    scope: str,
    scope_values: Mapping[str, Any],
    n_resamples: int,
) -> JsonObject | None:
    by_observation: dict[tuple[str, str, int], dict[str, ScoredContext]] = defaultdict(dict)
    for row in records:
        if row.treatment in {left, right}:
            by_observation[(row.case_id, row.checkpoint_id, row.budget_tokens)][row.treatment] = row
    clustered_values: list[tuple[Cluster, float]] = []
    for key, arms in sorted(by_observation.items()):
        if set(arms) != {left, right}:
            raise ScoreValidationError(f"paired comparison lacks an arm for observation {key!r}")
        left_value = accessor(arms[left])
        right_value = accessor(arms[right])
        if left_value is None or right_value is None:
            continue
        if arms[left].cluster != arms[right].cluster:
            raise ScoreValidationError(f"paired comparison changes cluster for {key!r}")
        clustered_values.append((arms[left].cluster, left_value - right_value))
    if not clustered_values:
        return None
    comparison = f"{left}-{right}"
    seed_parts: list[object] = [PROTOCOL_VERSION, "paired", comparison, metric, scope]
    for key, value in sorted(scope_values.items()):
        seed_parts.extend((key, value))
    point, low, high = clustered_bootstrap_mean_ci(
        clustered_values,
        seed=stable_seed(*seed_parts),
        n_resamples=n_resamples,
    )
    return {
        "comparison": comparison,
        "metric": metric,
        "scope": scope,
        **dict(scope_values),
        "n_pairs": len(clustered_values),
        "n_clusters": len({cluster for cluster, _ in clustered_values}),
        "delta": _round(point),
        "bootstrap95": [_round(low), _round(high)],
    }


def paired_deltas(
    records: Sequence[ScoredContext],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[JsonObject]:
    """Compute preregistered paired deltas in all reporting strata."""

    comparisons = (
        ("state_latest", "tail"),
        ("state_latest", "state_first"),
    )
    scopes: list[tuple[str, tuple[str, ...]]] = [
        ("overall", ()),
        ("family", ("family",)),
        ("turn", ("checkpoint_turn",)),
        ("budget", ("budget_tokens",)),
        ("family_turn_budget", ("family", "checkpoint_turn", "budget_tokens")),
    ]
    output: list[JsonObject] = []
    for scope, fields in scopes:
        grouped: dict[tuple[Any, ...], list[ScoredContext]] = defaultdict(list)
        if not fields:
            grouped[()] = list(records)
        else:
            for row in records:
                grouped[tuple(getattr(row, field) for field in fields)].append(row)
        for values in sorted(grouped):
            scope_values = dict(zip(fields, values))
            rows = grouped[values]
            for left, right in comparisons:
                for metric, accessor in _METRIC_ACCESSORS:
                    result = _paired_delta(
                        rows,
                        left=left,
                        right=right,
                        metric=metric,
                        accessor=accessor,
                        scope=scope,
                        scope_values=scope_values,
                        n_resamples=n_resamples,
                    )
                    if result is not None:
                        output.append(result)
    return output


def _b95_for_groups(
    records: Sequence[ScoredContext],
    *,
    dimensions: Sequence[str],
) -> list[JsonObject]:
    groups: dict[tuple[Any, ...], list[ScoredContext]] = defaultdict(list)
    for row in records:
        key = (row.treatment, *(getattr(row, field) for field in dimensions))
        groups[key].append(row)
    output: list[JsonObject] = []
    for key in sorted(groups):
        rows = groups[key]
        by_budget: dict[int, list[ScoredContext]] = defaultdict(list)
        for row in rows:
            by_budget[row.budget_tokens].append(row)
        rates = {
            budget: statistics.fmean(float(row.decision_ready) for row in budget_rows)
            for budget, budget_rows in sorted(by_budget.items())
        }
        qualifying = [budget for budget, rate in rates.items() if rate >= 0.95]
        labels = {"treatment": key[0]}
        labels.update(dict(zip(dimensions, key[1:])))
        output.append(
            {
                **labels,
                "b95": min(qualifying) if qualifying else None,
                "threshold": 0.95,
                "tested_budgets": [
                    {"budget_tokens": budget, "decision_ready_rate": _round(rate)}
                    for budget, rate in rates.items()
                ],
            }
        )
    return output


def _turns_to_first_failure(records: Sequence[ScoredContext]) -> list[JsonObject]:
    sessions: dict[tuple[str, int, str, int], list[ScoredContext]] = defaultdict(list)
    for row in records:
        sessions[(row.case_id, row.seed, row.treatment, row.budget_tokens)].append(row)
    grouped: dict[tuple[str, int], list[int | None]] = defaultdict(list)
    max_turn_by_group: dict[tuple[str, int], int] = {}
    for (_, _, treatment, budget), rows in sorted(sessions.items()):
        ordered = sorted(rows, key=lambda row: row.checkpoint_turn)
        failure = next((row.checkpoint_turn for row in ordered if not row.decision_ready), None)
        grouped[(treatment, budget)].append(failure)
        max_turn_by_group[(treatment, budget)] = max(row.checkpoint_turn for row in ordered)
    output: list[JsonObject] = []
    for (treatment, budget), failures in sorted(grouped.items()):
        observed = [turn for turn in failures if turn is not None]
        output.append(
            {
                "treatment": treatment,
                "budget_tokens": budget,
                "sessions": len(failures),
                "failed_sessions": len(observed),
                "censored_sessions": sum(turn is None for turn in failures),
                "mean_first_failure_turn": _round(statistics.fmean(observed)) if observed else None,
                "median_first_failure_turn": _round(statistics.median(observed)) if observed else None,
                "last_checkpoint_turn": max_turn_by_group[(treatment, budget)],
            }
        )
    return output


def _special_paired_delta(
    records: Sequence[ScoredContext],
    *,
    left: str,
    right: str,
    label: str,
    predicate: Callable[[ScoredContext], bool],
    n_resamples: int,
) -> JsonObject | None:
    rows = [row for row in records if predicate(row)]
    return _paired_delta(
        rows,
        left=left,
        right=right,
        metric="decision_ready",
        accessor=lambda row: float(row.decision_ready),
        scope=label,
        scope_values={},
        n_resamples=n_resamples,
    )


def _preregistered_checks(
    records: Sequence[ScoredContext],
    *,
    n_resamples: int,
    validity_passed: bool,
    public_source_validated: bool,
) -> JsonObject:
    h1 = _special_paired_delta(
        records,
        left="state_latest",
        right="tail",
        label="h1_long_update_dependency",
        predicate=lambda row: row.family in {"update-heavy", "dependency-heavy"}
        and row.checkpoint_turn >= 32,
        n_resamples=n_resamples,
    )
    h2 = _special_paired_delta(
        records,
        left="state_latest",
        right="state_first",
        label="h2_update_heavy",
        predicate=lambda row: row.family == "update-heavy",
        n_resamples=n_resamples,
    )
    chronology = _special_paired_delta(
        records,
        left="state_latest",
        right="tail",
        label="chronology_control",
        predicate=lambda row: row.family == "chronology-sensitive",
        n_resamples=n_resamples,
    )
    insufficient_rows = [row for row in records if row.expected_disposition == "insufficient"]
    h5_violations = sum(row.fabricated_sufficiency for row in insufficient_rows)
    h1_pass = bool(h1 and h1["bootstrap95"][0] is not None and h1["bootstrap95"][0] > 0.0)
    h2_pass = bool(h2 and h2["bootstrap95"][0] is not None and h2["bootstrap95"][0] > 0.0)
    chronology_delta = chronology["delta"] if chronology is not None else None
    chronology_pass = chronology_delta is not None and chronology_delta >= -0.05
    h5_pass = public_source_validated and h5_violations == 0
    main_claim = (
        validity_passed
        and public_source_validated
        and h1_pass
        and h2_pass
        and h5_pass
        and chronology_pass
    )
    return {
        "h1": {"pass": h1_pass, "paired_delta": h1},
        "h2": {"pass": h2_pass, "paired_delta": h2},
        "h5": {
            "pass": h5_pass,
            "public_source_validated": public_source_validated,
            "insufficient_contexts": len(insufficient_rows),
            "fabricated_sufficiency_violations": h5_violations,
        },
        "chronology_noninferiority": {
            "pass": chronology_pass,
            "margin": -0.05,
            "paired_delta": chronology,
        },
        "main_claim_supported": main_claim,
        "interpretation": "supported" if main_claim else "mixed_or_null",
    }


def aggregate_scores(
    records: Sequence[ScoredContext],
    *,
    validity: Mapping[str, Any] | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> JsonObject:
    """Build the deterministic preregistered RCWT-S aggregate artifact."""

    if not records:
        raise ScoreValidationError("cannot aggregate an empty score set")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    ordered = sorted(records, key=lambda row: row.key)
    validity_result = dict(
        validity
        or {
            "status": "UNVERIFIED",
            "valid": False,
            "public_source_validation": False,
        }
    )
    breakdowns = {
        "treatment": _breakdown(
            ordered,
            (("treatment", lambda row: row.treatment),),
            scope="treatment",
            n_resamples=n_resamples,
        ),
        "family": _breakdown(
            ordered,
            (
                ("treatment", lambda row: row.treatment),
                ("family", lambda row: row.family),
            ),
            scope="family",
            n_resamples=n_resamples,
        ),
        "turn": _breakdown(
            ordered,
            (
                ("treatment", lambda row: row.treatment),
                ("checkpoint_turn", lambda row: row.checkpoint_turn),
            ),
            scope="turn",
            n_resamples=n_resamples,
        ),
        "budget": _breakdown(
            ordered,
            (
                ("treatment", lambda row: row.treatment),
                ("budget_tokens", lambda row: row.budget_tokens),
            ),
            scope="budget",
            n_resamples=n_resamples,
        ),
        "family_turn_budget": _breakdown(
            ordered,
            (
                ("treatment", lambda row: row.treatment),
                ("family", lambda row: row.family),
                ("checkpoint_turn", lambda row: row.checkpoint_turn),
                ("budget_tokens", lambda row: row.budget_tokens),
            ),
            scope="family_turn_budget",
            n_resamples=n_resamples,
        ),
    }
    return {
        "protocol": PROTOCOL_VERSION,
        "analysis": {
            "primary_interval_method": (
                "percentile bootstrap over case/seed cluster means"
            ),
            "paired_bootstrap_method": (
                "paired percentile bootstrap over case/seed cluster means"
            ),
            "bootstrap_resamples": n_resamples,
            "bootstrap_confidence": CONFIDENCE,
            "naive_binary_interval_method": (
                "Wilson score interval over correlated records; descriptive only"
            ),
            "naive_binary_interval_confidence": CONFIDENCE,
            "b95_definition": "smallest tested budget with point decision-readiness rate >= 0.95",
            "contradiction_method": (
                "explicit stale-current pairs when supplied; otherwise stale exposure "
                "is counted conservatively as contradiction"
            ),
            "zero_required_recall": "null and excluded from recall means",
        },
        "validity": validity_result,
        "counts": {
            "records": len(ordered),
            "cases": len({row.cluster for row in ordered}),
            "checkpoint_pairs": len({(row.case_id, row.checkpoint_id) for row in ordered}),
            "families": sorted({row.family for row in ordered}),
            "turns": sorted({row.checkpoint_turn for row in ordered}),
            "budgets": sorted({row.budget_tokens for row in ordered}),
            "treatments": sorted({row.treatment for row in ordered}),
        },
        "overall": summarize_cell(
            ordered,
            scope="overall",
            n_resamples=n_resamples,
        ),
        "breakdowns": breakdowns,
        "b95": {
            "overall": _b95_for_groups(ordered, dimensions=()),
            "by_family": _b95_for_groups(ordered, dimensions=("family",)),
            "by_turn": _b95_for_groups(ordered, dimensions=("checkpoint_turn",)),
        },
        "turns_to_first_failure": _turns_to_first_failure(ordered),
        "paired_deltas": paired_deltas(ordered, n_resamples=n_resamples),
        "preregistered_checks": _preregistered_checks(
            ordered,
            n_resamples=n_resamples,
            validity_passed=bool(validity_result.get("valid")),
            public_source_validated=bool(
                validity_result.get("public_source_validation")
            ),
        ),
    }


def score_records(
    context_records: Sequence[Any],
    oracle_records: Sequence[Any],
    manifest: Mapping[str, Any],
    *,
    public_cases: Sequence[Any],
    manifest_checks: Sequence[str] = (),
    input_sha256: Mapping[str, str] | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> JsonObject:
    """Validate a complete in-memory run, score it, and aggregate it."""

    normalized_contexts = [
        _as_mapping(record, f"contexts:{index}")
        for index, record in enumerate(context_records, start=1)
    ]
    oracles = _load_oracles(oracle_records)
    sources = _load_public_cases(public_cases)
    scored, validity = _validate_matrix(
        normalized_contexts,
        oracles,
        sources,
        manifest,
    )
    validity["manifest_checks"] = list(manifest_checks)
    validity["input_sha256"] = dict(input_sha256 or {})
    return aggregate_scores(scored, validity=validity, n_resamples=n_resamples)


def score_files(
    contexts_path: Path,
    oracle_path: Path,
    manifest_path: Path,
    *,
    cases_path: Path,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> JsonObject:
    """Load, validate, score, and aggregate the four locked input artifacts."""

    contexts = load_jsonl(contexts_path, label="contexts")
    oracle = load_jsonl(oracle_path, label="oracle")
    public_cases = load_jsonl(cases_path, label="public_cases")
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreValidationError(f"cannot read manifest at {manifest_path}: {exc}") from exc
    if not isinstance(manifest_value, dict):
        raise ScoreValidationError("manifest must contain a JSON object")
    checks = _validate_manifest(
        manifest_value,
        manifest_path=manifest_path,
        contexts_path=contexts_path,
        oracle_path=oracle_path,
        cases_path=cases_path,
    )

    actual_contexts_digest = file_sha256(contexts_path)
    actual_oracle_digest = file_sha256(oracle_path)
    actual_cases_digest = file_sha256(cases_path)
    actual_manifest_digest = file_sha256(manifest_path)
    for index, context in enumerate(contexts, start=1):
        declared = context.get("manifest_sha256")
        if declared is not None:
            if not isinstance(declared, str) or _SHA256_RE.fullmatch(declared.lower()) is None:
                raise ScoreValidationError(f"contexts:{index}.manifest_sha256 is invalid")
            canonical_self = _digest_value(manifest_value.get("manifest_sha256"))
            accepted = {actual_manifest_digest}
            if canonical_self is not None:
                accepted.add(canonical_self)
            if declared.lower() not in accepted:
                raise ScoreValidationError(f"contexts:{index}.manifest_sha256 mismatch")
    return score_records(
        contexts,
        oracle,
        manifest_value,
        public_cases=public_cases,
        manifest_checks=checks,
        input_sha256={
            "contexts": actual_contexts_digest,
            "oracle": actual_oracle_digest,
            "public_cases": actual_cases_digest,
            "manifest": actual_manifest_digest,
        },
        n_resamples=n_resamples,
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and score the deterministic RCWT-S session experiment"
    )
    parser.add_argument("--contexts", type=Path, required=True, help="ContextRecord JSONL")
    parser.add_argument("--oracle", type=Path, required=True, help="separate OracleCase JSONL")
    parser.add_argument("--cases", type=Path, required=True, help="public_cases.jsonl source")
    parser.add_argument("--manifest", type=Path, required=True, help="run manifest JSON")
    parser.add_argument("--output", type=Path, required=True, help="aggregate JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        aggregate = score_files(
            args.contexts,
            args.oracle,
            args.manifest,
            cases_path=args.cases,
        )
        _atomic_write_json(args.output, aggregate)
    except ScoreValidationError as exc:
        print(f"RCWT-S validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        "RCWT-S score PASS: "
        f"records={aggregate['counts']['records']} "
        f"cases={aggregate['counts']['cases']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
