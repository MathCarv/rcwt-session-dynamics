"""Build deterministic, token-bounded RCWT-S treatment contexts.

This runner is intentionally oracle-blind: it consumes only ``public_cases``.
``tail`` preserves the newest complete event lines, while ``state_latest`` and
``state_first`` select query-relevant state, close its dependency graph, and
render dependencies before dependants with recency as the tie-breaker.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import tiktoken

from rcwt_session_types import (
    DEFAULT_BUDGETS,
    DEFAULT_CHECKPOINTS,
    TREATMENTS,
    Checkpoint,
    ContextRecord,
    Event,
    PublicCase,
    Treatment,
    bytes_sha256,
    canonical_json,
    canonical_sha256,
    jsonl_bytes,
    load_public_cases,
)

ENCODING_NAME = "cl100k_base"
PROTOCOL_VERSION = "session-v1"


@dataclass(frozen=True, slots=True)
class StateResolution:
    """Public-data derivation of the evidence needed for one checkpoint."""

    target_event_ids: tuple[str, ...]
    required_event_ids: tuple[str, ...]
    stale_event_ids: tuple[str, ...]
    order_edges: tuple[tuple[str, str], ...]
    complete: bool


def count_tokens(text: str) -> int:
    """Count context tokens with the preregistered ``cl100k_base`` encoding."""

    return len(tiktoken.get_encoding(ENCODING_NAME).encode(text))


def _events_at_checkpoint(case: PublicCase, checkpoint: Checkpoint) -> tuple[Event, ...]:
    return tuple(event for event in case.events if event.turn <= checkpoint.turn)


def _topological_recent(events: Iterable[Event]) -> tuple[Event, ...]:
    """Order events causally; among currently valid nodes, prefer recency."""

    unique = {event.event_id: event for event in events}
    outgoing: dict[str, list[str]] = {event_id: [] for event_id in unique}
    indegree = {event_id: 0 for event_id in unique}
    for event in unique.values():
        for dependency in event.depends_on:
            if dependency not in unique:
                continue
            outgoing[dependency].append(event.event_id)
            indegree[event.event_id] += 1

    available: list[tuple[int, str]] = []
    for event_id, degree in indegree.items():
        if degree == 0:
            event = unique[event_id]
            heapq.heappush(available, (-event.turn, event_id))

    ordered: list[Event] = []
    while available:
        _, event_id = heapq.heappop(available)
        ordered.append(unique[event_id])
        for child_id in sorted(outgoing[event_id]):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                child = unique[child_id]
                heapq.heappush(available, (-child.turn, child_id))

    if len(ordered) != len(unique):
        raise ValueError("event dependency graph contains a cycle")
    return tuple(ordered)


def resolve_state(
    case: PublicCase,
    checkpoint: Checkpoint,
    *,
    version: Literal["latest", "first"] = "latest",
) -> StateResolution:
    """Resolve query state and dependency closure using public event metadata.

    A state version is chosen independently for every ``(entity, field)`` key.
    ``latest`` first removes explicitly superseded versions (including grant /
    revoke transitions) and then chooses the newest remaining head.  ``first``
    chooses the oldest root, forming the preregistered stale-state ablation.
    """

    if version not in ("latest", "first"):
        raise ValueError(f"unsupported state version: {version!r}")

    eligible = _events_at_checkpoint(case, checkpoint)
    by_id = {event.event_id: event for event in eligible}
    targets: list[Event] = []
    stale: set[str] = set()
    complete = True

    for entity in checkpoint.query_entities:
        entity_events = [event for event in eligible if event.entity == entity]
        if not entity_events:
            complete = False
            continue
        by_key: dict[tuple[str, str], list[Event]] = {}
        for event in entity_events:
            by_key.setdefault((event.entity, event.field), []).append(event)

        for key_events in by_key.values():
            key_events.sort(key=lambda event: (event.turn, event.event_id))
            if version == "latest":
                superseded = {
                    event.supersedes
                    for event in key_events
                    if event.supersedes is not None
                }
                heads = [
                    event for event in key_events if event.event_id not in superseded
                ]
                chosen = max(heads or key_events, key=lambda event: (event.turn, event.event_id))
                stale.update(
                    event.event_id for event in key_events if event.event_id != chosen.event_id
                )
            else:
                key_ids = {event.event_id for event in key_events}
                roots = [
                    event
                    for event in key_events
                    if event.supersedes is None or event.supersedes not in key_ids
                ]
                chosen = min(roots or key_events, key=lambda event: (event.turn, event.event_id))
            targets.append(chosen)

    closure: dict[str, Event] = {}
    visiting: set[str] = set()

    def add_with_dependencies(event: Event) -> None:
        nonlocal complete
        if event.event_id in closure:
            return
        if event.event_id in visiting:
            raise ValueError(f"dependency cycle reaches {event.event_id}")
        visiting.add(event.event_id)
        for dependency_id in event.depends_on:
            dependency = by_id.get(dependency_id)
            if dependency is None:
                complete = False
                continue
            add_with_dependencies(dependency)
        visiting.remove(event.event_id)
        closure[event.event_id] = event

    for target in targets:
        add_with_dependencies(target)

    ordered = _topological_recent(closure.values())
    required_ids = {event.event_id for event in ordered}
    # Superseded transitions may still be necessary to establish chronology.
    # Required causal evidence cannot simultaneously be a stale conflict.
    stale.difference_update(required_ids)
    edges = sorted(
        {
            (dependency_id, event.event_id)
            for event in ordered
            for dependency_id in event.depends_on
            if dependency_id in required_ids
        }
    )
    target_ids = tuple(
        event.event_id
        for event in sorted(targets, key=lambda item: (item.turn, item.event_id))
    )
    return StateResolution(
        target_event_ids=target_ids,
        required_event_ids=tuple(event.event_id for event in ordered),
        stale_event_ids=tuple(sorted(stale)),
        order_edges=tuple(edges),
        complete=complete and bool(targets),
    )


def _render_context(
    checkpoint: Checkpoint,
    events: Sequence[Event],
    *,
    ready: bool,
    missing_reason: str = "evidence_or_budget",
) -> str:
    lines = [
        f"CHECKPOINT {checkpoint.checkpoint_id} TURN {checkpoint.turn}",
        f"QUERY {checkpoint.query}",
        *(event.rendered for event in events),
        "DISPOSITION ready"
        if ready
        else f"DISPOSITION MISSING ({missing_reason})",
    ]
    return "\n".join(lines) + "\n"


def _minimal_missing(checkpoint: Checkpoint, budget_tokens: int) -> str:
    """Produce the most informative MISSING marker that fits a tiny budget."""

    candidates = (
        _render_context(checkpoint, (), ready=False, missing_reason="budget"),
        f"CHECKPOINT {checkpoint.checkpoint_id}\nDISPOSITION MISSING (budget)\n",
        "DISPOSITION MISSING\n",
        "MISSING",
    )
    for candidate in candidates:
        if count_tokens(candidate) <= budget_tokens:
            return candidate
    encoder = tiktoken.get_encoding(ENCODING_NAME)
    return encoder.decode(encoder.encode("MISSING")[:budget_tokens])


def _tail_selection(
    case: PublicCase,
    checkpoint: Checkpoint,
    budget_tokens: int,
) -> tuple[tuple[Event, ...], str]:
    eligible = _events_at_checkpoint(case, checkpoint)
    requirement = resolve_state(case, checkpoint, version="latest")
    required = set(requirement.required_event_ids)
    for count in range(len(eligible), -1, -1):
        selected = eligible[len(eligible) - count :] if count else ()
        selected_ids = {event.event_id for event in selected}
        ready = requirement.complete and required.issubset(selected_ids)
        rendered = _render_context(
            checkpoint,
            selected,
            ready=ready,
            missing_reason="tail_omitted_required_evidence",
        )
        if count_tokens(rendered) <= budget_tokens:
            return selected, rendered
    return (), _minimal_missing(checkpoint, budget_tokens)


def _state_selection(
    case: PublicCase,
    checkpoint: Checkpoint,
    budget_tokens: int,
    *,
    version: Literal["latest", "first"],
) -> tuple[tuple[Event, ...], str]:
    resolution = resolve_state(case, checkpoint, version=version)
    eligible_by_id = {
        event.event_id: event for event in _events_at_checkpoint(case, checkpoint)
    }
    ordered = tuple(
        eligible_by_id[event_id]
        for event_id in resolution.required_event_ids
        if event_id in eligible_by_id
    )
    reason = "insufficient_evidence" if not resolution.complete else "budget"
    selected: tuple[Event, ...] = ()
    ready = False
    if resolution.complete:
        complete_render = _render_context(checkpoint, ordered, ready=True)
        if count_tokens(complete_render) <= budget_tokens:
            selected = ordered
            ready = True

    if not ready:
        # A prefix of the causal order is dependency-closed.  Keeping the
        # largest fitting prefix never emits a dependant without its evidence.
        for count in range(len(ordered), -1, -1):
            candidate_prefix = ordered[:count]
            candidate_render = _render_context(
                checkpoint,
                candidate_prefix,
                ready=False,
                missing_reason=reason,
            )
            if count_tokens(candidate_render) <= budget_tokens:
                selected = candidate_prefix
                break
        else:
            return (), _minimal_missing(checkpoint, budget_tokens)

    # The preregistered treatment fills residual capacity by recency.  Query
    # entity alternatives are excluded so filler cannot reintroduce a stale or
    # competing version after the state resolver made its explicit choice.
    eligible = _events_at_checkpoint(case, checkpoint)
    eligible_by_id = {event.event_id: event for event in eligible}
    selected_by_id = {event.event_id: event for event in selected}
    query_entities = set(checkpoint.query_entities)

    def filler_closure(event: Event) -> tuple[Event, ...] | None:
        pending = [event]
        closure: dict[str, Event] = {}
        while pending:
            current = pending.pop()
            if current.event_id in selected_by_id or current.event_id in closure:
                continue
            if current.entity in query_entities:
                return None
            closure[current.event_id] = current
            for dependency_id in current.depends_on:
                dependency = eligible_by_id.get(dependency_id)
                if dependency is None:
                    return None
                pending.append(dependency)
        return _topological_recent(closure.values())

    for filler in sorted(eligible, key=lambda item: (item.turn, item.event_id), reverse=True):
        if filler.event_id in selected_by_id or filler.entity in query_entities:
            continue
        closure = filler_closure(filler)
        if closure is None:
            continue
        proposed_by_id = dict(selected_by_id)
        proposed_by_id.update((event.event_id, event) for event in closure)
        proposed = _topological_recent(proposed_by_id.values())
        proposed_render = _render_context(
            checkpoint,
            proposed,
            ready=ready,
            missing_reason=reason,
        )
        if count_tokens(proposed_render) <= budget_tokens:
            selected_by_id = proposed_by_id

    selected = _topological_recent(selected_by_id.values())
    rendered = _render_context(
        checkpoint,
        selected,
        ready=ready,
        missing_reason=reason,
    )
    if count_tokens(rendered) > budget_tokens:
        raise AssertionError("internal error: state filler exceeded token budget")
    return selected, rendered


def build_context_record(
    case: PublicCase,
    checkpoint: Checkpoint,
    treatment: Treatment,
    budget_tokens: int,
) -> ContextRecord:
    """Build one treatment cell without reading or deriving a private oracle."""

    if treatment not in TREATMENTS:
        raise ValueError(f"unsupported treatment: {treatment!r}")
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    minimum_missing_tokens = count_tokens("DISPOSITION MISSING\n")
    if budget_tokens < minimum_missing_tokens:
        raise ValueError(
            f"budget_tokens must be >= {minimum_missing_tokens} so an explicit "
            "MISSING disposition can always be emitted"
        )
    if checkpoint not in case.checkpoints:
        raise ValueError(f"checkpoint {checkpoint.checkpoint_id} is not in {case.case_id}")

    if treatment == "tail":
        selected, rendered = _tail_selection(case, checkpoint, budget_tokens)
    else:
        version: Literal["latest", "first"] = (
            "latest" if treatment == "state_latest" else "first"
        )
        selected, rendered = _state_selection(
            case,
            checkpoint,
            budget_tokens,
            version=version,
        )
    token_count = count_tokens(rendered)
    if token_count > budget_tokens:  # Defensive invariant for alternate encoders.
        raise AssertionError("internal error: rendered context exceeds its budget")
    return ContextRecord.create(
        case_id=case.case_id,
        checkpoint_id=checkpoint.checkpoint_id,
        family=case.family,
        seed=case.seed,
        treatment=treatment,
        budget_tokens=budget_tokens,
        included_event_ids=[event.event_id for event in selected],
        rendered_context=rendered,
        token_count=token_count,
        source_case_sha256=case.sha256,
    )


def generate_context_records(
    cases: Sequence[PublicCase],
    *,
    treatments: Sequence[Treatment] = TREATMENTS,
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> Iterator[ContextRecord]:
    """Yield the deterministic case/checkpoint/treatment/budget cross-product."""

    treatment_order = tuple(dict.fromkeys(treatments))
    checkpoint_order = tuple(dict.fromkeys(checkpoints))
    budget_order = tuple(dict.fromkeys(budgets))
    if not treatment_order or any(item not in TREATMENTS for item in treatment_order):
        raise ValueError("treatments must contain only tail,state_latest,state_first")
    if not checkpoint_order or any(turn <= 0 for turn in checkpoint_order):
        raise ValueError("checkpoints must contain positive turns")
    if not budget_order or any(budget <= 0 for budget in budget_order):
        raise ValueError("budgets must contain positive token counts")

    wanted_turns = set(checkpoint_order)
    for case in sorted(cases, key=lambda item: item.case_id):
        available = {checkpoint.turn: checkpoint for checkpoint in case.checkpoints}
        missing = wanted_turns.difference(available)
        if missing:
            raise ValueError(
                f"case {case.case_id} lacks requested checkpoints: {sorted(missing)}"
            )
        for checkpoint_turn in checkpoint_order:
            checkpoint = available[checkpoint_turn]
            for treatment in treatment_order:
                for budget in budget_order:
                    yield build_context_record(case, checkpoint, treatment, budget)


def _manifest_mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _manifest_sequence(
    config: Mapping[str, Any], key: str, where: str
) -> tuple[object, ...]:
    value = config.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{where}.{key} must be a JSON array")
    return tuple(value)


def _digest(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{where} must be a lowercase SHA-256 digest")
    return normalized


def _declared_public_digest(manifest: Mapping[str, Any]) -> str:
    """Return one consistent required digest for the public-case artifact."""

    candidates: list[str] = []
    if "public_cases_sha256" in manifest:
        candidates.append(
            _digest(manifest["public_cases_sha256"], "manifest.public_cases_sha256")
        )
    hashes_raw = manifest.get("hashes")
    if hashes_raw is not None:
        hashes = _manifest_mapping(hashes_raw, "manifest.hashes")
        if "public_cases" in hashes:
            candidates.append(
                _digest(hashes["public_cases"], "manifest.hashes.public_cases")
            )
    artifacts_raw = manifest.get("artifacts")
    if artifacts_raw is not None:
        artifacts = _manifest_mapping(artifacts_raw, "manifest.artifacts")
        if "public_cases" in artifacts:
            public_artifact = _manifest_mapping(
                artifacts["public_cases"], "manifest.artifacts.public_cases"
            )
            candidates.append(
                _digest(
                    public_artifact.get("sha256"),
                    "manifest.artifacts.public_cases.sha256",
                )
            )
    if not candidates:
        raise ValueError("manifest does not declare public_cases SHA-256")
    if len(set(candidates)) != 1:
        raise ValueError("manifest has conflicting public_cases SHA-256 declarations")
    return candidates[0]


def load_and_validate_manifest(
    manifest_path: Path,
    cases_path: Path,
    cases: Sequence[PublicCase],
    *,
    treatments: Sequence[Treatment],
    checkpoints: Sequence[int],
    budgets: Sequence[int],
) -> dict[str, Any]:
    """Validate that a manifest seals the exact requested context run."""

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {manifest_path}: {exc}") from exc
    manifest = dict(_manifest_mapping(raw, "manifest"))
    protocols = [
        manifest[key]
        for key in ("protocol", "protocol_version")
        if key in manifest
    ]
    if not protocols or any(value != PROTOCOL_VERSION for value in protocols):
        raise ValueError(
            f"manifest protocol must be {PROTOCOL_VERSION!r} in every declaration"
        )
    if "manifest_sha256" in manifest:
        raise ValueError(
            "manifest_sha256 is incompatible with post-run sealing; remove the self-hash"
        )

    config = _manifest_mapping(manifest.get("config"), "manifest.config")
    declared_config_digest = _digest(
        manifest.get("config_sha256"), "manifest.config_sha256"
    )
    actual_config_digest = canonical_sha256(config)
    if declared_config_digest != actual_config_digest:
        raise ValueError(
            "manifest config hash mismatch: "
            f"declared={declared_config_digest} actual={actual_config_digest}"
        )
    hashes_raw = manifest.get("hashes")
    if hashes_raw is not None:
        hashes = _manifest_mapping(hashes_raw, "manifest.hashes")
        if "config" in hashes and _digest(
            hashes["config"], "manifest.hashes.config"
        ) != actual_config_digest:
            raise ValueError("manifest has conflicting config SHA-256 declarations")

    expected_treatments = tuple(treatments)
    expected_checkpoints = tuple(checkpoints)
    expected_budgets = tuple(budgets)
    declared_treatments = _manifest_sequence(
        config, "treatments", "manifest.config"
    )
    declared_checkpoints = _manifest_sequence(
        config, "checkpoints", "manifest.config"
    )
    declared_budgets = _manifest_sequence(config, "budgets", "manifest.config")
    if declared_treatments != expected_treatments:
        raise ValueError(
            "requested treatments differ from sealed config: "
            f"requested={expected_treatments!r} declared={declared_treatments!r}"
        )
    if declared_checkpoints != expected_checkpoints:
        raise ValueError(
            "requested checkpoints differ from sealed config: "
            f"requested={expected_checkpoints!r} declared={declared_checkpoints!r}"
        )
    if declared_budgets != expected_budgets:
        raise ValueError(
            "requested budgets differ from sealed config: "
            f"requested={expected_budgets!r} declared={declared_budgets!r}"
        )
    if config.get("encoding") != ENCODING_NAME:
        raise ValueError(f"manifest.config.encoding must be {ENCODING_NAME!r}")
    if config.get("cases") != len(cases):
        raise ValueError(
            f"manifest case count mismatch: declared={config.get('cases')!r} "
            f"actual={len(cases)}"
        )

    declared_public_digest = _declared_public_digest(manifest)
    actual_public_digest = bytes_sha256(cases_path.read_bytes())
    if declared_public_digest != actual_public_digest:
        raise ValueError(
            "public_cases digest mismatch: "
            f"declared={declared_public_digest} actual={actual_public_digest}"
        )
    case_hashes = _manifest_mapping(
        manifest.get("case_sha256"), "manifest.case_sha256"
    )
    if set(case_hashes) != {case.case_id for case in cases}:
        raise ValueError("manifest.case_sha256 does not cover exactly the loaded cases")
    for case in cases:
        declared_case_digest = _digest(
            case_hashes.get(case.case_id),
            f"manifest.case_sha256.{case.case_id}",
        )
        if declared_case_digest != case.sha256:
            raise ValueError(f"source case hash mismatch for {case.case_id}")
    return manifest


def seal_contexts_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    contexts_path: Path,
    contexts_payload: bytes,
    record_count: int,
) -> dict[str, Any]:
    """Return a manifest copy sealed to the newly written contexts artifact."""

    sealed = json.loads(canonical_json(manifest))
    if not isinstance(sealed, dict):  # Defensive; input was already a mapping.
        raise AssertionError("canonical manifest copy is not an object")
    digest = bytes_sha256(contexts_payload)
    try:
        artifact_path = contexts_path.resolve().relative_to(
            manifest_path.parent.resolve()
        ).as_posix()
    except ValueError:
        artifact_path = contexts_path.name

    artifacts = dict(
        _manifest_mapping(sealed.get("artifacts", {}), "manifest.artifacts")
    )
    artifacts["contexts"] = {
        "path": artifact_path,
        "records": record_count,
        "sha256": digest,
    }
    hashes = dict(_manifest_mapping(sealed.get("hashes", {}), "manifest.hashes"))
    hashes["contexts"] = digest
    sealed["artifacts"] = artifacts
    sealed["hashes"] = hashes
    sealed["contexts_sha256"] = digest
    counts_raw = sealed.get("counts")
    if counts_raw is not None:
        counts = dict(_manifest_mapping(counts_raw, "manifest.counts"))
        counts["contexts"] = record_count
        sealed["counts"] = counts
    return sealed


def _parse_csv_ints(raw: str, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{label} must contain positive integers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{label} must not contain duplicates")
    return values


def _parse_treatments(raw: str) -> tuple[Treatment, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    unsupported = [value for value in values if value not in TREATMENTS]
    if not values or unsupported:
        raise argparse.ArgumentTypeError(
            "treatments must be a comma-separated subset of " + ",".join(TREATMENTS)
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("treatments must not contain duplicates")
    return values  # type: ignore[return-value]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the oracle-blind context generation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="public_cases.jsonl")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="generator manifest to validate and seal with the contexts digest",
    )
    parser.add_argument(
        "--treatments",
        type=_parse_treatments,
        default=TREATMENTS,
        help="comma-separated treatments (default: tail,state_latest,state_first)",
    )
    parser.add_argument(
        "--checkpoints",
        type=lambda value: _parse_csv_ints(value, "checkpoints"),
        default=DEFAULT_CHECKPOINTS,
        help="comma-separated checkpoint turns (default: 8,16,32,64)",
    )
    parser.add_argument(
        "--budgets",
        type=lambda value: _parse_csv_ints(value, "budgets"),
        default=DEFAULT_BUDGETS,
        help="comma-separated token budgets (default: 256,512,1024)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("contexts.jsonl"), help="output JSONL path"
    )
    return parser.parse_args(argv)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit status for easy smoke testing."""

    args = parse_args(argv)
    try:
        resolved_output = args.output.resolve()
        if resolved_output in {args.cases.resolve(), args.manifest.resolve()}:
            raise ValueError("--output must differ from --cases and --manifest")
        cases = load_public_cases(args.cases)
        if not cases:
            raise ValueError("public case file is empty")
        manifest = load_and_validate_manifest(
            args.manifest,
            args.cases,
            cases,
            treatments=args.treatments,
            checkpoints=args.checkpoints,
            budgets=args.budgets,
        )
        records = list(
            generate_context_records(
                cases,
                treatments=args.treatments,
                checkpoints=args.checkpoints,
                budgets=args.budgets,
            )
        )
        contexts_payload = jsonl_bytes(records)
        _atomic_write(args.output, contexts_payload)
        sealed_manifest = seal_contexts_manifest(
            manifest,
            manifest_path=args.manifest,
            contexts_path=args.output,
            contexts_payload=contexts_payload,
            record_count=len(records),
        )
        _atomic_write(
            args.manifest,
            canonical_json(sealed_manifest).encode("utf-8") + b"\n",
        )
    except (OSError, ValueError) as exc:
        print(f"rcwt_session_run: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {len(records)} contexts to {args.output} "
        f"(encoding={ENCODING_NAME})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
