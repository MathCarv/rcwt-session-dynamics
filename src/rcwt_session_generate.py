"""Generate the deterministic public corpus and separate RCWT-S oracle.

The default corpus contains 64 cases (16 per family), 64 turns per case, and
checkpoints at turns 8, 16, 32, and 64.  Critical evidence alternates between
early and late positions while distractors reuse the same actors, fields, and
value vocabulary.  No provider or model API is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rcwt_session_types import (
    DEFAULT_BUDGETS,
    FAMILIES,
    SCHEMA_VERSION,
    TREATMENTS,
    Checkpoint,
    Event,
    Family,
    Oracle,
    PublicCase,
    bytes_sha256,
    canonical_json,
    canonical_sha256,
    jsonl_bytes,
)

ENCODING_NAME = "cl100k_base"
DEFAULT_SEED = 20260911
DEFAULT_CASES = 64
DEFAULT_TURNS = 64
PREREGISTRATION_RELATIVE_PATH = Path("docs/rcwt_session_preregistration.md")


@dataclass(frozen=True, slots=True)
class _EventSpec:
    actor: str
    kind: str
    entity: str
    field: str
    value: str
    dependency_turns: tuple[int, ...] = ()
    supersedes_previous: bool = True


@dataclass(frozen=True, slots=True)
class _GoldResolution:
    """Generator-private gold semantics, independent of treatment code."""

    required_event_ids: tuple[str, ...]
    stale_event_ids: tuple[str, ...]
    order_edges: tuple[tuple[str, str], ...]
    complete: bool


def _derived_seed(master_seed: int, family: Family, ordinal: int) -> int:
    payload = f"rcwt-s|{master_seed}|{family}|{ordinal}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def checkpoint_turns(turns: int) -> tuple[int, ...]:
    """Scale the preregistered 1/8, 1/4, 1/2, and final checkpoints.

    At the canonical 64 turns this is exactly ``(8, 16, 32, 64)``.  A minimum
    of 64 turns keeps the six-event early/late dependency layout disjoint.
    """

    if turns < 64:
        raise ValueError("turns must be >= 64")
    values = (turns // 8, turns // 4, turns // 2, turns)
    if len(set(values)) != 4:
        raise ValueError("turn count does not yield four distinct checkpoints")
    return values


def _put(plan: dict[int, _EventSpec], turn: int, spec: _EventSpec) -> None:
    if turn in plan:
        raise AssertionError(f"event plan collision at turn {turn}")
    plan[turn] = spec


def _update_plan(
    case_id: str,
    checkpoints: tuple[int, ...],
    ordinal: int,
    rng: random.Random,
) -> tuple[dict[int, _EventSpec], tuple[Checkpoint, ...]]:
    plan: dict[int, _EventSpec] = {}
    queries: list[Checkpoint] = []
    previous = 0
    values = ("manual_review", "approved", "held", "released")
    actors = ("risk-agent", "routing-agent", "policy-agent")
    for index, turn in enumerate(checkpoints):
        start = previous + 1
        alpha = f"{case_id}:account-alpha"
        beta = f"{case_id}:account-beta"
        _put(
            plan,
            start,
            _EventSpec(
                actor=rng.choice(actors),
                kind="state_update",
                entity=alpha,
                field="routing_status",
                value=values[(ordinal + index) % len(values)],
            ),
        )
        _put(
            plan,
            turn,
            _EventSpec(
                actor=rng.choice(actors),
                kind="state_update",
                entity=beta,
                field="routing_status",
                value=values[(ordinal + index + 2) % len(values)],
            ),
        )
        chosen = alpha if (ordinal + index) % 2 == 0 else beta
        queries.append(
            Checkpoint(
                checkpoint_id=f"cp-{turn:04d}",
                turn=turn,
                query=f"What is the current routing_status for {chosen}?",
                query_entities=(chosen,),
            )
        )
        previous = turn
    return plan, tuple(queries)


def _dependency_chain(
    plan: dict[int, _EventSpec],
    turns: tuple[int, int, int],
    entity: str,
    value_index: int,
    rng: random.Random,
) -> None:
    evidence_turn, assessment_turn, decision_turn = turns
    _put(
        plan,
        evidence_turn,
        _EventSpec(
            actor=rng.choice(("telemetry-agent", "ledger-agent")),
            kind="evidence",
            entity=f"{entity}:evidence",
            field="signal",
            value=("matched", "verified", "consistent")[value_index % 3],
        ),
    )
    _put(
        plan,
        assessment_turn,
        _EventSpec(
            actor="validation-agent",
            kind="validate",
            entity=f"{entity}:assessment",
            field="verdict",
            value=("clear", "review", "clear")[(value_index + 1) % 3],
            dependency_turns=(evidence_turn,),
        ),
    )
    _put(
        plan,
        decision_turn,
        _EventSpec(
            actor="settlement-agent",
            kind="commit",
            entity=entity,
            field="release_state",
            value=("ready", "paused", "ready")[(value_index + 2) % 3],
            dependency_turns=(assessment_turn,),
        ),
    )


def _dependency_plan(
    case_id: str,
    checkpoints: tuple[int, ...],
    ordinal: int,
    rng: random.Random,
) -> tuple[dict[int, _EventSpec], tuple[Checkpoint, ...]]:
    plan: dict[int, _EventSpec] = {}
    queries: list[Checkpoint] = []
    previous = 0
    for index, turn in enumerate(checkpoints):
        start = previous + 1
        alpha = f"{case_id}:release-alpha"
        beta = f"{case_id}:release-beta"
        _dependency_chain(
            plan,
            (start, start + 1, start + 2),
            alpha,
            ordinal + index,
            rng,
        )
        _dependency_chain(
            plan,
            (turn - 2, turn - 1, turn),
            beta,
            ordinal + index + 1,
            rng,
        )
        chosen = alpha if (ordinal + index) % 2 == 0 else beta
        queries.append(
            Checkpoint(
                checkpoint_id=f"cp-{turn:04d}",
                turn=turn,
                query=(
                    f"Reconstruct the current release_state for {chosen} "
                    "from its dependency chain."
                ),
                query_entities=(chosen,),
            )
        )
        previous = turn
    return plan, tuple(queries)


def _chronology_plan(
    case_id: str,
    checkpoints: tuple[int, ...],
    ordinal: int,
    rng: random.Random,
) -> tuple[dict[int, _EventSpec], tuple[Checkpoint, ...]]:
    plan: dict[int, _EventSpec] = {}
    queries: list[Checkpoint] = []
    previous = 0
    previous_alpha_turn: int | None = None
    previous_beta_turn: int | None = None
    for index, turn in enumerate(checkpoints):
        start = previous + 1
        alpha = f"{case_id}:authorization-alpha"
        beta = f"{case_id}:authorization-beta"
        alpha_first_active = (ordinal + index) % 2 == 0
        alpha_current_active = not alpha_first_active
        beta_first_active = (ordinal + index + 1) % 2 == 0
        beta_current_active = not beta_first_active
        _put(
            plan,
            start,
            _EventSpec(
                actor=rng.choice(("authorization-agent", "policy-agent")),
                kind="grant" if alpha_first_active else "revoke",
                entity=alpha,
                field="authorization_state",
                value="authorized" if alpha_first_active else "revoked",
                dependency_turns=(
                    (previous_alpha_turn,) if previous_alpha_turn is not None else ()
                ),
            ),
        )
        _put(
            plan,
            start + 1,
            _EventSpec(
                actor=rng.choice(("authorization-agent", "policy-agent")),
                kind="grant" if alpha_current_active else "revoke",
                entity=alpha,
                field="authorization_state",
                value="authorized" if alpha_current_active else "revoked",
                dependency_turns=(start,),
            ),
        )
        _put(
            plan,
            turn - 1,
            _EventSpec(
                actor=rng.choice(("authorization-agent", "policy-agent")),
                kind="grant" if beta_first_active else "revoke",
                entity=beta,
                field="authorization_state",
                value="authorized" if beta_first_active else "revoked",
                dependency_turns=(
                    (previous_beta_turn,) if previous_beta_turn is not None else ()
                ),
            ),
        )
        _put(
            plan,
            turn,
            _EventSpec(
                actor=rng.choice(("authorization-agent", "policy-agent")),
                kind="grant" if beta_current_active else "revoke",
                entity=beta,
                field="authorization_state",
                value="authorized" if beta_current_active else "revoked",
                dependency_turns=(turn - 1,),
            ),
        )
        chosen = alpha if (ordinal + index) % 2 == 0 else beta
        queries.append(
            Checkpoint(
                checkpoint_id=f"cp-{turn:04d}",
                turn=turn,
                query=(
                    f"What is the current authorization_state for {chosen} "
                    "after all grants and revocations?"
                ),
                query_entities=(chosen,),
            )
        )
        previous_alpha_turn = start + 1
        previous_beta_turn = turn
        previous = turn
    return plan, tuple(queries)


def _insufficient_plan(
    case_id: str,
    checkpoints: tuple[int, ...],
    ordinal: int,
    rng: random.Random,
) -> tuple[dict[int, _EventSpec], tuple[Checkpoint, ...]]:
    plan: dict[int, _EventSpec] = {}
    queries: list[Checkpoint] = []
    previous = 0
    for index, turn in enumerate(checkpoints):
        start = previous + 1
        missing = f"{case_id}:approval-{index + 1}:canonical"
        near_match = f"{case_id}:approval-{index + 1}:candidate"
        evidence_turn = start if (ordinal + index) % 2 == 0 else turn
        _put(
            plan,
            evidence_turn,
            _EventSpec(
                actor=rng.choice(("approval-agent", "review-agent")),
                kind="state_update",
                entity=near_match,
                field="approval_state",
                value=rng.choice(("approved", "held", "review")),
            ),
        )
        queries.append(
            Checkpoint(
                checkpoint_id=f"cp-{turn:04d}",
                turn=turn,
                query=(
                    f"Determine approval_state for {missing}; report MISSING "
                    "when no evidence exists."
                ),
                query_entities=(missing,),
            )
        )
        previous = turn
    return plan, tuple(queries)


def _distractor_spec(
    case_id: str,
    family: Family,
    turn: int,
    rng: random.Random,
) -> _EventSpec:
    """Create same-vocabulary distractors without touching queried entities."""

    index = turn % 7
    if family == "update-heavy":
        return _EventSpec(
            actor=rng.choice(("risk-agent", "routing-agent", "policy-agent")),
            kind="state_update",
            entity=f"{case_id}:account-distractor-{index}",
            field="routing_status",
            value=rng.choice(("manual_review", "approved", "held", "released")),
        )
    if family == "dependency-heavy":
        return _EventSpec(
            actor=rng.choice(("telemetry-agent", "validation-agent", "settlement-agent")),
            kind=rng.choice(("evidence", "validate", "commit")),
            entity=f"{case_id}:release-distractor-{index}",
            field=rng.choice(("signal", "verdict", "release_state")),
            value=rng.choice(("matched", "clear", "review", "ready", "paused")),
        )
    if family == "chronology-sensitive":
        active = rng.choice((True, False))
        return _EventSpec(
            actor=rng.choice(("authorization-agent", "policy-agent")),
            kind="grant" if active else "revoke",
            entity=f"{case_id}:authorization-distractor-{index}",
            field="authorization_state",
            value="authorized" if active else "revoked",
        )
    return _EventSpec(
        actor=rng.choice(("approval-agent", "review-agent")),
        kind="state_update",
        entity=f"{case_id}:approval-distractor-{index}",
        field="approval_state",
        value=rng.choice(("approved", "held", "review")),
    )


def _render_event(
    *,
    event_id: str,
    turn: int,
    spec: _EventSpec,
    supersedes: str | None,
    depends_on: tuple[str, ...],
) -> str:
    dependency_text = ",".join(depends_on) if depends_on else "-"
    return (
        f"EVENT {event_id} | turn={turn:04d} | actor={spec.actor} | "
        f"kind={spec.kind} | entity={spec.entity} | field={spec.field} | "
        f"value={json.dumps(spec.value, ensure_ascii=False)} | "
        f"supersedes={supersedes or '-'} | depends_on={dependency_text}"
    )


def _materialize_events(
    case_id: str,
    family: Family,
    turns: int,
    plan: dict[int, _EventSpec],
    rng: random.Random,
) -> tuple[Event, ...]:
    previous_by_key: dict[tuple[str, str], str] = {}
    events: list[Event] = []
    for turn in range(1, turns + 1):
        spec = plan.get(turn) or _distractor_spec(case_id, family, turn, rng)
        event_id = f"{case_id}-event-{turn:04d}"
        key = (spec.entity, spec.field)
        supersedes = previous_by_key.get(key) if spec.supersedes_previous else None
        depends_on = tuple(
            f"{case_id}-event-{dependency_turn:04d}"
            for dependency_turn in spec.dependency_turns
        )
        event = Event(
            event_id=event_id,
            turn=turn,
            actor=spec.actor,
            kind=spec.kind,
            entity=spec.entity,
            field=spec.field,
            value=spec.value,
            supersedes=supersedes,
            depends_on=depends_on,
            rendered=_render_event(
                event_id=event_id,
                turn=turn,
                spec=spec,
                supersedes=supersedes,
                depends_on=depends_on,
            ),
        )
        events.append(event)
        previous_by_key[key] = event_id
    return tuple(events)


def generate_case(
    *,
    master_seed: int,
    family: Family,
    ordinal: int,
    turns: int = DEFAULT_TURNS,
) -> PublicCase:
    """Generate one deterministic public case for a family-local ordinal."""

    if family not in FAMILIES:
        raise ValueError(f"unsupported family: {family!r}")
    if ordinal < 0:
        raise ValueError("ordinal must be non-negative")
    checkpoints = checkpoint_turns(turns)
    case_seed = _derived_seed(master_seed, family, ordinal)
    rng = random.Random(case_seed)
    family_code = {
        "update-heavy": "update",
        "dependency-heavy": "dependency",
        "chronology-sensitive": "chronology",
        "insufficient-evidence": "insufficient",
    }[family]
    case_id = f"rcwts-{family_code}-{ordinal + 1:03d}"
    if family == "update-heavy":
        plan, queries = _update_plan(case_id, checkpoints, ordinal, rng)
    elif family == "dependency-heavy":
        plan, queries = _dependency_plan(case_id, checkpoints, ordinal, rng)
    elif family == "chronology-sensitive":
        plan, queries = _chronology_plan(case_id, checkpoints, ordinal, rng)
    else:
        plan, queries = _insufficient_plan(case_id, checkpoints, ordinal, rng)
    events = _materialize_events(case_id, family, turns, plan, rng)
    return PublicCase(
        case_id=case_id,
        family=family,
        seed=case_seed,
        events=events,
        checkpoints=queries,
    )


def _derive_gold(case: PublicCase, checkpoint: Checkpoint) -> _GoldResolution:
    """Derive oracle truth directly from event semantics, never from a treatment.

    This implementation intentionally does not share the runner's resolver or
    ordering helper.  Gold selection follows explicit supersession heads and a
    dependency-first DFS, making an implementation error in a treatment unable
    to silently define its own answer key.
    """

    eligible = tuple(event for event in case.events if event.turn <= checkpoint.turn)
    by_id = {event.event_id: event for event in eligible}
    targets: list[Event] = []
    stale: set[str] = set()
    complete = True
    for entity in checkpoint.query_entities:
        entity_events = [event for event in eligible if event.entity == entity]
        if not entity_events:
            complete = False
            continue
        keys: dict[tuple[str, str], list[Event]] = {}
        for event in entity_events:
            keys.setdefault((event.entity, event.field), []).append(event)
        for versions in keys.values():
            versions.sort(key=lambda event: (event.turn, event.event_id))
            superseded_ids = {
                event.supersedes for event in versions if event.supersedes is not None
            }
            heads = [event for event in versions if event.event_id not in superseded_ids]
            current = max(
                heads or versions,
                key=lambda event: (event.turn, event.event_id),
            )
            targets.append(current)
            stale.update(
                event.event_id for event in versions if event.event_id != current.event_id
            )

    ordered: list[Event] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(event: Event) -> None:
        nonlocal complete
        if event.event_id in visited:
            return
        if event.event_id in visiting:
            raise ValueError(f"gold dependency cycle reaches {event.event_id}")
        visiting.add(event.event_id)
        for dependency_id in event.depends_on:
            dependency = by_id.get(dependency_id)
            if dependency is None:
                complete = False
                continue
            visit(dependency)
        visiting.remove(event.event_id)
        visited.add(event.event_id)
        ordered.append(event)

    for target in sorted(targets, key=lambda event: (event.turn, event.event_id)):
        visit(target)
    required = {event.event_id for event in ordered}
    # Historical transitions can be superseded as state values while remaining
    # indispensable causal evidence.  Such events are required, not conflicts.
    stale.difference_update(required)
    edges = sorted(
        {
            (dependency_id, event.event_id)
            for event in ordered
            for dependency_id in event.depends_on
            if dependency_id in required
        }
    )
    return _GoldResolution(
        required_event_ids=tuple(event.event_id for event in ordered),
        stale_event_ids=tuple(sorted(stale)),
        order_edges=tuple(edges),
        complete=complete and bool(targets),
    )


def _oracles_for_case(case: PublicCase) -> tuple[Oracle, ...]:
    rows: list[Oracle] = []
    for checkpoint in case.checkpoints:
        resolution = _derive_gold(case, checkpoint)
        rows.append(
            Oracle(
                case_id=case.case_id,
                checkpoint_id=checkpoint.checkpoint_id,
                required_current_event_ids=resolution.required_event_ids,
                stale_conflict_event_ids=resolution.stale_event_ids,
                required_order_edges=resolution.order_edges,
                expected_disposition="ready" if resolution.complete else "insufficient",
            )
        )
    return tuple(rows)


def generate_corpus(
    *,
    seed: int = DEFAULT_SEED,
    cases: int = DEFAULT_CASES,
    turns: int = DEFAULT_TURNS,
) -> tuple[tuple[PublicCase, ...], tuple[Oracle, ...]]:
    """Generate a balanced deterministic corpus and its separate oracle rows."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if cases <= 0:
        raise ValueError("cases must be positive")
    checkpoint_turns(turns)
    family_ordinals: Counter[Family] = Counter()
    public_cases: list[PublicCase] = []
    for index in range(cases):
        family = FAMILIES[index % len(FAMILIES)]
        ordinal = family_ordinals[family]
        family_ordinals[family] += 1
        public_cases.append(
            generate_case(
                master_seed=seed,
                family=family,
                ordinal=ordinal,
                turns=turns,
            )
        )
    oracles = tuple(
        oracle for case in public_cases for oracle in _oracles_for_case(case)
    )
    return tuple(public_cases), oracles


def _source_commit(repository_root: Path, source_files: dict[str, str]) -> str:
    """Return the latest commit that changed a fingerprinted protocol input.

    Using ``HEAD`` here would make a committed result impossible to reproduce:
    the result commit itself would change the next manifest despite leaving the
    protocol untouched.  Path-scoped history identifies the actual source
    revision and remains stable across evidence-only commits.
    """

    tracked_paths = [*source_files, PREREGISTRATION_RELATIVE_PATH.as_posix()]
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "log",
                "-1",
                "--format=%H",
                "--",
                *tracked_paths,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip().lower()
    return commit if len(commit) == 40 else "unknown"


def _source_fingerprints(repository_root: Path) -> tuple[dict[str, str], str]:
    """Fingerprint the actual working sources, including uncommitted content."""

    relative_paths = (
        Path("src/rcwt_session_types.py"),
        Path("src/rcwt_session_generate.py"),
        Path("src/rcwt_session_run.py"),
        Path("src/rcwt_session_score.py"),
    )
    files: dict[str, str] = {}
    for relative_path in relative_paths:
        absolute_path = repository_root / relative_path
        if absolute_path.is_file():
            files[relative_path.as_posix()] = bytes_sha256(absolute_path.read_bytes())
    return files, canonical_sha256(files)


def _source_state(repository_root: Path, source_files: dict[str, str]) -> str:
    """Describe whether the fingerprinted protocol sources match ``HEAD``."""

    tracked_paths = [*source_files, PREREGISTRATION_RELATIVE_PATH.as_posix()]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain", "--", *tracked_paths],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return "working-tree" if result.stdout.strip() else "clean"


def build_manifest(
    *,
    seed: int,
    cases: Sequence[PublicCase],
    oracles: Sequence[Oracle],
    turns: int,
    public_payload: bytes,
    oracle_payload: bytes,
    repository_root: Path,
) -> dict[str, object]:
    """Build a self-describing manifest without attempting a self-hash."""

    checkpoints = checkpoint_turns(turns)
    family_counts = Counter(case.family for case in cases)
    config: dict[str, object] = {
        "seed": seed,
        "cases": len(cases),
        "turns": turns,
        "families": list(FAMILIES),
        "family_counts": {family: family_counts.get(family, 0) for family in FAMILIES},
        "checkpoints": list(checkpoints),
        "treatments": list(TREATMENTS),
        "budgets": list(DEFAULT_BUDGETS),
        "encoding": ENCODING_NAME,
    }
    public_digest = bytes_sha256(public_payload)
    oracle_digest = bytes_sha256(oracle_payload)
    config_digest = canonical_sha256(config)
    preregistration_path = repository_root / PREREGISTRATION_RELATIVE_PATH
    preregistration_digest = (
        bytes_sha256(preregistration_path.read_bytes())
        if preregistration_path.is_file()
        else None
    )
    source_files, source_tree_digest = _source_fingerprints(repository_root)
    source_commit = _source_commit(repository_root, source_files)
    source_state = _source_state(repository_root, source_files)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "session-v1",
        "protocol_version": "session-v1",
        "oracle_derivation": "independent-gold-v1",
        "source_commit": source_commit,
        "source_state": source_state,
        "source_files_sha256": source_files,
        "source_tree_sha256": source_tree_digest,
        "source": {
            "commit": source_commit,
            "state": source_state,
            "files": source_files,
            "tree_sha256": source_tree_digest,
        },
        "preregistration_path": PREREGISTRATION_RELATIVE_PATH.as_posix(),
        "preregistration_sha256": preregistration_digest,
        "config": config,
        "config_sha256": config_digest,
        "public_cases_sha256": public_digest,
        "oracle_cases_sha256": oracle_digest,
        "hashes": {
            "config": config_digest,
            "public_cases": public_digest,
            "oracle_cases": oracle_digest,
            "preregistration": preregistration_digest,
        },
        "artifacts": {
            "public_cases": {
                "path": "public_cases.jsonl",
                "records": len(cases),
                "sha256": public_digest,
            },
            "oracle_cases": {
                "path": "oracle_cases.jsonl",
                "records": len(oracles),
                "sha256": oracle_digest,
            },
        },
        "case_sha256": {case.case_id: case.sha256 for case in cases},
        "counts": {
            "public_cases": len(cases),
            "oracle_records": len(oracles),
            "events": sum(len(case.events) for case in cases),
            "checkpoints": sum(len(case.checkpoints) for case in cases),
        },
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_corpus(
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    cases: int = DEFAULT_CASES,
    turns: int = DEFAULT_TURNS,
) -> dict[str, object]:
    """Generate and atomically publish all three deterministic artifacts."""

    public_cases, oracles = generate_corpus(seed=seed, cases=cases, turns=turns)
    public_payload = jsonl_bytes(public_cases)
    oracle_payload = jsonl_bytes(oracles)
    repository_root = Path(__file__).resolve().parents[1]
    manifest = build_manifest(
        seed=seed,
        cases=public_cases,
        oracles=oracles,
        turns=turns,
        public_payload=public_payload,
        oracle_payload=oracle_payload,
        repository_root=repository_root,
    )
    manifest_payload = canonical_json(manifest).encode("utf-8") + b"\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_dir / "public_cases.jsonl", public_payload)
    _atomic_write(output_dir / "oracle_cases.jsonl", oracle_payload)
    _atomic_write(output_dir / "manifest.json", manifest_payload)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deterministic corpus-generation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; no network credentials or model providers are consulted."""

    args = parse_args(argv)
    try:
        manifest = write_corpus(
            args.output_dir,
            seed=args.seed,
            cases=args.cases,
            turns=args.turns,
        )
    except (OSError, ValueError) as exc:
        print(f"rcwt_session_generate: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {manifest['counts']['public_cases']} cases and "  # type: ignore[index]
        f"{manifest['counts']['oracle_records']} oracle rows to {args.output_dir}"  # type: ignore[index]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
