"""Protocol-level regression tests for the deterministic RCWT-S experiment."""

from __future__ import annotations

import ast
import io
import json
import os
import socket
import tempfile
import unittest
import urllib.request
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

import rcwt_session_generate as generator
import rcwt_session_run as runner
from rcwt_session_generate import DEFAULT_CASES, DEFAULT_SEED, DEFAULT_TURNS
from rcwt_session_score import (
    ScoreValidationError,
    context_sha256,
    count_construction_tokens,
    score_context,
    score_files,
    score_records,
)
from rcwt_session_types import (
    DEFAULT_BUDGETS,
    DEFAULT_CHECKPOINTS,
    FAMILIES,
    TREATMENTS,
    Checkpoint,
    Event,
    PublicCase,
    bytes_sha256,
    canonical_json,
    canonical_sha256,
    jsonl_bytes,
    load_context_records,
    load_public_cases,
)


ROOT = Path(__file__).resolve().parents[1]
ORACLE_ONLY_FIELDS = {
    "required_current_event_ids",
    "stale_conflict_event_ids",
    "required_order_edges",
    "expected_disposition",
    "stale_current_pairs",
    "oracle",
}


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for child in value.values():
            keys.update(_nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_nested_keys(child))
    return keys


def _recount_and_rehash(context: dict[str, Any]) -> dict[str, Any]:
    """Seal an intentionally changed context so semantic checks must catch it."""

    context["token_count"] = runner.count_tokens(str(context["rendered_context"]))
    context["context_sha256"] = context_sha256(context)
    return context


def _render_event(event: Event, *, turn: int | None = None) -> str:
    rendered_turn = event.turn if turn is None else turn
    dependencies = ",".join(event.depends_on) if event.depends_on else "-"
    return (
        f"EVENT {event.event_id} | turn={rendered_turn:04d} | actor={event.actor} | "
        f"kind={event.kind} | entity={event.entity} | field={event.field} | "
        f"value={json.dumps(event.value, ensure_ascii=False)} | "
        f"supersedes={event.supersedes or '-'} | depends_on={dependencies}"
    )


def _event(
    event_id: str,
    turn: int,
    entity: str,
    field: str,
    value: str,
    *,
    depends_on: Iterable[str] = (),
) -> Event:
    event = Event(
        event_id=event_id,
        turn=turn,
        actor="fixture-agent",
        kind="state_update",
        entity=entity,
        field=field,
        value=value,
        supersedes=None,
        depends_on=tuple(depends_on),
        rendered="pending",
    )
    return replace(event, rendered=_render_event(event))


def _append_only_case() -> PublicCase:
    events = (
        _event("fact-evidence", 1, "fixture:evidence", "signal", "verified"),
        _event(
            "fact-status",
            2,
            "fixture:account",
            "status",
            "approved",
            depends_on=("fact-evidence",),
        ),
        _event("fact-region", 3, "fixture:account", "region", "south"),
        *(
            _event(
                f"distractor-{turn}",
                turn,
                f"fixture:distractor:{turn}",
                "status",
                f"value-{turn}",
            )
            for turn in range(4, 9)
        ),
    )
    return PublicCase(
        case_id="append-only-fixture",
        family="dependency-heavy",
        seed=7,
        events=events,
        checkpoints=(
            Checkpoint(
                checkpoint_id="cp-0008",
                turn=8,
                query="Resolve the current status and region for fixture:account.",
                query_entities=("fixture:account",),
            ),
        ),
    )


def _permute_distractors(case: PublicCase) -> PublicCase:
    """Reverse distractor positions while preserving their facts and identities."""

    distractors = [event for event in case.events if event.event_id.startswith("distractor-")]
    reversed_turns = list(reversed([event.turn for event in distractors]))
    turn_by_id = {
        event.event_id: turn for event, turn in zip(distractors, reversed_turns)
    }
    permuted: list[Event] = []
    for event in case.events:
        turn = turn_by_id.get(event.event_id, event.turn)
        permuted.append(replace(event, turn=turn, rendered=_render_event(event, turn=turn)))
    return replace(case, events=tuple(sorted(permuted, key=lambda event: event.turn)))


def _independent_oracle(case: PublicCase, checkpoint: Checkpoint) -> dict[str, Any]:
    """Test-side gold derivation that never calls the treatment resolver."""

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
        by_key: dict[tuple[str, str], list[Event]] = {}
        for event in entity_events:
            by_key.setdefault((event.entity, event.field), []).append(event)
        for versions in by_key.values():
            superseded = {
                event.supersedes for event in versions if event.supersedes is not None
            }
            heads = [event for event in versions if event.event_id not in superseded]
            chosen = max(heads or versions, key=lambda event: (event.turn, event.event_id))
            targets.append(chosen)
            stale.update(
                event.event_id for event in versions if event.event_id != chosen.event_id
            )

    required: set[str] = set()
    stack = list(targets)
    while stack:
        event = stack.pop()
        if event.event_id in required:
            continue
        required.add(event.event_id)
        for dependency_id in event.depends_on:
            dependency = by_id.get(dependency_id)
            if dependency is None:
                complete = False
            else:
                stack.append(dependency)
    # A superseded transition can still be mandatory causal evidence; it is
    # therefore required rather than a stale conflict.
    stale.difference_update(required)
    edges = {
        (dependency_id, event_id)
        for event_id in required
        for dependency_id in by_id[event_id].depends_on
        if dependency_id in required
    }
    return {
        "required": required,
        "stale": stale,
        "edges": edges,
        "disposition": "ready" if complete and bool(targets) else "insufficient",
    }


class RcwtSessionProtocolTests(unittest.TestCase):
    """Lock the preregistered corpus, builder, and scorer contracts together."""

    @classmethod
    def setUpClass(cls) -> None:
        # Warm the pinned tokenizer before tests deliberately disable networking.
        runner.count_tokens("RCWT-S tokenizer warmup")
        cls.cases, cls.oracles = generator.generate_corpus(
            seed=DEFAULT_SEED,
            cases=len(FAMILIES),
            turns=DEFAULT_TURNS,
        )
        cls.contexts = tuple(runner.generate_context_records(cls.cases))
        cls.case_by_id = {case.case_id: case for case in cls.cases}
        cls.oracle_by_key = {
            (oracle.case_id, oracle.checkpoint_id): oracle for oracle in cls.oracles
        }

    @classmethod
    def _manifest(cls, *, budgets: Iterable[int] = DEFAULT_BUDGETS) -> dict[str, Any]:
        return {
            "protocol_version": "session-v1",
            "config": {
                "treatments": list(TREATMENTS),
                "budgets": list(budgets),
                "checkpoints": list(DEFAULT_CHECKPOINTS),
            },
            "case_sha256": {case.case_id: case.sha256 for case in cls.cases},
        }

    def test_same_seed_is_byte_for_byte_deterministic(self) -> None:
        cases_again, oracles_again = generator.generate_corpus(
            seed=DEFAULT_SEED,
            cases=len(FAMILIES),
            turns=DEFAULT_TURNS,
        )
        contexts_again = tuple(runner.generate_context_records(cases_again))

        self.assertEqual(jsonl_bytes(self.cases), jsonl_bytes(cases_again))
        self.assertEqual(jsonl_bytes(self.oracles), jsonl_bytes(oracles_again))
        self.assertEqual(jsonl_bytes(self.contexts), jsonl_bytes(contexts_again))

    def test_public_records_and_builder_are_oracle_blind(self) -> None:
        public_keys = _nested_keys([case.to_dict() for case in self.cases])
        context_keys = _nested_keys([record.to_dict() for record in self.contexts])
        self.assertTrue(ORACLE_ONLY_FIELDS.isdisjoint(public_keys))
        self.assertTrue(ORACLE_ONLY_FIELDS.isdisjoint(context_keys))

        tree = ast.parse((ROOT / "src" / "rcwt_session_run.py").read_text(encoding="utf-8"))
        forbidden_imports: list[str] = []
        forbidden_symbols: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                forbidden_imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name in {"rcwt_session_generate", "rcwt_session_score"}
                )
            elif isinstance(node, ast.ImportFrom):
                if node.module in {"rcwt_session_generate", "rcwt_session_score"}:
                    forbidden_imports.append(str(node.module))
                forbidden_symbols.extend(
                    alias.name
                    for alias in node.names
                    if alias.name in {"Oracle", "OracleCase", "load_oracles"}
                )
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "load_oracles":
                    forbidden_symbols.append(node.func.id)
        self.assertEqual(forbidden_imports, [])
        self.assertEqual(forbidden_symbols, [])

    def test_locked_default_corpus_and_full_cross_product(self) -> None:
        cases, oracles = generator.generate_corpus()
        cases_again, oracles_again = generator.generate_corpus()
        self.assertEqual(len(cases), DEFAULT_CASES)
        self.assertEqual(len(oracles), DEFAULT_CASES * len(DEFAULT_CHECKPOINTS))
        self.assertEqual(Counter(case.family for case in cases), Counter({f: 16 for f in FAMILIES}))
        self.assertTrue(all(len(case.events) == DEFAULT_TURNS for case in cases))
        self.assertTrue(
            all(tuple(cp.turn for cp in case.checkpoints) == DEFAULT_CHECKPOINTS for case in cases)
        )
        self.assertEqual(jsonl_bytes(cases), jsonl_bytes(cases_again))
        self.assertEqual(jsonl_bytes(oracles), jsonl_bytes(oracles_again))

        def cheap_cell(
            case: PublicCase,
            checkpoint: Checkpoint,
            treatment: str,
            budget: int,
        ) -> tuple[str, str, str, int]:
            return case.case_id, checkpoint.checkpoint_id, treatment, budget

        with patch.object(runner, "build_context_record", side_effect=cheap_cell):
            observed = tuple(runner.generate_context_records(cases))
        expected = {
            (case.case_id, checkpoint.checkpoint_id, treatment, budget)
            for case in cases
            for checkpoint in case.checkpoints
            for treatment in TREATMENTS
            for budget in DEFAULT_BUDGETS
        }
        self.assertEqual(len(observed), DEFAULT_CASES * 4 * 3 * 3)
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(set(observed), expected)

    def test_real_matrix_is_complete_and_scorer_rejects_a_missing_cell(self) -> None:
        expected = {
            (case.case_id, checkpoint.checkpoint_id, treatment, budget)
            for case in self.cases
            for checkpoint in case.checkpoints
            for treatment in TREATMENTS
            for budget in DEFAULT_BUDGETS
        }
        observed = {
            (record.case_id, record.checkpoint_id, record.treatment, record.budget_tokens)
            for record in self.contexts
        }
        self.assertEqual(observed, expected)
        aggregate = score_records(
            self.contexts,
            self.oracles,
            self._manifest(),
            public_cases=self.cases,
            n_resamples=16,
        )
        self.assertEqual(aggregate["validity"]["status"], "PASS")
        self.assertEqual(aggregate["counts"]["records"], len(expected))
        self.assertTrue(aggregate["validity"]["treatment_matrix_complete"])
        rate = aggregate["overall"]["decision_ready"]
        self.assertEqual(rate["n_clusters"], len(self.cases))
        self.assertEqual(len(rate["cluster_bootstrap95"]), 2)
        self.assertEqual(len(rate["naive_wilson95"]), 2)
        self.assertEqual(
            aggregate["overall"]["decision_ready_ci95"],
            rate["cluster_bootstrap95"],
        )
        self.assertIn("cluster means", aggregate["analysis"]["primary_interval_method"])
        self.assertIn("descriptive only", aggregate["analysis"]["naive_binary_interval_method"])

        with self.assertRaisesRegex(ScoreValidationError, "treatment matrix mismatch"):
            score_records(
                self.contexts[:-1],
                self.oracles,
                self._manifest(),
                public_cases=self.cases,
                n_resamples=2,
            )

    def test_budget_token_recount_source_binding_and_self_hash(self) -> None:
        for record in self.contexts:
            with self.subTest(
                case=record.case_id,
                checkpoint=record.checkpoint_id,
                treatment=record.treatment,
                budget=record.budget_tokens,
            ):
                case = self.case_by_id[record.case_id]
                checkpoint = case.checkpoint(record.checkpoint_id)
                eligible_ids = {
                    event.event_id for event in case.events if event.turn <= checkpoint.turn
                }
                self.assertLessEqual(record.token_count, record.budget_tokens)
                self.assertEqual(record.token_count, runner.count_tokens(record.rendered_context))
                self.assertEqual(
                    record.token_count,
                    count_construction_tokens(record.rendered_context),
                )
                self.assertEqual(record.source_case_sha256, case.sha256)
                self.assertTrue(set(record.included_event_ids).issubset(eligible_ids))
                self.assertTrue(record.verify_hash())
                self.assertEqual(
                    record.context_sha256,
                    canonical_sha256(record.to_dict(include_hash=False)),
                )
                scored = score_context(
                    record,
                    self.oracle_by_key[(record.case_id, record.checkpoint_id)],
                    case,
                )
                self.assertEqual(scored.realized_tokens, record.token_count)

    def test_h3_append_only_latest_and_first_render_identical_bytes(self) -> None:
        case = _append_only_case()
        checkpoint = case.checkpoints[0]
        latest = runner.build_context_record(case, checkpoint, "state_latest", 1024)
        first = runner.build_context_record(case, checkpoint, "state_first", 1024)

        self.assertFalse(any(event.supersedes for event in case.events))
        self.assertEqual(latest.rendered_context.encode("utf-8"), first.rendered_context.encode("utf-8"))
        self.assertEqual(latest.included_event_ids, first.included_event_ids)
        self.assertEqual(latest.token_count, first.token_count)

    def test_h4_distractor_permutation_preserves_latest_critical_set(self) -> None:
        case = _append_only_case()
        permuted = _permute_distractors(case)
        checkpoint = case.checkpoints[0]
        original = runner.resolve_state(case, checkpoint, version="latest")
        shuffled = runner.resolve_state(permuted, checkpoint, version="latest")
        critical = {"fact-evidence", "fact-status", "fact-region"}

        self.assertEqual(set(original.required_event_ids), critical)
        self.assertEqual(set(shuffled.required_event_ids), critical)
        self.assertEqual(original.order_edges, shuffled.order_edges)
        for budget in DEFAULT_BUDGETS:
            left = runner.build_context_record(case, checkpoint, "state_latest", budget)
            right = runner.build_context_record(permuted, checkpoint, "state_latest", budget)
            self.assertEqual(
                set(left.included_event_ids) & critical,
                set(right.included_event_ids) & critical,
            )

    def test_h5_insufficient_cases_emit_missing_without_fabrication(self) -> None:
        insufficient_cases = {
            case.case_id: case
            for case in self.cases
            if case.family == "insufficient-evidence"
        }
        rows = [record for record in self.contexts if record.case_id in insufficient_cases]
        self.assertEqual(len(rows), 4 * len(TREATMENTS) * len(DEFAULT_BUDGETS))

        for record in rows:
            case = insufficient_cases[record.case_id]
            checkpoint = case.checkpoint(record.checkpoint_id)
            events = {event.event_id: event for event in case.events}
            source_lines = tuple(events[event_id].rendered for event_id in record.included_event_ids)
            rendered_lines = tuple(
                line for line in record.rendered_context.splitlines() if line.startswith("EVENT ")
            )
            self.assertIn("MISSING", record.rendered_context)
            self.assertNotIn("DISPOSITION ready", record.rendered_context)
            self.assertEqual(rendered_lines, source_lines)
            self.assertTrue(
                all(
                    events[event_id].entity not in checkpoint.query_entities
                    for event_id in record.included_event_ids
                )
            )
            scored = score_context(
                record,
                self.oracle_by_key[(record.case_id, record.checkpoint_id)],
                case,
            )
            self.assertTrue(scored.explicit_missing)
            self.assertFalse(scored.fabricated_sufficiency)

    def test_chronology_oracles_define_and_latest_preserves_causal_edges(self) -> None:
        full_cases, full_oracles = generator.generate_corpus()
        chronology_ids = {
            case.case_id for case in full_cases if case.family == "chronology-sensitive"
        }
        chronology_oracles = [
            oracle for oracle in full_oracles if oracle.case_id in chronology_ids
        ]
        self.assertEqual(len(chronology_oracles), 16 * len(DEFAULT_CHECKPOINTS))
        self.assertTrue(
            all(oracle.required_order_edges for oracle in chronology_oracles),
            "chronology-sensitive checkpoints must encode at least one causal edge",
        )

        cached_chronology_ids = {
            case.case_id for case in self.cases if case.family == "chronology-sensitive"
        }
        latest_rows = [
            record
            for record in self.contexts
            if record.case_id in cached_chronology_ids
            and record.treatment == "state_latest"
            and record.budget_tokens == max(DEFAULT_BUDGETS)
        ]
        self.assertEqual(len(latest_rows), len(DEFAULT_CHECKPOINTS))
        for record in latest_rows:
            oracle = self.oracle_by_key[(record.case_id, record.checkpoint_id)]
            positions = {
                event_id: index for index, event_id in enumerate(record.included_event_ids)
            }
            for before, after in oracle.required_order_edges:
                self.assertIn(before, positions)
                self.assertIn(after, positions)
                self.assertLess(positions[before], positions[after])
            scored = score_context(record, oracle, self.case_by_id[record.case_id])
            self.assertEqual(scored.causal_edges_preserved, scored.causal_edges_total)
            self.assertGreater(scored.causal_edges_total, 0)

    def test_outputs_ignore_provider_env_and_network_after_tokenizer_bootstrap(self) -> None:
        # tiktoken's first cl100k_base load may populate its own upstream cache.
        # setUpClass deliberately completes that bootstrap before this test proves
        # that RCWT-S itself is provider-free and network-independent thereafter.
        forbidden_roots = {"anthropic", "google", "httpx", "openai", "requests", "socket", "urllib"}
        imported_roots: set[str] = set()
        for source_name in ("rcwt_session_generate.py", "rcwt_session_run.py"):
            tree = ast.parse((ROOT / "src" / source_name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.split(".")[0])
        self.assertTrue(forbidden_roots.isdisjoint(imported_roots))

        def artifacts() -> tuple[bytes, bytes, bytes]:
            cases, oracles = generator.generate_corpus(
                seed=DEFAULT_SEED,
                cases=len(FAMILIES),
                turns=DEFAULT_TURNS,
            )
            contexts = tuple(
                runner.generate_context_records(
                    cases[:1],
                    checkpoints=(8,),
                    budgets=(256,),
                )
            )
            return jsonl_bytes(cases), jsonl_bytes(oracles), jsonl_bytes(contexts)

        baseline = artifacts()
        fake_credentials = {
            "OPENAI_API_KEY": "must-not-be-read",
            "ANTHROPIC_API_KEY": "must-not-be-read",
            "GOOGLE_API_KEY": "must-not-be-read",
        }
        network_error = AssertionError("RCWT-S attempted network access")
        with (
            patch.dict(os.environ, fake_credentials, clear=False),
            patch("socket.create_connection", side_effect=network_error),
            patch.object(socket.socket, "connect", side_effect=network_error),
            patch.object(urllib.request, "urlopen", side_effect=network_error),
        ):
            isolated = artifacts()
        self.assertEqual(isolated, baseline)

    def test_context_tampering_fails_closed_even_before_scoring(self) -> None:
        record = self.contexts[0]
        oracle = self.oracle_by_key[(record.case_id, record.checkpoint_id)]
        tampered = record.to_dict()
        tampered["rendered_context"] += "TAMPERED\n"

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contexts.jsonl"
            path.write_text(canonical_json(tampered) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "context_sha256 mismatch"):
                load_context_records(path)
        with self.assertRaisesRegex(
            ScoreValidationError,
            "context_sha256 mismatch|non-canonical disposition",
        ):
            score_context(tampered, oracle, self.case_by_id[record.case_id])

        # Recomputing the self-hash must not let inconsistent redundant fields pass.
        forged = record.to_dict()
        forged["included_event_ids"] = [*forged["included_event_ids"], "invented-event"]
        forged["context_sha256"] = context_sha256(forged)
        with self.assertRaisesRegex(ScoreValidationError, "EVENT order|ordered EVENT lines"):
            score_context(forged, oracle, self.case_by_id[record.case_id])

    def test_rehashed_query_and_event_tampering_still_fails_closed(self) -> None:
        record = next(
            item
            for item in self.contexts
            if item.checkpoint_id == "cp-0008"
            and item.treatment == "tail"
            and item.budget_tokens == max(DEFAULT_BUDGETS)
            and item.included_event_ids
        )
        case = self.case_by_id[record.case_id]
        checkpoint = case.checkpoint(record.checkpoint_id)
        oracle = self.oracle_by_key[(record.case_id, record.checkpoint_id)]

        query_tampered = record.to_dict()
        exact_query = f"QUERY {checkpoint.query}"
        query_tampered["rendered_context"] = str(
            query_tampered["rendered_context"]
        ).replace(exact_query, exact_query + " altered", 1)
        _recount_and_rehash(query_tampered)
        self.assertEqual(query_tampered["context_sha256"], context_sha256(query_tampered))
        self.assertLessEqual(query_tampered["token_count"], query_tampered["budget_tokens"])
        with self.assertRaisesRegex(ScoreValidationError, "exact public checkpoint query"):
            score_context(query_tampered, oracle, case)

        event_id = record.included_event_ids[0]
        source_event = next(event for event in case.events if event.event_id == event_id)
        event_tampered = record.to_dict()
        event_tampered["rendered_context"] = str(
            event_tampered["rendered_context"]
        ).replace(source_event.rendered, source_event.rendered + " | altered=true", 1)
        _recount_and_rehash(event_tampered)
        self.assertEqual(event_tampered["context_sha256"], context_sha256(event_tampered))
        self.assertLessEqual(event_tampered["token_count"], event_tampered["budget_tokens"])
        with self.assertRaisesRegex(ScoreValidationError, "payload differs from public event"):
            score_context(event_tampered, oracle, case)

    def test_missing_marker_cannot_hide_an_invented_current_fact(self) -> None:
        case = next(case for case in self.cases if case.family == "insufficient-evidence")
        checkpoint = case.checkpoints[0]
        record = next(
            item
            for item in self.contexts
            if item.case_id == case.case_id
            and item.checkpoint_id == checkpoint.checkpoint_id
            and item.treatment == "state_latest"
            and item.budget_tokens == max(DEFAULT_BUDGETS)
        )
        invented_id = f"{case.case_id}-invented-current"
        invented_event = (
            f"EVENT {invented_id} | turn={checkpoint.turn:04d} | actor=forger | "
            f"kind=state_update | entity={checkpoint.query_entities[0]} | "
            'field=approval_state | value="approved" | supersedes=- | depends_on=-'
        )
        forged = record.to_dict()
        forged["included_event_ids"] = [invented_id]
        forged["rendered_context"] = (
            f"CHECKPOINT {checkpoint.checkpoint_id} TURN {checkpoint.turn}\n"
            f"QUERY {checkpoint.query}\n"
            f"{invented_event}\n"
            "DISPOSITION MISSING (insufficient_evidence)\n"
        )
        _recount_and_rehash(forged)
        self.assertIn("MISSING", forged["rendered_context"])
        self.assertEqual(forged["context_sha256"], context_sha256(forged))
        with self.assertRaisesRegex(ScoreValidationError, "unknown or future event"):
            score_context(
                forged,
                self.oracle_by_key[(case.case_id, checkpoint.checkpoint_id)],
                case,
            )

    def test_runner_validates_manifest_and_seals_context_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            generator.write_corpus(
                directory,
                seed=DEFAULT_SEED,
                cases=len(FAMILIES),
                turns=DEFAULT_TURNS,
            )
            cases_path = directory / "public_cases.jsonl"
            oracle_path = directory / "oracle_cases.jsonl"
            manifest_path = directory / "manifest.json"
            contexts_path = directory / "contexts.jsonl"
            arguments = [
                "--cases",
                str(cases_path),
                "--manifest",
                str(manifest_path),
                "--output",
                str(contexts_path),
            ]

            with redirect_stdout(io.StringIO()):
                status = runner.main(arguments)
            self.assertEqual(status, 0)
            contexts_payload = contexts_path.read_bytes()
            digest = bytes_sha256(contexts_payload)
            sealed = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_records = len(FAMILIES) * 4 * 3 * 3
            self.assertEqual(sealed["contexts_sha256"], digest)
            self.assertEqual(sealed["hashes"]["contexts"], digest)
            self.assertEqual(sealed["artifacts"]["contexts"]["sha256"], digest)
            self.assertEqual(sealed["artifacts"]["contexts"]["records"], expected_records)
            self.assertEqual(sealed["counts"]["contexts"], expected_records)

            first_manifest_bytes = manifest_path.read_bytes()
            with redirect_stdout(io.StringIO()):
                rerun_status = runner.main(arguments)
            self.assertEqual(rerun_status, 0)
            self.assertEqual(contexts_path.read_bytes(), contexts_payload)
            self.assertEqual(manifest_path.read_bytes(), first_manifest_bytes)

            aggregate = score_files(
                contexts_path,
                oracle_path,
                manifest_path,
                cases_path=cases_path,
                n_resamples=4,
            )
            self.assertEqual(aggregate["validity"]["status"], "PASS")

            loaded_cases = load_public_cases(cases_path)
            config_tampered = json.loads(json.dumps(sealed))
            config_tampered["config"]["seed"] += 1
            tampered_path = directory / "manifest-config-tampered.json"
            tampered_path.write_text(
                canonical_json(config_tampered) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "config hash mismatch"):
                runner.load_and_validate_manifest(
                    tampered_path,
                    cases_path,
                    loaded_cases,
                    treatments=TREATMENTS,
                    checkpoints=DEFAULT_CHECKPOINTS,
                    budgets=DEFAULT_BUDGETS,
                )
            with self.assertRaisesRegex(ValueError, "requested budgets differ"):
                runner.load_and_validate_manifest(
                    manifest_path,
                    cases_path,
                    loaded_cases,
                    treatments=TREATMENTS,
                    checkpoints=DEFAULT_CHECKPOINTS,
                    budgets=(256,),
                )

    def test_manifest_config_oracle_and_source_hash_tampering_fail_closed(self) -> None:
        source_case = self.cases[0]
        checkpoint = source_case.checkpoints[0]
        case = replace(source_case, checkpoints=(checkpoint,))
        contexts = tuple(
            runner.build_context_record(case, checkpoint, treatment, 256)
            for treatment in TREATMENTS
        )
        oracle = self.oracle_by_key[(case.case_id, checkpoint.checkpoint_id)]
        contexts_payload = jsonl_bytes(contexts)
        oracle_payload = jsonl_bytes((oracle,))
        cases_payload = jsonl_bytes((case,))
        config: dict[str, Any] = {
            "seed": DEFAULT_SEED,
            "cases": 1,
            "turns": DEFAULT_TURNS,
            "treatments": list(TREATMENTS),
            "budgets": [256],
            "checkpoints": [checkpoint.turn],
        }
        config_digest = canonical_sha256(config)
        manifest: dict[str, Any] = {
            "protocol_version": "session-v1",
            "config": config,
            "config_sha256": config_digest,
            "hashes": {"config": config_digest},
            "case_sha256": {case.case_id: case.sha256},
            "artifacts": {
                "contexts": {
                    "path": "contexts.jsonl",
                    "sha256": bytes_sha256(contexts_payload),
                },
                "oracle_cases": {
                    "path": "oracle_cases.jsonl",
                    "sha256": bytes_sha256(oracle_payload),
                },
                "public_cases": {
                    "path": "public_cases.jsonl",
                    "sha256": bytes_sha256(cases_payload),
                },
            },
        }

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contexts_path = directory / "contexts.jsonl"
            oracle_path = directory / "oracle_cases.jsonl"
            cases_path = directory / "public_cases.jsonl"
            manifest_path = directory / "manifest.json"
            contexts_path.write_bytes(contexts_payload)
            oracle_path.write_bytes(oracle_payload)
            cases_path.write_bytes(cases_payload)
            manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
            valid = score_files(
                contexts_path,
                oracle_path,
                manifest_path,
                cases_path=cases_path,
                n_resamples=4,
            )
            self.assertEqual(valid["validity"]["status"], "PASS")

            missing_context_digest = json.loads(json.dumps(manifest))
            del missing_context_digest["artifacts"]["contexts"]
            manifest_path.write_text(
                canonical_json(missing_context_digest) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ScoreValidationError, "does not declare.*contexts"):
                score_files(
                    contexts_path,
                    oracle_path,
                    manifest_path,
                    cases_path=cases_path,
                    n_resamples=2,
                )

            config_tampered = json.loads(json.dumps(manifest))
            config_tampered["config"]["seed"] += 1
            manifest_path.write_text(
                canonical_json(config_tampered) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ScoreValidationError, "config hash mismatch"):
                score_files(
                    contexts_path,
                    oracle_path,
                    manifest_path,
                    cases_path=cases_path,
                    n_resamples=2,
                )

            manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
            oracle_path.write_bytes(oracle_payload + b"\n")
            with self.assertRaisesRegex(ScoreValidationError, "digest mismatch"):
                score_files(
                    contexts_path,
                    oracle_path,
                    manifest_path,
                    cases_path=cases_path,
                    n_resamples=2,
                )

        source_tampered = self._manifest()
        source_tampered["case_sha256"][source_case.case_id] = "0" * 64
        with self.assertRaisesRegex(ScoreValidationError, "source case hash mismatch"):
            score_records(
                self.contexts,
                self.oracles,
                source_tampered,
                public_cases=self.cases,
                n_resamples=2,
            )

    def test_oracle_generation_is_independent_of_treatment_resolver(self) -> None:
        generator_tree = ast.parse(
            (ROOT / "src" / "rcwt_session_generate.py").read_text(encoding="utf-8")
        )
        runner_imports = [
            node
            for node in ast.walk(generator_tree)
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "rcwt_session_run"
            )
            or (
                isinstance(node, ast.Import)
                and any(alias.name == "rcwt_session_run" for alias in node.names)
            )
        ]
        self.assertEqual(runner_imports, [])

        poison = AssertionError("oracle consulted the treatment resolver")
        with (
            patch.object(runner, "resolve_state", side_effect=poison),
            patch.object(generator, "resolve_state", side_effect=poison, create=True),
        ):
            cases, oracles = generator.generate_corpus(
                seed=DEFAULT_SEED,
                cases=len(FAMILIES),
                turns=DEFAULT_TURNS,
            )

        by_key = {(oracle.case_id, oracle.checkpoint_id): oracle for oracle in oracles}
        for case in cases:
            for checkpoint in case.checkpoints:
                expected = _independent_oracle(case, checkpoint)
                actual = by_key[(case.case_id, checkpoint.checkpoint_id)]
                self.assertEqual(set(actual.required_current_event_ids), expected["required"])
                self.assertEqual(set(actual.stale_conflict_event_ids), expected["stale"])
                self.assertEqual(set(actual.required_order_edges), expected["edges"])
                self.assertEqual(actual.expected_disposition, expected["disposition"])


if __name__ == "__main__":
    unittest.main()
