"""Offline parser/metric tests; synthetic fixtures are NOT R2 evidence.

The one explicitly named R1 test reads only the already-public redacted R1
log. Its 1,248 calls test b10809 parser compatibility, not R2 completion,
learning, execution safety, private custody or physical timing attestation.
Persistence tests replace build_accounting and the runner with explicit fake
fixtures. No test invokes a real campaign, runner replay, model or endpoint.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import verify_r2_accounting as audit


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_R1_LOG = ROOT / "results/agent_v4_replication_public_runtime/server.redacted.txt"
PUBLIC_R1_SHA256 = "64b74addd5505f9f3d0e3a0543fa35fa8055c81504f76fd72ac0aa3d139dc34e"
STARTUP = (
    "0.00.000.001 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'\n"
    "0.00.000.002 I srv  llama_server: model loaded\n"
    "0.00.000.003 I srv  llama_server: listening on http://127.0.0.1:18085\n"
)


def fake_call(index=0):
    """Deliberately invented measurements; no model or recorded R2 output."""
    return {"purpose": "EXPLICIT-SYNTHETIC-NOT-R2-EVIDENCE:" + str(index),
            "prompt_tokens": 10 + index % 13, "completion_tokens": 5 + index % 7,
            "wall_seconds": .05, "api_cost_usd": 0,
            "timings": {"prompt_n": 10 + index % 13, "predicted_n": 5 + index % 7,
                        "cache_n": 0, "prompt_ms": 10.124, "predicted_ms": 20.456}}


def fake_task(index=0):
    call = fake_call(index)
    total = call["prompt_tokens"] + call["completion_tokens"]
    prefix = f"1.00.000.001 I slot {{event}}: id 0 | task {index * 100} | "
    return "\n".join([
        prefix.format(event="launch_slot_") + "processing task, is_child = 0",
        prefix.format(event="print_timing") + "n_gen = 2, tg = 1.00 t/s, tg_3s = 2.00 t/s",
        prefix.format(event="print_timing") + f"prompt eval time = 10.12 ms / {call['prompt_tokens']} tokens ( 1.00 ms per token, 10.00 tokens per second)",
        prefix.format(event="print_timing") + f"eval time = 20.46 ms / {call['completion_tokens']} tokens ( 1.00 ms per token, 10.00 tokens per second)",
        prefix.format(event="print_timing") + f"total time = 30.58 ms / {total} tokens",
        prefix.format(event="print_timing") + "graphs reused = 3",
        prefix.format(event="release") + f"stop processing: n_tokens = {total - 1}, truncated = 0",
        "",
    ])


def fake_log(count=1):
    return (STARTUP + "".join(fake_task(index) for index in range(count))).encode("utf-8")


class R2LogParserTests(unittest.TestCase):
    def reject(self, data, count=1):
        with self.assertRaises(ValueError):
            audit.parse_server_log(data, count)

    def test_exact_2250_synthetic_calls_reconcile_without_io(self):
        data = fake_log(2250)
        calls = [fake_call(index) for index in range(2250)]
        before = copy.deepcopy(calls)
        with patch("socket.create_connection", side_effect=AssertionError("No network")), \
                patch("subprocess.Popen", side_effect=AssertionError("No process")), \
                patch("builtins.open", side_effect=AssertionError("No file IO")):
            parsed = audit.parse_server_log(data, 2250)
            result = audit.compare_calls(calls, parsed)
        self.assertEqual(result["matched_calls"], 2250)
        self.assertEqual(result["prompt_tokens"], sum(row["prompt_tokens"] for row in calls))
        self.assertEqual(result["completion_tokens"], sum(row["completion_tokens"] for row in calls))
        self.assertEqual(result["total_tokens"], result["prompt_tokens"] + result["completion_tokens"])
        self.assertEqual(result["server_eval_seconds"], 68.805)
        self.assertEqual(result["client_inference_seconds"], 112.5)
        self.assertEqual(result["maximum_timing_delta_ms"], .004)
        self.assertEqual(result["api_cost_usd"], 0)
        self.assertIsNone(result["total_monetary_cost_usd"])
        self.assertEqual(calls, before)
        self.assertEqual(data, fake_log(2250))
        self.assertEqual([row["task_id"] for row in parsed], [i * 100 for i in range(2250)])
        for row in parsed:
            positions = [row[key] for key in ("launch_line", "prompt_line", "eval_line", "total_line", "release_line")]
            self.assertEqual(positions, sorted(set(positions)))

    def test_expected_count_is_an_exact_positive_integer(self):
        for value in (True, False, 0, -1, 1.0, "1", None):
            with self.subTest(value=value):
                self.reject(fake_log(), value)

    def test_extra_and_missing_tasks_reject(self):
        self.reject(fake_log(3), 2)
        self.reject(fake_log(2), 3)

    def test_canceled_session_rejects_even_after_complete_calls(self):
        for word in ("cancel", "canceled", "cancelled", "canceling", "cancellation"):
            with self.subTest(word=word):
                self.reject(fake_log() + f"2.00.000.001 W srv note: {word} task\n".encode())

    def test_invalid_utf8_or_partial_final_line_reject(self):
        self.reject(fake_log() + b"\xff\n")
        self.reject(fake_log().rstrip(b"\n"))

    def test_each_incomplete_lifecycle_prefix_rejects(self):
        lines = fake_task().splitlines(keepends=True)
        for length in range(1, len(lines)):
            with self.subTest(length=length):
                self.reject((STARTUP + "".join(lines[:length])).encode())

    def test_overlapping_and_duplicate_tasks_reject(self):
        unfinished = "".join(fake_task().splitlines(keepends=True)[:-1])
        self.reject((STARTUP + unfinished + fake_task(1)).encode(), 2)
        self.reject((STARTUP + fake_task() + fake_task()).encode(), 2)

    def test_child_or_second_slot_or_unmatched_task_reject(self):
        for original, changed in ((b"is_child = 0", b"is_child = 1"),
                                  (b"id 0", b"id 1"),
                                  (b"task 0 | prompt eval", b"task 99 | prompt eval")):
            with self.subTest(changed=changed):
                self.reject(fake_log().replace(original, changed))

    def test_startup_must_be_single_complete_and_precede_generation(self):
        for data in ((STARTUP + STARTUP + fake_task()).encode(),
                     (fake_task() + STARTUP).encode(),
                     fake_log().replace(b"n_slots = 1", b"n_slots = 2"),
                     fake_log().replace(b"n_ctx_slot = 4096", b"n_ctx_slot = 8192"),
                     fake_log().replace(b"18085", b"18086"),
                     fake_log().replace(b"model loaded", b"model loading")):
            with self.subTest(data=data[:80]):
                self.reject(data)

    def test_unparsed_or_unknown_lifecycle_record_rejects(self):
        for data in (fake_log().replace(b"I slot launch_slot_", b"I other launch_slot_"),
                     fake_log().replace(b"slot print_timing", b"slot unexpected"),
                     fake_log() + b"2.00.000.001 I srv note: id_task = 999\n"):
            with self.subTest(data=data[-100:]):
                self.reject(data)

    def test_duplicate_final_timings_reject(self):
        lines = fake_task().splitlines(keepends=True)
        for position in (2, 3, 4):
            with self.subTest(position=position):
                extra = lines[:position] + [lines[position]] + lines[position:]
                self.reject((STARTUP + "".join(extra)).encode())

    def test_out_of_order_timings_reject(self):
        lines = fake_task().splitlines(keepends=True)
        for left, right in ((2, 3), (3, 4), (1, 4), (2, 5)):
            swapped = list(lines)
            swapped[left], swapped[right] = swapped[right], swapped[left]
            with self.subTest(pair=(left, right)):
                self.reject((STARTUP + "".join(swapped)).encode())

    def test_unparseable_nonfinite_and_negative_server_timings_reject(self):
        for value in (b"NaN", b"Infinity", b"-1.00", b"1e3", b"10.12 ms = 3"):
            with self.subTest(value=value):
                self.reject(fake_log().replace(b"10.12", value))

    def test_final_total_tokens_must_equal_prompt_plus_generated(self):
        self.reject(fake_log().replace(b"30.58 ms / 15 tokens", b"30.58 ms / 16 tokens"))

    def test_total_time_rounding_boundary_is_exact_decimal(self):
        audit.parse_server_log(fake_log().replace(b"30.58", b"30.59"), 1)
        self.reject(fake_log().replace(b"30.58", b"30.591"))
        audit.parse_server_log(fake_log().replace(b"30.58", b"30.57"), 1)
        self.reject(fake_log().replace(b"30.58", b"30.569"))

    def test_eval_runs_crlf_lru_and_final_kv_count_are_compatible(self):
        data = fake_log().replace(b"20.46 ms / 5 tokens", b"20.46 ms / 5 runs")
        lru = b"0.00.000.004 I slot get_availabl: id 0 | task -1 | selected slot by LRU, t_last = -1\n"
        data = data.replace(b"1.00.000.001 I slot launch_slot_", lru + b"1.00.000.001 I slot launch_slot_", 1)
        parsed = audit.parse_server_log(data.replace(b"\n", b"\r\n"), 1)
        self.assertEqual(parsed[0]["prompt_tokens"], 10)
        self.assertEqual(parsed[0]["completion_tokens"], 5)
        # KV release count 14 is deliberately not mistaken for billed total 15.
        self.assertEqual(audit.compare_calls([fake_call()], parsed)["total_tokens"], 15)


class R2CallMetricTests(unittest.TestCase):
    def setUp(self):
        self.call = fake_call()
        self.observed = audit.parse_server_log(fake_log(), 1)

    def check_rejected(self, call):
        with self.assertRaises(ValueError):
            audit.compare_calls([call], self.observed)

    def test_missing_extra_and_reordered_calls_reject(self):
        for calls, observed in (([], []), ([self.call], []), ([], self.observed),
                                ([self.call, self.call], self.observed),
                                ([fake_call(1), fake_call(0)], audit.parse_server_log(fake_log(2), 2))):
            with self.subTest(counts=(len(calls), len(observed))):
                with self.assertRaises(ValueError):
                    audit.compare_calls(calls, observed)

    def test_token_counts_are_exact_positive_integers(self):
        for field in ("prompt_tokens", "completion_tokens"):
            for value in (0, -1, True, False, 1.0, "1", None):
                call = copy.deepcopy(self.call)
                call[field] = value
                with self.subTest(field=field, value=value):
                    self.check_rejected(call)

    def test_token_mismatch_rejects(self):
        for field in ("prompt_tokens", "completion_tokens"):
            call = copy.deepcopy(self.call)
            call[field] += 1
            self.check_rejected(call)

    def test_timing_counts_must_match_uncached_call_counts(self):
        for field in ("prompt_n", "predicted_n"):
            for value in (0, -1, True, 10.0, "10", 999):
                call = copy.deepcopy(self.call)
                call["timings"][field] = value
                with self.subTest(field=field, value=value):
                    self.check_rejected(call)

    def test_cache_count_must_be_exact_integer_zero(self):
        for value in (1, -1, False, True, 0.0, "0", None):
            call = copy.deepcopy(self.call)
            call["timings"]["cache_n"] = value
            with self.subTest(value=value):
                self.check_rejected(call)
        call = copy.deepcopy(self.call)
        del call["timings"]["cache_n"]
        self.check_rejected(call)

    def test_client_timing_rounding_boundary_is_exact_decimal(self):
        for field, accepted, rejected in (("prompt_ms", 10.13, 10.131),
                                           ("predicted_ms", 20.45, 20.449)):
            call = copy.deepcopy(self.call)
            call["timings"][field] = accepted
            result = audit.compare_calls([call], self.observed)
            self.assertEqual(result["maximum_timing_delta_ms"], .01)
            call["timings"][field] = rejected
            self.check_rejected(call)

    def test_client_timing_and_wall_time_are_finite_nonnegative_numbers(self):
        for field in ("prompt_ms", "predicted_ms", "wall_seconds"):
            for value in (-1, float("nan"), float("inf"), True, "10.12", None):
                call = copy.deepcopy(self.call)
                (call if field == "wall_seconds" else call["timings"])[field] = value
                with self.subTest(field=field, value=value):
                    self.check_rejected(call)

    def test_nonzero_or_invalid_api_cost_rejects(self):
        for value in (.01, -1, float("nan"), float("inf"), True, "0", None):
            call = copy.deepcopy(self.call)
            call["api_cost_usd"] = value
            with self.subTest(value=value):
                self.check_rejected(call)

    def test_canonical_call_digest_covers_unmetered_fields(self):
        original = audit.compare_calls([self.call], self.observed)
        expected = hashlib.sha256(json.dumps(self.call, sort_keys=True, ensure_ascii=False,
                                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(original["call_pairs"][0]["call_sha256"], expected)
        self.call["synthetic_extra"] = "fixture annotation, not a measured output"
        changed = audit.compare_calls([self.call], self.observed)
        self.assertNotEqual(changed["call_pairs"][0]["call_sha256"], expected)
        self.assertEqual(changed["total_tokens"], original["total_tokens"])

    def test_nonfinite_unmetered_fields_cannot_be_hidden_in_call_digest(self):
        self.call["synthetic_extra"] = float("nan")
        self.check_rejected(self.call)


class PublicR1ParserCompatibilityTests(unittest.TestCase):
    def test_public_r1_1248_calls_are_parser_compatibility_not_r2_evidence(self):
        data = PUBLIC_R1_LOG.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), PUBLIC_R1_SHA256)
        with patch("socket.create_connection", side_effect=AssertionError("No network")), \
                patch("subprocess.Popen", side_effect=AssertionError("No process")):
            calls = audit.parse_server_log(data, expected_calls=1248)
        self.assertEqual(len(calls), 1248)
        self.assertEqual(len({call["task_id"] for call in calls}), 1248)
        self.assertEqual(sum(call["prompt_tokens"] for call in calls), 1906468)
        self.assertEqual(sum(call["completion_tokens"] for call in calls), 231058)
        self.assertEqual(PUBLIC_R1_LOG.read_bytes(), data)
        # In particular this source must NOT satisfy the R2 count contract.
        with self.assertRaises(ValueError):
            audit.parse_server_log(data, expected_calls=2250)


class SyntheticAccountingPersistenceTests(unittest.TestCase):
    """Post-write guards only; full replay and source pins are mocked, not proved."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="explicit-synthetic-r2-accounting-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "FAKE-run-not-evidence"
        self.run.mkdir()
        (self.run / "fixture.txt").write_bytes(b"EXPLICIT SYNTHETIC RUN; NO R2 EVIDENCE\n")
        self.log = self.root / "FAKE-server.log"
        self.log.write_bytes(b"EXPLICIT SYNTHETIC LOG; NOT A REAL SESSION\n")
        (self.root / ".runs").mkdir()
        self.output = self.root / ".runs" / "r2_accounting_FAKE"
        self.enterContext(patch.object(audit, "ROOT", self.root))
        self.helper_bytes = b"EXPLICIT SYNTHETIC ACCOUNTING SOURCE PIN"

        def read_bytes(path):
            return self.helper_bytes if Path(path) == Path(audit.__file__) else Path(path).read_bytes()

        def snapshot(directory):
            return dict(sorted((path.relative_to(directory).as_posix(), digest(path.read_bytes()))
                               for path in Path(directory).rglob("*") if path.is_file()))

        def write(path, data, *, raw=False):
            self.assertIs(raw, True)
            with Path(path).open("xb") as handle:
                handle.write(data)

        def digest(data):
            return hashlib.sha256(data).hexdigest()

        self.fake_runner = SimpleNamespace(
            _local=lambda path, **kwargs: Path(path).resolve(),
            _bytes=read_bytes, _snapshot=snapshot, _write=write, digest=digest,
            json_bytes=lambda value: (json.dumps(value, sort_keys=True, ensure_ascii=False,
                                                indent=2, allow_nan=False) + "\n").encode(),
            validate_protocol=Mock(return_value={"provenance": "MOCK ONLY; NOT REAL REPLAY"}),
        )
        self.result = {"schema": "EXPLICIT-SYNTHETIC-FIXTURE-NOT-R2-EVIDENCE",
                       "run_files_sha256": snapshot(self.run),
                       "server_log_sha256": digest(self.log.read_bytes()),
                       "source_sha256": digest(self.helper_bytes),
                       "call_pairs": [{"purpose": "EXPLICIT SYNTHETIC"}],
                       "fixture_notice": "No complete R2 campaign or source gate was executed"}
        self.build = self.enterContext(patch.object(audit, "build_accounting", return_value=self.result))
        self.enterContext(patch.dict("sys.modules", {"rcwt_r2": self.fake_runner}))
        self.enterContext(patch("socket.create_connection", side_effect=AssertionError("No network")))
        self.enterContext(patch("subprocess.Popen", side_effect=AssertionError("No process")))

    def run_audit(self, *, verify=False):
        return audit.audit(self.run, self.log, self.output, verify=verify)

    def after_write(self, callback):
        original = self.fake_runner._write

        def write(path, data, **kwargs):
            original(path, data, **kwargs)
            callback(Path(path))

        self.fake_runner._write = write

    def test_new_output_exact_bytes_verify_and_final_protocol_guard(self):
        summary = self.run_audit()
        saved = (self.output / "accounting.json").read_bytes()
        self.assertEqual(saved, self.fake_runner.json_bytes(self.result))
        self.assertNotIn("call_pairs", summary)
        self.assertNotIn("run_files_sha256", summary)
        self.assertEqual(summary["schema"], "EXPLICIT-SYNTHETIC-FIXTURE-NOT-R2-EVIDENCE")
        self.assertEqual(self.run_audit(verify=True), summary)
        self.assertEqual((self.output / "accounting.json").read_bytes(), saved)
        self.assertEqual(self.fake_runner.validate_protocol.call_args_list, [((self.run,),), ((self.run,),)])
        self.assertEqual(self.build.call_count, 2)

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_bytes(b"KEEP SYNTHETIC FILE")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.run_audit()
        self.assertEqual(marker.read_bytes(), b"KEEP SYNTHETIC FILE")
        self.build.assert_not_called()

    def test_new_output_cannot_modify_source_or_existing_experiment_trees(self):
        for prefix in ("results/agent_v4_replication", "src", "tools", "docs", ".git", ".runs/old_run"):
            target = self.root / prefix / "new-output"
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, "immediate"):
                audit.audit(self.run, self.log, target)
            self.assertFalse(target.exists())
        self.build.assert_not_called()

    def test_output_cannot_overlap_run_or_contain_log(self):
        for target in (self.run, self.run / "nested", self.root):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "separate"):
                audit.audit(self.run, self.log, target)
        separate_run = self.root / "other" / "nested-run"
        separate_run.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "separate"):
            audit.audit(separate_run, self.log, self.log.parent)
        self.build.assert_not_called()

    def test_corrupt_written_bytes_are_rejected(self):
        self.after_write(lambda path: path.write_bytes(b"CORRUPTED SYNTHETIC OUTPUT"))
        with self.assertRaisesRegex(ValueError, "Written accounting bytes"):
            self.run_audit()
        self.fake_runner.validate_protocol.assert_not_called()

    def test_extra_output_file_is_rejected(self):
        self.after_write(lambda path: (path.parent / "unexpected.txt").write_bytes(b"EXTRA SYNTHETIC FILE"))
        with self.assertRaisesRegex(ValueError, "Written accounting bytes"):
            self.run_audit()

    def test_run_changed_during_write_is_rejected(self):
        self.after_write(lambda path: (self.run / "fixture.txt").write_bytes(b"CHANGED SYNTHETIC INPUT"))
        with self.assertRaisesRegex(ValueError, "Input changed"):
            self.run_audit()

    def test_log_appended_during_write_is_rejected(self):
        self.after_write(lambda path: self.log.write_bytes(self.log.read_bytes() + b"SYNTHETIC APPEND\n"))
        with self.assertRaisesRegex(ValueError, "Input changed"):
            self.run_audit()

    def test_helper_source_changed_during_write_is_rejected(self):
        self.after_write(lambda path: setattr(self, "helper_bytes", b"CHANGED SYNTHETIC SOURCE PIN"))
        with self.assertRaisesRegex(ValueError, "Input changed"):
            self.run_audit()

    def test_verify_requires_exact_output_bytes_and_no_extra_inventory(self):
        self.run_audit()
        path = self.output / "accounting.json"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "Saved accounting differs"):
            self.run_audit(verify=True)
        path.write_bytes(original)
        (self.output / "unexpected.txt").write_bytes(b"EXTRA SYNTHETIC FILE")
        with self.assertRaisesRegex(ValueError, "Saved accounting differs"):
            self.run_audit(verify=True)

    def test_verify_detects_input_mutation_during_output_read(self):
        self.run_audit()
        original = self.fake_runner._snapshot

        def snapshot(path):
            value = original(path)
            if path == self.output:
                self.log.write_bytes(self.log.read_bytes() + b"SYNTHETIC APPEND DURING VERIFY\n")
            return value

        self.fake_runner._snapshot = snapshot
        with self.assertRaisesRegex(ValueError, "Input changed"):
            self.run_audit(verify=True)

    def test_final_protocol_pin_failure_propagates_after_write_and_verify(self):
        self.fake_runner.validate_protocol.side_effect = ValueError("SYNTHETIC external source pin drift")
        with self.assertRaisesRegex(ValueError, "source pin drift"):
            self.run_audit()
        with self.assertRaisesRegex(ValueError, "source pin drift"):
            self.run_audit(verify=True)


if __name__ == "__main__":
    unittest.main()
