"""Synthetic accounting fixtures only; no R1 corpus, trace or live log is read.

The replay gate and source pins are explicitly mocked for these parser/audit
unit tests. They do not provide confirmation, runtime, model or timing evidence.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import audit_v4_replication_calls as audit


PROVENANCE = "EXPLICIT SYNTHETIC FIXTURE; no model, measured timing or real R1 data"
STARTUP = (
    "0.00.000.001 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'\n"
    "0.00.000.002 I srv  llama_server: model loaded\n"
    "0.00.000.003 I srv  llama_server: listening on http://127.0.0.1:18085\n"
)


def synthetic_call(index):
    return {"purpose": "SYNTHETIC:" + str(index), "prompt_tokens": 10 + index % 13,
            "completion_tokens": 5 + index % 7, "wall_seconds": 0.05,
            "api_cost_usd": 0, "timings": {"prompt_n": 10 + index % 13,
                "predicted_n": 5 + index % 7, "cache_n": 0,
                "prompt_ms": 10.124, "predicted_ms": 20.456}}


def synthetic_task(index):
    task = index * 100
    prefix = f"1.00.000.001 I slot {{event}}: id  0 | task {task} | "
    call = synthetic_call(index)
    total = call["prompt_tokens"] + call["completion_tokens"]
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


def synthetic_log(count=1248):
    return (STARTUP + "".join(synthetic_task(index) for index in range(count))).encode("utf-8")


def synthetic_rows():
    rows, position = [], 0
    for index in range(512):
        count = 3 if index < 224 else 2
        events = [{"method": "tokenize", "result": {"provenance": PROVENANCE}}]
        events.extend({"method": "complete", "result": synthetic_call(call_index)}
                      for call_index in range(position, position + count))
        rows.append({"sha256": f"SYNTHETIC-ROW-{index}", "client_events": events})
        position += count
    assert position == 1248
    return rows


class ReplicationCallAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log_fixture = synthetic_log()
        cls.rows_fixture = synthetic_rows()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="explicit-synthetic-r1-accounting-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "synthetic-run"
        self.run_dir.mkdir()
        self.output = self.root / "synthetic-audit"
        self.server_log = self.root / "synthetic-server.log"
        self.server_log.write_bytes(self.log_fixture)
        (self.run_dir / "protocol.json").write_bytes(audit._json_bytes({"provenance": PROVENANCE}))
        self.write_rows(copy.deepcopy(self.rows_fixture))
        self.gate = {"status": "PASS", "read_only": True, "integrity_only": True,
                     "gain": "NOT_EVALUATED", "verified_steps": 512,
                     "verified_episode_summaries": 64, "generation_calls": 1248,
                     "inference_calls": 0,
                     "protocol_sha256": audit._digest((self.run_dir / "protocol.json").read_bytes()),
                     "prompt_tokens": sum(synthetic_call(i)["prompt_tokens"] for i in range(1248)),
                     "completion_tokens": sum(synthetic_call(i)["completion_tokens"] for i in range(1248)),
                     "provenance": PROVENANCE}
        self.patches = {}
        for name, patcher in {
            "root": patch.object(audit, "ROOT", self.root),
            "pins": patch.object(audit, "_source_pins", return_value={"provenance": PROVENANCE}),
            "gate": patch.object(audit, "verify_run", side_effect=lambda directory: copy.deepcopy(self.gate)),
            "network": patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Network forbidden")),
            "socket": patch("socket.create_connection", side_effect=AssertionError("Socket forbidden")),
        }.items():
            self.patches[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def write_rows(self, rows):
        (self.run_dir / "traces.jsonl").write_bytes(
            b"".join(json.dumps(row, allow_nan=False).encode() + b"\n" for row in rows))

    def build(self):
        return audit.build_accounting(self.run_dir, self.server_log)

    def test_synthetic_complete_accounting_requires_full_gate_and_changes_no_inputs(self):
        before = audit._snapshot(self.run_dir)
        result = self.build()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["accounting_only"])
        self.assertEqual(result["gain"], "NOT_EVALUATED")
        self.assertEqual(result["matched_trace_calls"], 1248)
        self.assertEqual(result["server_task_starts"], 1248)
        self.assertEqual(result["server_final_timing_pairs"], 1248)
        self.assertEqual(result["prompt_tokens"], self.gate["prompt_tokens"])
        self.assertEqual(result["completion_tokens"], self.gate["completion_tokens"])
        self.assertAlmostEqual(result["recorded_client_inference_seconds"], 62.4)
        self.assertIsNone(result["total_monetary_cost_usd"])
        self.assertEqual(result["inference_calls"], 0)
        self.assertEqual(result["maximum_observed_timing_delta_ms"], 0.004)
        self.assertEqual(result["call_pairs"][0]["event_index"], 1)
        self.assertEqual(before, audit._snapshot(self.run_dir))
        self.patches["gate"].assert_called_once_with(self.run_dir)
        self.patches["network"].assert_not_called()
        self.patches["socket"].assert_not_called()

    def test_exclusive_output_and_verify_exact_saved_bytes(self):
        result = audit.audit(self.run_dir, self.server_log, self.output)
        self.assertEqual(result["output_mode"], "CREATED_EXCLUSIVELY")
        saved = audit._snapshot(self.output)
        checked = audit.audit(self.run_dir, self.server_log, self.output, verify=True)
        self.assertEqual(checked["output_mode"], "VERIFIED_EXACT_BYTES")
        self.assertEqual(saved, audit._snapshot(self.output))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            audit.audit(self.run_dir, self.server_log, self.output)
        self.assertEqual(saved, audit._snapshot(self.output))

    def test_changed_saved_json_bytes_are_not_repaired(self):
        audit.audit(self.run_dir, self.server_log, self.output)
        saved = self.output / audit.OUTPUT_NAMES[0]
        saved.write_bytes(saved.read_bytes() + b" ")
        changed = audit._snapshot(self.output)
        with self.assertRaisesRegex(ValueError, "Saved audit bytes differ"):
            audit.audit(self.run_dir, self.server_log, self.output, verify=True)
        self.assertEqual(changed, audit._snapshot(self.output))

    def test_changed_saved_markdown_bytes_are_not_repaired(self):
        audit.audit(self.run_dir, self.server_log, self.output)
        saved = self.output / audit.OUTPUT_NAMES[1]
        saved.write_bytes(saved.read_bytes().replace(b"unknown", b"zero"))
        with self.assertRaisesRegex(ValueError, "Saved audit bytes differ"):
            audit.audit(self.run_dir, self.server_log, self.output, verify=True)

    def test_extra_output_file_is_rejected(self):
        audit.audit(self.run_dir, self.server_log, self.output)
        (self.output / "extra.txt").write_text(PROVENANCE)
        with self.assertRaisesRegex(ValueError, "additional"):
            audit.audit(self.run_dir, self.server_log, self.output, verify=True)

    def test_output_must_be_outside_input_evidence(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            audit.audit(self.run_dir, self.server_log, self.run_dir / "audit")
        self.patches["gate"].assert_not_called()

    def test_extra_or_missing_started_call_fails(self):
        for count in (1247, 1249):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "1248"):
                audit.parse_server_log(synthetic_log(count))

    def test_cancel_fails_even_with_other_complete_timings(self):
        with self.assertRaisesRegex(ValueError, "Canceled"):
            audit.parse_server_log(self.log_fixture + b"9.00.000.001 W srv stop: cancel task, id_task = 555\n")

    def test_malformed_launch_and_child_task_fail(self):
        for replacement in (b"processing task, is_child = unknown", b"processing task, is_child = 1"):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "malformed"):
                audit.parse_server_log(self.log_fixture.replace(b"processing task, is_child = 0", replacement, 1))

    def test_malformed_final_timing_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "Unparseable"):
            audit.parse_server_log(self.log_fixture.replace(b"prompt eval time = 10.12", b"prompt eval time = NaN", 1))

    def test_unknown_slot_task_record_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            audit.parse_server_log(self.log_fixture + b"9.00.000.001 I slot print_timing: id 0 | task 999999 | eval time = 20.00 ms / 5 tokens ( 1 ms per token)\n")

    def test_unmatched_final_timing_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            audit.parse_server_log(self.log_fixture.replace(b"task 0 | prompt eval", b"task 9 | prompt eval", 1))

    def test_duplicate_final_timing_is_rejected(self):
        prompt_line = next(line for line in self.log_fixture.splitlines(keepends=True) if b"prompt eval time" in line)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit.parse_server_log(self.log_fixture.replace(prompt_line, prompt_line + prompt_line, 1))

    def test_incomplete_final_timing_and_trailing_line_are_rejected(self):
        evaluation = next(line for line in self.log_fixture.splitlines(keepends=True) if b"| eval time" in line)
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            audit.parse_server_log(self.log_fixture.replace(evaluation, b"", 1))
        with self.assertRaisesRegex(ValueError, "incomplete line"):
            audit.parse_server_log(self.log_fixture[:-1])

    def test_duplicate_id_overlap_and_nonzero_slot_are_rejected(self):
        cases = [self.log_fixture.replace(b"task 100 |", b"task 0 |"),
                 self.log_fixture.replace(b"id  0 |", b"id  1 |", 1)]
        for data in cases:
            with self.subTest(), self.assertRaises(ValueError):
                audit.parse_server_log(data)
        first_release = next(line for line in self.log_fixture.splitlines(keepends=True) if b"release:" in line)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            audit.parse_server_log(self.log_fixture.replace(first_release, b"", 1))

    def test_session_restart_or_missing_startup_fails(self):
        for data in (self.log_fixture + STARTUP.encode(), self.log_fixture[len(STARTUP.encode()):]):
            with self.subTest(), self.assertRaises(ValueError):
                audit.parse_server_log(data)

    def test_eval_runs_unit_and_one_token_release_difference_are_accepted(self):
        data = self.log_fixture.replace(b"eval time = 20.46 ms / 5 tokens", b"eval time = 20.46 ms / 5 runs")
        self.assertEqual(len(audit.parse_server_log(data)), 1248)

    def test_rounding_tolerance_accepts_exact_boundary_and_rejects_larger_delta(self):
        rows = copy.deepcopy(self.rows_fixture)
        rows[0]["client_events"][1]["result"]["timings"]["prompt_ms"] = 10.13
        self.write_rows(rows)
        self.assertEqual(self.build()["maximum_observed_timing_delta_ms"], 0.01)
        rows[0]["client_events"][1]["result"]["timings"]["prompt_ms"] = 10.131
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "timing mismatch"):
            self.build()

    def test_ordered_token_mismatch_fails(self):
        rows = copy.deepcopy(self.rows_fixture)
        events = rows[0]["client_events"]
        events[1], events[2] = events[2], events[1]
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "token mismatch"):
            self.build()

    def test_invalid_count_type_and_cache_metering_fail(self):
        for key, value in (("prompt_n", True), ("predicted_n", 5.0), ("cache_n", 1)):
            rows = copy.deepcopy(self.rows_fixture)
            rows[0]["client_events"][1]["result"]["timings"][key] = value
            self.write_rows(rows)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.build()

    def test_nonfinite_or_string_trace_timing_fails(self):
        for value in ("10.124", -1, True):
            rows = copy.deepcopy(self.rows_fixture)
            rows[0]["client_events"][1]["result"]["timings"]["prompt_ms"] = value
            self.write_rows(rows)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.build()
        self.write_rows(copy.deepcopy(self.rows_fixture))
        traces = self.run_dir / "traces.jsonl"
        traces.write_bytes(traces.read_bytes().replace(b"10.124", b"1e999", 1))
        with self.assertRaises(ValueError):
            self.build()

    def test_output_bytes_are_reread_after_creation(self):
        original_snapshot = audit._snapshot
        def mutate_output_before_readback(directory):
            if directory == self.output:
                target = directory / audit.OUTPUT_NAMES[0]
                target.write_bytes(target.read_bytes() + b" SYNTHETIC MUTATION")
            return original_snapshot(directory)
        with patch.object(audit, "_snapshot", side_effect=mutate_output_before_readback):
            with self.assertRaisesRegex(ValueError, "Output bytes changed"):
                audit.audit(self.run_dir, self.server_log, self.output)

    def test_missing_or_extra_trace_calls_and_wrong_step_count_fail(self):
        rows = copy.deepcopy(self.rows_fixture)
        rows[0]["client_events"].pop()
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "1248"):
            self.build()
        self.write_rows(rows[:-1])
        with self.assertRaisesRegex(ValueError, "512"):
            self.build()

    def test_replay_must_be_complete_read_only_integrity_gate(self):
        for key, value in (("verified_steps", 173), ("verified_steps", 512.0),
                           ("generation_calls", 1247), ("inference_calls", 1),
                           ("read_only", False), ("gain", "PASS"),
                           ("protocol_sha256", "changed")):
            previous = self.gate[key]
            self.gate[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "complete read-only"):
                self.build()
            self.gate[key] = previous

    def test_server_log_changed_during_gate_fails_without_output(self):
        def mutate(directory):
            self.server_log.write_bytes(self.log_fixture + b"SYNTHETIC mutation\n")
            return self.gate
        self.patches["gate"].side_effect = mutate
        with self.assertRaisesRegex(ValueError, "changed during audit"):
            audit.audit(self.run_dir, self.server_log, self.output)
        self.assertFalse(self.output.exists())

    def test_trace_bytes_changed_during_gate_fails(self):
        def mutate(directory):
            target = self.run_dir / "protocol.json"
            target.write_bytes(target.read_bytes() + b" ")
            return self.gate
        self.patches["gate"].side_effect = mutate
        with self.assertRaisesRegex(ValueError, "changed during audit"):
            self.build()

    def test_source_pins_changed_during_audit_fail(self):
        self.patches["pins"].side_effect = [{"provenance": PROVENANCE}, {"provenance": "CHANGED SYNTHETIC PIN"}]
        with self.assertRaisesRegex(ValueError, "changed during audit"):
            self.build()

    def test_changed_original_log_invalidates_saved_audit_even_if_accounting_same(self):
        audit.audit(self.run_dir, self.server_log, self.output)
        self.server_log.write_bytes(self.log_fixture + b"SYNTHETIC extra informational line\n")
        with self.assertRaisesRegex(ValueError, "Saved audit bytes differ"):
            audit.audit(self.run_dir, self.server_log, self.output, verify=True)


if __name__ == "__main__":
    unittest.main()
