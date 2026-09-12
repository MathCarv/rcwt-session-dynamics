"""Entirely synthetic archives + fake verifier; never real experiment evidence."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import report_online_development as report


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class DevelopmentReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rcwt-development-index-test-", dir=report.ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        baseline = self.root / "results" / "agent_v2"
        baseline.mkdir(parents=True)
        for name in ("protocol.json", "RESULTS.md", "DIAGNOSIS.md"):
            (baseline / name).write_text("SYNTHETIC verifier dependency; not evidence.\n", encoding="utf-8")
        self.output = self.root / "index"
        self.calls = []
        self.addCleanup(patch.stopall)
        patch.object(report.archive_verifier, "ROOT", self.root).start()
        patch.object(report.archive_verifier, "verify_archive", self.fake_verify).start()

    def fake_verify(self, directory):
        self.calls.append(directory)
        return {"status": "PASS", "mode": "EXPLICIT SYNTHETIC TEST VERIFIER; no model or real replay",
                "protocol_sha256": report.digest(directory / "protocol.json"),
                "verification": {"status": "PASS", "verified_steps": 4, "fixture_only": True},
                "inference_calls": 0, "limit": "Fake verifier for isolated unit tests, not evidence"}

    def fixture(self, name, *, candidate_successes=2, candidate_unsafe=0, candidate_cents=0):
        directory = self.root / name
        (directory / "sources").mkdir(parents=True)
        source = directory / "sources" / "fixture.py"
        source.write_text("# Synthetic fixture, never executed.\n", encoding="utf-8")
        protocol = {"schema": "rcwt-online-memory/3.2", "mode": "development", "split": "train",
                    "count": 1, "steps_per_episode": 2, "policies": ["summary", "structured"],
                    "dataset_seed": 123, "inference_seed": 456, "model": "synthetic-never-executed",
                    "memory_budget": 256, "api_cost_usd": 0, "total_monetary_cost_usd": None,
                    "source_sha256": {"src/fixture.py": report.digest(source)},
                    "candidate_origin": "Synthetic fixture only", "baseline": "Synthetic fixture only",
                    "fixture_attempt": name}
        write_json(directory / "protocol.json", protocol)
        records = []
        for policy in ("summary", "structured"):
            records.append({"episode_id": "same-reused-training-episode", "family": "synthetic",
                            "split": "train", "policy": policy, "steps": 2,
                            "successes": 1 if policy == "summary" else candidate_successes,
                            "unsafe_actions": 0 if policy == "summary" else candidate_unsafe,
                            "unsafe_booked_cents": 0 if policy == "summary" else candidate_cents,
                            "model_calls": 3 if policy == "summary" else 2,
                            "prompt_tokens": 100, "completion_tokens": 20,
                            "inference_seconds": 0.3 if policy == "summary" else 0.2,
                            "episode_seconds": 0.4 if policy == "summary" else 0.3})
        (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        for name in report.CORE_INPUTS:
            path = directory / name
            if not path.exists():
                write_json(path, {"synthetic_fixture": True})
        totals = {row["policy"]: {key: row[key] for key in (
            "successes", "steps", "unsafe_actions", "unsafe_booked_cents")} for row in records}
        screen = {"passed": candidate_successes > 1 and candidate_unsafe == 0 and candidate_cents == 0,
                  "totals": totals, "scope": "development-only screening; not held-out evidence"}
        write_json(directory / "development-screen.json", screen)
        return directory

    def test_generate_and_verify_preserve_all_attempts_and_separate_quality(self):
        runs = [self.fixture("pilot01", candidate_successes=1), self.fixture("pilot02")]
        self.output.mkdir()
        (self.output / "unrelated-subdir").mkdir()
        marker = self.output / "unrelated-subdir" / "keep.txt"
        marker.write_text("preserve", encoding="utf-8")
        self.assertEqual(report.generate(runs, self.output)["status"], "PASS")
        saved = report._read(self.output / report.JSON_NAME)
        self.assertEqual([attempt["screen_status"] for attempt in saved["attempts"]], ["FAIL", "PASS"])
        self.assertEqual([attempt["by_policy"]["structured"]["successes"] for attempt in saved["attempts"]], [1, 2])
        total = saved["development_expenditure"]["all_policies"]
        self.assertEqual(total["model_calls"], 10)
        self.assertEqual(total["total_tokens"], 480)
        self.assertAlmostEqual(total["inference_seconds"], 1.0)
        self.assertAlmostEqual(total["episode_wall_seconds"], 1.4)
        for field in ("accuracy", "successes", "steps", "unsafe_actions", "unsafe_booked_cents"):
            self.assertNotIn(field, total)
        self.assertIn("EXPLICIT SYNTHETIC", saved["attempts"][0]["archive_verification"]["mode"])
        before = {path: path.read_bytes() for path in (self.output / report.JSON_NAME, self.output / report.MARKDOWN_NAME)}
        mtimes = {path: path.stat().st_mtime_ns for path in before}
        self.assertEqual(report.verify_index(runs, self.output)["status"], "PASS")
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in before})
        self.assertEqual(marker.read_text(), "preserve")
        self.assertEqual(len(self.calls), 4)

    def test_provenance_binds_inputs_generator_helper_and_relative_paths(self):
        run = self.fixture("pilot01")
        index = report.build_index([run], self.output)
        source = index["provenance"]["report_source"]
        helper = index["provenance"]["archive_verifier_source"]
        self.assertEqual(source["sha256"], report.digest(Path(report.__file__)))
        self.assertEqual(helper["sha256"], report.digest(Path(report.archive_verifier.__file__)))
        manifest = index["attempts"][0]["input_manifest"]
        self.assertEqual(len(manifest), len(report.CORE_INPUTS) + 1)
        for item in manifest + [source, helper] + index["provenance"]["archive_verifier_baseline_inputs"]:
            self.assertFalse(Path(item["path"]).is_absolute())
            self.assertNotIn(":", item["path"])
            self.assertNotIn("\\", item["path"])
            self.assertEqual(len(item["sha256"]), 64)
        self.assertEqual(index["attempts"][0]["run_path"], "../pilot01")

    def test_generation_never_overwrites_either_existing_report_file(self):
        run = self.fixture("pilot01")
        self.output.mkdir()
        for name in (report.JSON_NAME, report.MARKDOWN_NAME):
            target = self.output / name
            target.write_bytes(b"USER CONTENT")
            with self.subTest(name=name), self.assertRaises(FileExistsError):
                report.generate([run], self.output)
            self.assertEqual(target.read_bytes(), b"USER CONTENT")
            target.unlink()
        self.assertEqual(self.calls, [])

    def test_json_markdown_and_resealed_counter_tampering_fail_verification(self):
        run = self.fixture("pilot01")
        report.generate([run], self.output)
        json_path, md_path = self.output / report.JSON_NAME, self.output / report.MARKDOWN_NAME
        original_json, original_md = json_path.read_bytes(), md_path.read_bytes()
        md_path.write_bytes(original_md + b"tampered\n")
        with self.assertRaisesRegex(ValueError, "Markdown"):
            report.verify_index([run], self.output)
        md_path.write_bytes(original_md)
        saved = json.loads(original_json)
        saved["development_expenditure"]["all_policies"]["model_calls"] += 100
        json_path.write_bytes(report._json_bytes(saved))
        md_path.write_text(report.render_report(saved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON"):
            report.verify_index([run], self.output)

    def test_input_cost_and_provenance_mutation_fail_saved_index(self):
        run = self.fixture("pilot01")
        report.generate([run], self.output)
        path = run / "episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["prompt_tokens"] += 1
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON"):
            report.verify_index([run], self.output)

    def test_saved_screen_is_recomputed_not_trusted_as_a_pass_label(self):
        run = self.fixture("pilot01", candidate_successes=1)
        screen = report._read(run / "development-screen.json")
        screen["passed"] = True
        write_json(run / "development-screen.json", screen)
        with self.assertRaisesRegex(ValueError, "screen"):
            report.generate([run], self.output)
        self.assertFalse(self.output.exists())

    def test_missing_pairs_duplicate_episodes_and_invalid_values_fail_closed(self):
        run = self.fixture("pilot01")
        path = run / "episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutations = [rows[:1], rows + [rows[0]]]
        for field, value in (("steps", True), ("successes", 3), ("unsafe_actions", -1),
                             ("prompt_tokens", 1.5), ("episode_seconds", -1),
                             ("inference_seconds", 99), ("split", "test")):
            altered = copy.deepcopy(rows)
            altered[0][field] = value
            mutations.append(altered)
        for altered in mutations:
            with self.subTest(altered=altered):
                path.write_text("".join(json.dumps(row) + "\n" for row in altered), encoding="utf-8")
                with self.assertRaises(ValueError):
                    report.build_index([run], self.output)

    def test_heldout_protocol_rejected_before_verifier_or_corpus_read(self):
        run = self.fixture("pilot01")
        protocol = report._read(run / "protocol.json")
        protocol.update(mode="confirmatory", split="test")
        write_json(run / "protocol.json", protocol)
        (run / "oracle.json").unlink()
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            report.build_index([run], self.output)
        self.assertEqual(self.calls, [])

    def test_failed_verifier_or_nonzero_new_inference_receipt_rejected(self):
        run = self.fixture("pilot01")
        for field, value in (("status", "FAIL"), ("inference_calls", 1), ("protocol_sha256", "0" * 64)):
            receipt = self.fake_verify(run)
            receipt[field] = value
            with self.subTest(field=field), patch.object(report.archive_verifier, "verify_archive", return_value=receipt):
                with self.assertRaisesRegex(ValueError, "offline PASS"):
                    report.build_index([run], self.output)

    def test_duplicate_runs_and_copied_archives_cannot_inflate_cost_totals(self):
        first = self.fixture("pilot01")
        second = self.fixture("pilot02")
        with self.assertRaisesRegex(ValueError, "distinct"):
            report.build_index([first, first], self.output)
        (second / "protocol.json").write_bytes((first / "protocol.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "Duplicate archive"):
            report.build_index([first, second], self.output)

    def test_cli_generation_and_exact_verify_use_the_same_explicit_run_list(self):
        run = self.fixture("pilot01")
        arguments = ["--runs", str(run), "--output-dir", str(self.output)]
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            report.main(arguments)
        self.assertEqual(json.loads(stream.getvalue())["status"], "PASS")
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            report.main(arguments + ["--verify"])
        self.assertEqual(json.loads(stream.getvalue())["new_model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
