"""Offline consistency checks for the exploratory pilot receipt package.

These tests read captured artifacts only. Hashes and declared loopback metadata
do not independently attest physical model execution, network traffic, or
held-out performance. No runtime endpoint or frozen campaign source is accessed.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import unittest
from pathlib import Path
from urllib.parse import urlsplit


PILOTS = Path(__file__).resolve().parents[1] / "results" / "agent_development" / "pilots"
EXPECTED = {
    "action-schema-diagnostic-receipt.json": (
        "cdb614c1be39282bbbfea8b8f352fe2fc16dce1164ffd928d12e2d8285135615",
        10, 5404, 473, "rcwt-local-qwen3-4b",
    ),
    "thinking-pilot-receipt.json": (
        "7b770c7d54fa67b6d558abe8bd500be72f084ee95db27de6ed873ed981bb6cd3",
        6, 5443, 2984, "rcwt-local-qwen3-4b",
    ),
    "qwen35-pilot-receipt.json": (
        "2cf7d20573e7b92ff5df845e52b5726cdcc241765455a32eaefd4ee6b6b42220",
        6, 5521, 371, "rcwt-local-qwen35-4b",
    ),
    "evidence-action-pilot-receipt.json": (
        "35113f5908cce6985c9081fccdc2300ccb58978d3f5d07740fc5790250d4af70",
        6, 5827, 836, "rcwt-local-qwen35-4b",
    ),
}
LITERAL_FIXTURES = {
    "literal_no_schema", "literal_action_schema", "literal_reversed_enums",
    "simple_missing_action_schema",
}


def _walk(value):
    """Visit scalar leaves, including JSON objects encoded inside prompt strings."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return
        if isinstance(decoded, (dict, list)):
            yield from _walk(decoded)


def _call_parts(record):
    if "call" in record:
        call = record["call"]
        return call["request"], call, call["wall_seconds"]
    if "response" in record:
        return record["request"], record["response"]["usage"], record["wall_seconds"]
    return record["request"], record["usage"], record["wall_seconds"]


class AgentPilotArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = json.loads((PILOTS / "index.json").read_text(encoding="utf-8"))
        cls.entries = {entry["file"]: entry for entry in cls.index["receipts"]}
        cls.receipts = {
            name: json.loads((PILOTS / name).read_text(encoding="utf-8"))
            for name in EXPECTED
        }

    def assert_loopback(self, url):
        parsed = urlsplit(url)
        self.assertEqual(parsed.scheme, "http")
        self.assertTrue(ipaddress.ip_address(parsed.hostname).is_loopback)
        self.assertEqual(parsed.port, 18085)
        self.assertIsNone(parsed.username)
        self.assertIsNone(parsed.password)

    def test_four_receipts_are_byte_locked_and_indexed_once(self):
        self.assertEqual(len(self.index["receipts"]), 4)
        self.assertEqual(set(self.entries), set(EXPECTED))
        self.assertEqual(
            {path.name for path in PILOTS.glob("*-receipt.json")}, set(EXPECTED)
        )
        for name, (digest, *_rest) in EXPECTED.items():
            with self.subTest(receipt=name):
                data = (PILOTS / name).read_bytes()
                entry = self.entries[name]
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
                self.assertEqual(entry["published_sha256"], digest)
                self.assertEqual(entry["published_bytes"], len(data))
                self.assertRegex(entry["source_sha256"], r"^[0-9a-f]{64}$")

    def test_metering_recomputes_the_known_lower_bound(self):
        calls = input_tokens = output_tokens = 0
        wall_times = []
        for name, receipt in self.receipts.items():
            with self.subTest(receipt=name):
                entry = self.entries[name]
                parts = [_call_parts(record) for record in receipt["records"]]
                prompt = sum(usage["prompt_tokens"] for _, usage, _ in parts)
                completion = sum(usage["completion_tokens"] for _, usage, _ in parts)
                wall = math.fsum(seconds for _, _, seconds in parts)
                self.assertEqual((len(parts), prompt, completion), EXPECTED[name][1:4])
                self.assertEqual(entry["calls"], len(parts))
                self.assertEqual(entry["input_tokens"], prompt)
                self.assertEqual(entry["output_tokens"], completion)
                self.assertAlmostEqual(entry["wall_seconds"], wall, delta=1e-9)
                for _, usage, seconds in parts:
                    self.assertIs(type(usage["prompt_tokens"]), int)
                    self.assertIs(type(usage["completion_tokens"]), int)
                    self.assertGreater(usage["prompt_tokens"], 0)
                    self.assertGreater(usage["completion_tokens"], 0)
                    self.assertTrue(math.isfinite(seconds) and seconds > 0)
                calls += len(parts)
                input_tokens += prompt
                output_tokens += completion
                wall_times.extend(seconds for _, _, seconds in parts)
        self.assertEqual((calls, input_tokens, output_tokens), (28, 22195, 4664))
        total = self.index["known_metered_lower_bound"]
        self.assertEqual(total["calls"], calls)
        self.assertEqual(total["input_tokens"], input_tokens)
        self.assertEqual(total["output_tokens"], output_tokens)
        self.assertEqual(total["total_tokens"], input_tokens + output_tokens)
        self.assertAlmostEqual(total["summed_call_wall_seconds"], math.fsum(wall_times), delta=1e-9)

    def test_api_charges_are_zero_without_claiming_zero_total_cost(self):
        self.assertEqual(self.index["inference_policy"], "local_only_no_paid_api")
        self.assertEqual(self.index["known_metered_lower_bound"]["api_provider_cost_usd"], 0)
        for name, receipt in self.receipts.items():
            self.assertEqual(self.entries[name]["api_provider_cost_usd"], 0)
            for record in receipt["records"]:
                if "call" in record:
                    self.assertEqual(record["call"]["api_cost_usd"], 0)
                elif "response" not in record:
                    self.assertEqual(record["api_cost_usd"], 0)
        exclusions = {row["category"]: row for row in self.index["exclusions"]}
        self.assertIsNone(exclusions["electricity_hardware_depreciation_and_human_time"]["cost_usd"])
        diagnostics = exclusions["four_additional_root_non_greedy_diagnostics"]
        self.assertEqual(diagnostics["calls"], 4)
        self.assertIs(diagnostics["metering_available_in_this_package"], False)
        for key in ("input_tokens", "output_tokens", "wall_seconds"):
            self.assertIsNone(diagnostics[key])
        self.assertIn("aborted_greedy_development_campaign", exclusions)
        self.assertIn("runtime_setup_downloads_warmup_and_other_unreceipted_probes", exclusions)

    def test_recorded_endpoint_and_all_request_aliases_are_local(self):
        self.assert_loopback(self.index["inference_endpoint"])
        for name, receipt in self.receipts.items():
            alias = EXPECTED[name][4]
            if "server" in receipt:
                server = receipt["server"]
                self.assert_loopback(server["base_url"])
                self.assertEqual(server["model_alias"], alias)
                self.assertEqual(server["inference_policy"], "local_only_no_paid_api")
                self.assertIn("--host 127.0.0.1 --port 18085", server["arguments"])
                self.assertIn("--offline", server["arguments"])
            else:
                # This capture has no independent server snapshot; do not invent one.
                self.assertEqual(name, "evidence-action-pilot-receipt.json")
            for record in receipt["records"]:
                request, _, _ = _call_parts(record)
                self.assertEqual(request["model"], alias)
                self.assertIs(request["stream"], False)
                self.assertIs(request["cache_prompt"], False)
                if "response" in record:
                    self.assertEqual(record["response"]["model"], alias)
                if "call" in record:
                    self.assertEqual(record["call"]["model"], alias)
                for key, value in _walk(request):
                    if key.lower() in {"url", "endpoint", "base_url", "api_base", "proxy"}:
                        self.assert_loopback(value)
                    if isinstance(value, str):
                        for url in re.findall(r"https?://[^\s\"<>]+", value):
                            self.assert_loopback(url)

    def test_no_raw_hidden_reasoning_credentials_or_private_paths(self):
        prohibited = {
            "reasoning_content", "reasoning", "thinking", "analysis", "thoughts",
            "chain_of_thought", "password", "secret", "authorization", "api_key",
            "access_token", "private_key", "cookie",
        }
        for name, receipt in self.receipts.items():
            with self.subTest(receipt=name):
                for key, value in _walk(receipt):
                    if key.lower() in prohibited:
                        self.assertIn(value, (None, "", [], {}))
                    if isinstance(value, str):
                        self.assertNotRegex(value, r"(?i)</?think(?:\s|>)")
                        self.assertNotRegex(value, r"(?i)(?<![a-z])[a-z]:[\\/]|/Users/|/home/")
                    if key == "reasoning_sha256" and value is not None:
                        self.assertRegex(value, r"^[0-9a-f]{64}$")
                    if key in {"reasoning_characters", "reasoning_chars"}:
                        self.assertIs(type(value), int)
                        self.assertGreaterEqual(value, 0)

    def test_inputs_contain_training_cases_or_declared_literal_fixtures_not_labels(self):
        forbidden_keys = {
            "expected_action", "expected_answer", "oracle", "oracle_action",
            "ground_truth", "gold_action", "gold_answer", "score", "failure_category",
            "evaluation_only_not_in_request",
        }
        for name, receipt in self.receipts.items():
            if "training_episode_id" in receipt:
                self.assertRegex(receipt["training_episode_id"], r"^rcwta-train-")
                self.assertIs(receipt["private_oracle_accessed"], False)
                self.assertIs(receipt["future_steps_accessed"], False)
            for record in receipt["records"]:
                with self.subTest(receipt=name, experiment=record.get("experiment"), episode=record.get("episode_id")):
                    if "episode_id" in record:
                        self.assertRegex(record["episode_id"], r"^rcwta-train-")
                    request, _, _ = _call_parts(record)
                    text = json.dumps(request, ensure_ascii=False)
                    self.assertNotRegex(text, r"(?i)\b(?:test|validation|held[-_]?out)[-_][a-z0-9]")
                    if record.get("experiment") not in LITERAL_FIXTURES:
                        self.assertRegex(text, r"\binv-train-[a-z0-9-]+")
                    for key, value in _walk(request):
                        self.assertNotIn(key.lower(), forbidden_keys)
                        if key == "split":
                            self.assertEqual(value, "train")

    def test_scores_remain_outside_requests_and_are_not_pooled_with_final_results(self):
        self.assertEqual(self.index["classification"], "exploratory_development_pilots_not_final_benchmark")
        self.assertIs(self.index["included_in_final_benchmark_metrics"], False)
        expected_scores = {
            "thinking-pilot-receipt.json": 4,
            "qwen35-pilot-receipt.json": 3,
            "evidence-action-pilot-receipt.json": 5,
        }
        for name, count in expected_scores.items():
            records = self.receipts[name]["records"]
            scores = [record.get("evaluation_only_not_in_request", record.get("score")) for record in records]
            self.assertEqual(sum(score["success"] for score in scores), count)
            self.assertEqual(self.entries[name]["task_successes"], count)
        self.assertIsNone(self.entries["action-schema-diagnostic-receipt.json"]["task_successes"])
        self.assertEqual(sum(row["redacted_values"] for row in self.entries.values()), 20)
        for entry in self.entries.values():
            self.assertEqual(len(entry["redacted_json_paths"]), entry["redacted_values"])
        verification = self.index["verification"]
        self.assertEqual(verification["new_inference_calls_for_packaging"], 0)
        self.assertIs(verification["source_receipts_modified"], False)
        self.assertIn("does not independently attest", verification["hash_limit"])


if __name__ == "__main__":
    unittest.main()
