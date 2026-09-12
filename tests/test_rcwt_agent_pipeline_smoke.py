"""End-to-end integrity smoke with an explicitly fake transport, never model evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import rcwt_agent_run as runner
from rcwt_agent_env import generate_episodes
from rcwt_local_model import LocalModelClient


class OfflineTransport(LocalModelClient):
    def _request(self, path, payload=None):
        if path == "/tokenize":
            return {"tokens": list(payload["content"].encode("utf-8"))}
        if path == "/detokenize":
            return {"content": bytes(payload["tokens"]).decode("utf-8", errors="replace")}
        if path != "/v1/chat/completions":
            raise AssertionError("Unexpected test transport request")
        if "response_format" in payload:
            public = json.loads(payload["messages"][1]["content"])["current_step"]
            text = json.dumps({"evidence_check": {
                "invoice_amount_cents": None, "account_status": "unknown", "ownership_match": "unknown",
                "payment_status": "unknown", "return_status": "unknown", "operation_already_booked": "no_record"
            }, "tool": "record_decision", "arguments": {
                "case_id": public["task"]["case_id"], "decision": "ask_info",
                "amount_cents": 0, "reason_code": "missing_evidence"}})
        else:
            text = "Keep current facts and unknowns."
        return {"model": self.model, "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "timings": {"cache_n": 0}}


class PipelineSmokeTests(unittest.TestCase):
    def test_entire_pipeline_replays_without_network(self):
        with tempfile.TemporaryDirectory() as temporary, patch("builtins.print"):
            directory = Path(temporary) / "explicit-fake-test-only"
            args = SimpleNamespace(train_count=4, validation_count=4, test_count=4, seed=31,
                                   memory_budget=256, model="rcwt-local-qwen35-4b",
                                   runtime_receipt=runner.ROOT / "docs/rcwt_agent_runtime.json")
            protocol = runner.prepare(directory, args)
            client = OfflineTransport(seed=args.seed)
            runner.run_cohort(client, generate_episodes("train", 4, args.seed),
                              [runner.Policy("summary", "summary")], 256, directory / "train", args.seed)
            runner.validate_stage(client, directory, protocol)
            runner.test_stage(client, directory, protocol)
            result = runner.verify_evidence(directory)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["verified_steps"], 224)


if __name__ == "__main__":
    unittest.main()
