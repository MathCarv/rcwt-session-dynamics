import unittest
from unittest.mock import patch

from rcwt_local_model import LocalModelClient, LocalModelError, canonical_hash, validate_endpoint


class LocalClientTests(unittest.TestCase):
    def test_remote_and_ambiguous_endpoints_rejected(self):
        for endpoint in ("https://api.openai.com", "http://localhost:1234", "http://192.168.1.5:1234",
                         "http://127.0.0.1:1234/v1", "http://u:p@127.0.0.1:1234",
                         "http://127.0.0.1:1234?key=oops", "https://127.0.0.1:1234"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_endpoint(endpoint)
        self.assertEqual(validate_endpoint("http://127.0.0.1:18085/"), "http://127.0.0.1:18085")
        self.assertEqual(validate_endpoint("http://[::1]:8080"), "http://[::1]:8080")

    def test_usage_is_provider_reported_not_estimated(self):
        response = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 18, "completion_tokens": 2}, "timings": {"predicted_ms": 12}}
        client = LocalModelClient()
        with patch.object(client, "_request", return_value=response) as request:
            result = client.complete([{"role": "user", "content": "hi"}], 20, purpose="test")
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (18, 2))
        self.assertEqual(result.api_cost_usd, 0)
        self.assertFalse(request.call_args.args[1]["cache_prompt"])
        self.assertNotIn("api_key", request.call_args.args[1])

    def test_no_silent_fallback_when_usage_absent(self):
        client = LocalModelClient()
        with patch.object(client, "_request", return_value={"choices": [{"message": {"content": "{}"}}]}):
            with self.assertRaises(LocalModelError):
                client.complete([], 20)

    def test_hash_order_independent(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
