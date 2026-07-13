"""Regression tests for RCWT scoring and token accounting."""

from __future__ import annotations

import json
import unittest

from rcwt_intact_scoring import score_response
from rcwt_token_accounting import fixed_budget_allocation


class IntactScoringTests(unittest.TestCase):
    """Ensure deterministic scoring is isolated by response field."""

    def test_does_not_credit_value_from_another_key(self) -> None:
        response = json.dumps(
            {
                "pg_limit": "unknown",
                "redis_pubsub_rate": "100K messages/sec; unrelated note: 8KB",
            }
        )

        scores = score_response(response)

        self.assertEqual(scores["pg_limit"], 0)
        self.assertEqual(scores["redis_pubsub_rate"], 1)


class TokenAccountingTests(unittest.TestCase):
    """Lock the historical q-to-p conversion used by the paper."""

    def test_main_q90_allocation(self) -> None:
        allocation = fixed_budget_allocation(4096, 0.90)

        self.assertEqual(allocation.coordination_tokens, 3383)
        self.assertEqual(allocation.reference_tokens, 376)
        self.assertEqual(allocation.residual_task_tokens, 713)
        self.assertAlmostEqual(allocation.coordination_share, 3383 / 4096)


if __name__ == "__main__":
    unittest.main()
