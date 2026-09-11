"""Lock the released RCWT-S confirmatory evidence to its frozen protocol."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "session_v1"
SOURCE_COMMIT = "6153fbe9d154483346abbecde52b9172b2fbde07"
EXPECTED_HASHES = {
    "public_cases.jsonl": "d44cdfb9bf11043b97c0ed3e023545e48ef69a08456b1f090b618cd77571bd5a",
    "oracle_cases.jsonl": "9a5dcdb08e049b60f7ab1a4a222d0505b02509088a6260299f6dae23de087d09",
    "contexts.jsonl": "07514a0fa27bd863de5517de7468e09b81f9ee3aace0825ed8b9c761ad0e4b28",
    "manifest.json": "587cc15307575fd9f9e9c2b4d9fdc0d6c36a890e04635626f82f3eff2006a6af",
    "aggregates.json": "6637e468a6185c021012d04c93bea229b5d8fab4691520bd9bbba6c876733429",
    "decision_readiness.svg": "50a6bfaf07c9ee058eedb64a652438014cbf2f37c60fdb1a1849487e2098cb21",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SessionReleaseEvidenceTests(unittest.TestCase):
    """Prevent silent drift in released inputs, outputs, or headline claims."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest: dict[str, Any] = json.loads(
            (RESULTS / "manifest.json").read_text(encoding="utf-8")
        )
        cls.aggregate: dict[str, Any] = json.loads(
            (RESULTS / "aggregates.json").read_text(encoding="utf-8")
        )

    def test_all_release_artifacts_are_byte_locked(self) -> None:
        for name, expected in EXPECTED_HASHES.items():
            with self.subTest(artifact=name):
                path = RESULTS / name
                self.assertTrue(path.is_file())
                self.assertEqual(_sha256(path), expected)

    def test_manifest_records_clean_source_and_complete_matrix(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["protocol"], "session-v1")
        self.assertEqual(manifest["source_commit"], SOURCE_COMMIT)
        self.assertEqual(manifest["source_state"], "clean")
        self.assertEqual(manifest["oracle_derivation"], "independent-gold-v1")
        self.assertEqual(
            manifest["counts"],
            {
                "checkpoints": 256,
                "contexts": 2304,
                "events": 4096,
                "oracle_records": 256,
                "public_cases": 64,
            },
        )
        for name in ("public_cases", "oracle_cases", "contexts"):
            artifact = manifest["artifacts"][name]
            self.assertEqual(
                artifact["sha256"], EXPECTED_HASHES[f"{name}.jsonl"]
            )

    def test_scorer_rederived_every_release_record(self) -> None:
        validity = self.aggregate["validity"]
        self.assertEqual(validity["status"], "PASS")
        self.assertTrue(validity["valid"])
        self.assertTrue(validity["public_source_validation"])
        self.assertTrue(validity["treatment_matrix_complete"])
        for key in (
            "context_hashes_rederived",
            "token_counts_rederived",
            "rendered_event_lists_rederived",
            "public_source_records_checked",
        ):
            self.assertEqual(validity[key], 2304)
        self.assertEqual(validity["source_hashes_checked"], 64)
        self.assertEqual(
            validity["input_sha256"],
            {
                "contexts": EXPECTED_HASHES["contexts.jsonl"],
                "manifest": EXPECTED_HASHES["manifest.json"],
                "oracle": EXPECTED_HASHES["oracle_cases.jsonl"],
                "public_cases": EXPECTED_HASHES["public_cases.jsonl"],
            },
        )

    def test_headline_rates_and_paired_effect_are_exact(self) -> None:
        by_treatment = {
            row["treatment"]: row
            for row in self.aggregate["breakdowns"]["treatment"]
        }
        expected = {
            "state_latest": (0.8333333333, [0.7903645833, 0.8763020833]),
            "tail": (0.5416666667, [0.4700520833, 0.61328125]),
            "state_first": (0.375, [0.28515625, 0.46875]),
        }
        for treatment, (rate, interval) in expected.items():
            with self.subTest(treatment=treatment):
                self.assertEqual(by_treatment[treatment]["decision_ready_rate"], rate)
                self.assertEqual(by_treatment[treatment]["decision_ready_ci95"], interval)

        effect = next(
            row
            for row in self.aggregate["paired_deltas"]
            if row["scope"] == "overall"
            and row["comparison"] == "state_latest-tail"
            and row["metric"] == "decision_ready"
        )
        self.assertEqual(effect["delta"], 0.2916666667)
        self.assertEqual(effect["bootstrap95"], [0.24609375, 0.3359375])
        self.assertEqual(effect["n_clusters"], 64)

    def test_preregistered_claim_and_controls_pass(self) -> None:
        checks = self.aggregate["preregistered_checks"]
        self.assertTrue(checks["main_claim_supported"])
        self.assertEqual(checks["interpretation"], "supported")
        self.assertTrue(checks["h1"]["pass"])
        self.assertEqual(
            checks["h1"]["paired_delta"]["bootstrap95"],
            [0.3854166667, 0.4427083333],
        )
        self.assertTrue(checks["h2"]["pass"])
        self.assertTrue(checks["h5"]["pass"])
        self.assertEqual(checks["h5"]["fabricated_sufficiency_violations"], 0)
        self.assertEqual(checks["h5"]["insufficient_contexts"], 576)
        self.assertTrue(checks["chronology_noninferiority"]["pass"])

    def test_only_latest_state_reaches_b95(self) -> None:
        b95 = {
            row["treatment"]: row["b95"]
            for row in self.aggregate["b95"]["overall"]
        }
        self.assertEqual(
            b95,
            {"state_first": None, "state_latest": 1024, "tail": None},
        )


if __name__ == "__main__":
    unittest.main()
