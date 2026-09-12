"""Require the actual published R1 cohort and preserve its exact evidence bytes.

These tests inspect the completed recording; they never generate a cohort or
call a model. Detailed semantic replay and demonstration are strict CI steps.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/agent_v4_replication"
EXPECTED = {
    "analysis.json": "5841fa137bf2ea45d6257bf8a47cc6f3882816186242345ea4cf5c0ae0634ce1",
    "completion.json": "575283d1ca997218d60caca8eb5900a8156fb066b969e64490e8af1eeb1f7692",
    "episodes.jsonl": "c5295e80e942b06eb74790aab91d38de5303de26c11998d800ed86121cc0e723",
    "freeze.json": "8cadf1b7296addd8400eccfdadfafcadc67a80f0426eeca092b216a9d0762c0c",
    "oracle.json": "1312fa9d65d3409b634e79462d329ef18cdb0157a62b2360078282834d3b490c",
    "protocol.json": "5fdb1fcb9b81bd4fc073d9429c90e199a44688dfbdf9b2bdfedc74f1c9092524",
    "public.json": "d1295b465ecfe5120a84e5dc0790d642cd95ac746088ed4aba8003b88c660c32",
    "registration.md": "0afb3d5b4ce22f27c06a91a963914be409ec61d51ea57f192a9469a0f1282f12",
    "report-started.json": "aed22f3ca7adca0cbcc99470dbc0fe099733fdd18609b836c376e34001a47704",
    "RESULTS.md": "e790f88a839362534e6b776150a03969e267051feede9ba710f84788f50f18fb",
    "schedule.json": "19673db3abafebea623772acdc0c54208d9080519a32a334f18aa748c156952a",
    "started.json": "e71f020ab47e553fa43150a8831e976504e0eed6642008126944ec229c3e1dbe",
    "traces.jsonl": "5d54c01d2911ee42f0f69c47c05ba053d5796912726ff8277d92da8e3ce6b8d9",
    "verification.json": "0c886a8c6b502967814d038231abe0746de50b33f80d8cd5f135f05c1a2c7e5a",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R1ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if digest(DIRECTORY / "protocol.json") != EXPECTED["protocol.json"]:
            raise ValueError("The registered published R1 protocol is missing or changed")
        cls.protocol = read(DIRECTORY / "protocol.json")
        cls.analysis = read(DIRECTORY / "analysis.json")

    def test_all_32_artifacts_are_required_and_byte_locked(self):
        expected = dict(EXPECTED)
        expected.update({"sources/" + Path(name).name: value
                         for name, value in self.protocol["source_sha256"].items()})
        expected.update({"orchestration/" + Path(name).name: value
                         for name, value in self.protocol["orchestrator_sha256"].items()})
        actual = {path.relative_to(DIRECTORY).as_posix(): digest(path)
                  for path in DIRECTORY.rglob("*") if path.is_file()}
        self.assertEqual(len(expected), 32)
        self.assertEqual(actual, expected)
        self.assertEqual(sum(path.stat().st_size for path in DIRECTORY.rglob("*") if path.is_file()), 27571092)

    def test_candidate_orchestration_and_all_147_history_pins_are_portable(self):
        pins = self.protocol["historical_evidence"]["input_sha256"]
        self.assertEqual(len(pins), 147)
        self.assertEqual(len(self.protocol["source_sha256"]), 15)
        self.assertEqual(len(self.protocol["orchestrator_sha256"]), 3)
        for name, expected in pins.items():
            with self.subTest(path=name):
                self.assertFalse(Path(name).is_absolute())
                self.assertNotIn("..", Path(name).parts)
                self.assertNotIn("runtime", Path(name).parts)
                self.assertEqual(digest(ROOT / name), expected)

    def test_complete_cohort_and_fixed_endpoints_remain_published(self):
        protocol = self.protocol
        self.assertEqual((protocol["schema"], protocol["replication_id"], protocol["split"]),
                         ("rcwt-online-replication/1", "R1", "test"))
        self.assertEqual(protocol["dataset_seed"], 2026091210)
        self.assertEqual(protocol["bootstrap_samples"], 10000)
        self.assertFalse(protocol["historical_pooling"])
        for name, count in (("traces.jsonl", 512), ("episodes.jsonl", 64)):
            with (DIRECTORY / name).open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for line in handle if line.strip()), count)
        receipt = read(DIRECTORY / "verification.json")
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(receipt["integrity_only"])
        self.assertEqual(receipt["generation_calls"], 1248)
        self.assertEqual(receipt["inference_calls"], 0)

    def test_interrupted_history_is_retained_unpooled_and_not_a_complete_result(self):
        old = self.protocol["historical_evidence"]["interrupted"]
        self.assertEqual(old["path"], "results/agent_v4_interrupted")
        receipt = old["verification"]
        self.assertEqual(receipt["status"], "PARTIAL_VERIFIED")
        self.assertEqual(receipt["verified_steps"], 173)
        self.assertEqual(receipt["gain"], "NOT_EVALUATED")
        self.assertEqual(receipt["recorded_completed_step_resources"]["generation_calls"], 421)
        self.assertFalse((ROOT / old["path"] / "completion.json").exists())


if __name__ == "__main__":
    unittest.main()
