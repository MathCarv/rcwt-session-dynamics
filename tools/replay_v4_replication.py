"""Show the first recorded R1 episode pair after complete offline verification."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
POLICIES = ("summary", "structured")


def verify_run(directory: Path) -> dict:
    from rcwt_replication_v4 import verify_run as verify
    return verify(directory)


def _snapshot(directory: Path) -> dict:
    from verify_v4_replication_report import _snapshot as snapshot
    return snapshot(directory)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _calls(row: dict) -> list[dict]:
    return [event["result"] for event in row["client_events"] if event["method"] == "complete"]


def _resources(rows: list[dict]) -> dict:
    calls = [call for row in rows for call in _calls(row)]
    for call in calls:
        for key in ("prompt_tokens", "completion_tokens"):
            if type(call[key]) is not int or call[key] < 0:
                raise ValueError("Invalid recorded token count")
        for key in ("wall_seconds", "api_cost_usd"):
            value = call[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid recorded inference time or charge")
        if call["api_cost_usd"] != 0:
            raise ValueError("R1 allows local inference only")
    return {
        "decisions": len(rows), "generation_calls": len(calls),
        "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
        "completion_tokens": sum(call["completion_tokens"] for call in calls),
        "inference_seconds": math.fsum(call["wall_seconds"] for call in calls),
        "sum_step_seconds": math.fsum(row["step_seconds"] for row in rows),
        "tokenizer_operations": sum(event["method"] in {"tokenize", "detokenize"}
                                    for row in rows for event in row["client_events"]),
        "api_cost_usd": 0, "total_monetary_cost_usd": None,
    }


def _actor_calls(row: dict) -> tuple[dict, dict]:
    calls = _calls(row)
    expected = ["draft:" + row["policy"], "action:" + row["policy"]]
    if row["policy"] == "summary" and row["step_index"] < 7:
        expected.append("memory:summary")
    if [call["purpose"] for call in calls] != expected:
        raise ValueError("Require the complete recorded two-pass actor and compaction sequence")
    return calls[0], calls[1]


def render_replay(directory: Path) -> str:
    """Always display manifest index zero; no best-case selection or live requests."""
    directory = Path(directory)
    before = _snapshot(directory)
    verified = verify_run(directory)
    if (not isinstance(verified, dict) or verified.get("status") != "PASS"
            or verified.get("read_only") is not True or verified.get("integrity_only") is not True
            or verified.get("inference_calls") != 0 or verified.get("verified_steps") != 512
            or verified.get("verified_episode_summaries") != 64 or verified.get("generation_calls") != 1248):
        raise ValueError("Require a complete offline-verified R1 before displaying decisions")
    protocol, public = read(directory / "protocol.json"), read(directory / "public.json")
    if (protocol.get("schema") != "rcwt-online-replication/1" or protocol.get("replication_id") != "R1"
            or protocol.get("mode") != "replication" or protocol.get("split") != "test"
            or protocol.get("count") != 32 or protocol.get("dataset_seed") != 2026091210
            or protocol.get("policies") != list(POLICIES) or len(public) != 32):
        raise ValueError("Require the fixed R1 protocol and complete public manifest")
    rows = read_jsonl(directory / "traces.jsonl")
    summaries = read_jsonl(directory / "episodes.jsonl")
    if len(rows) != 512 or len(summaries) != 64:
        raise ValueError("Missing complete-cohort recorded decisions or summaries")
    selected = public[0]
    episode_id = selected["episode_id"]
    if len(selected["steps"]) != 8:
        raise ValueError("Require all eight decisions in the first manifest episode")
    pairs = {}
    for line, row in enumerate(rows, 1):
        if row["episode_id"] != episode_id:
            continue
        key = (row["policy"], row["step_index"])
        if (key in pairs or row["policy"] not in POLICIES or type(row["step_index"]) is not int
                or not 0 <= row["step_index"] < 8 or row["split"] != "test"
                or row["public_step"] != selected["steps"][row["step_index"]]):
            raise ValueError("Duplicate or mismatched first-episode trajectory")
        pairs[key] = (row, line)
    if set(pairs) != {(policy, step) for policy in POLICIES for step in range(8)}:
        raise ValueError("Both complete first-episode trajectories are required")
    output = [
        "RECORDED R1 REPLAY / NO MODEL CALLS",
        f"Offline complete-evidence integrity: PASS. Fixed manifest episode index 0: {episode_id}.",
        "REPLICATION R1 / TEST: a recording of the first pair, not a new run or an improvement verdict.",
        "The first manifest episode is fixed regardless of its score. Every decision in both arms is shown.",
        "All observations, accounts, tools and amounts are fictional. Private grades were not given to the actor.",
        "The raw plan was not executed. Only the final response supplied the attempted action, including invalid or unsafe attempts.",
        "Each arm's reference follows its own prior simulated ledger. Interpret gain using the complete report and interval.",
    ]
    for step in range(8):
        output += [f"\nSTEP {step + 1}/8", "PUBLIC OBSERVATIONS AND REQUEST: " + _json(selected["steps"][step])]
        for policy in POLICIES:
            row, line = pairs[(policy, step)]
            draft, final = _actor_calls(row)
            output += [
                "\n" + policy.upper(),
                f"TRACE: traces.jsonl line {line}; sha256={row['sha256']}",
                "STORED MEMORY BEFORE: " + row["memory_before"],
                "EXACT CONTEXT SENT TO ACTOR: " + row["actor_memory_before"],
                "RAW PLAN (NOT EXECUTED): " + draft["text"],
                "PLAN FINISH REASON: " + draft["finish_reason"],
                "RAW FINAL RESPONSE: " + final["text"],
                "FINAL FINISH REASON: " + final["finish_reason"],
                "FINAL SELF-REPORTED EVIDENCE CHECK: " + _json(row["evidence_check"]),
                "ACTUAL ACTION (FINAL ONLY): " + _json(row["action"]),
                "ACTUAL FICTIONAL TOOL RECEIPT: " + _json(row["tool_result"]),
                "PRIVATE EVALUATION: " + _json(row["score"]),
                "RETAINED MEMORY AFTER: " + row["memory_after"],
                f"TOKEN CAPS: stored_after={row['memory_tokens']}/256; actor_context={row['actor_memory_tokens']}/256; memory_truncated={row['memory_truncated']}",
            ]
            for call in _calls(row):
                output.append("RECORDED GENERATION CALL: " + _json({key: call[key] for key in (
                    "purpose", "prompt_tokens", "completion_tokens", "wall_seconds", "api_cost_usd")}))
            output.append("ALL-CALL STEP RESOURCES: " + _json(_resources([row])))
    output.append("\nSELECTED EPISODE TOTALS (BOTH COMPLETE TRAJECTORIES):")
    for policy in POLICIES:
        selected_rows = [pairs[(policy, step)][0] for step in range(8)]
        output.append(policy + ": " + _json({"successes": sum(int(row["score"]["success"]) for row in selected_rows),
                                            **_resources(selected_rows)}))
    output.append("\nWHOLE-R1 COHORT RESOURCE TOTALS (NO EARLIER-ATTEMPT POOLING):")
    for policy in POLICIES:
        arm = [row for row in rows if row["policy"] == policy]
        wall = math.fsum(row["episode_seconds"] for row in summaries if row["policy"] == policy)
        output.append(policy + ": " + _json({**_resources(arm), "accounted_episode_wall_seconds": wall}))
    output += [
        "The preserved 173/512-decision interruption and all development costs are separate from these R1 totals.",
        "API charge US$0; electricity, hardware and total monetary cost remain unknown.",
        "Recorded local timings depend on hardware and load; they are not independent physical attestation.",
        "A single displayed pair does not establish improvement, production safety, new-domain transfer or autonomous learning.",
    ]
    if before != _snapshot(directory):
        raise ValueError("R1 evidence changed during display preparation")
    return "\n".join(output) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, choices=(0,), default=0,
                        help="R1 demonstration is fixed to the first manifest episode")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    print(render_replay(args.run_dir), end="")


if __name__ == "__main__":
    main()
