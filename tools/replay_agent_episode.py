"""Replay a verified recorded episode without starting a model or server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcwt_agent_run import read, read_jsonl, verify_evidence  # noqa: E402

POLICIES = ("summary", "learned", "tail")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def replay_episode(run_dir: Path, episode_index: int = 0, policy: str = "summary",
                   show_memory: bool = False, output: TextIO | None = None) -> dict[str, Any]:
    """Render the manifest-selected episode only after complete offline verification.

    Index zero means the first episode in the frozen public test manifest. No
    search for the best result, generation, sleep or re-timing is performed.
    """
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("episode_index must be a nonnegative integer")
    if policy not in POLICIES:
        raise ValueError("policy must be summary, learned or tail")
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "test-freeze.json").is_file():
        raise ValueError("A held-out freeze is required before replay; the run is not ready")
    verification = verify_evidence(run_dir)
    if verification.get("status") != "PASS":
        raise ValueError("Offline evidence verification did not pass; no replay was rendered")
    protocol = read(run_dir / "protocol.json")
    manifest = read(run_dir / "test-public.json")
    if episode_index >= len(manifest):
        raise ValueError(f"episode_index out of range: the frozen test manifest has {len(manifest)} episodes")
    episode = manifest[episode_index]
    episode_id = episode["episode_id"]
    rows = [row for row in read_jsonl(run_dir / "test/traces.jsonl")
            if row["episode_id"] == episode_id and row["policy"] == policy]
    summaries = [row for row in read_jsonl(run_dir / "test/episodes.jsonl")
                 if row["episode_id"] == episode_id and row["policy"] == policy]
    if len(rows) != len(episode["steps"]) or len(summaries) != 1:
        raise ValueError("Verified episode evidence is incomplete")
    freeze = read(run_dir / "test-freeze.json")
    stream = output if output is not None else sys.stdout

    def emit(text: str = "") -> None:
        print(text, file=stream)

    emit("RECORDED REPLAY / NO MODEL CALLS")
    emit("All money and bookings are fictional. Output below is saved experimental evidence.")
    emit(f"Offline evidence verification: PASS ({verification['verified_steps']} recorded steps verified)")
    emit(f"Model: {protocol['model']}")
    emit(f"Frozen test time: {freeze['frozen_at_utc']}")
    emit(f"Manifest episode index: {episode_index} | Episode: {episode_id} | Family: {episode['family']}")
    emit(f"Recorded policy arm: {policy} | Stored-memory budget: {protocol['memory_budget']} tokens")
    if policy == "learned":
        emit("Frozen validation-selected policy: " + compact_json(freeze["selected"]))
    emit("Latency and token counts below are recorded values, not newly measured performance.")
    for row in rows:
        public = row["public_step"]
        task = public["task"]
        emit()
        emit(f"STEP {row['step_index'] + 1}/{len(rows)} (recorded step_index={row['step_index']})")
        emit(f"Request: {task['request_id']} | Case: {task['case_id']} | Operation: {task['operation']}")
        emit("New public observations:")
        for observation in public["observations"]:
            source = observation.get("tool", observation["source"])
            emit(f"  {observation['event_id']} [{source}] " + compact_json(observation["content"]))
        emit("Model self-check (self-reported; not reference truth): " + compact_json(row["evidence_check"]))
        emit("Actual extracted tool call: " + row["action"])
        emit("Actual simulated tool receipt: " + compact_json(row["tool_result"]))
        state = row["state_after"]
        emit("Fictional ledger totals (cents): " + compact_json({
            "payout_cents": state["payout_cents"], "refund_cents": state["refund_cents"],
            "unsafe_booked_cents": state["unsafe_booked_cents"],
            "completed_operations": len(state["completed"]),
        }))
        emit("PRIVATE EVALUATION - not supplied to the acting model or memory compressor:")
        emit("  " + compact_json(row["score"]))
        calls = row["model_calls"]
        emit(f"Recorded resources: calls={len(calls)}; prompt_tokens={sum(c['prompt_tokens'] for c in calls)}; "
             f"completion_tokens={sum(c['completion_tokens'] for c in calls)}; "
             f"decision_seconds={calls[0]['wall_seconds']:.6f}; step_seconds={row['step_seconds']:.6f}")
        emit(f"Stored memory after step: {row['memory_tokens']} tokens; truncated={str(row['memory_truncated']).lower()}")
        if show_memory:
            emit("Retained memory BEFORE: " + compact_json(row["memory_before"]))
            emit("Retained memory AFTER: " + compact_json(row["memory_after"]))
    summary = summaries[0]
    emit()
    emit("RECORDED EPISODE SUMMARY")
    emit(compact_json({key: summary[key] for key in (
        "episode_id", "policy", "steps", "successes", "failures", "valid_actions",
        "unsafe_actions", "unsafe_booked_cents", "prompt_tokens", "completion_tokens",
        "model_calls", "episode_seconds", "last_trace_sha256",
    )}))
    emit("API charge was zero under the local protocol; electricity and total monetary cost are unknown.")
    emit("Replay complete. No inference was performed and no result was changed.")
    return {"episode_id": episode_id, "policy": policy, "steps": len(rows),
            "verification": verification, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/agent_v2")
    parser.add_argument("--episode-index", type=int, default=0,
                        help="Zero-based index in the frozen public test manifest; default is the first episode")
    parser.add_argument("--policy", choices=POLICIES, default="summary")
    parser.add_argument("--show-memory", action="store_true")
    args = parser.parse_args(argv)
    try:
        replay_episode(args.run_dir, args.episode_index, args.policy, args.show_memory)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Replay unavailable: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
