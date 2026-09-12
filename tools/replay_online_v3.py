"""Recorded paired-agent demonstration; no inference or external tools."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcwt_agent_run import read, read_jsonl
from rcwt_agent_actor import extract_action, extract_evidence_check
from rcwt_local_model import canonical_hash
from rcwt_online_v3 import POLICIES, verify_run


def replay(directory: Path, index: int = 0, show_memory: bool = False) -> None:
    verified = verify_run(directory)
    if not isinstance(verified, dict) or verified.get("status") != "PASS":
        raise ValueError("A verified complete run is required before demonstrating recorded decisions")
    public = read(directory / "public.json")
    if type(index) is not int or not 0 <= index < len(public):
        raise ValueError("Episode index must exist in the frozen manifest")
    episode_id = public[index]["episode_id"]
    rows = read_jsonl(directory / "traces.jsonl")
    pairs = {(r["policy"], r["step_index"]): r for r in rows if r["episode_id"] == episode_id}
    selections = {}
    for key, row in pairs.items():
        policy = row["policy"]
        calls = [event["result"] for event in row["client_events"] if event["method"] == "complete"]
        finals = [(position, call) for position, call in enumerate(calls) if call.get("purpose") == f"action:{policy}"]
        drafts = [(position, call) for position, call in enumerate(calls) if call.get("purpose") == f"draft:{policy}"]
        if len(finals) != 1 or len(drafts) > 1 or (drafts and drafts[0][0] >= finals[0][0]):
            raise ValueError("Require exactly one final action completion and at most one preceding draft")
        final = finals[0][1]
        if (row["action"] != extract_action(final["text"], final["finish_reason"])
                or canonical_hash(row["evidence_check"]) != canonical_hash(extract_evidence_check(final["text"], final["finish_reason"]))):
            raise ValueError("Displayed action or check is not bound to the final raw completion")
        selections[key] = (final, drafts[0][1] if drafts else None)
    print("RECORDED PAIRED REPLAY / NO MODEL CALLS")
    print(f"Offline verification: {verified['status']}; episode index {index}; {episode_id}")
    print("Fictional tools and money. The default is the FIRST manifest episode, not a best-case example.")
    print("Private evaluation shown below was never provided to the actor or its memory reducer.")
    print("Plans and legacy JSON drafts, when present, were NOT executed. Only the final action:policy completion was submitted to the simulator.")
    for step in range(len(public[index]["steps"])):
        print(f"\nSTEP {step + 1}/8")
        print("PUBLIC OBSERVATIONS AND REQUEST:")
        print(json.dumps(pairs[("summary", step)]["public_step"], ensure_ascii=False, indent=2))
        for policy in POLICIES:
            row = pairs[(policy, step)]
            final, draft = selections[(policy, step)]
            print(f"\n{policy.upper()}")
            if show_memory:
                print("STORED MEMORY BEFORE: " + row["memory_before"])
                print("MEMORY VIEW SENT TO ACTOR: " + row.get("actor_memory_before", row["memory_before"]))
            if draft is not None:
                if "response_format" not in draft.get("request", {}):
                    print("PLAN OUTPUT (NOT EXECUTED; NO TOOL PARSING): " + draft["text"])
                    print("PLAN FINISH REASON (NOT A FINAL-ACTION GRADE): " + draft["finish_reason"])
                else:
                    print("DRAFT OUTPUT (NOT EXECUTED): " + draft["text"])
                    print("DRAFT PROPOSED ACTION (NOT EXECUTED): " + json.dumps(extract_action(draft["text"], draft["finish_reason"]), ensure_ascii=False))
            print("FINAL ACTOR SELF-REPORTED CHECK: " + json.dumps(row["evidence_check"], ensure_ascii=False))
            print("ACTUAL ACTION (FINAL): " + json.dumps(row["action"], ensure_ascii=False))
            print("ACTUAL FICTIONAL TOOL RECEIPT: " + json.dumps(row["tool_result"], ensure_ascii=False))
            print("PRIVATE EVALUATION: " + json.dumps(row["score"], ensure_ascii=False))
            calls = [e["result"] for e in row["client_events"] if e["method"] == "complete"]
            print(f"RECORDED: {row['step_seconds']:.3f}s full step; "
                  f"{sum(c['prompt_tokens'] + c['completion_tokens'] for c in calls)} model tokens; "
                  f"{len(calls)} generation calls; {row['memory_tokens']}/256 memory tokens")
            if show_memory:
                print("RETAINED MEMORY AFTER: " + row["memory_after"])
    print("\nA single episode is not proof of general gain; use the complete paired aggregate and its interval.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/agent_v3"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--show-memory", action="store_true")
    args = parser.parse_args()
    replay(args.run_dir, args.episode_index, args.show_memory)


if __name__ == "__main__":
    main()
