"""Display a verified recorded v4 episode pair; never run a model or repair actions."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from rcwt_agent_actor import extract_action, extract_evidence_check

POLICIES = ("summary", "structured")


def verify_archive(directory: Path) -> dict:
    """Use the versioned archive gate, including archived sources, not live replay."""
    from verify_online_v4_archive import verify_archive as verify
    return verify(directory)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _calls(row: dict) -> list[dict]:
    return [event["result"] for event in row["client_events"] if event["method"] == "complete"]


def _resources(rows: list[dict]) -> dict:
    calls = [call for row in rows for call in _calls(row)]
    for call in calls:
        for key in ("prompt_tokens", "completion_tokens"):
            if type(call[key]) is not int or call[key] < 0:
                raise ValueError("Invalid recorded generation token count")
        for key in ("wall_seconds", "api_cost_usd"):
            value = call[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid recorded generation cost or duration")
        if call["api_cost_usd"] != 0:
            raise ValueError("Expected exclusively local generation with zero API charge")
    for row in rows:
        seconds = row["step_seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid recorded step duration")
    return {
        "steps": len(rows), "generation_calls": len(calls),
        "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
        "completion_tokens": sum(call["completion_tokens"] for call in calls),
        "model_tokens": sum(call["prompt_tokens"] + call["completion_tokens"] for call in calls),
        "inference_seconds": sum(call["wall_seconds"] for call in calls),
        "sum_step_seconds": sum(row["step_seconds"] for row in rows),
        "tokenizer_operations": sum(event["method"] in {"tokenize", "detokenize"}
                                    for row in rows for event in row["client_events"]),
        "api_cost_usd": sum(call["api_cost_usd"] for call in calls),
        "total_monetary_cost_usd": None,
    }


def _selected_calls(row: dict) -> tuple[dict, dict, dict]:
    calls = _calls(row)
    policy = row["policy"]
    drafts = [(i, call) for i, call in enumerate(calls) if call.get("purpose") == f"draft:{policy}"]
    finals = [(i, call) for i, call in enumerate(calls) if call.get("purpose") == f"action:{policy}"]
    if len(drafts) != 1 or len(finals) != 1 or drafts[0][0] >= finals[0][0]:
        raise ValueError("V4 requires one plan followed by exactly one final action completion")
    draft, final = drafts[0][1], finals[0][1]
    if "response_format" in draft.get("request", {}):
        raise ValueError("V4 first pass must be a schema-free text plan")
    if (row["action"] != extract_action(final["text"], final["finish_reason"])
            or _json(row["evidence_check"]) != _json(extract_evidence_check(final["text"], final["finish_reason"]))):
        raise ValueError("Displayed final action or check is not bound to its raw completion")
    try:
        schema = final["request"]["response_format"]["json_schema"]["schema"]
        arguments = schema["properties"]["arguments"]["properties"]
        task = row["public_step"]["task"]
        vocabulary = ["hold", "ask_info", "approve" if task["operation"] == "payout" else "refund"]
        if arguments["case_id"]["const"] != task["case_id"] or arguments["decision"]["enum"] != vocabulary:
            raise ValueError("Displayed final schema is not bound to the public task")
        schema_excerpt = {key: arguments[key] for key in ("case_id", "decision", "reason_code", "amount_cents")}
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing recorded task-bound final generation schema") from exc
    return draft, final, schema_excerpt


def render_replay(directory: Path, index: int = 0) -> str:
    """Verify first, then render both complete trajectories for a manifest index.

    This function reads completed recorded artifacts only. It never opens the
    private oracle, generates a corpus, calls a model, or modifies an artifact.
    The archive verifier owns full transcript/source integrity validation.
    """
    verified = verify_archive(directory)
    if (not isinstance(verified, dict) or verified.get("status") != "PASS"
            or verified.get("inference_calls") != 0):
        raise ValueError("A verified complete offline archive is required before displaying decisions")
    protocol = read(directory / "protocol.json")
    expected_split = {"development": "train", "confirmatory": "test"}
    if (protocol.get("schema") != "rcwt-online-memory/4"
            or protocol.get("mode") not in expected_split
            or protocol.get("split") != expected_split[protocol["mode"]]
            or protocol.get("policies") != list(POLICIES)):
        raise ValueError("Require a v4 protocol with explicit matching mode and split")
    public = read(directory / "public.json")
    if not isinstance(public, list) or len(public) != protocol["count"]:
        raise ValueError("Public manifest disagrees with protocol episode count")
    if type(index) is not int or not 0 <= index < len(public):
        raise ValueError("Episode index must exist in the frozen public manifest")
    selected = public[index]
    episode_id = selected["episode_id"]
    rows = read_jsonl(directory / "traces.jsonl")
    manifest = {episode["episode_id"]: episode for episode in public}
    if len(manifest) != len(public):
        raise ValueError("Duplicate public episode identity")
    pairs, line_numbers = {}, {}
    for line, row in enumerate(rows, 1):
        key = (row["episode_id"], row["policy"], row["step_index"])
        episode = manifest.get(row["episode_id"])
        if (key in pairs or episode is None or row["policy"] not in POLICIES
                or type(row["step_index"]) is not int
                or not 0 <= row["step_index"] < len(episode["steps"])
                or row["split"] != protocol["split"]
                or _json(row["public_step"]) != _json(episode["steps"][row["step_index"]])):
            raise ValueError("Duplicate, mismatched, or unpaired recorded trajectory")
        pairs[key], line_numbers[key] = row, line
    expected = {(episode["episode_id"], policy, step)
                for episode in public for policy in POLICIES for step in range(len(episode["steps"]))}
    if set(pairs) != expected:
        raise ValueError("Complete paired trajectories are required")
    selections = {key: _selected_calls(row) for key, row in pairs.items() if key[0] == episode_id}
    run_resources = {policy: _resources([row for row in rows if row["policy"] == policy]) for policy in POLICIES}

    output = [
        "RECORDED V4 PAIRED REPLAY / NO MODEL CALLS",
        f"Offline archive verification: PASS; episode index {index}; {episode_id}",
        f"Protocol: {protocol['schema']}; mode={protocol['mode']}; split={protocol['split']}; manifest episodes={len(public)}.",
        ("DEVELOPMENT / TRAIN: this is not a new held-out result."
         if protocol["mode"] == "development" else
         "CONFIRMATORY / TEST: this is a recorded episode, not a new run or new evaluation."),
        "Fictional tools and money. Default selection is the FIRST manifest episode, never the best-scoring episode.",
        "Both arms use the same two-pass actor and public task-bound schema; schema eligibility is NOT computed.",
        "Structured context combines retained and current PUBLIC facts. It is transient, not additional persistent memory.",
        "Private grades shown below were never fed to the actor, context builder, or memory writer.",
        "The raw plan is unexecuted, even if it resembles a tool call; only the final action:policy response was executed.",
    ]
    for step_index in range(len(selected["steps"])):
        output.extend([f"\nSTEP {step_index + 1}/{len(selected['steps'])}",
                       "PUBLIC OBSERVATIONS AND REQUEST: " + _json(selected["steps"][step_index])])
        for policy in POLICIES:
            key = (episode_id, policy, step_index)
            row = pairs[key]
            draft, final, schema_excerpt = selections[key]
            output.extend([
                f"\n{policy.upper()}",
                f"TRACE: traces.jsonl line {line_numbers[key]}; sha256={row.get('sha256', 'not supplied')}",
                "STORED MEMORY BEFORE: " + row["memory_before"],
                "EXACT CONTEXT SENT TO ACTOR: " + row["actor_memory_before"],
                "RAW PLAN (NOT EXECUTED; NO TOOL PARSING): " + draft["text"],
                "PLAN FINISH REASON (NOT FINAL GRADE): " + draft["finish_reason"],
                "RECORDED TASK_SCHEMA ARGUMENT CONSTRAINTS (PUBLIC ID/OPERATION; NOT ELIGIBILITY): " + _json(schema_excerpt),
                "RAW FINAL RESPONSE: " + final["text"],
                "FINAL FINISH REASON: " + final["finish_reason"],
                "FINAL ACTOR SELF-REPORTED CHECK: " + _json(row["evidence_check"]),
                "ACTUAL ACTION (FINAL ONLY): " + _json(row["action"]),
                "ACTUAL FICTIONAL TOOL RECEIPT: " + _json(row["tool_result"]),
                "PRIVATE EVALUATION: " + _json(row["score"]),
            ])
            for position, call in enumerate(_calls(row), 1):
                output.append("RECORDED GENERATION CALL: " + _json({
                    "position": position, "purpose": call["purpose"],
                    "prompt_tokens": call["prompt_tokens"], "completion_tokens": call["completion_tokens"],
                    "wall_seconds": call["wall_seconds"], "api_cost_usd": call["api_cost_usd"]}))
            output.extend([
                "ALL-CALL STEP RESOURCES: " + _json(_resources([row])),
                f"TOKEN CAPS: stored_after={row['memory_tokens']}/{protocol['memory_budget']}; "
                f"actor_context={row['actor_memory_tokens']}/{protocol['memory_budget']}; memory_evicted_or_truncated={row['memory_truncated']}",
                "RETAINED MEMORY AFTER: " + row["memory_after"],
            ])
    output.append("\nSELECTED EPISODE TOTALS (BOTH COMPLETE TRAJECTORIES; ALL GENERATION CALLS):")
    for policy in POLICIES:
        chosen = [pairs[(episode_id, policy, step)] for step in range(len(selected["steps"]))]
        output.append(policy + ": " + _json({"successes": sum(int(row["score"]["success"]) for row in chosen),
                                            **_resources(chosen)}))
    output.append("\nWHOLE-RUN RESOURCE TOTALS BY ARM (COST ONLY; NO CROSS-ATTEMPT POOLING):")
    output.extend(policy + ": " + _json(run_resources[policy]) for policy in POLICIES)
    output.extend([
        "API charge zero does not mean zero electricity, hardware, or total monetary cost; total cost is unmeasured.",
        "sum_step_seconds includes recorded per-step processing, not complete wall-clock runtime or startup overhead.",
        "Token counts and times are recorded local-runtime claims, not independent physical attestation; latency depends on hardware.",
        "A single episode is not proof of general gain. Use the complete paired-episode analysis and its uncertainty interval.",
        "This replay makes no causal memory-failure, production-safety, cross-model, or cross-domain claim.",
    ])
    return "\n".join(output) + "\n"


def replay(directory: Path, index: int = 0) -> None:
    print(render_replay(directory, index), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/agent_v4"))
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()
    replay(args.run_dir, args.episode_index)


if __name__ == "__main__":
    main()
