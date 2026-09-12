"""Offline R2 server/trace accounting, separate from quality and custody.

The parser follows the pinned b10809 single-slot lifecycle already used for
R1. This new auditor does not modify or parameterize the immutable R1 auditor.
Require a stable, complete log from the separately closed owned R2 runtime.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

TOLERANCE_MS = Decimal("0.01")
SLOT = re.compile(r"^\d+(?:\.\d+){3}\s+[IWEFD]\s+slot\s+(?P<event>\w+):\s+id\s+(?P<slot>\d+)\s+\|\s+task\s+(?P<task>-?\d+)\s+\|\s*(?P<body>.*)$")
NUMBER = r"\d+(?:\.\d+)?"
PROMPT = re.compile(rf"prompt eval time\s*=\s*({NUMBER}) ms\s*/\s*(\d+) tokens\s+\([^\r\n]*\)")
EVAL = re.compile(rf"eval time\s*=\s*({NUMBER}) ms\s*/\s*(\d+) (?:tokens|runs)\s+\([^\r\n]*\)")
TOTAL = re.compile(rf"total time\s*=\s*({NUMBER}) ms\s*/\s*(\d+) tokens")
RELEASE = re.compile(r"stop processing: n_tokens = \d+, truncated = [01]")
PROGRESS = re.compile(rf"n_gen\s*=\s*\d+, tg\s*=\s*{NUMBER} t/s, tg_3s\s*=\s*{NUMBER} t/s")
GRAPHS = re.compile(r"graphs reused\s*=\s*\d+")


def parse_server_log(data: bytes, expected_calls: int) -> list[dict]:
    """Fail on extra, canceled, incomplete, overlapping or ambiguous tasks."""
    if type(expected_calls) is not int or expected_calls <= 0:
        raise ValueError("Require an exact positive expected generation count")
    text = data.decode("utf-8", errors="strict")
    if not text.endswith("\n"):
        raise ValueError("Incomplete last server-log line")
    calls, active, seen = [], None, set()
    loaded = listening = configured = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if re.search(r"\bcancel(?:led|ed|ing|lation)?\b", line, re.I):
            raise ValueError("Canceled task in a supposedly complete session")
        if re.search(r"\bsrv\s+llama_server:\s+model loaded$", line):
            loaded += 1
        if re.search(r"\bsrv\s+llama_server:\s+listening on http://127\.0\.0\.1:18085$", line):
            listening += 1
        if re.search(r"\bsrv\s+load_model:\s+initializing, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'$", line):
            configured += 1
        match = SLOT.fullmatch(line)
        if not match:
            if any(marker in line for marker in ("launch_slot_", "processing task", "prompt eval time", "eval time =", "total time =", "stop processing:", "id_task")):
                raise ValueError(f"Unparseable task accounting at line {line_number}")
            continue
        event, body = match["event"], match["body"].strip()
        slot, task = int(match["slot"]), int(match["task"])
        if slot != 0:
            raise ValueError("Unexpected second slot")
        if event == "get_availabl" and task == -1 and body.startswith("selected slot by LRU,"):
            continue
        if event == "launch_slot_":
            if body != "processing task, is_child = 0" or task < 0 or task in seen or active is not None:
                raise ValueError("Duplicate, child, overlapping or malformed task launch")
            if (loaded, listening, configured) != (1, 1, 1):
                raise ValueError("Exactly one pinned startup must precede generation")
            seen.add(task)
            active = {"slot_id": slot, "task_id": task, "launch_line": line_number}
            continue
        if active is None or active["task_id"] != task:
            raise ValueError("Unmatched task lifecycle record")
        if event == "print_timing":
            prompt, evaluated, total = PROMPT.fullmatch(body), EVAL.fullmatch(body), TOTAL.fullmatch(body)
            if prompt:
                if "prompt_tokens" in active:
                    raise ValueError("Duplicate final prompt timing")
                active.update(prompt_tokens=int(prompt[2]), prompt_ms=prompt[1], prompt_line=line_number)
            elif evaluated:
                if "prompt_tokens" not in active or "completion_tokens" in active:
                    raise ValueError("Unmatched or duplicate final evaluation timing")
                active.update(completion_tokens=int(evaluated[2]), predicted_ms=evaluated[1], eval_line=line_number)
            elif total:
                if "completion_tokens" not in active or "total_ms" in active:
                    raise ValueError("Unmatched or duplicate total timing")
                if int(total[2]) != active["prompt_tokens"] + active["completion_tokens"]:
                    raise ValueError("Final total tokens differ")
                if abs(Decimal(total[1]) - Decimal(active["prompt_ms"]) - Decimal(active["predicted_ms"])) > TOLERANCE_MS:
                    raise ValueError("Final total timing exceeds rounding tolerance")
                active.update(total_ms=total[1], total_line=line_number)
            elif PROGRESS.fullmatch(body):
                if "prompt_tokens" in active:
                    raise ValueError("Generation progress after final timings")
            elif GRAPHS.fullmatch(body):
                if "total_ms" not in active:
                    raise ValueError("Graph summary before final timings")
            else:
                raise ValueError("Unparseable or ambiguous final timing")
        elif event == "release":
            if not RELEASE.fullmatch(body) or "total_ms" not in active:
                raise ValueError("Task released without all final timings")
            # The final consumed KV count may be one below generated+prompt.
            active["release_line"] = line_number
            calls.append(active)
            active = None
        else:
            raise ValueError("Unrecognized task lifecycle event")
    if active is not None or (loaded, listening, configured) != (1, 1, 1):
        raise ValueError("Incomplete task or repeated/incomplete runtime startup")
    if len(seen) != expected_calls or len(calls) != expected_calls:
        raise ValueError("Server starts/completions differ from the exact registered count")
    return calls


def _positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError("Invalid positive integer: " + label)
    return value


def _decimal(value, label):
    if type(value) not in (int, float):
        raise ValueError("Invalid metered number: " + label)
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("Invalid finite nonnegative number: " + label)
    return parsed


def compare_calls(trace_calls: list[dict], server_calls: list[dict]) -> dict:
    if not trace_calls or len(trace_calls) != len(server_calls):
        raise ValueError("Cannot omit or add calls in ordered reconciliation")
    pairs = []
    prompt_total = completion_total = 0
    server_ms = client_seconds = maximum_delta = Decimal(0)
    for index, (call, observed) in enumerate(zip(trace_calls, server_calls)):
        prompt = _positive(call["prompt_tokens"], "prompt tokens")
        completion = _positive(call["completion_tokens"], "completion tokens")
        if (prompt, completion) != (observed["prompt_tokens"], observed["completion_tokens"]):
            raise ValueError(f"Server/trace token mismatch at call {index}")
        timing = call["timings"]
        if (_positive(timing["prompt_n"], "prompt_n") != prompt
                or _positive(timing["predicted_n"], "predicted_n") != completion
                or type(timing.get("cache_n")) is not int or timing["cache_n"] != 0
                or _decimal(call["api_cost_usd"], "API charge") != 0):
            raise ValueError("Nonlocal, cached or inconsistent token accounting")
        deltas = [abs(_decimal(timing[name], name) - Decimal(observed[name]))
                  for name in ("prompt_ms", "predicted_ms")]
        if max(deltas) > TOLERANCE_MS:
            raise ValueError(f"Server/trace timing mismatch at call {index}")
        maximum_delta = max(maximum_delta, *deltas)
        prompt_total += prompt
        completion_total += completion
        server_ms += Decimal(observed["prompt_ms"]) + Decimal(observed["predicted_ms"])
        client_seconds += _decimal(call["wall_seconds"], "client wall time")
        pairs.append({"index": index, "purpose": call["purpose"],
                      "call_sha256": hashlib.sha256(json.dumps(call, sort_keys=True,
                          ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                      **observed, "maximum_timing_delta_ms": float(max(deltas))})
    return {"matched_calls": len(pairs), "prompt_tokens": prompt_total,
            "completion_tokens": completion_total, "total_tokens": prompt_total + completion_total,
            "server_eval_seconds": float(server_ms / 1000), "client_inference_seconds": float(client_seconds),
            "rounding_tolerance_ms": float(TOLERANCE_MS), "maximum_timing_delta_ms": float(maximum_delta),
            "api_cost_usd": 0, "total_monetary_cost_usd": None, "call_pairs": pairs}


def build_accounting(directory: Path, server_log: Path) -> dict:
    import rcwt_r2 as runner
    directory, server_log = runner._local(directory), runner._local(server_log)
    before = runner._snapshot(directory)
    log = runner._bytes(server_log)
    helper_hash = runner.digest(runner._bytes(Path(__file__)))
    verification = runner.verify_run(directory)
    protocol = runner.read(directory / "protocol.json")
    if (protocol["evidence_kind"] != "real_local_model" or not verification["complete"]
            or verification["generation_calls"] != 2250 or verification["verified_steps"] != 1120
            or verification["inference_calls"] != 0):
        raise ValueError("Require complete real-model R2 recorded-evidence verification")
    calls = []
    for stage in runner.STAGES:
        rows = (runner.read_jsonl(directory / "proposals.jsonl") if stage == "propose"
                else runner.all_rows(directory, runner.PHASE[stage]))
        for row in rows:
            for event in row["client_events"]:
                if event["method"] == "complete":
                    calls.append(event["result"])
    observed = parse_server_log(log, 2250)
    result = compare_calls(calls, observed)
    if (result["prompt_tokens"] != verification["prompt_tokens"]
            or result["completion_tokens"] != verification["completion_tokens"]):
        raise ValueError("Reconciled totals differ from complete replay")
    if (before != runner._snapshot(directory) or log != runner._bytes(server_log)
            or helper_hash != runner.digest(runner._bytes(Path(__file__)))):
        raise ValueError("Evidence, log or accounting source changed during audit")
    runner.validate_protocol(directory)
    return {"schema": "rcwt-r2-server-accounting/1", "status": "PASS",
            "accounting_only": True, "gain": "NOT_EVALUATED", "inference_calls": 0,
            "source_sha256": helper_hash, "run_files_sha256": before,
            "server_log_sha256": runner.digest(log), "server_log_bytes": len(log),
            "complete_offline_replay": verification, **result,
            "limitations": [
                "Recorded consistency is not independent hardware attestation or proof of no unlisted runs.",
                "A stable full-session log is required; closure/custody and absence of later appends are separate checks.",
                "This audit does not evaluate improvement or production safety.",
                "Provider API charge is zero; energy, hardware and total monetary cost are unknown.",
            ]}


def audit(directory: Path, server_log: Path, output_dir: Path, *, verify=False) -> dict:
    import rcwt_r2 as runner
    directory, server_log = runner._local(directory), runner._local(server_log)
    output_dir = runner._local(output_dir, missing=not verify)
    if (output_dir == directory or output_dir.is_relative_to(directory)
            or directory.is_relative_to(output_dir) or server_log.is_relative_to(output_dir)):
        raise ValueError("Accounting output must be separate from original evidence")
    if not verify and (output_dir.parent != (ROOT / ".runs").resolve()
                       or re.fullmatch(r"r2_accounting_[A-Za-z0-9_-]+", output_dir.name) is None):
        raise ValueError("New accounting output must be an immediate .runs/r2_accounting_* directory")
    if not verify and output_dir.exists():
        raise ValueError("Refusing accounting overwrite")
    result = build_accounting(directory, server_log)
    data = runner.json_bytes(result)
    if verify:
        if runner._snapshot(output_dir) != {"accounting.json": runner.digest(data)}:
            raise ValueError("Saved accounting differs from complete recomputation")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        runner._write(output_dir / "accounting.json", data, raw=True)
    if runner._snapshot(output_dir) != {"accounting.json": runner.digest(data)}:
        raise ValueError("Written accounting bytes differ from recomputation")
    if (result["run_files_sha256"] != runner._snapshot(directory)
            or result["server_log_sha256"] != runner.digest(runner._bytes(server_log))
            or result["source_sha256"] != runner.digest(runner._bytes(Path(__file__)))):
        raise ValueError("Input changed while materializing or verifying accounting")
    runner.validate_protocol(directory)
    return {key: value for key, value in result.items() if key not in {"call_pairs", "run_files_sha256"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir, args.server_log, args.output_dir, verify=args.verify),
                     indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
