"""Reconcile one completed R1 server log with its 1248 recorded model calls.

Offline accounting only: no model construction, endpoint request, repair,
statistical analysis, or gain evaluation. Inputs must be stable local copies.
New output directories are exclusive; --verify compares exact saved bytes.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import importlib
import json
from pathlib import Path
import re
import stat
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
EXPECTED_CALLS = 1248
EXPECTED_STEPS = 512
TOLERANCE_MS = Decimal("0.01")
OUTPUT_NAMES = ("call-accounting.json", "CALL-ACCOUNTING.md")
SLOT = re.compile(r"^\d+(?:\.\d+){3}\s+[IWEFD]\s+slot\s+(?P<event>\w+):\s+id\s+(?P<slot>\d+)\s+\|\s+task\s+(?P<task>-?\d+)\s+\|\s*(?P<body>.*)$")
NUMBER = r"\d+(?:\.\d+)?"
PROMPT = re.compile(rf"prompt eval time\s*=\s*({NUMBER}) ms\s*/\s*(\d+) tokens\s+\([^\r\n]*\)")
EVAL = re.compile(rf"eval time\s*=\s*({NUMBER}) ms\s*/\s*(\d+) (?:tokens|runs)\s+\([^\r\n]*\)")
TOTAL = re.compile(rf"total time\s*=\s*({NUMBER}) ms\s*/\s*(\d+) tokens")
RELEASE = re.compile(r"stop processing: n_tokens = \d+, truncated = [01]")
PROGRESS = re.compile(rf"n_gen\s*=\s*\d+, tg\s*=\s*{NUMBER} t/s, tg_3s\s*=\s*{NUMBER} t/s")
GRAPHS = re.compile(r"graphs reused\s*=\s*\d+")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ordinary(path: Path, *, missing=False) -> Path:
    """Reject links, junctions, reparse ancestors and network paths before I/O."""
    path = Path(path).absolute()
    if str(path).startswith(("\\\\", "//")) or ".." in path.parts:
        raise ValueError("Require an ordinary local path without traversal")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if missing:
                continue
            raise ValueError("Missing required local path: " + str(current)) from None
        if stat.S_ISLNK(info.st_mode) or (getattr(info, "st_file_attributes", 0)
                                          & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError("Links and reparse points are forbidden")
    return path


def _read(path: Path) -> bytes:
    path = _ordinary(path)
    if not path.is_file():
        raise ValueError("Expected a regular input file")
    return path.read_bytes()


def _json(data: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    value = json.loads(data, object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    return value


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _snapshot(directory: Path) -> dict[str, str]:
    directory = _ordinary(directory)
    if not directory.is_dir():
        raise ValueError("Require an existing ordinary evidence directory")
    result = {}
    def visit(folder):
        for path in sorted(folder.iterdir()):
            checked = _ordinary(path)
            if checked.is_dir():
                visit(checked)
            elif checked.is_file():
                result[checked.relative_to(directory).as_posix()] = _digest(_read(checked))
            else:
                raise ValueError("Nonregular evidence path")
    visit(directory)
    return result


def _trusted_module(name: str, relative_path: str):
    expected = _ordinary(ROOT / relative_path)
    module = importlib.import_module(name)
    if Path(module.__file__).resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError("Loaded verification source came from another checkout")
    return module


def _source_pins(protocol: dict) -> dict:
    checker = _trusted_module("tools.verify_v4_replication_report", "tools/verify_v4_replication_report.py")
    checker._fixed_protocol(protocol)
    candidates = [ROOT / "src" / name for name in checker.SOURCE_NAMES]
    orchestrators = [ROOT / name for name in checker.ORCHESTRATOR_NAMES]
    for path in [*candidates, *orchestrators, checker.DEVELOPMENT_PROTOCOL, checker.REGISTRATION]:
        if not _ordinary(path).is_file():
            raise ValueError("Require ordinary source and registration files before pin reads")
    pins = checker._source_pins(protocol)
    # Validate every source path before importing the runner or executing replay.
    for group in (pins["candidate"], pins["orchestration"]):
        for relative, expected in group.items():
            if _digest(_read(ROOT / relative)) != expected:
                raise ValueError("Pinned source bytes changed")
    return pins


def verify_run(directory: Path) -> dict:
    runner = _trusted_module("rcwt_replication_v4", "src/rcwt_replication_v4.py")
    with runner.partial._offline():
        return runner.verify_run(directory)


def _integer(value, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("Invalid positive integer: " + label)
    return value


def _decimal(value, label: str) -> Decimal:
    if type(value) not in (int, float):
        raise ValueError("Invalid recorded number: " + label)
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("Invalid recorded number: " + label) from None
    if not result.is_finite() or result < 0:
        raise ValueError("Invalid finite nonnegative number: " + label)
    return result


def parse_server_log(data: bytes) -> list[dict]:
    """Parse the pinned llama.cpp task lifecycle; reject ambiguous accounting."""
    text = data.decode("utf-8", errors="strict")
    if not text.endswith("\n"):
        raise ValueError("Server log ends with an incomplete line")
    calls, active, seen = [], None, set()
    loaded = listening = configured = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if re.search(r"\bcancel(?:led|ed|ing|lation)?\b", line, re.I):
            raise ValueError("Canceled server task in complete R1 log")
        if re.search(r"\bsrv\s+llama_server:\s+model loaded$", line):
            loaded += 1
        if re.search(r"\bsrv\s+llama_server:\s+listening on http://127\.0\.0\.1:18085$", line):
            listening += 1
        if re.search(r"\bsrv\s+load_model:\s+initializing, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'$", line):
            configured += 1
        match = SLOT.fullmatch(line)
        if not match:
            if any(marker in line for marker in ("launch_slot_", "processing task", "prompt eval time", "eval time =", "total time =", "stop processing:", "id_task")):
                raise ValueError(f"Unparseable task accounting at server log line {line_number}")
            continue
        event, body = match["event"], match["body"].strip()
        slot, task = int(match["slot"]), int(match["task"])
        if slot != 0:
            raise ValueError("Unexpected slot in single-slot R1 runtime")
        if event == "get_availabl" and task == -1 and body.startswith("selected slot by LRU,"):
            continue
        if event == "launch_slot_":
            if body != "processing task, is_child = 0" or task < 0 or task in seen or active is not None:
                raise ValueError("Duplicate, child, overlapping or malformed task launch")
            if (loaded, listening, configured) != (1, 1, 1):
                raise ValueError("Complete single-session startup must precede every generation")
            seen.add(task)
            active = {"slot_id": slot, "task_id": task, "launch_line": line_number}
            continue
        if active is None or active["task_id"] != task:
            raise ValueError("Unmatched server task lifecycle record")
        if event == "print_timing":
            parsed_prompt, parsed_eval, parsed_total = PROMPT.fullmatch(body), EVAL.fullmatch(body), TOTAL.fullmatch(body)
            if parsed_prompt:
                if "prompt_tokens" in active:
                    raise ValueError("Duplicate final prompt timing")
                active.update(prompt_tokens=int(parsed_prompt[2]), prompt_ms=parsed_prompt[1], prompt_line=line_number)
            elif parsed_eval:
                if "prompt_tokens" not in active or "completion_tokens" in active:
                    raise ValueError("Unmatched or duplicate final evaluation timing")
                active.update(completion_tokens=int(parsed_eval[2]), predicted_ms=parsed_eval[1], eval_line=line_number)
            elif parsed_total:
                if "completion_tokens" not in active or "total_ms" in active:
                    raise ValueError("Unmatched or duplicate total timing")
                if int(parsed_total[2]) != active["prompt_tokens"] + active["completion_tokens"]:
                    raise ValueError("Final total token count differs from prompt and evaluation")
                if abs(Decimal(parsed_total[1]) - Decimal(active["prompt_ms"]) - Decimal(active["predicted_ms"])) > TOLERANCE_MS:
                    raise ValueError("Final total timing exceeds rounding tolerance")
                active.update(total_ms=parsed_total[1], total_line=line_number)
            elif PROGRESS.fullmatch(body):
                if "prompt_tokens" in active:
                    raise ValueError("Generation progress occurs after final timings")
            elif GRAPHS.fullmatch(body):
                if "total_ms" not in active:
                    raise ValueError("Graph summary occurs before final timings")
            else:
                raise ValueError("Unparseable or ambiguous timing record")
        elif event == "release":
            if not RELEASE.fullmatch(body) or "total_ms" not in active:
                raise ValueError("Incomplete server task released without final timings")
            # release.n_tokens is deliberately not compared: the pinned server's
            # final consumed KV count can be one below prompt + generated tokens.
            active["release_line"] = line_number
            calls.append(active)
            active = None
        else:
            raise ValueError("Unrecognized task lifecycle event")
    if active is not None:
        raise ValueError("Unfinished server task at end of complete log")
    if (loaded, listening, configured) != (1, 1, 1):
        raise ValueError("Require exactly one complete pinned runtime session")
    if len(seen) != EXPECTED_CALLS or len(calls) != EXPECTED_CALLS:
        raise ValueError("Require exactly 1248 starts and 1248 complete final timings")
    return calls


def _trace_calls(data: bytes) -> list[dict]:
    if not data.endswith(b"\n"):
        raise ValueError("Incomplete trace line")
    rows = [_json(line) for line in data.splitlines()]
    if len(rows) != EXPECTED_STEPS:
        raise ValueError("Require exactly 512 trace decisions")
    calls = []
    for trace_index, row in enumerate(rows):
        for event_index, event in enumerate(row["client_events"]):
            if event["method"] == "complete":
                calls.append({"trace_index": trace_index, "event_index": event_index,
                              "trace_sha256": row["sha256"], "result": event["result"]})
    if len(calls) != EXPECTED_CALLS:
        raise ValueError("Require exactly 1248 recorded model calls")
    return calls


def _assert_stable(run_dir, server_log, files_before, log_hash, pins, helper_hash, protocol):
    if (files_before != _snapshot(run_dir) or log_hash != _digest(_read(server_log))
            or pins != _source_pins(protocol) or helper_hash != _digest(_read(Path(__file__)))):
        raise ValueError("Evidence, server log, helper or pinned source bytes changed during audit")


def build_accounting(run_dir: Path, server_log: Path) -> dict:
    run_dir, server_log = _ordinary(run_dir), _ordinary(server_log)
    if not run_dir.is_relative_to(ROOT.absolute()):
        raise ValueError("R1 evidence must belong to this pinned project")
    files_before = _snapshot(run_dir)
    protocol = _json(_read(run_dir / "protocol.json"))
    pins = _source_pins(protocol)
    helper_hash = _digest(_read(Path(__file__)))
    log_bytes = _read(server_log)
    log_hash = _digest(log_bytes)
    replay = verify_run(run_dir)
    if (not isinstance(replay, dict) or replay.get("status") != "PASS"
            or replay.get("read_only") is not True or replay.get("integrity_only") is not True
            or replay.get("gain") != "NOT_EVALUATED"
            or any(type(replay.get(key)) is not int or replay[key] != expected for key, expected in {
                "verified_steps": 512, "verified_episode_summaries": 64,
                "generation_calls": 1248, "inference_calls": 0}.items())
            or replay.get("protocol_sha256") != files_before.get("protocol.json")):
        raise ValueError("Require complete read-only offline replay of all 512 decisions and 1248 calls")
    trace_calls = _trace_calls(_read(run_dir / "traces.jsonl"))
    server_calls = parse_server_log(log_bytes)
    pairs = []
    prompt_total = completion_total = 0
    wall_total = server_prompt_total = server_eval_total = Decimal(0)
    max_delta = Decimal(0)
    for index, (trace, observed) in enumerate(zip(trace_calls, server_calls)):
        call = trace["result"]
        prompt = _integer(call["prompt_tokens"], "trace prompt tokens")
        completion = _integer(call["completion_tokens"], "trace completion tokens")
        if prompt != observed["prompt_tokens"] or completion != observed["completion_tokens"]:
            raise ValueError(f"Server/trace token mismatch at call {index}")
        timings = call["timings"]
        if (_integer(timings["prompt_n"], "runtime prompt count") != prompt
                or _integer(timings["predicted_n"], "runtime completion count") != completion
                or type(timings.get("cache_n")) is not int or timings["cache_n"] != 0):
            raise ValueError(f"Runtime token metering mismatch at call {index}")
        if _decimal(call["api_cost_usd"], "API cost") != 0:
            raise ValueError("Require zero recorded provider API cost for local R1")
        deltas = [abs(_decimal(timings[name], name) - Decimal(observed[name]))
                  for name in ("prompt_ms", "predicted_ms")]
        if max(deltas) > TOLERANCE_MS:
            raise ValueError(f"Server/trace timing mismatch at call {index}")
        max_delta = max(max_delta, *deltas)
        prompt_total += prompt
        completion_total += completion
        wall_total += _decimal(call["wall_seconds"], "client inference duration")
        server_prompt_total += Decimal(observed["prompt_ms"])
        server_eval_total += Decimal(observed["predicted_ms"])
        pairs.append({"call_index": index, "trace_index": trace["trace_index"],
                      "event_index": trace["event_index"], "trace_sha256": trace["trace_sha256"],
                      "purpose": call["purpose"], **observed,
                      "max_timing_delta_ms": float(max(deltas))})
    if replay.get("prompt_tokens") != prompt_total or replay.get("completion_tokens") != completion_total:
        raise ValueError("Reconciled token totals differ from complete replay")
    _assert_stable(run_dir, server_log, files_before, log_hash, pins, helper_hash, protocol)
    return {
        "schema": "rcwt-r1-server-call-accounting/1", "status": "PASS", "accounting_only": True,
        "gain": "NOT_EVALUATED", "inference_calls": 0, "read_only_inputs": True,
        "verified_steps": 512, "verified_episode_summaries": 64,
        "server_task_starts": len(server_calls), "server_final_timing_pairs": len(server_calls),
        "matched_trace_calls": len(pairs), "canceled_tasks": 0, "extra_tasks": 0,
        "prompt_tokens": prompt_total, "completion_tokens": completion_total,
        "total_tokens": prompt_total + completion_total,
        "recorded_client_inference_seconds": float(wall_total),
        "server_prompt_eval_seconds": float(server_prompt_total / 1000),
        "server_generation_eval_seconds": float(server_eval_total / 1000),
        "server_recorded_eval_seconds": float((server_prompt_total + server_eval_total) / 1000),
        "api_provider_cost_usd": 0, "total_monetary_cost_usd": None,
        "rounding_tolerance_ms": float(TOLERANCE_MS), "maximum_observed_timing_delta_ms": float(max_delta),
        "inputs": {"run_directory": run_dir.relative_to(ROOT.absolute()).as_posix(),
                   "run_files_sha256": files_before, "server_log_sha256": log_hash,
                   "server_log_bytes": len(log_bytes), "source_pins": pins,
                   "audit_helper_sha256": helper_hash},
        "complete_offline_replay": replay, "call_pairs": pairs,
        "limitations": [
            "PASS means complete recorded call accounting only; quality or improvement was not evaluated.",
            "The supplied complete log and traces are local claims, not independent hardware or global run attestation.",
            "Process start/stop, log-copy completeness and absence of later appends require separate runtime custody receipts.",
            "Client inference duration includes request overhead; server evaluation time is reported separately.",
            "Provider API charges are zero; electricity, hardware and total monetary cost are unknown.",
            "This audit neither pools nor replaces the preserved interrupted confirmation.",
        ],
    }


def render_markdown(result: dict) -> str:
    return "\n".join([
        "# R1 server-call accounting", "", "PASS — recorded accounting only. Gain: NOT_EVALUATED.", "",
        f"Complete offline replay: {result['verified_steps']} decisions and {result['verified_episode_summaries']} trajectory summaries.",
        f"Server starts / complete final timing pairs / matched trace calls: {result['server_task_starts']} / {result['server_final_timing_pairs']} / {result['matched_trace_calls']}.",
        "Canceled or extra tasks: 0. No inference or endpoint request was made by this audit.", "",
        f"Prompt tokens: {result['prompt_tokens']}. Completion tokens: {result['completion_tokens']}. Total: {result['total_tokens']}.",
        f"Recorded client inference duration: {result['recorded_client_inference_seconds']:.6f} s.",
        f"Server prompt evaluation: {result['server_prompt_eval_seconds']:.6f} s; generation evaluation: {result['server_generation_eval_seconds']:.6f} s.",
        f"Per-field timing rounding tolerance: {result['rounding_tolerance_ms']:.2f} ms; maximum observed delta: {result['maximum_observed_timing_delta_ms']:.6f} ms.",
        "Provider API cost: USD 0. Total monetary cost: unknown (electricity and hardware are not measured).", "",
        f"Server-log SHA-256: `{result['inputs']['server_log_sha256']}`.",
        f"Protocol SHA-256: `{result['inputs']['run_files_sha256']['protocol.json']}`.",
        f"Audit-helper SHA-256: `{result['inputs']['audit_helper_sha256']}`.", "",
        "All 1248 ordered task-to-trace mappings and the input/source hashes are preserved in call-accounting.json.", "",
        *result["limitations"], "",
    ])


def audit(run_dir: Path, server_log: Path, output_dir: Path, *, verify=False) -> dict:
    run_dir, server_log = _ordinary(run_dir), _ordinary(server_log)
    output_dir = _ordinary(output_dir, missing=not verify)
    if (output_dir == run_dir or output_dir.is_relative_to(run_dir) or run_dir.is_relative_to(output_dir)
            or server_log.is_relative_to(output_dir)):
        raise ValueError("Audit output must be separate from all original evidence")
    if not verify and output_dir.exists():
        raise ValueError("Output already exists; refusing audit overwrite")
    saved_before = _snapshot(output_dir) if verify else None
    result = build_accounting(run_dir, server_log)
    expected = {OUTPUT_NAMES[0]: _json_bytes(result), OUTPUT_NAMES[1]: render_markdown(result).encode("utf-8")}
    if verify:
        if set(saved_before) != set(OUTPUT_NAMES):
            raise ValueError("Saved audit files are missing or additional")
        for name, content in expected.items():
            if _read(output_dir / name) != content:
                raise ValueError("Saved audit bytes differ from recomputed accounting: " + name)
        if saved_before != _snapshot(output_dir):
            raise ValueError("Saved audit bytes changed during verification")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        for name, content in expected.items():
            with (output_dir / name).open("xb") as handle:
                handle.write(content)
        if _snapshot(output_dir) != {name: _digest(content) for name, content in expected.items()}:
            raise ValueError("Output bytes changed during audit publication")
    inputs = result["inputs"]
    _assert_stable(run_dir, server_log, inputs["run_files_sha256"], inputs["server_log_sha256"],
                   inputs["source_pins"], inputs["audit_helper_sha256"],
                   _json(_read(run_dir / "protocol.json")))
    return {"status": "PASS", "accounting_only": True, "gain": "NOT_EVALUATED",
            "matched_trace_calls": result["matched_trace_calls"], "inference_calls": 0,
            "output_mode": "VERIFIED_EXACT_BYTES" if verify else "CREATED_EXCLUSIVELY",
            "output_files_sha256": {name: _digest(content) for name, content in expected.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir, args.server_log, args.output_dir, verify=args.verify), indent=2))


if __name__ == "__main__":
    main()
