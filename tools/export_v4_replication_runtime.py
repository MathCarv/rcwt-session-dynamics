"""Export a derived R1 log with only its local model-directory prefix redacted.

The original accountant and custody are verified first. Every line, task,
timestamp and timing remains present. This never changes original evidence,
calls a model, or treats the derived log as the original runtime log.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path, PureWindowsPath
import re
import stat
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUDITOR_RELATIVE = "tools/audit_v4_replication_calls.py"
OUTPUT_NAMES = ("server.redacted.txt", "manifest.json")
METADATA_MODEL = re.compile(
    rb"(?P<header>\d+(?:\.\d+){3}\s+[IWEFD]\s+srv\s+load_model:\s+loading model ')"
    rb"(?P<path>[^'\r\n]+)(?P<ending>'\r?\n)"
)
SLOT_LINE = re.compile(rb"^\d+(?:\.\d+){3}\s+[IWEFD]\s+slot\s+")
PRIVATE_PATH = re.compile(rb"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]|/(?:home|Users|mnt)/|(?<!\\)\\\\[^\s\\]+\\")
CREDENTIAL = re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ordinary(path: Path, *, missing=False) -> Path:
    path = Path(path).absolute()
    if str(path).startswith(("\\\\", "//")) or ".." in path.parts:
        raise ValueError("Only ordinary local paths are accepted")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if missing:
                continue
            raise ValueError("A required input is missing") from None
        if stat.S_ISLNK(info.st_mode) or (getattr(info, "st_file_attributes", 0)
                                          & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError("Links and reparse paths are forbidden")
    return path


def _bytes(path: Path) -> bytes:
    path = _ordinary(path)
    if not path.is_file():
        raise ValueError("A required input is not a regular file")
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
        raise ValueError("Require an existing input directory")
    files = {}
    def visit(folder):
        for path in sorted(folder.iterdir()):
            checked = _ordinary(path)
            if checked.is_dir():
                visit(checked)
            elif checked.is_file():
                files[checked.relative_to(directory).as_posix()] = _sha(_bytes(checked))
            else:
                raise ValueError("Nonregular input entry")
    visit(directory)
    return files


def _auditor():
    expected = _ordinary(ROOT / AUDITOR_RELATIVE)
    module = importlib.import_module("tools.audit_v4_replication_calls")
    if Path(module.__file__).resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError("The original auditor came from another checkout")
    return module


def verify_original_accounting(run_dir, server_log, accounting_dir):
    return _auditor().audit(run_dir, server_log, accounting_dir, verify=True)


def verify_custody(custody: dict, runtime_dir: Path, snapshot: dict, accounting: dict) -> dict:
    """Verify only preserved project copies; never open private source paths."""
    if (custody.get("schema") != "rcwt-r1-runtime-custody/1"
            or custody.get("status") != "STOPPED_AND_COPIED_PENDING_CALL_RECONCILIATION"
            or custody.get("listeners_after") != []
            or any(custody.get(name) is not True for name in (
                "runner_absent_before_stop", "server_absent_after_stop", "input_and_copy_hashes_rechecked",
                "source_handles_exclusive_during_copy", "copied_files_flush_to_disk"))
            or any(type(custody.get(name)) is not int or custody[name] != expected for name, expected in {
                "stop_actions": 1, "files_deleted": 0, "other_processes_targeted": 0,
                "endpoint_calls": 0, "inference_calls": 0}.items())):
        raise ValueError("Require verified post-exit exclusive-copy custody")
    run_hashes = accounting["inputs"]["run_files_sha256"]
    if any(run_hashes.get(name) != digest for name, digest in custody["run_files_sha256_before_and_after"].items()):
        raise ValueError("Custody run bindings differ from the original accounting")
    required_run_bindings = {"protocol.json", "completion.json", "traces.jsonl", "episodes.jsonl",
                             "analysis.json", "verification.json", "RESULTS.md"}
    if set(custody["run_files_sha256_before_and_after"]) != required_run_bindings:
        raise ValueError("Custody must bind all complete run/report files")
    records = custody.get("copied_files")
    if not isinstance(records, list) or len(records) != 22:
        raise ValueError("Require the original 22-file custody inventory")
    copied = {}
    for entry in records:
        destination = _ordinary(Path(entry["copy"]))
        if not destination.is_relative_to(runtime_dir):
            raise ValueError("Custody copy leaves the supplied runtime bundle")
        relative = destination.relative_to(runtime_dir).as_posix()
        if relative in copied or relative in {"custody.json", "close-intent.json"}:
            raise ValueError("Duplicate or invalid custody copy binding")
        digest = snapshot.get(relative)
        if (digest is None or any(entry.get(name) != digest for name in (
                "source_sha256_before", "source_sha256_after", "copy_sha256"))
                or type(entry.get("bytes")) is not int or entry["bytes"] != len(_bytes(destination))
                or entry.get("source_exclusive_handle") is not True
                or entry.get("destination_flush_to_disk") is not True):
            raise ValueError("Preserved custody copy bytes or exclusive-copy flags differ")
        copied[relative] = digest
    if set(snapshot) != set(copied) | {"custody.json", "close-intent.json"}:
        raise ValueError("Runtime custody bundle has missing or additional files")
    if (copied.get("logs/server.stderr.log") != accounting["inputs"]["server_log_sha256"]
            or not isinstance(custody.get("observed_identity_before_stop"), dict)):
        raise ValueError("The source log or runtime identity is not bound by custody")
    return custody["observed_identity_before_stop"]


def redact_metadata_prefix(source: bytes, identity: dict) -> tuple[bytes, dict]:
    """The single allowed edit is the directory prefix in model-load metadata."""
    source.decode("utf-8", errors="strict")
    if not source.endswith(b"\n") or CREDENTIAL.search(source):
        raise ValueError("Incomplete log or credential-like content cannot be exported")
    model = PureWindowsPath(identity["Model"])
    binary = PureWindowsPath(identity["Binary"])
    runtime = model.parent.parent
    if (not model.is_absolute() or model.name != "Qwen3.5-4B-Q4_K_M.gguf"
            or model.parent.name != "models" or binary.parent.parent != runtime
            or binary.name != "llama-server.exe" or binary.parent.name != "llama-b10809"):
        raise ValueError("Unexpected pinned runtime metadata layout")
    private_prefix = str(runtime).encode("utf-8")
    model_bytes = str(model).encode("utf-8")
    expected_tail = model_bytes[len(private_prefix):]
    if not expected_tail.startswith(b"\\models\\"):
        raise ValueError("Unexpected model-prefix boundary")
    original_lines = source.splitlines(keepends=True)
    redacted_lines, changed = [], []
    for number, line in enumerate(original_lines, 1):
        match = METADATA_MODEL.fullmatch(line)
        if match:
            if match["path"] != model_bytes or not match["path"].startswith(private_prefix + b"\\"):
                raise ValueError("Model-load metadata differs from the custody identity")
            line = match["header"] + b"<LOCAL_RUNTIME>" + expected_tail + match["ending"]
            changed.append(number)
        if PRIVATE_PATH.search(line) or private_prefix in line:
            raise ValueError("A local path remains outside the permitted metadata-prefix edit")
        redacted_lines.append(line)
    if len(changed) != 1 or len(redacted_lines) != len(original_lines):
        raise ValueError("Require exactly one metadata-prefix replacement and every original line")
    for number, (original, derived) in enumerate(zip(original_lines, redacted_lines), 1):
        if number not in changed and original != derived:
            raise ValueError("A nonmetadata line changed")
    task_original = b"".join(line for line in original_lines if SLOT_LINE.match(line))
    task_derived = b"".join(line for line in redacted_lines if SLOT_LINE.match(line))
    if task_original != task_derived:
        raise ValueError("Task-event bytes changed during metadata redaction")
    redacted = b"".join(redacted_lines)
    # The unchanged parser must see identical starts, task IDs, tokens and timings.
    parsed_original = _auditor().parse_server_log(source)
    if parsed_original != _auditor().parse_server_log(redacted) or len(parsed_original) != 1248:
        raise ValueError("Redacted and original task accounting differ")
    details = {
        "source_line_count": len(original_lines), "derived_line_count": len(redacted_lines),
        "changed_metadata_line_numbers": changed, "unchanged_line_count": len(original_lines) - len(changed),
        "unchanged_task_event_line_count": sum(bool(SLOT_LINE.match(line)) for line in original_lines),
        "unchanged_task_event_bytes_sha256": _sha(task_original),
        "parsed_task_count": len(parsed_original), "line_order_preserved": True,
        "line_endings_preserved": True, "task_event_bytes_preserved": True,
        "redaction_categories": [{"category": "local_runtime_directory_prefix", "replacement": "<LOCAL_RUNTIME>",
                                  "occurrences": 1, "metadata_line_count": 1}],
    }
    return redacted, details


def build_export(run_dir: Path, runtime_dir: Path, accounting_dir: Path) -> tuple[bytes, dict, dict]:
    directories = tuple(_ordinary(path) for path in (run_dir, runtime_dir, accounting_dir))
    if any(not path.is_relative_to(ROOT.absolute()) for path in directories):
        raise ValueError("All original bundles must belong to the pinned project")
    run_dir, runtime_dir, accounting_dir = directories
    before = [_snapshot(path) for path in directories]
    exporter_sha = _sha(_bytes(Path(__file__)))
    auditor_sha = _sha(_bytes(ROOT / AUDITOR_RELATIVE))
    accounting_bytes = _bytes(accounting_dir / "call-accounting.json")
    accounting = _json(accounting_bytes)
    if (accounting.get("schema") != "rcwt-r1-server-call-accounting/1"
            or accounting.get("status") != "PASS" or accounting.get("accounting_only") is not True
            or accounting.get("gain") != "NOT_EVALUATED"
            or type(accounting.get("matched_trace_calls")) is not int or accounting["matched_trace_calls"] != 1248
            or accounting["inputs"]["audit_helper_sha256"] != auditor_sha):
        raise ValueError("Require the original PASS accounting and its unchanged auditor")
    server_log = runtime_dir / "logs/server.stderr.log"
    original_verification = verify_original_accounting(run_dir, server_log, accounting_dir)
    if (not isinstance(original_verification, dict) or original_verification.get("status") != "PASS"
            or original_verification.get("output_mode") != "VERIFIED_EXACT_BYTES"
            or original_verification.get("accounting_only") is not True
            or original_verification.get("gain") != "NOT_EVALUATED"
            or type(original_verification.get("matched_trace_calls")) is not int
            or original_verification["matched_trace_calls"] != 1248
            or original_verification.get("inference_calls") != 0):
        raise ValueError("Complete exact-byte original accounting verification is required")
    custody_bytes = _bytes(runtime_dir / "custody.json")
    identity = verify_custody(_json(custody_bytes), runtime_dir, before[1], accounting)
    source = _bytes(server_log)
    redacted, details = redact_metadata_prefix(source, identity)
    if before != [_snapshot(path) for path in directories] or exporter_sha != _sha(_bytes(Path(__file__))) or auditor_sha != _sha(_bytes(ROOT / AUDITOR_RELATIVE)):
        raise ValueError("Original bundles or helper sources changed during export validation")
    manifest = {
        "schema": "rcwt-r1-derived-runtime-log/1", "status": "PASS_DERIVED_EXPORT_INTEGRITY",
        "artifact_kind": "DERIVED_LOG_WITH_METADATA_PATH_PREFIX_REDACTION", "is_original_runtime_log": False,
        "inference_calls": 0, "original_accounting_verified_exact_bytes": True,
        "original_custody_copy_inventory_verified": True,
        "source_log_sha256": _sha(source), "source_log_bytes": len(source),
        "custody_sha256": _sha(custody_bytes), "original_accounting_json_sha256": _sha(accounting_bytes),
        "original_accounting_markdown_sha256": _sha(_bytes(accounting_dir / "CALL-ACCOUNTING.md")),
        "original_auditor_sha256": auditor_sha, "exporter_sha256": exporter_sha,
        "derived_file": OUTPUT_NAMES[0], "derived_log_sha256": _sha(redacted), "derived_log_bytes": len(redacted),
        **details,
        "limitations": [
            "This is a derived log with one metadata path-prefix replacement, not the original log or independent hardware attestation.",
            "Original custody and source hashes identify private originals that are intentionally outside the public package.",
            "Public call accounting must be created separately with the unchanged auditor against this derived file.",
            "All task events and recorded timings remain byte-identical; no call, task line or timestamp was selected, omitted or rewritten.",
            "Export integrity does not evaluate quality, gain, privacy beyond the declared path edit, or unlisted runs.",
        ],
    }
    if PRIVATE_PATH.search(_json_bytes(manifest)) or CREDENTIAL.search(_json_bytes(manifest)):
        raise ValueError("The public manifest would expose a local path or credential-like value")
    state = {"directories": directories, "snapshots": before, "exporter_sha256": exporter_sha, "auditor_sha256": auditor_sha}
    return redacted, manifest, state


def _assert_unchanged(state):
    if (state["snapshots"] != [_snapshot(path) for path in state["directories"]]
            or state["exporter_sha256"] != _sha(_bytes(Path(__file__)))
            or state["auditor_sha256"] != _sha(_bytes(ROOT / AUDITOR_RELATIVE))):
        raise ValueError("Original input or source bytes changed during publication")


def export_runtime(run_dir: Path, runtime_dir: Path, accounting_dir: Path, output_dir: Path, *, verify=False) -> dict:
    inputs = tuple(_ordinary(path) for path in (run_dir, runtime_dir, accounting_dir))
    output_dir = _ordinary(output_dir, missing=not verify)
    if (not output_dir.is_relative_to(ROOT.absolute())
            or any(output_dir == path or output_dir.is_relative_to(path) or path.is_relative_to(output_dir) for path in inputs)):
        raise ValueError("Use a separate project-local output directory")
    if not verify and output_dir.exists():
        raise ValueError("Export output already exists; overwrite is forbidden")
    saved_before = _snapshot(output_dir) if verify else None
    redacted, manifest, state = build_export(*inputs)
    expected = {OUTPUT_NAMES[0]: redacted, OUTPUT_NAMES[1]: _json_bytes(manifest)}
    if verify:
        if set(saved_before) != set(expected) or any(_bytes(output_dir / name) != content for name, content in expected.items()):
            raise ValueError("Saved derived export bytes differ")
        if saved_before != _snapshot(output_dir):
            raise ValueError("Saved export changed during verification")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        for name, content in expected.items():
            with (output_dir / name).open("xb") as handle:
                handle.write(content)
        if _snapshot(output_dir) != {name: _sha(content) for name, content in expected.items()}:
            raise ValueError("Export readback differs from the validated bytes")
    _assert_unchanged(state)
    return {"status": "PASS_DERIVED_EXPORT_INTEGRITY", "is_original_runtime_log": False,
            "mode": "VERIFIED_EXACT_BYTES" if verify else "CREATED_EXCLUSIVELY", "inference_calls": 0,
            "metadata_prefix_replacements": 1, "preserved_tasks": 1248,
            "output_files_sha256": {name: _sha(content) for name, content in expected.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--accounting-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        result = export_runtime(args.run_dir, args.runtime_dir, args.accounting_dir, args.output_dir, verify=args.verify)
    except Exception as exc:
        # Underlying filesystem exceptions may contain private paths. Never
        # print their message, input values, traceback or original custody data.
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__,
                          "message": "Export refused by input, custody, redaction or integrity validation."}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
