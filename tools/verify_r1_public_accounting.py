"""Verify the fixed public R1 release offline without writing artifacts.

The original accountant stays unchanged. Released JSON and Markdown bytes are
SHA-256 pinned; recomputation permits only JSON object-key ordering differences.
This is not a claim that the original byte formatter is cross-platform stable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
RUN = "results/agent_v4_replication"
RUNTIME = "results/agent_v4_replication_public_runtime"
ACCOUNTING = "results/agent_v4_replication_public_accounting"
AUDITOR = "tools/audit_v4_replication_calls.py"
EXPORTER = "tools/export_v4_replication_runtime.py"
RELEASE_SHA256 = {
    AUDITOR: "1e99ec11151944d285abda1b15f9411636a17f6e626ec265cb81b20274f281ac",
    EXPORTER: "6919f6f19ee680411ece9ab5627d0154111b2d88f59fb810ff9a009643a4e972",
    ACCOUNTING + "/call-accounting.json": "73557b9bfa29f3a9917856cafae76ffe9f356432d84785cb7f851257ff32896d",
    ACCOUNTING + "/CALL-ACCOUNTING.md": "d33e7cf2d887f8c3aa798eb37d3d35b692a98cb279ba7292ab92e3d39313be40",
    RUNTIME + "/manifest.json": "8ace0f7e3d59073ad4a67a61fe7afaa9d813df5a7ef0c5520317e1977a6e0d59",
    RUNTIME + "/server.redacted.txt": "64b74addd5505f9f3d0e3a0543fa35fa8055c81504f76fd72ac0aa3d139dc34e",
}


def _read(path: Path) -> bytes:
    """Check ancestors before reading, including before importing the auditor."""
    path = Path(path).absolute()
    if str(path).startswith(("\\\\", "//")) or ".." in path.parts:
        raise ValueError("Require an ordinary local release path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or (getattr(info, "st_file_attributes", 0)
                                          & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError("Links and reparse points are forbidden")
    if not path.is_file():
        raise ValueError("Require an ordinary release file")
    return path.read_bytes()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if not isinstance(name, str) or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError("Invalid project-relative release reference")
    return root.joinpath(*relative.parts)


def canonical_json(value) -> bytes:
    """Preserve JSON values, numeric/Boolean types and array order exactly."""
    def validate(item):
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be strings")
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("Require exact JSON value types")
    validate(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def require_same_accounting(recomputed: dict, saved: dict) -> None:
    if canonical_json(recomputed) != canonical_json(saved):
        raise ValueError("Recomputed accounting differs beyond JSON object-key ordering")


def _released_bytes(root: Path) -> dict[str, bytes]:
    result = {}
    for relative, expected in RELEASE_SHA256.items():
        data = _read(_relative(root, relative))
        if _digest(data) != expected:
            raise ValueError("Published release byte pin differs")
        result[relative] = data
    return result


def _load_accountant(root: Path):
    if root != ROOT.absolute():
        raise ValueError("Verify the release in the helper's own project")
    if _digest(_read(root / AUDITOR)) != RELEASE_SHA256[AUDITOR]:
        raise ValueError("Original auditor byte pin differs before import")
    sys.path.insert(0, str(root))
    accountant = importlib.import_module("tools.audit_v4_replication_calls")
    if Path(accountant.__file__).absolute() != root / AUDITOR or accountant.ROOT.absolute() != root:
        raise ValueError("Original auditor was loaded from another project")
    return accountant


def _state(root: Path, accountant, saved: dict) -> dict:
    released = _released_bytes(root)
    run = accountant._snapshot(root / RUN)
    if len(run) != 32 or canonical_json(run) != canonical_json(saved["inputs"]["run_files_sha256"]):
        raise ValueError("Require the exact released 32-file R1 snapshot")
    runtime = accountant._snapshot(root / RUNTIME)
    accounting = accountant._snapshot(root / ACCOUNTING)
    for directory, snapshot in ((RUNTIME, runtime), (ACCOUNTING, accounting)):
        expected = {name[len(directory) + 1:]: digest for name, digest in RELEASE_SHA256.items()
                    if name.startswith(directory + "/")}
        if canonical_json(snapshot) != canonical_json(expected):
            raise ValueError("Public release directory has missing or additional files")
    protocol = accountant._json(_read(root / RUN / "protocol.json"))
    historical = protocol["historical_evidence"]["input_sha256"]
    if type(historical) is not dict or len(historical) != 147:
        raise ValueError("Require all 147 frozen historical input pins")
    historical_now = {name: _digest(_read(_relative(root, name))) for name in historical}
    if canonical_json(historical_now) != canonical_json(historical):
        raise ValueError("Frozen historical input bytes differ")
    pins = accountant._source_pins(protocol)
    if canonical_json(pins) != canonical_json(saved["inputs"]["source_pins"]):
        raise ValueError("Frozen source and registration pins differ")
    return {"run": run, "runtime": runtime, "accounting": accounting,
            "historical": historical_now, "source_pins": pins,
            "released": {name: _digest(data) for name, data in released.items()},
            "verifier_sha256": _digest(_read(Path(__file__)))}


def verify_public_accounting(root: Path = ROOT) -> dict:
    root = Path(root).absolute()
    # Byte pins are checked before any code from the original auditor is loaded.
    released = _released_bytes(root)
    accountant = _load_accountant(root)
    saved = accountant._json(released[ACCOUNTING + "/call-accounting.json"])
    before = _state(root, accountant, saved)
    recomputed = accountant.build_accounting(root / RUN, root / RUNTIME / "server.redacted.txt")
    require_same_accounting(recomputed, saved)
    markdown = accountant.render_markdown(recomputed).encode("utf-8")
    if markdown != released[ACCOUNTING + "/CALL-ACCOUNTING.md"]:
        raise ValueError("Original Markdown rendering differs from exact published bytes")
    if canonical_json(before) != canonical_json(_state(root, accountant, saved)):
        raise ValueError("Original inputs, public inputs or verification sources changed")
    return {"status": "PASS", "accounting_only": True, "gain": "NOT_EVALUATED",
            "read_only": True, "inference_calls": 0, "matched_trace_calls": 1248,
            "comparison": "EXACT_RELEASED_BYTES_AND_JSON_VALUES_EXCEPT_OBJECT_KEY_ORDER",
            "original_cross_platform_formatter_claim": False,
            "published_json_sha256": RELEASE_SHA256[ACCOUNTING + "/call-accounting.json"],
            "published_markdown_sha256": RELEASE_SHA256[ACCOUNTING + "/CALL-ACCOUNTING.md"]}


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        print(json.dumps(verify_public_accounting(), indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        # Missing/private local paths must never be printed by CLI diagnostics.
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__,
                          "message": "Fixed public R1 release verification failed"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
