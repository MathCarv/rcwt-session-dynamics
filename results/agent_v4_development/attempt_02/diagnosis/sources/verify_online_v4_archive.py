"""Offline verification of this project's trusted local v4 source archives.

Only local bundles below this checkout's .runs/ or results/ are accepted. Hashes
check integrity, not authorship: never place downloaded/untrusted Python in a
trusted experiment bundle. The recorded project sources execute in a temporary
copy, exclusively in verify mode, with network connections disabled. This is
not a security sandbox or independent attestation of historical model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "rcwt_agent_actor.py", "rcwt_agent_analysis.py", "rcwt_agent_env.py",
    "rcwt_agent_memory.py", "rcwt_agent_run.py", "rcwt_local_model.py",
    "rcwt_memory_v3.py", "rcwt_online_v3.py", "rcwt_analysis_v3.py",
    "rcwt_retrieval_v3.py", "rcwt_review_v3.py", "rcwt_context_v4.py",
    "rcwt_decision_v4.py", "rcwt_online_v4.py", "rcwt_analysis_v4.py",
)
INPUTS = ("protocol.json", "freeze.json", "started.json", "completion.json",
          "schedule.json", "public.json", "oracle.json", "traces.jsonl", "episodes.jsonl")
HISTORY = {
    "v2_result_sha256": "results/agent_v2/RESULTS.md",
    "v2_diagnosis_sha256": "results/agent_v2/DIAGNOSIS.md",
    "v3_protocol_sha256": "results/agent_v3_development/attempt_05/protocol.json",
    "v3_index_sha256": "results/agent_v3_development/development.json",
    "v3_closure_sha256": "results/agent_v3_development/RESULTS.md",
}
_VERIFY_ONLY = """import runpy, socket, sys
from pathlib import Path
def blocked(*args, **kwargs):
    raise RuntimeError('Network is forbidden during archived offline verification')
socket.create_connection = blocked
socket.getaddrinfo = blocked
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.socket.sendto = blocked
script = sys.argv[1]
sys.path.insert(0, str(Path(script).parent))
sys.argv = sys.argv[1:]
if len(sys.argv) != 5 or sys.argv[1:4] != ['--stage', 'verify', '--output-dir']:
    raise RuntimeError('Only the offline verify stage is allowed')
runpy.run_path(script, run_name='__main__')
"""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return _digest(Path(path).read_bytes())


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field in archive")
        result[key] = value
    return result


def _json(data):
    def reject(value):
        raise ValueError("Non-finite archive JSON: " + value)
    return json.loads(data, object_pairs_hook=_object, parse_constant=reject)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _local(path: Path) -> Path:
    boundary = ROOT.resolve(strict=True)
    original = path.absolute()
    try:
        parts = original.relative_to(boundary).parts
    except ValueError as exc:
        raise ValueError("Only this project's trusted local files are accepted") from exc
    current = boundary
    for part in parts:
        current = current / part
        info = current.lstat()
        if current.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("Archive paths cannot contain symbolic links or reparse points")
    resolved = original.resolve(strict=True)
    if not resolved.is_relative_to(boundary):
        raise ValueError("Archive path leaves this project")
    return resolved


def _read(path: Path, observed: dict[Path, bytes]) -> bytes:
    path = _local(path)
    if not path.is_file():
        raise ValueError("Archive input must be a regular local file")
    data = path.read_bytes()
    if path in observed and observed[path] != data:
        raise ValueError("Archive input changed during verification")
    observed[path] = data
    return data


def _hashes(value, names):
    if (not isinstance(value, dict) or set(value) != {"src/" + name for name in names}
            or any(not isinstance(item, str) or re.fullmatch("[0-9a-f]{64}", item) is None for item in value.values())):
        raise ValueError("Expected the exact complete source hash manifest")


def _bundle(directory: Path, observed: dict[Path, bytes], *, development_only=False):
    if any((directory / name).exists() for name in ("aborted.json", "partial-step.json")):
        raise ValueError("An aborted or partial archive is not complete")
    files = {name: _read(directory / name, observed) for name in INPUTS}
    protocol = _json(files["protocol.json"])
    modes = {"development": ("train", 8), "confirmatory": ("test", 32)}
    mode = protocol.get("mode")
    if (protocol.get("schema") != "rcwt-online-memory/4" or mode not in modes
            or type(protocol.get("count")) is not int or (protocol.get("split"), protocol["count"]) != modes[mode]
            or (development_only and mode != "development")):
        raise ValueError("Expected a v4 development-8 or confirmation-32 archive")
    expected = protocol.get("source_sha256")
    _hashes(expected, NAMES)
    for name in NAMES:
        data = _read(directory / "sources" / name, observed)
        if _digest(data) != expected["src/" + name]:
            raise ValueError("Source archive hash mismatch")
        files["sources/" + name] = data
    freeze, started, completion = (_json(files[name + ".json"]) for name in ("freeze", "started", "completion"))
    protocol_hash = _digest(files["protocol.json"])
    if any(item.get("protocol_sha256") != protocol_hash for item in (freeze, started, completion)):
        raise ValueError("Freeze, start and completion must bind the protocol")
    if freeze.get("schedule_sha256") != _digest(files["schedule.json"]):
        raise ValueError("Frozen schedule hash mismatch")
    if any(completion.get(name + "_sha256") != _digest(files[name + ".jsonl"]) for name in ("traces", "episodes")):
        raise ValueError("Completed evidence hash mismatch")
    protocols = [protocol]
    if mode == "confirmatory":
        development_files, development_protocols = _bundle(directory / "development", observed, development_only=True)
        development = protocol.get("development")
        if (not isinstance(development, dict)
                or development.get("protocol_sha256") != _digest(development_files["protocol.json"])
                or development.get("completion_sha256") != _digest(development_files["completion.json"])
                or development_protocols[0]["source_sha256"] != expected):
            raise ValueError("Copied development is not bound to the confirmation")
        files.update({"development/" + name: data for name, data in development_files.items()})
        protocols.extend(development_protocols)
    elif protocol.get("development") is not None:
        raise ValueError("Development archive cannot contain selection evidence")
    return files, protocols


def verify_archive(directory: Path) -> dict:
    directory = _local(Path(directory))
    relative = directory.relative_to(ROOT.resolve())
    if len(relative.parts) < 2 or relative.parts[0] not in {".runs", "results"} or not directory.is_dir():
        raise ValueError("Use a trusted project bundle below .runs/ or results/")
    observed: dict[Path, bytes] = {}
    run_files, protocols = _bundle(directory, observed)
    historical = {path: _read(ROOT / path, observed) for path in HISTORY.values()}
    v2_name = "results/agent_v2/protocol.json"
    historical[v2_name] = _read(ROOT / v2_name, observed)
    inherited_v2 = _json(historical[v2_name])["source_sha256"]
    inherited_v3 = _json(historical[HISTORY["v3_protocol_sha256"]])["source_sha256"]
    _hashes(inherited_v2, NAMES[:6])
    _hashes(inherited_v3, NAMES[:11])
    runtime_name = "docs/rcwt_agent_runtime.json"
    historical[runtime_name] = _read(ROOT / runtime_name, observed)
    runtime = _json(historical[runtime_name])
    for protocol in protocols:
        if any(protocol.get(label) != _digest(historical[path]) for label, path in HISTORY.items()):
            raise ValueError("Pinned historical v2/v3 evidence hash mismatch")
        if any(protocol["source_sha256"].get(path) != value
               for inherited in (inherited_v2, inherited_v3) for path, value in inherited.items()):
            raise ValueError("Archived inherited sources differ from frozen v2/v3")
        if (not isinstance(runtime, dict) or runtime.get("inference_policy") != "local_only_no_paid_api"
                or _canonical(protocol.get("runtime")) != _canonical(runtime)):
            raise ValueError("Pinned local runtime receipt mismatch")
    with tempfile.TemporaryDirectory(prefix="rcwt-v4-archive-") as temporary:
        workspace = Path(temporary)
        copied = {"run/" + name: data for name, data in run_files.items()}
        copied.update(historical)
        copied.update({"src/" + name: run_files["sources/" + name] for name in NAMES})
        for name, data in copied.items():
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if target.read_bytes() != data:
                raise ValueError("Temporary verification copy differs from the archive")
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", _VERIFY_ONLY, str(workspace / "src/rcwt_online_v4.py"),
             "--stage", "verify", "--output-dir", str(workspace / "run")],
            cwd=workspace, env=environment, capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode:
            raise ValueError("Archived offline verification failed: " + result.stderr[-2000:])
        verified = _json(result.stdout)
    if (not isinstance(verified, dict) or verified.get("status") != "PASS"
            or type(verified.get("verified_steps")) is not int or verified["verified_steps"] != protocols[0]["count"] * 16):
        raise ValueError("Archive did not produce the complete expected PASS receipt")
    if any(_read(path, observed) != data for path, data in list(observed.items())):
        raise ValueError("Original evidence changed during archived verification")
    return {"status": "PASS", "mode": "trusted archived source / offline replay",
            "protocol_sha256": _digest(run_files["protocol.json"]), "verified_steps": verified["verified_steps"],
            "verification": verified, "inference_calls": 0, "read_only": True,
            "limit": "Trusted local project sources only; hashes do not authenticate authorship. No independent model, token or hardware attestation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
