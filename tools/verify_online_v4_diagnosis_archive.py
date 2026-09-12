"""Reverify a historical v4 diagnosis with its trusted, hash-pinned tools.

Only completed project bundles below .runs/ or results/ are accepted. The
diagnosis must be an immediate child of its run so its relative links remain
unchanged. Original files are read-only; a temporary workspace executes the
archived inspector exclusively in --verify mode, with networking disabled.
Hashes do not authenticate authorship. Never archive untrusted Python here;
this is not a security sandbox or independent model/hardware attestation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_online_v4_archive as archive

TOOLS = ("inspect_online_v4.py", "inspect_agent_traces.py", "verify_online_v4_archive.py")
_VERIFY_ONLY = """import runpy, socket, sys
from pathlib import Path
def blocked(*args, **kwargs):
    raise RuntimeError('Network is forbidden during archived diagnosis verification')
socket.create_connection = blocked
socket.getaddrinfo = blocked
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.socket.sendto = blocked
script = sys.argv[1]
sys.path.insert(0, str(Path(script).parent))
sys.argv = sys.argv[1:]
if (len(sys.argv) != 6 or sys.argv[1] != '--run-dir'
        or sys.argv[3] != '--output-dir' or sys.argv[5] != '--verify'):
    raise RuntimeError('Only the read-only diagnosis verification mode is allowed')
runpy.run_path(script, run_name='__main__')
"""


def verify_diagnosis_archive(run_dir: Path, diagnosis_dir: Path) -> dict:
    run_dir, diagnosis_dir = archive._local(Path(run_dir)), archive._local(Path(diagnosis_dir))
    relative = run_dir.relative_to(archive.ROOT.resolve())
    if (len(relative.parts) < 2 or relative.parts[0] not in {".runs", "results"}
            or not run_dir.is_dir() or not diagnosis_dir.is_dir() or diagnosis_dir.parent != run_dir):
        raise ValueError("Use a trusted run below .runs/ or results/ and its immediate diagnosis subdirectory")
    core = archive.verify_archive(run_dir)
    if (not isinstance(core, dict) or core.get("status") != "PASS"
            or type(core.get("inference_calls")) is not int or core["inference_calls"] != 0):
        raise ValueError("Core archive verification must PASS without inference before diagnosis reads")
    observed: dict[Path, bytes] = {}
    run_files, protocols = archive._bundle(run_dir, observed)
    protocol = protocols[0]
    if core.get("protocol_sha256") != archive._digest(run_files["protocol.json"]):
        raise ValueError("Core receipt does not bind the captured protocol")
    artifacts = {name: archive._read(diagnosis_dir / name, observed)
                 for name in ("diagnosis.json", "DIAGNOSIS.md")}
    saved = archive._json(artifacts["diagnosis.json"])
    if not isinstance(saved, dict) or saved.get("schema_version") != "rcwt-online-trace-diagnosis/4":
        raise ValueError("Require a saved v4 diagnosis")
    provenance = saved.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Missing diagnostic provenance")
    snapshots = {"tools/" + name: archive._read(diagnosis_dir / "sources" / name, observed) for name in TOOLS}
    tool_hashes = {name: archive._digest(data) for name, data in snapshots.items()}
    input_hashes = {name: archive._digest(run_files[name]) for name in archive.INPUTS}
    if (provenance.get("diagnostic_sources_sha256") != tool_hashes
            or provenance.get("inspector_path") != "tools/inspect_online_v4.py"
            or provenance.get("inspector_sha256") != tool_hashes["tools/inspect_online_v4.py"]):
        raise ValueError("The exact three diagnostic tool snapshots must match their pinned hashes")
    if (provenance.get("files_sha256") != input_hashes
            or provenance.get("input_manifest_sha256") != archive._digest(archive._canonical(input_hashes).encode("utf-8"))
            or provenance.get("frozen_sources_sha256") != protocol["source_sha256"]):
        raise ValueError("Diagnostic input/source provenance does not match the captured run")
    historical = {name: archive._read(archive.ROOT / name, observed) for name in (
        *archive.HISTORY.values(), "results/agent_v2/protocol.json", "docs/rcwt_agent_runtime.json")}
    with tempfile.TemporaryDirectory(prefix="rcwt-v4-diagnosis-archive-") as temporary:
        workspace = Path(temporary)
        run_name = "results/archived-v4-run"
        copied = {run_name + "/" + name: data for name, data in run_files.items()}
        copied.update(historical)
        copied.update(snapshots)
        copied.update({"src/" + name: run_files["sources/" + name] for name in archive.NAMES})
        copied.update({run_name + "/" + diagnosis_dir.name + "/" + name: data for name, data in artifacts.items()})
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
            [sys.executable, "-I", "-B", "-c", _VERIFY_ONLY,
             str(workspace / "tools/inspect_online_v4.py"), "--run-dir", str(workspace / run_name),
             "--output-dir", str(workspace / run_name / diagnosis_dir.name), "--verify"],
            cwd=workspace, env=environment, capture_output=True, text=True, timeout=60, check=False)
        if result.returncode:
            raise ValueError("Archived diagnosis verification failed: " + result.stderr[-2000:])
        verified = archive._json(result.stdout)
    if (not isinstance(verified, dict) or verified.get("status") != "PASS"
            or verified.get("read_only") is not True or verified.get("feedback_to_agent") is not False
            or type(verified.get("n_episodes")) is not int or verified["n_episodes"] != protocol["count"]
            or type(verified.get("decisions")) is not int or verified["decisions"] != protocol["count"] * 16
            or verified.get("diagnosis_sha256") != archive._digest(artifacts["diagnosis.json"])
            or verified.get("report_sha256") != archive._digest(artifacts["DIAGNOSIS.md"])):
        raise ValueError("Archived diagnosis did not return the complete, byte-bound PASS receipt")
    for path, data in list(observed.items()):
        if archive._read(path, observed) != data:
            raise ValueError("Original evidence changed during diagnosis verification")
    return {"status": "PASS", "mode": "trusted archived diagnosis / offline replay",
            "protocol_sha256": archive._digest(run_files["protocol.json"]),
            "diagnostic_sources_sha256": tool_hashes, "frozen_sources_sha256": protocol["source_sha256"],
            "diagnosis_sha256": verified["diagnosis_sha256"], "report_sha256": verified["report_sha256"],
            "archive_verifier_sha256": archive.digest(Path(archive.__file__)),
            "diagnosis_archive_verifier_sha256": archive.digest(Path(__file__)),
            "core_verification": core, "verification": verified, "inference_calls": 0,
            "original_writes": 0, "read_only": True,
            "limit": "Trusted local snapshots only; integrity is not authorship, model-call/hardware attestation, or evidence of performance gain."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--diagnosis-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_diagnosis_archive(args.run_dir, args.diagnosis_dir), indent=2))


if __name__ == "__main__":
    main()
