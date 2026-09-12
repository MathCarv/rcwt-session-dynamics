"""Verify a trusted local v3 development archive with its own source snapshot.

This executes archived Python from this repository, exclusively in verify mode.
Use only the project's trusted experiment bundles, not downloaded/untrusted
source snapshots. It neither invokes a model nor changes the archived evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("rcwt_agent_actor.py", "rcwt_agent_analysis.py", "rcwt_agent_env.py",
         "rcwt_agent_memory.py", "rcwt_agent_run.py", "rcwt_local_model.py",
         "rcwt_memory_v3.py", "rcwt_online_v3.py", "rcwt_analysis_v3.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_archive(directory: Path) -> dict:
    directory = directory.resolve(strict=True)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    expected = protocol["source_sha256"]
    names = NAMES + (("rcwt_retrieval_v3.py",) if "src/rcwt_retrieval_v3.py" in expected else ())
    names += (("rcwt_review_v3.py",) if "src/rcwt_review_v3.py" in expected else ())
    if protocol["mode"] != "development" or set(expected) != {"src/" + name for name in names}:
        raise ValueError("Expected a complete development source snapshot")
    for name in names:
        if digest(directory / "sources" / name) != expected["src/" + name]:
            raise ValueError("Source archive hash mismatch")
    # The experiment deliberately retained the six original v2 dependencies.
    inherited = json.loads((ROOT / "results/agent_v2/protocol.json").read_text(encoding="utf-8"))["source_sha256"]
    if any(expected.get(path) != value for path, value in inherited.items()):
        raise ValueError("Archived inherited code differs from the frozen v2 baseline")
    with tempfile.TemporaryDirectory(prefix="rcwt-v3-archive-") as temporary:
        temporary_root = Path(temporary)
        (temporary_root / "src").mkdir()
        (temporary_root / "docs").mkdir()
        (temporary_root / "results/agent_v2").mkdir(parents=True)
        for name in names:
            shutil.copyfile(directory / "sources" / name, temporary_root / "src" / name)
        (temporary_root / "docs/rcwt_agent_runtime.json").write_text(
            json.dumps(protocol["runtime"], ensure_ascii=False), encoding="utf-8")
        for name in ("protocol.json", "RESULTS.md", "DIAGNOSIS.md"):
            shutil.copyfile(ROOT / "results/agent_v2" / name, temporary_root / "results/agent_v2" / name)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run([sys.executable, str(temporary_root / "src/rcwt_online_v3.py"),
                                 "--stage", "verify", "--output-dir", str(directory)],
                                cwd=temporary_root, env=environment, text=True, capture_output=True,
                                timeout=60, check=False)
        if result.returncode:
            raise ValueError("Archived offline verification failed: " + result.stderr[-2000:])
        verified = json.loads(result.stdout)
    if verified.get("status") != "PASS":
        raise ValueError("Archive did not produce a PASS receipt")
    return {"status": "PASS", "mode": "trusted archived source / offline replay",
            "protocol_sha256": digest(directory / "protocol.json"),
            "verification": verified, "inference_calls": 0,
            "limit": "Executes the recorded trusted project source; not independent hardware or token attestation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
