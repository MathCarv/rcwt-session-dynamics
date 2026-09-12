"""Reconcile completed trusted v4 bundles with one explicit local server log.

Scope is this log and immediate .runs/v4_* directories, not global attestation
that no other endpoint, directory, log, or unrecorded run exists. No model calls
or new corpus generation occur; archived verification replays recorded inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import verify_online_v4_archive as archive_verifier

SCHEMA = "rcwt-v4-campaign-audit/1"
MARKER = "processing task, is_child = 0"
CORE = archive_verifier.INPUTS
OUTPUTS = ("campaign.json", "CAMPAIGN.md")
LIMIT = ("Accounting for exactly the supplied endpoint log and immediate project .runs/v4_* directories. "
         "Not independent inference attestation, proof of global non-repetition, confirmation of model gain, "
         "or assurance that other endpoints, directories, omitted/rotated logs, or unrecorded runs do not exist. "
         "Reverification depends on the local inventory; this is not portable global replay.")


class CampaignBlocked(ValueError):
    pass


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return archive_verifier._json(Path(path).read_bytes())


def _relative(path):
    try:
        return Path(path).resolve(strict=True).relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise CampaignBlocked("Receipt inputs, including the server log, must be copied inside project ROOT") from exc


def _manifest(directory):
    flags = [name for name in ("aborted.json", "partial-step.json") if (directory / name).exists()]
    missing = [name for name in CORE if not (directory / name).is_file()]
    if flags or missing:
        raise CampaignBlocked(f"Incomplete run {_relative(directory)}: flags={flags}, missing={missing}")
    return {name: _hash(directory / name) for name in (*CORE, *("sources/" + name for name in archive_verifier.NAMES))}


def _inventory():
    parent = ROOT / ".runs"
    if not parent.is_dir():
        raise CampaignBlocked("The scoped .runs directory is absent")
    records, problems = [], []
    for directory in sorted(parent.iterdir()):
        if not directory.name.startswith("v4_") or not directory.is_dir():
            continue
        try:
            if directory.is_symlink() or directory.resolve().parent != parent.resolve():
                raise CampaignBlocked(f"Redirected inventory directory: {directory.name}")
            records.append({"path": _relative(directory), "files_sha256": _manifest(directory)})
        except (OSError, ValueError) as exc:
            problems.append(str(exc))
    if problems:
        raise CampaignBlocked("Inventory blocks complete accounting: " + "; ".join(problems))
    return records


def _totals(rows, count):
    if not isinstance(rows, list) or len(rows) != count * 2:
        raise CampaignBlocked("Incomplete episode summaries")
    totals = {policy: {key: 0 for key in ("successes", "steps", "unsafe_actions", "unsafe_booked_cents", "model_calls")}
              for policy in ("summary", "structured")}
    pairs = set()
    for row in rows:
        policy = row.get("policy")
        pair = (row.get("episode_id"), policy)
        if policy not in totals or not isinstance(pair[0], str) or pair in pairs or row.get("steps") != 8:
            raise CampaignBlocked("Malformed or duplicate episode/policy pair")
        pairs.add(pair)
        for key in totals[policy]:
            value = row.get(key)
            if type(value) is not int or value < 0:
                raise CampaignBlocked("Missing or invalid recorded counter: " + key)
            totals[policy][key] += value
    if ({episode for episode, policy in pairs if policy == "summary"}
            != {episode for episode, policy in pairs if policy == "structured"}):
        raise CampaignBlocked("Unpaired episode cohort")
    base, candidate = totals["summary"], totals["structured"]
    passed = (candidate["successes"] > base["successes"] and candidate["unsafe_actions"] <= base["unsafe_actions"]
              and candidate["unsafe_booked_cents"] <= base["unsafe_booked_cents"])
    return totals, passed


def campaign_facts(runs, server_log):
    paths = sorted({Path(path).resolve(strict=True) for path in runs})
    if not paths:
        raise CampaignBlocked("Explicit completed run paths are required")
    relative_paths = [_relative(path) for path in paths]
    inventory = _inventory()
    server_log = Path(server_log).resolve(strict=True)
    log_relative = _relative(server_log)
    log_bytes = server_log.read_bytes()
    log_hash = hashlib.sha256(log_bytes).hexdigest()
    script_hashes = {"tools/audit_online_v4_campaign.py": _hash(Path(__file__)),
                     "tools/verify_online_v4_archive.py": _hash(Path(archive_verifier.__file__))}
    unique, observations = {}, {}
    for directory in paths:
        manifest = _manifest(directory)
        verified = archive_verifier.verify_archive(directory)
        if (not isinstance(verified, dict) or verified.get("status") != "PASS" or verified.get("inference_calls") != 0
                or verified.get("protocol_sha256") != manifest["protocol.json"]):
            raise CampaignBlocked("Archive verification must bind a complete offline PASS")
        protocol = _read(directory / "protocol.json")
        mode, count = protocol.get("mode"), protocol.get("count")
        if (protocol.get("schema") != "rcwt-online-memory/4" or type(count) is not int
                or (mode, count, protocol.get("split")) not in (("development", 8, "train"), ("confirmatory", 32, "test"))
                or verified.get("verified_steps") != count * 16):
            raise CampaignBlocked("Only complete v4 development-8 and confirmation-32 runs are permitted")
        timestamp = datetime.fromisoformat(protocol["frozen_at_utc"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise CampaignBlocked("Run ordering requires timezone-aware frozen timestamps")
        rows = [archive_verifier._json(line) for line in (directory / "episodes.jsonl").read_bytes().splitlines()]
        totals, passed = _totals(rows, count)
        identity = manifest["protocol.json"]
        record = {"protocol_sha256": identity, "mode": mode, "count": count, "frozen_at_utc": timestamp.isoformat(),
                  "files_sha256": manifest, "source_sha256": protocol["source_sha256"], "verification": verified,
                  "model_calls": sum(row["model_calls"] for row in rows), "policy_totals": totals,
                  "development_screen_passed": passed if mode == "development" else None,
                  "selected_development": protocol.get("development"), "paths": [_relative(directory)]}
        if identity in unique:
            if unique[identity]["files_sha256"] != manifest:
                raise CampaignBlocked("Repeated protocol has different evidence; it is not a release copy")
            unique[identity]["paths"].append(_relative(directory))
        else:
            unique[identity] = record
        observations[directory] = manifest
    inventory_ids = {item["files_sha256"]["protocol.json"] for item in inventory}
    if inventory_ids != set(unique):
        raise CampaignBlocked("Explicit protocol hashes must exactly cover the scoped .runs/v4_* inventory")
    for item in inventory:
        if unique[item["files_sha256"]["protocol.json"]]["files_sha256"] != item["files_sha256"]:
            raise CampaignBlocked("Inventory copy differs from explicitly verified evidence")
    ordered = sorted(unique.values(), key=lambda item: (datetime.fromisoformat(item["frozen_at_utc"]), item["protocol_sha256"]))
    dev = [item for item in ordered if item["mode"] == "development"]
    conf = [item for item in ordered if item["mode"] == "confirmatory"]
    if not 1 <= len(dev) <= 2 or len(conf) > 1:
        raise CampaignBlocked("Campaign permits at most two development attempts and one confirmation")
    passed = [index for index, item in enumerate(dev) if item["development_screen_passed"]]
    if passed and passed[0] != len(dev) - 1:
        raise CampaignBlocked("Development continued after the first PASS")
    selected = dev[-1]["protocol_sha256"] if passed else None
    if conf and (selected is None or not isinstance(conf[0]["selected_development"], dict)
                 or conf[0]["selected_development"].get("protocol_sha256") != selected
                 or datetime.fromisoformat(conf[0]["frozen_at_utc"]) <= datetime.fromisoformat(dev[-1]["frozen_at_utc"])):
        raise CampaignBlocked("Confirmation must follow and select the first passing development attempt")
    tasks = sum(MARKER in line for line in log_bytes.decode("utf-8", errors="strict").splitlines())
    calls = sum(item["model_calls"] for item in ordered)
    if tasks != calls:
        raise CampaignBlocked(f"Server-log accounting mismatch: {tasks} parent tasks versus {calls} recorded model calls")
    if (_inventory() != inventory or _hash(server_log) != log_hash
            or any(_manifest(path) != manifest for path, manifest in observations.items())
            or script_hashes["tools/audit_online_v4_campaign.py"] != _hash(Path(__file__))
            or script_hashes["tools/verify_online_v4_archive.py"] != _hash(Path(archive_verifier.__file__))):
        raise CampaignBlocked("Campaign evidence or verifier changed during audit")
    return {"schema": SCHEMA, "status": "PASS", "scope": LIMIT, "runs": ordered,
            "explicit_paths": relative_paths, "inventory": inventory,
            "unique_protocols": len(ordered), "copies_not_double_counted": len(paths) - len(ordered),
            "selected_development_protocol_sha256": selected, "confirmation_present": bool(conf),
            "model_calls": calls, "server_log": {"path": log_relative, "sha256": log_hash,
                                                  "literal_marker": MARKER, "matching_lines": tasks},
            "source_sha256": script_hashes, "inference_calls": 0, "api_cost_usd": 0,
            "total_monetary_cost_usd": None, "global_non_repetition_proven": False}


def render_report(facts):
    lines = ["# V4 campaign accounting", "", "Status: PASS (scoped accounting, not model gain).", "", facts["scope"], "",
             f"Recorded calls: {facts['model_calls']}; matching server parent-task lines: {facts['server_log']['matching_lines']}.",
             f"Unique protocols: {facts['unique_protocols']}; release copies not counted twice: {facts['copies_not_double_counted']}.",
             f"Selected development: `{facts['selected_development_protocol_sha256']}`; confirmation present: {facts['confirmation_present']}.", ""]
    for run in facts["runs"]:
        lines.append(f"- {run['mode']}: `{run['protocol_sha256']}`; {run['count']} episodes; {run['model_calls']} calls; development screen={run['development_screen_passed']}.")
    lines += ["", f"Server log SHA-256: `{facts['server_log']['sha256']}`.",
              "No failed attempt is excluded and no quality outcomes are pooled. API charge: US$0; electricity, hardware and total monetary cost are unknown, not zero.", ""]
    return "\n".join(lines)


def audit_campaign(runs, server_log, output_dir, *, verify=False):
    facts = campaign_facts(runs, server_log)
    output_dir = Path(output_dir)
    contents = (_canonical_bytes(facts), render_report(facts).encode("utf-8"))
    if verify:
        if any((output_dir / name).read_bytes() != content for name, content in zip(OUTPUTS, contents)):
            raise CampaignBlocked("Saved campaign receipt or report differs from recomputed evidence")
        return {"status": "PASS", "read_only": True, "model_calls": facts["model_calls"], "inference_calls": 0, "scope": LIMIT}
    if any((output_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError("Campaign artifacts exist; use --verify or a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in zip(OUTPUTS, contents):
        with (output_dir / name).open("xb") as handle:
            handle.write(content)
    return facts


def _canonical_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        result = audit_campaign(args.runs, args.server_log, args.output_dir, verify=args.verify)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc), "scope": LIMIT}))
        raise SystemExit(1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
