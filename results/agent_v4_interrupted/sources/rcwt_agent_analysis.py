"""Paired episode-cluster analysis for the local, online-memory experiment.

This module never invokes a model. Its input is one complete evaluation split
with one summary per (episode_id, policy). Bootstrap samples draw paired episode
indices, not individual decisions. Monetary API cost is zero for this explicitly
local-inference protocol; electricity, hardware and total cost remain unknown.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from statistics import fmean
from typing import Any


SCHEMA_VERSION = "rcwt-agent-analysis/1"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Each episode summary must be a dictionary")
    clean = {
        key: _name(record.get(key), key)
        for key in ("episode_id", "family", "split", "policy")
    }
    if clean["split"] not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    clean["steps"] = _integer(record.get("steps"), "steps", 1)
    for key in ("successes", "valid_actions", "unsafe_actions", "memory_truncations"):
        clean[key] = _integer(record.get(key), key)
        if clean[key] > clean["steps"]:
            raise ValueError(f"{key} cannot exceed steps")
    if clean["successes"] > clean["valid_actions"]:
        raise ValueError("successes cannot exceed valid_actions")
    if clean["unsafe_actions"] > clean["valid_actions"]:
        raise ValueError("unsafe_actions cannot exceed valid_actions")
    failures = record.get("failures")
    if not isinstance(failures, dict):
        raise ValueError("failures must map failure classes to integer counts")
    clean["failures"] = {
        _name(key, "failure class"): _integer(value, "failure count")
        for key, value in failures.items()
    }
    if sum(clean["failures"].values()) != clean["steps"] - clean["successes"]:
        raise ValueError("Failure counts must equal steps minus successes")
    for key in ("prompt_tokens", "completion_tokens", "model_calls"):
        clean[key] = _integer(record.get(key), key)
    clean["unsafe_booked_cents"] = _integer(record.get("unsafe_booked_cents", 0), "unsafe_booked_cents")
    for key in ("decision_seconds", "step_seconds"):
        values = record.get(key)
        if not isinstance(values, list) or len(values) != clean["steps"]:
            raise ValueError(f"{key} must contain exactly one value per step")
        clean[key] = [_number(value, key) for value in values]
    if any(
        step + 1e-9 < decision
        for step, decision in zip(clean["step_seconds"], clean["decision_seconds"])
    ):
        raise ValueError("step_seconds must include decision_seconds")
    clean["episode_seconds"] = _number(record.get("episode_seconds"), "episode_seconds")
    if clean["episode_seconds"] + 1e-8 < math.fsum(clean["step_seconds"]):
        raise ValueError("episode_seconds must include all step_seconds")
    return clean


def _quantile(values: list[float], probability: float) -> float:
    """Linearly interpolated sample quantile (inclusive endpoints / type 7)."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "p50": _quantile(values, 0.5),
        "p95": _quantile(values, 0.95),
        "total": math.fsum(values),
    }


def _policy_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    steps = sum(record["steps"] for record in records)
    successes = sum(record["successes"] for record in records)
    full_successes = sum(record["successes"] == record["steps"] for record in records)
    failures: Counter[str] = Counter()
    for record in records:
        failures.update(record["failures"])
    prompt_tokens = sum(record["prompt_tokens"] for record in records)
    completion_tokens = sum(record["completion_tokens"] for record in records)
    model_calls = sum(record["model_calls"] for record in records)
    valid_actions = sum(record["valid_actions"] for record in records)
    unsafe_actions = sum(record["unsafe_actions"] for record in records)
    unsafe_booked_cents = sum(record["unsafe_booked_cents"] for record in records)
    truncations = sum(record["memory_truncations"] for record in records)
    return {
        "n_episodes": len(records),
        "steps": steps,
        "successes": successes,
        "macro_episode_success_rate": fmean(record["successes"] / record["steps"] for record in records),
        "micro_step_success_rate": successes / steps,
        "full_success_episodes": full_successes,
        "full_success_rate": full_successes / len(records),
        "failure_counts": dict(sorted(failures.items())),
        "valid_actions": valid_actions,
        "valid_action_rate": valid_actions / steps,
        "unsafe_actions": unsafe_actions,
        "unsafe_action_rate": unsafe_actions / steps,
        "unsafe_booked_cents": unsafe_booked_cents,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_per_step": prompt_tokens / steps,
        "completion_tokens_per_step": completion_tokens / steps,
        "tokens_per_step": (prompt_tokens + completion_tokens) / steps,
        "model_calls": model_calls,
        "model_calls_per_step": model_calls / steps,
        "decision_latency_seconds": _latency([value for record in records for value in record["decision_seconds"]]),
        "step_latency_seconds": _latency([value for record in records for value in record["step_seconds"]]),
        "episode_latency_seconds": _latency([record["episode_seconds"] for record in records]),
        "memory_truncations": truncations,
        "memory_truncation_rate": truncations / steps,
        "api_cost_usd": 0.0,
        "total_monetary_cost_usd": None,
    }


def _comparison(
    candidate_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    # Pairs are already validated and sorted by episode_id. Subtracting each
    # pair before sampling is equivalent to sampling the same IDs in both arms.
    differences = [
        100 * (candidate["successes"] / candidate["steps"] - reference["successes"] / reference["steps"])
        for candidate, reference in zip(candidate_records, reference_records)
    ]
    count = len(differences)
    rng = random.Random(seed)
    bootstrap = [
        fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(bootstrap_samples)
    ]
    # Both arms have the same step count within each pair. Integer differences
    # avoid classifying an exact tie through floating-point rounding.
    wins = sum(candidate["successes"] > reference["successes"] for candidate, reference in zip(candidate_records, reference_records))
    losses = sum(candidate["successes"] < reference["successes"] for candidate, reference in zip(candidate_records, reference_records))
    return {
        "candidate_policy": candidate_records[0]["policy"],
        "reference_policy": reference_records[0]["policy"],
        "n_episodes": count,
        "delta_percentage_points": fmean(differences),
        "ci95_percentage_points": [_quantile(bootstrap, 0.025), _quantile(bootstrap, 0.975)],
        "wins": wins,
        "ties": count - wins - losses,
        "losses": losses,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "paired_episode",
        "bootstrap_method": "percentile",
        "bootstrap_seed": seed,
    }


def analyze(
    episodes: list[dict[str, Any]],
    baseline: str = "summary",
    candidate: str = "learned",
    bootstrap_samples: int = 5000,
    seed: int = 20260911,
) -> dict[str, Any]:
    """Analyze one complete split; missing pairs or invalid summaries fail closed.

    All present policies must cover identical IDs, family labels and step counts.
    The requested baseline, candidate and tail must be present. Primary gain is
    the unweighted mean of per-episode success-rate differences, in percentage
    points. Families are also reported, without claims of multiplicity control.
    """

    baseline, candidate = _name(baseline, "baseline"), _name(candidate, "candidate")
    if baseline == candidate:
        raise ValueError("Candidate and baseline must be distinct policies")
    _integer(bootstrap_samples, "bootstrap_samples", 1)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episodes must be a non-empty list")
    records = [_validate_record(record) for record in episodes]
    splits = {record["split"] for record in records}
    if len(splits) != 1:
        raise ValueError("Analyze one split at a time; training and held-out data cannot be pooled")
    by_policy: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        group = by_policy.setdefault(record["policy"], {})
        if record["episode_id"] in group:
            raise ValueError(f"Duplicate episode/policy pair: {record['episode_id']} / {record['policy']}")
        group[record["episode_id"]] = record
    missing_policies = {baseline, candidate, "tail"} - by_policy.keys()
    if missing_policies:
        raise ValueError(f"Missing required policies: {', '.join(sorted(missing_policies))}")
    expected_ids = set(by_policy[baseline])
    for policy, group in by_policy.items():
        if set(group) != expected_ids:
            raise ValueError(f"Incomplete paired episode cohort for policy {policy}")
        for episode_id, record in group.items():
            reference = by_policy[baseline][episode_id]
            if (record["family"], record["steps"]) != (reference["family"], reference["steps"]):
                raise ValueError(f"Mismatched family or steps for paired episode {episode_id}")
    ids = sorted(expected_ids)
    families = sorted({by_policy[baseline][episode_id]["family"] for episode_id in ids})
    policies: dict[str, Any] = {}
    for policy, group in sorted(by_policy.items()):
        ordered = [group[episode_id] for episode_id in ids]
        policies[policy] = _policy_metrics(ordered)
        policies[policy]["by_family"] = {
            family: _policy_metrics([record for record in ordered if record["family"] == family])
            for family in families
        }
    comparisons: dict[str, Any] = {}
    for reference_policy in dict.fromkeys((baseline, "tail")):
        if reference_policy == candidate:
            continue
        candidate_records = [by_policy[candidate][episode_id] for episode_id in ids]
        reference_records = [by_policy[reference_policy][episode_id] for episode_id in ids]
        result = _comparison(candidate_records, reference_records, bootstrap_samples, seed)
        result["by_family"] = {
            family: _comparison(
                [record for record in candidate_records if record["family"] == family],
                [record for record in reference_records if record["family"] == family],
                bootstrap_samples,
                seed,
            )
            for family in families
        }
        comparisons[f"{candidate}_vs_{reference_policy}"] = result
    primary = comparisons[f"{candidate}_vs_{baseline}"]
    normalized = sorted(records, key=lambda record: (record["policy"], record["episode_id"]))
    fingerprint = hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "split": next(iter(splits)),
        "input_sha256": fingerprint,
        "n_episodes": len(ids),
        "families": families,
        "policies": policies,
        "comparisons": comparisons,
        "improvement_gate": {
            "candidate_policy": candidate,
            "reference_policy": baseline,
            "criterion": "lower endpoint of paired episode bootstrap 95% CI > 0 percentage points",
            "passed": primary["ci95_percentage_points"][0] > 0,
            "accuracy_only": True,
        },
        "limitations": [
            "Synthetic workflow tasks, one local quantized model, and no weight updates; not evidence of general recursive self-improvement.",
            "Confidence intervals cover episode sampling from this generator, not transfer to other domains, models, or stochastic inference runs.",
            "Latency depends on hardware, runtime, load, cache state, and execution order; it is not a universal model property.",
            "Zero API charge does not mean zero electricity, hardware, or total monetary cost; total monetary cost is unknown.",
            "Totals cover the supplied evaluation episodes; one-time training and policy-search costs must be reported separately.",
            "The accuracy gate does not certify latency, token-efficiency, safety, or cost improvements; family intervals are descriptive and not multiplicity-adjusted.",
        ],
    }


def render_report(analysis: dict[str, Any]) -> str:
    """Render the supplied analysis without changing or recomputing evidence."""

    if analysis.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported agent analysis schema")
    gate = analysis["improvement_gate"]
    selected = analysis.get("selected_policy")
    baseline_retained = isinstance(selected, dict) and selected.get("kind") == "summary"
    status = "PASS" if gate["passed"] and not baseline_retained else "NOT DEMONSTRATED"
    lines = [
        "# Local online-memory agent evaluation",
        "",
        f"Split: `{analysis['split']}`. Paired episodes: {analysis['n_episodes']}.",
        "",
        f"Accuracy improvement against `{gate['reference_policy']}`: **{status}**.",
        f"Gate: {gate['criterion']}.",
    ]
    if selected:
        lines += ["", f"Frozen validation choice: `{selected['name']}` (kind `{selected['kind']}`)."]
        if baseline_retained:
            lines += ["The summary baseline was retained: no memory-policy change was selected. The arm named `learned` is the unchanged baseline, not evidence of a learned improvement."]
        else:
            lines += ["The selected memory instruction was frozen before held-out evaluation; this changes a compression instruction, not model weights."]
    if "verification" in analysis:
        verification = analysis["verification"]
        lines += ["", f"Offline evidence verification: **{verification['status']}**; verified steps: {verification.get('verified_steps', 'not reported')}.", f"Verification scope: {verification.get('scope', 'not reported')}."]
    lines += ["", "| Policy | Episode-macro success | Entire episode success | Tokens / step | Calls | Decision p50 / p95 (s) | Step p50 / p95 (s) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for policy, metrics in analysis["policies"].items():
        decision, step = metrics["decision_latency_seconds"], metrics["step_latency_seconds"]
        lines.append(
            f"| {policy} | {100 * metrics['macro_episode_success_rate']:.1f}% | "
            f"{100 * metrics['full_success_rate']:.1f}% | {metrics['tokens_per_step']:.1f} | "
            f"{metrics['model_calls']} | {decision['p50']:.3f} / {decision['p95']:.3f} | "
            f"{step['p50']:.3f} / {step['p95']:.3f} |"
        )
    lines += ["", "Step latency includes memory compaction; decision latency is reported separately.", "", "## Paired comparisons", ""]
    for comparison in analysis["comparisons"].values():
        low, high = comparison["ci95_percentage_points"]
        lines.append(
            f"- `{comparison['candidate_policy']}` vs `{comparison['reference_policy']}`: "
            f"{comparison['delta_percentage_points']:+.2f} pp, 95% CI [{low:+.2f}, {high:+.2f}]; "
            f"n={comparison['n_episodes']} episodes; wins/ties/losses="
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']}."
        )
    lines += ["", "Bootstrap unit: paired episode; the same sampled episode indices are used in both arms. Decisions are not treated as independent samples.", "", "## By family", "", "| Family | Policy | Episode-macro success | Entire episode success |", "| --- | --- | ---: | ---: |"]
    for family in analysis["families"]:
        for policy, metrics in analysis["policies"].items():
            family_metrics = metrics["by_family"][family]
            lines.append(f"| {family} | {policy} | {100 * family_metrics['macro_episode_success_rate']:.1f}% | {100 * family_metrics['full_success_rate']:.1f}% |")
    lines += ["", "## Failure and resource accounting", ""]
    for policy, metrics in analysis["policies"].items():
        failures = ", ".join(f"{name}={count}" for name, count in metrics["failure_counts"].items()) or "none"
        lines.append(
            f"- `{policy}`: failures {failures}; valid actions {metrics['valid_actions']}/{metrics['steps']}; "
            f"unsafe monetary-action attempts {metrics['unsafe_actions']}; "
            f"unsafe amount actually booked {metrics['unsafe_booked_cents']} fictional cents; prompt/completion tokens "
            f"{metrics['prompt_tokens']}/{metrics['completion_tokens']}; memory truncations "
            f"{metrics['memory_truncations']}; episode wall time {metrics['episode_latency_seconds']['total']:.3f} s."
        )
    lines += ["", "Unsafe monetary-action attempts include rejected calls. Booked amounts are separate fictional ledger counters, not real transactions.", "", "API charge: US$0 under the local-inference protocol. Total monetary cost: unknown, not zero."]
    if "training_and_search_cost" in analysis:
        overhead = analysis["training_and_search_cost"]
        evaluation_calls = sum(metrics["model_calls"] for metrics in analysis["policies"].values())
        evaluation_prompt = sum(metrics["prompt_tokens"] for metrics in analysis["policies"].values())
        evaluation_completion = sum(metrics["completion_tokens"] for metrics in analysis["policies"].values())
        evaluation_seconds = math.fsum(metrics["episode_latency_seconds"]["total"] for metrics in analysis["policies"].values())
        lines += [
            "", "## Training and policy-search overhead", "",
            "This includes training episodes, the policy-proposal call, and validation of all candidates, including candidates not selected.", "",
            f"- Training/search: {overhead['model_calls']} model calls; {overhead['prompt_tokens']} prompt + {overhead['completion_tokens']} completion tokens ({overhead['prompt_tokens'] + overhead['completion_tokens']} total); {overhead['wall_seconds']:.3f} s accounted wall time.",
            f"- Held-out evaluation, all arms: {evaluation_calls} model calls; {evaluation_prompt} prompt + {evaluation_completion} completion tokens ({evaluation_prompt + evaluation_completion} total); {evaluation_seconds:.3f} s accounted wall time.",
            f"- Combined training/search and evaluation: {overhead['model_calls'] + evaluation_calls} model calls; {overhead['prompt_tokens'] + overhead['completion_tokens'] + evaluation_prompt + evaluation_completion} tokens; {overhead['wall_seconds'] + evaluation_seconds:.3f} s accounted wall time.",
            "", "API charge remains US$0. Electricity, hardware depreciation, downloads/setup, and total monetary cost are not priced; these are not included in a claim of free execution.",
        ]
    lines += ["", "## Limits", ""]
    lines.extend(f"- {limitation}" for limitation in analysis["limitations"])
    lines += ["", f"Normalized input SHA-256: `{analysis['input_sha256']}`.", ""]
    return "\n".join(lines)
