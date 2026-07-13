"""Re-score saved model responses with a different judge (L2 cross-family check).

Loads existing rcwt_controlled_responses.jsonl and re-judges with a specified
model, then compares aggregate scores to the original CSV.

Usage:
    PYTHONPATH=src python src/rescore_responses.py \
        --responses results/haiku/rcwt_controlled_responses.jsonl \
        --original-csv results/haiku/rcwt_controlled.csv \
        --judge-model gpt-4.1-mini \
        --output-dir results/l2_rescore/haiku_judged_by_gpt
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rescore_responses")

# ---------------------------------------------------------------------------
# Import judge infrastructure from rcwt_controlled
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from rcwt_controlled import (
    JUDGE_ITEMS,
    EFFECTIVE_ITEMS,
    NOISE_ITEMS,
    JUDGE_PROMPT,
    MODELS,
    call_model,
    redacted_metering,
    wilson_ci,
)

REASONING_TASK_SHORT = "Technical specification recall task."


def judge_response(judge_model: str, response: str) -> dict[str, int]:
    """Re-judge a response with the specified model."""
    user_msg = f"## Original Task\n{REASONING_TASK_SHORT}\n\n## Response to Evaluate\n{response}"
    global _JUDGE_MODEL_OVERRIDE
    _JUDGE_MODEL_OVERRIDE = judge_model

    # Temporarily patch JUDGE_MODEL via direct call
    text, in_tok, out_tok = call_model(
        judge_model, JUDGE_PROMPT, user_msg, max_tokens=200, temperature=0.0
    )
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        scores = json.loads(text[start:end])
        return {k: min(1, max(0, int(scores.get(k, 0)))) for k in JUDGE_ITEMS}
    except (ValueError, json.JSONDecodeError):
        logger.warning("judge_parse_failed raw=%s", text[:200])
        return {k: 0 for k in JUDGE_ITEMS}


def judge_response_with_retry(
    judge_model: str,
    response: str,
    max_retries: int,
    retry_sleep_seconds: float,
) -> dict[str, int]:
    """Re-judge a response, backing off on provider rate-limit/transient errors."""
    for attempt in range(max_retries + 1):
        try:
            return judge_response(judge_model, response)
        except Exception as exc:
            if attempt >= max_retries:
                logger.error(
                    "judge_failed judge_model=%s attempt=%d error_type=%s error=%s",
                    judge_model,
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                )
                raise
            sleep_for = retry_sleep_seconds * (2 ** attempt)
            logger.warning(
                "judge_retry judge_model=%s attempt=%d sleep_seconds=%.1f "
                "error_type=%s error=%s",
                judge_model,
                attempt + 1,
                sleep_for,
                type(exc).__name__,
                exc,
            )
            time.sleep(sleep_for)
    return {k: 0 for k in JUDGE_ITEMS}


def load_responses(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def key_for_record(rec: dict[str, Any]) -> tuple[float, str, int]:
    """Stable trial key used across CSV, JSONL, and resume files."""
    return (float(rec["proportion"]), str(rec["order"]), int(rec["trial_index"]))


def key_to_str(key: tuple[float, str, int]) -> str:
    return f"{key[0]:.6f}|{key[1]}|{key[2]}"


def load_partial_scores(path: Path) -> dict[tuple[float, str, int], dict[str, int]]:
    """Load incrementally saved judge scores from a JSONL resume file."""
    scores: dict[tuple[float, str, int], dict[str, int]] = {}
    if not path.exists():
        return scores
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (float(rec["proportion"]), str(rec["order"]), int(rec["trial_index"]))
            scores[key] = {k: int(rec["scores"].get(k, 0)) for k in JUDGE_ITEMS}
    return scores


def load_original_scores(csv_path: Path) -> dict[tuple[float, str, int], dict[str, int]]:
    """Load original judge scores keyed by (proportion, order, trial_index)."""
    scores: dict[tuple[float, str, int], dict[str, int]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (float(row["proportion"]), row["order"], int(row["trial_index"]))
            scores[key] = {item: int(row[item]) for item in JUDGE_ITEMS if item in row}
    return scores


def aggregate_by_proportion(
    records: list[dict],
    new_scores: dict[tuple[float, str, int], dict[str, int]],
) -> dict[float, dict]:
    """Pool new scores by proportion."""
    by_prop: dict[float, list] = {}
    for rec in records:
        key = (rec["proportion"], rec["order"], rec["trial_index"])
        s = new_scores.get(key)
        if s is None:
            continue
        prop = rec["proportion"]
        by_prop.setdefault(prop, []).append(s)

    result = {}
    for prop, score_list in by_prop.items():
        n = len(score_list)
        hits = sum(s[i] for s in score_list for i in EFFECTIVE_ITEMS)
        possible = n * len(EFFECTIVE_ITEMS)
        mean_eff = hits / possible if possible > 0 else 0.0
        ci = wilson_ci(hits, possible)
        result[prop] = {"mean_effective": mean_eff, "ci95": ci, "n": n}
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Re-score saved responses with cross-family judge")
    p.add_argument("--responses", required=True, help="Path to rcwt_controlled_responses.jsonl")
    p.add_argument("--original-csv", required=True, help="Path to original rcwt_controlled.csv")
    p.add_argument("--judge-model", required=True,
                   help="Model to use as judge (e.g. gpt-4.1-mini, gemini-2.5-flash)")
    p.add_argument("--output-dir", required=True, help="Output directory for results")
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep after each successful judge call; useful for rate-limited providers.",
    )
    p.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=20.0,
        help="Initial exponential-backoff sleep for failed judge calls.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per judge call.",
    )
    p.add_argument(
        "--max-new-calls",
        type=int,
        default=None,
        help="Stop after this many newly executed judge calls; skipped resume rows do not count.",
    )
    args = p.parse_args()

    responses_path = Path(args.responses)
    original_csv_path = Path(args.original_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_model = args.judge_model
    judge_provider = MODELS.get(judge_model, {}).get("provider")
    if not judge_provider:
        logger.error("Unknown judge model: %s", judge_model)
        sys.exit(1)

    env_checks = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GEMINI_API_KEY",
    }
    if not os.environ.get(env_checks.get(judge_provider, "")):
        logger.error("Missing API key for judge provider: %s", judge_provider)
        sys.exit(1)

    logger.info("Loading responses from %s", responses_path)
    records = load_responses(responses_path)
    logger.info("Loaded %d responses", len(records))

    logger.info("Loading original scores from %s", original_csv_path)
    original_scores = load_original_scores(original_csv_path)

    safe_judge = judge_model.replace("/", "-")
    partial_path = output_dir / f"rescore_partial_{safe_judge}.jsonl"
    new_scores = load_partial_scores(partial_path)
    if new_scores:
        logger.info("loaded_partial_scores path=%s records=%d", partial_path, len(new_scores))

    total_metering_redacted = 0.0
    t0 = time.monotonic()
    new_calls = 0

    for i, rec in enumerate(records):
        key = key_for_record(rec)
        if key in new_scores:
            scores = new_scores[key]
            skipped = True
        else:
            if args.max_new_calls is not None and new_calls >= args.max_new_calls:
                logger.info(
                    "max_new_calls_reached max_new_calls=%d scored_records=%d",
                    args.max_new_calls,
                    len(new_scores),
                )
                break
            scores = judge_response_with_retry(
                judge_model,
                rec["response"],
                max_retries=args.max_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            new_scores[key] = scores
            new_calls += 1
            skipped = False
            with open(partial_path, "a") as f:
                json.dump(
                    {
                        "key": key_to_str(key),
                        "proportion": key[0],
                        "order": key[1],
                        "trial_index": key[2],
                        "judge_model": judge_model,
                        "scores": scores,
                    },
                    f,
                )
                f.write("\n")

        orig = original_scores.get(key, {})
        mean_new = sum(scores[it] for it in EFFECTIVE_ITEMS) / len(EFFECTIVE_ITEMS)
        mean_orig = sum(orig.get(it, 0) for it in EFFECTIVE_ITEMS) / len(EFFECTIVE_ITEMS) if orig else float("nan")

        # Rough metering placeholder for a judge prompt plus response excerpt
        total_metering_redacted += redacted_metering(judge_model, 200, 50)

        if (i + 1) % 20 == 0:
            elapsed = time.monotonic() - t0
            logger.info(
                "progress index=%d total=%d prop=%.0f%% new=%.3f orig=%.3f "
                "metering=%.3f elapsed=%.0fs skipped=%s",
                i + 1,
                len(records),
                rec["proportion"] * 100,
                mean_new,
                mean_orig,
                total_metering_redacted,
                elapsed,
                skipped,
            )
        if not skipped and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    elapsed = time.monotonic() - t0
    logger.info("rescoring_done records=%d metering=%.3f time=%.0fs", len(records), total_metering_redacted, elapsed)

    # Aggregate by proportion: new vs original
    new_by_prop = aggregate_by_proportion(records, new_scores)
    orig_by_prop = aggregate_by_proportion(records, original_scores)

    proportions = sorted(new_by_prop.keys())

    print("\n" + "=" * 72)
    print(f"L2 CROSS-FAMILY JUDGE COMPARISON")
    print(f"  Judge model (new): {judge_model}")
    print(f"  Original judge: claude-haiku-4-5 (same-family)")
    print(f"  Responses: {len(records)}")
    print(f"  Total rescore metering redacted: {total_metering_redacted:.3f}")
    print("=" * 72)
    print(f"\n  {'Prop':>6}  {'Original':>10}  {'New Judge':>10}  {'Δ':>8}  {'N':>4}")
    print(f"  {'-' * 50}")

    max_delta = 0.0
    for prop in proportions:
        nd = new_by_prop.get(prop, {})
        od = orig_by_prop.get(prop, {})
        n_mean = nd.get("mean_effective", float("nan"))
        o_mean = od.get("mean_effective", float("nan"))
        delta = n_mean - o_mean
        max_delta = max(max_delta, abs(delta))
        flag = " ⚠" if abs(delta) > 0.05 else ""
        print(f"  {prop:>6.1%}  {o_mean:>10.3f}  {n_mean:>10.3f}  {delta:>+8.3f}{flag}  {nd.get('n', 0):>4}")

    print(f"\n  Max |Δ| across proportions: {max_delta:.3f}")
    if max_delta < 0.05:
        print("  → Same-family bias is MINIMAL (<5pp). L2 concern resolved.")
    elif max_delta < 0.10:
        print("  → Same-family bias is MODERATE (5-10pp). Note in limitations.")
    else:
        print("  → Same-family bias is SUBSTANTIAL (>10pp). Revisit Haiku scores.")
    print()

    # Save detailed comparison CSV
    comparison_path = output_dir / f"rescore_comparison_{judge_model.replace('/', '-')}.csv"
    with open(comparison_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "proportion", "order", "trial_index",
        ] + [f"orig_{it}" for it in JUDGE_ITEMS] + [f"new_{it}" for it in JUDGE_ITEMS])
        writer.writeheader()
        for rec in records:
            key = (rec["proportion"], rec["order"], rec["trial_index"])
            orig = original_scores.get(key, {k: -1 for k in JUDGE_ITEMS})
            new = new_scores.get(key, {k: -1 for k in JUDGE_ITEMS})
            row = {
                "proportion": rec["proportion"],
                "order": rec["order"],
                "trial_index": rec["trial_index"],
            }
            row.update({f"orig_{k}": orig.get(k, -1) for k in JUDGE_ITEMS})
            row.update({f"new_{k}": new.get(k, -1) for k in JUDGE_ITEMS})
            writer.writerow(row)

    # Save aggregates JSON
    agg_path = output_dir / f"rescore_aggregates_{judge_model.replace('/', '-')}.json"
    agg_data = []
    for prop in proportions:
        agg_data.append({
            "proportion": prop,
            "original_judge": {
                "mean_effective": orig_by_prop.get(prop, {}).get("mean_effective"),
                "ci95": orig_by_prop.get(prop, {}).get("ci95"),
                "n": orig_by_prop.get(prop, {}).get("n"),
            },
            "new_judge": {
                "model": judge_model,
                "mean_effective": new_by_prop.get(prop, {}).get("mean_effective"),
                "ci95": new_by_prop.get(prop, {}).get("ci95"),
                "n": new_by_prop.get(prop, {}).get("n"),
            },
        })
    with open(agg_path, "w") as f:
        json.dump(agg_data, f, indent=2)

    logger.info("comparison_csv_saved path=%s", comparison_path)
    logger.info("aggregates_saved path=%s", agg_path)


if __name__ == "__main__":
    main()
