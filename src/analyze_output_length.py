"""Analyze output length as a potential mediating variable for quality collapse.

Tests whether quality collapse (θ effect) is driven by models generating shorter
responses under compression — a simpler mechanism that would not require a reasoning
threshold explanation.

Logic:
  - If output length collapses at the SAME proportion as quality → confound possible.
  - If quality collapses BEFORE output length does → length compression alone cannot
    explain the quality drop; the θ interpretation is strengthened.

Datasets used:
  W=4096 (main): Gemini + Haiku (p=0..90), GPT (p=0..98 via cliff_n20)
  W=16384: all three models (p=0..99)

Output: summary table + per-model CSV for inclusion in paper Appendix E.

Usage:
    cd <artifact-root>
    python3 analyze_output_length.py
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import NamedTuple

import tiktoken

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parents[1] / "results"

# (label, window_size, model_short, jsonl_path, [extra_jsonl_paths])
DATASETS: list[tuple[str, int, str, list[Path]]] = [
    ("Gemini W=4K",  4096,  "gemini",
        [BASE / "rcwt_controlled_responses.jsonl"]),
    ("Haiku W=4K",   4096,  "haiku",
        [BASE / "haiku/rcwt_controlled_responses.jsonl"]),
    ("GPT W=4K",     4096,  "gpt",
        [BASE / "gpt/rcwt_controlled_responses.jsonl",
         BASE / "cliff_n20/rcwt_controlled_responses.jsonl"]),
    ("Gemini W=16K", 16384, "gemini",
        [BASE / "w16384/gemini/rcwt_controlled_responses.jsonl"]),
    ("Haiku W=16K",  16384, "haiku",
        [BASE / "w16384/haiku/rcwt_controlled_responses.jsonl"]),
    ("GPT W=16K",    16384, "gpt",
        [BASE / "w16384/gpt/rcwt_controlled_responses.jsonl"]),
]

# Empirical quality cliff proportions from paper (Table in §6.5)
# These are the p₀ values where quality drops sharply.
QUALITY_CLIFF: dict[tuple[int, str], float] = {
    (4096,  "gemini"): 0.913,
    (4096,  "haiku"):  0.917,
    (4096,  "gpt"):    0.939,
    (16384, "gemini"): 0.978,
    (16384, "haiku"):  0.979,
    (16384, "gpt"):    0.984,
}

ENC = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Cell(NamedTuple):
    proportion: float
    mean_output_tokens: float
    std_output_tokens: float
    n: int
    mean_chars: float


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def token_count(text: str) -> int:
    return len(ENC.encode(text))


def aggregate_by_proportion(records: list[dict]) -> dict[float, Cell]:
    by_prop: dict[float, list[int]] = {}
    chars_by_prop: dict[float, list[int]] = {}
    for rec in records:
        prop = rec["proportion"]
        resp = rec.get("response", "")
        toks = token_count(resp)
        by_prop.setdefault(prop, []).append(toks)
        chars_by_prop.setdefault(prop, []).append(len(resp))

    result = {}
    for prop, tok_list in sorted(by_prop.items()):
        n = len(tok_list)
        mean = sum(tok_list) / n
        variance = sum((t - mean) ** 2 for t in tok_list) / n if n > 1 else 0.0
        std = math.sqrt(variance)
        mean_chars = sum(chars_by_prop[prop]) / n
        result[prop] = Cell(prop, mean, std, n, mean_chars)
    return result


def length_cliff_proportion(cells: dict[float, Cell], baseline_mean: float,
                             threshold: float = 0.30) -> float | None:
    """Return the first proportion where output length drops below
    (1 - threshold) * baseline_mean.  Returns None if never crossed."""
    cutoff = baseline_mean * (1 - threshold)
    for prop in sorted(cells):
        if cells[prop].mean_output_tokens < cutoff:
            return prop
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 78)
    print("OUTPUT LENGTH vs QUALITY COLLAPSE ANALYSIS")
    print("=" * 78)
    print(f"\n{'Dataset':<16} {'p₀ quality':>10} {'Length –30%':>12} {'Length –50%':>12}  {'Verdict'}")
    print("-" * 78)

    all_rows = []

    for label, W, model_short, paths in DATASETS:
        records = load_records(paths)
        if not records:
            print(f"{label:<16} {'N/A':>10} {'N/A':>12} {'N/A':>12}  skipped: response JSONL not included")
            continue
        cells = aggregate_by_proportion(records)
        baseline_prop = min(cells)
        baseline_mean = cells[baseline_prop].mean_output_tokens

        p0_quality = QUALITY_CLIFF.get((W, model_short))
        p_len_30 = length_cliff_proportion(cells, baseline_mean, threshold=0.30)
        p_len_50 = length_cliff_proportion(cells, baseline_mean, threshold=0.50)

        if p0_quality is not None and p_len_30 is not None:
            gap = p_len_30 - p0_quality
            if gap >= 0.02:
                verdict = f"✓ quality collapses {gap*100:.1f}pp BEFORE length (θ supported)"
            elif gap <= -0.02:
                verdict = f"⚠ length collapses {-gap*100:.1f}pp BEFORE quality (confound risk)"
            else:
                verdict = "~ collapse simultaneous (ambiguous)"
        else:
            verdict = "— data insufficient for cliff comparison"

        p_len_30_str = f"{p_len_30:.3f}" if p_len_30 is not None else "never"
        p_len_50_str = f"{p_len_50:.3f}" if p_len_50 is not None else "never"
        p0_str = f"{p0_quality:.3f}" if p0_quality is not None else "—"

        print(f"{label:<16} {p0_str:>10} {p_len_30_str:>12} {p_len_50_str:>12}  {verdict}")

        # Detailed per-proportion table
        all_rows.append({
            "dataset": label,
            "window": W,
            "model": model_short,
            "baseline_output_tokens": round(baseline_mean, 1),
            "p0_quality": p0_quality,
            "length_cliff_30pct": p_len_30,
            "length_cliff_50pct": p_len_50,
            "verdict": verdict,
            "cells": cells,
        })

    # Per-model detail tables
    print("\n\n" + "=" * 78)
    print("DETAILED OUTPUT LENGTH PER CELL")
    print("=" * 78)

    for row in all_rows:
        cells = row["cells"]
        W = row["window"]
        baseline = row["baseline_output_tokens"]
        p0q = row["p0_quality"]

        print(f"\n{row['dataset']}  (baseline mean: {baseline:.0f} tok)")
        print(f"  {'p':>6}  {'mean_tok':>9}  {'%baseline':>9}  {'std':>6}  {'N':>4}  {'flag'}")
        print(f"  {'-'*55}")
        for prop, cell in sorted(cells.items()):
            pct = cell.mean_output_tokens / baseline * 100 if baseline > 0 else 0
            flag = ""
            if p0q is not None and abs(prop - p0q) < 0.008:
                flag = "← quality cliff"
            elif pct < 70:
                flag = "⚠ length –30%+"
            print(f"  {prop:>6.1%}  {cell.mean_output_tokens:>9.1f}  {pct:>9.1f}%  "
                  f"{cell.std_output_tokens:>6.1f}  {cell.n:>4}  {flag}")

    # Save CSV for paper appendix
    output_csv = BASE / "output_length_analysis.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "window", "model", "proportion",
                         "mean_output_tokens", "std_output_tokens", "n",
                         "mean_chars", "pct_of_baseline"])
        for row in all_rows:
            cells = row["cells"]
            baseline = row["baseline_output_tokens"]
            for prop, cell in sorted(cells.items()):
                pct = cell.mean_output_tokens / baseline * 100 if baseline > 0 else 0
                writer.writerow([row["dataset"], row["window"], row["model"],
                                  prop, round(cell.mean_output_tokens, 2),
                                  round(cell.std_output_tokens, 2), cell.n,
                                  round(cell.mean_chars, 1), round(pct, 2)])
    print(f"\n\nCSV saved → {output_csv}")

    # Summary interpretation
    print("\n" + "=" * 78)
    print("INTERPRETATION SUMMARY")
    print("=" * 78)
    total = len(all_rows)
    supported = sum(1 for r in all_rows if "BEFORE length" in r["verdict"] and "✓" in r["verdict"])
    ambiguous = sum(1 for r in all_rows if "simultaneous" in r["verdict"])
    confound = sum(1 for r in all_rows if "⚠" in r["verdict"] and "BEFORE quality" in r["verdict"])
    insufficient = total - supported - ambiguous - confound

    print(f"\n  θ supported (quality drops before length):  {supported}/{total}")
    print(f"  Confound risk (length drops first):         {confound}/{total}")
    print(f"  Ambiguous (simultaneous):                   {ambiguous}/{total}")
    print(f"  Insufficient data:                          {insufficient}/{total}")

    if supported >= total * 0.75:
        print("\n  → STRONG SUPPORT for θ interpretation:")
        print("    Quality collapse precedes output length compression in most datasets.")
        print("    The observed cliff is not explained by models generating shorter answers.")
    elif confound >= total * 0.5:
        print("\n  → CONFOUND WARNING:")
        print("    Output length drops at similar or earlier proportions than quality.")
        print("    The θ claim requires qualification — length compression may drive results.")
    else:
        print("\n  → MIXED: results are dataset-dependent. Report per-model.")


if __name__ == "__main__":
    main()
