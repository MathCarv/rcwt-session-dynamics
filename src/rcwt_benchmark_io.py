"""Persistence and aggregation helpers for RCWT benchmark runners."""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path

from rcwt_benchmark_types import BenchmarkConfig, TrialRecord
from rcwt_controlled import MODELS, wilson_ci

logger = logging.getLogger("rcwt_benchmark")


def trial_key(record: dict[str, object]) -> tuple[str, str, str, float, str, int]:
    return (
        str(record["benchmark"]),
        str(record["model"]),
        str(record["item_id"]),
        float(record["proportion"]),
        str(record["order"]),
        int(record["trial_index"]),
    )


def load_existing(path: Path) -> dict[tuple[str, str, str, float, str, int], dict[str, object]]:
    records: dict[tuple[str, str, str, float, str, int], dict[str, object]] = {}
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[trial_key(rec)] = rec
    return records


def append_record(path: Path, record: TrialRecord) -> None:
    with open(path, "a") as f:
        json.dump(asdict(record), f)
        f.write("\n")


def write_csv(records: list[dict[str, object]], path: Path) -> None:
    if not records:
        return
    fieldnames = [key for key in records[0].keys() if key != "response"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: value for key, value in record.items() if key != "response"}
            writer.writerow(row)


def aggregate_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = {}
    for record in records:
        key = (str(record["model"]), float(record["proportion"]), str(record["order"]))
        grouped.setdefault(key, []).append(record)
    aggregates: list[dict[str, object]] = []
    for (model, proportion, order), group in sorted(grouped.items()):
        n = len(group)
        hits = sum(
            int(record.get("score_hits", 1 if bool(record["correct"]) else 0))
            for record in group
        )
        possible = sum(int(record.get("score_total", 1)) for record in group)
        mean = hits / possible if possible else 0.0
        ci_low, ci_high = wilson_ci(hits, possible)
        truncated_count = sum(1 for record in group if bool(record["task_truncated"]))
        total_metering_redacted = sum(float(record["metering_redacted"]) for record in group)
        aggregates.append(
            {
                "model": model,
                "provider": MODELS[model]["provider"],
                "proportion": proportion,
                "order": order,
                "n": n,
                "score_total": possible,
                "accuracy": mean,
                "ci95": [ci_low, ci_high],
                "task_truncated_rate": truncated_count / n if n else 0.0,
                "mean_metering_redacted": total_metering_redacted / n if n else 0.0,
            }
        )
    return aggregates


def finalize_outputs(output_dir: Path, config: BenchmarkConfig) -> None:
    jsonl_path = output_dir / f"rcwt_{config.name}_responses.jsonl"
    records = list(load_existing(jsonl_path).values())
    write_csv(records, output_dir / f"rcwt_{config.name}.csv")
    aggregates = aggregate_records(records)
    with open(output_dir / f"rcwt_{config.name}_aggregates.json", "w") as f:
        json.dump(
            {
                "benchmark": config.name,
                "records": len(records),
                "total_metering_redacted": round(sum(float(r["metering_redacted"]) for r in records), 4),
                "aggregates": aggregates,
            },
            f,
            indent=2,
        )
    logger.info(
        "outputs_saved benchmark=%s records=%d output_dir=%s",
        config.name,
        len(records),
        output_dir,
    )
