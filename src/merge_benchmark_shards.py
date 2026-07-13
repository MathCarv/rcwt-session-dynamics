"""Merge RCWT benchmark shard outputs and rebuild aggregate artifacts."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from rcwt_benchmark_io import finalize_outputs, trial_key
from rcwt_benchmark_types import BenchmarkConfig
from rcwt_drop_pack import CONFIG as DROP_PACK_CONFIG
from rcwt_gsm8k import CONFIG as GSM8K_CONFIG
from rcwt_gsm8k_pack import CONFIG as GSM8K_PACK_CONFIG
from rcwt_mmlu_pro import CONFIG as MMLU_PRO_CONFIG
from rcwt_mmlu_pro_pack import CONFIG as MMLU_PRO_PACK_CONFIG

logger = logging.getLogger("merge_benchmark_shards")

CONFIGS: dict[str, BenchmarkConfig] = {
    DROP_PACK_CONFIG.name: DROP_PACK_CONFIG,
    GSM8K_CONFIG.name: GSM8K_CONFIG,
    GSM8K_PACK_CONFIG.name: GSM8K_PACK_CONFIG,
    MMLU_PRO_CONFIG.name: MMLU_PRO_CONFIG,
    MMLU_PRO_PACK_CONFIG.name: MMLU_PRO_PACK_CONFIG,
}


def load_records(paths: list[Path]) -> dict[tuple[str, str, str, float, str, int], dict[str, Any]]:
    """Load unique records from shard JSONL files, keyed by benchmark trial."""
    records: dict[tuple[str, str, str, float, str, int], dict[str, Any]] = {}
    duplicates = 0
    for path in paths:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = trial_key(record)
                if key in records:
                    duplicates += 1
                records[key] = record
    if duplicates:
        logger.warning("duplicate_records_overwritten duplicates=%d", duplicates)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write merged benchmark records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            json.dump(record, handle)
            handle.write("\n")


def sorted_records(
    records: dict[tuple[str, str, str, float, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return stable deterministic order for merged outputs."""
    return [records[key] for key in sorted(records)]


def merge_shards(benchmark: str, source_root: Path, output_dir: Path) -> None:
    """Merge all shard JSONL files for a benchmark into one output directory."""
    config = CONFIGS[benchmark]
    jsonl_name = f"rcwt_{benchmark}_responses.jsonl"
    paths = sorted(source_root.glob(f"shard_*/{jsonl_name}"))
    if not paths:
        raise FileNotFoundError(f"No shard files found under {source_root}")

    records = sorted_records(load_records(paths))
    merged_path = output_dir / jsonl_name
    write_jsonl(merged_path, records)
    finalize_outputs(output_dir, config)
    logger.info(
        "merge_complete benchmark=%s shards=%d records=%d output_dir=%s",
        benchmark,
        len(paths),
        len(records),
        output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    merge_shards(args.benchmark, args.source_root, args.output_dir)


if __name__ == "__main__":
    main()
