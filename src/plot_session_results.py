"""Render the preregistered RCWT-S outcomes as a deterministic SVG.

The chart consumes only the scorer's ``aggregates.json``.  It rejects invalid
or incomplete treatment-by-budget matrices, fixes Matplotlib's SVG hash salt
and font configuration, strips metadata, and publishes the SVG atomically.

Example::

    PYTHONPATH=src python src/plot_session_results.py \
        --aggregates results/session_v1/aggregates.json \
        --output results/session_v1/decision_readiness.svg
"""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402  (backend must be selected first)
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

PROTOCOL_VERSION = "session-v1"
TREATMENT_ORDER = ("tail", "state_latest", "state_first")
TREATMENT_STYLE: dict[str, dict[str, str]] = {
    "tail": {
        "label": "Transcript tail",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "--",
    },
    "state_latest": {
        "label": "State · latest",
        "color": "#009E73",
        "marker": "s",
        "linestyle": "-",
    },
    "state_first": {
        "label": "State · first (ablation)",
        "color": "#D55E00",
        "marker": "^",
        "linestyle": ":",
    },
}

_SVG_METADATA_RE = re.compile(r"\s*<metadata>.*?</metadata>", re.DOTALL)
_SVG_ROOT_RE = re.compile(r"<svg\b([^>]*)>")


@dataclass(frozen=True, slots=True)
class MetricCell:
    """Validated metrics for one treatment/budget aggregate cell."""

    treatment: str
    budget: int
    n: int
    decision_rate: float
    decision_low: float
    decision_high: float
    stale_rate: float
    stale_low: float
    stale_high: float


@dataclass(frozen=True, slots=True)
class PlotData:
    """Validated plot inputs and scope labels derived from an aggregate."""

    cells: tuple[MetricCell, ...]
    budgets: tuple[int, ...]
    cases: int
    turns: tuple[int, ...]
    family_count: int


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _list(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a JSON array")
    return value


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _rate(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{where} must be finite and between 0 and 1")
    return number


def _interval(value: object, rate: float, where: str) -> tuple[float, float]:
    values = _list(value, where)
    if len(values) != 2:
        raise ValueError(f"{where} must contain [low, high]")
    low = _rate(values[0], f"{where}[0]")
    high = _rate(values[1], f"{where}[1]")
    if low > rate or rate > high:
        raise ValueError(f"{where} must contain its point estimate")
    return low, high


def _metric(
    row: Mapping[str, Any],
    *,
    name: str,
    flat_rate_key: str,
    flat_interval_keys: tuple[str, ...],
    where: str,
) -> tuple[float, float, float]:
    metric = _mapping(row.get(name), f"{where}.{name}")
    nested_rate = _rate(metric.get("rate"), f"{where}.{name}.rate")
    flat_rate = _rate(row.get(flat_rate_key), f"{where}.{flat_rate_key}")
    if not math.isclose(nested_rate, flat_rate, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{where} has inconsistent {name} point estimates")
    nested_interval = _interval(
        metric.get("cluster_bootstrap95"),
        flat_rate,
        f"{where}.{name}.cluster_bootstrap95",
    )
    for flat_interval_key in flat_interval_keys:
        flat_interval = _interval(
            row.get(flat_interval_key),
            flat_rate,
            f"{where}.{flat_interval_key}",
        )
        if flat_interval != nested_interval:
            raise ValueError(f"{where} has inconsistent {name} cluster intervals")
    return flat_rate, *nested_interval


def validate_aggregates(raw: object) -> PlotData:
    """Validate the scorer schema and return the exact plotting matrix."""

    aggregate = _mapping(raw, "aggregate")
    if aggregate.get("protocol") != PROTOCOL_VERSION:
        raise ValueError(
            f"aggregate.protocol must be {PROTOCOL_VERSION!r}, got "
            f"{aggregate.get('protocol')!r}"
        )
    validity = _mapping(aggregate.get("validity"), "aggregate.validity")
    if validity.get("valid") is not True or validity.get("status") != "PASS":
        raise ValueError("aggregate validity must be PASS before plotting")

    counts = _mapping(aggregate.get("counts"), "aggregate.counts")
    cases = _positive_int(counts.get("cases"), "aggregate.counts.cases")
    raw_budgets = _list(counts.get("budgets"), "aggregate.counts.budgets")
    budgets = tuple(
        sorted(_positive_int(value, "aggregate.counts.budgets[]") for value in raw_budgets)
    )
    if not budgets or len(budgets) != len(set(budgets)):
        raise ValueError("aggregate.counts.budgets must be non-empty and unique")
    raw_turns = _list(counts.get("turns"), "aggregate.counts.turns")
    turns = tuple(
        sorted(_positive_int(value, "aggregate.counts.turns[]") for value in raw_turns)
    )
    if not turns or len(turns) != len(set(turns)):
        raise ValueError("aggregate.counts.turns must be non-empty and unique")
    families = _list(counts.get("families"), "aggregate.counts.families")
    if not families or any(not isinstance(value, str) or not value for value in families):
        raise ValueError("aggregate.counts.families must contain family names")
    treatments = _list(counts.get("treatments"), "aggregate.counts.treatments")
    if set(treatments) != set(TREATMENT_ORDER) or len(treatments) != len(TREATMENT_ORDER):
        raise ValueError("aggregate must contain the three RCWT-S treatments exactly once")

    breakdowns = _mapping(aggregate.get("breakdowns"), "aggregate.breakdowns")
    rows = _list(breakdowns.get("budget"), "aggregate.breakdowns.budget")
    cells: dict[tuple[str, int], MetricCell] = {}
    for index, item in enumerate(rows):
        where = f"aggregate.breakdowns.budget[{index}]"
        row = _mapping(item, where)
        treatment = row.get("treatment")
        if treatment not in TREATMENT_ORDER:
            raise ValueError(f"{where}.treatment is unsupported: {treatment!r}")
        budget = _positive_int(row.get("budget_tokens"), f"{where}.budget_tokens")
        n = _positive_int(row.get("n"), f"{where}.n")
        decision_rate, decision_low, decision_high = _metric(
            row,
            name="decision_ready",
            flat_rate_key="decision_ready_rate",
            flat_interval_keys=(
                "decision_ready_ci95",
                "decision_ready_cluster_bootstrap95",
            ),
            where=where,
        )
        stale_rate, stale_low, stale_high = _metric(
            row,
            name="stale_exposure",
            flat_rate_key="stale_exposure_rate",
            flat_interval_keys=(),
            where=where,
        )
        key = (str(treatment), budget)
        if key in cells:
            raise ValueError(f"duplicate budget/treatment cell: {key!r}")
        cells[key] = MetricCell(
            treatment=str(treatment),
            budget=budget,
            n=n,
            decision_rate=decision_rate,
            decision_low=decision_low,
            decision_high=decision_high,
            stale_rate=stale_rate,
            stale_low=stale_low,
            stale_high=stale_high,
        )

    expected = {
        (treatment, budget)
        for treatment in TREATMENT_ORDER
        for budget in budgets
    }
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        extra = sorted(set(cells) - expected)
        raise ValueError(f"incomplete budget matrix: missing={missing!r} extra={extra!r}")
    return PlotData(
        cells=tuple(cells[key] for key in sorted(cells)),
        budgets=budgets,
        cases=cases,
        turns=turns,
        family_count=len(set(families)),
    )


def load_plot_data(path: Path) -> PlotData:
    """Load and validate one scorer aggregate JSON document."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    return validate_aggregates(raw)


def _series(data: PlotData, treatment: str) -> tuple[MetricCell, ...]:
    indexed = {
        cell.budget: cell for cell in data.cells if cell.treatment == treatment
    }
    return tuple(indexed[budget] for budget in data.budgets)


def _plot_metric(
    axis: Axes,
    data: PlotData,
    *,
    rate_attribute: str,
    low_attribute: str,
    high_attribute: str,
    title: str,
    ylabel: str,
) -> None:
    for treatment in TREATMENT_ORDER:
        cells = _series(data, treatment)
        rates = [float(getattr(cell, rate_attribute)) for cell in cells]
        lows = [float(getattr(cell, low_attribute)) for cell in cells]
        highs = [float(getattr(cell, high_attribute)) for cell in cells]
        style = TREATMENT_STYLE[treatment]
        axis.errorbar(
            data.budgets,
            rates,
            yerr=(
                [rate - low for rate, low in zip(rates, lows)],
                [high - rate for rate, high in zip(rates, highs)],
            ),
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=7.2,
            markeredgecolor="white",
            markeredgewidth=0.9,
            capsize=3.5,
            elinewidth=1.15,
            zorder=3,
        )
    axis.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=11)
    axis.set_xlabel("Construction budget (cl100k_base tokens)")
    axis.set_ylabel(ylabel)
    axis.set_xscale("log", base=2)
    axis.set_xticks(data.budgets, labels=[f"{budget:,}" for budget in data.budgets])
    axis.set_ylim(-0.035, 1.035)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="y", color="#D9E2EA", linewidth=0.8, alpha=0.9)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A99A8")
    axis.spines["bottom"].set_color("#8A99A8")
    axis.tick_params(colors="#334155")
    axis.set_axisbelow(True)


def _accessible_svg(svg: str, data: PlotData) -> str:
    """Remove metadata and add stable SVG accessibility semantics."""

    without_metadata = _SVG_METADATA_RE.sub("", svg)
    title = "RCWT-S decision readiness by context budget"
    description = (
        f"Two-panel line chart for {data.cases} synthetic sessions. "
        "The first panel compares decision readiness and the second compares "
        "stale-fact exposure for transcript tail, latest state, and first-state "
        "ablation treatments across construction token budgets."
    )

    def replace_root(match: re.Match[str]) -> str:
        attributes = match.group(1)
        return (
            f'<svg role="img" aria-labelledby="rcwts-title rcwts-desc"{attributes}>'
            f'<title id="rcwts-title">{html.escape(title)}</title>'
            f'<desc id="rcwts-desc">{html.escape(description)}</desc>'
        )

    accessible, replacements = _SVG_ROOT_RE.subn(replace_root, without_metadata, count=1)
    if replacements != 1:
        raise ValueError("Matplotlib produced an SVG without a root element")
    return accessible.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"


def render_svg(data: PlotData) -> bytes:
    """Render validated aggregate data to byte-deterministic SVG content."""

    rc = {
        "svg.hashsalt": "rcwt-session-v1-decision-readiness",
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10.5,
        "axes.labelcolor": "#1F2937",
        "axes.titlecolor": "#111827",
        "text.color": "#111827",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    }
    with matplotlib.rc_context(rc):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(12.0, 5.8),
            gridspec_kw={"width_ratios": (1.55, 1.0), "wspace": 0.28},
        )
        main_axis, stale_axis = axes
        _plot_metric(
            main_axis,
            data,
            rate_attribute="decision_rate",
            low_attribute="decision_low",
            high_attribute="decision_high",
            title="A  Decision-ready contexts",
            ylabel="Decision-ready rate",
        )
        _plot_metric(
            stale_axis,
            data,
            rate_attribute="stale_rate",
            low_attribute="stale_low",
            high_attribute="stale_high",
            title="B  Stale-fact exposure",
            ylabel="Stale-exposure rate",
        )
        figure.suptitle(
            "RCWT-S · Decision readiness under session compaction",
            x=0.07,
            y=0.975,
            ha="left",
            fontsize=17,
            fontweight="bold",
        )
        turns = "/".join(str(turn) for turn in data.turns)
        figure.text(
            0.07,
            0.925,
            (
                f"{data.cases} synthetic sessions · checkpoints {turns} · "
                f"{data.family_count} families · 95% case-cluster bootstrap intervals"
            ),
            ha="left",
            va="top",
            fontsize=10.5,
            color="#475569",
        )
        handles, labels = main_axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.015),
            columnspacing=2.1,
            handlelength=3.0,
        )
        figure.subplots_adjust(left=0.075, right=0.975, top=0.82, bottom=0.19)
        buffer = io.StringIO()
        figure.savefig(
            buffer,
            format="svg",
            metadata={"Date": None, "Creator": None},
        )
        plt.close(figure)
    return _accessible_svg(buffer.getvalue(), data).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def plot_results(aggregates_path: Path, output_path: Path) -> None:
    """Validate an aggregate and atomically write its deterministic SVG."""

    if output_path.suffix.lower() != ".svg":
        raise ValueError("--output must use the .svg extension")
    data = load_plot_data(aggregates_path)
    _atomic_write(output_path, render_svg(data))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the aggregate-to-SVG command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with concise fail-closed diagnostics."""

    args = parse_args(argv)
    try:
        plot_results(args.aggregates, args.output)
    except (OSError, ValueError) as exc:
        print(f"plot_session_results: {exc}", file=sys.stderr)
        return 2
    print(f"wrote deterministic SVG to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
