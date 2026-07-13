"""R(c, W, T) Task-2 experiment — algorithmic trace reasoning.

Validates that the coordination-overhead / quality degradation phenomenon
(logistic decay, c* ≈ 91-94%) generalises beyond technical-spec recall
(Task 1, rcwt_controlled.py) to a second distinct cognitive task type:
*generative computation* (trace a Python function through given inputs).

Design mirrors rcwt_controlled.py exactly:
  - 2×5 factorial: order × proportion
  - Proportions: {0%, 25%, 50%, 75%, 90%}
  - N=20 trials per (proportion, order) cell
  - coord_first | reason_first order randomisation
  - Call-level uncertainty summarized by the release analysis
  - Haiku judge (binary 0/1 per question)

New task: given process_batch(items=[3,10,15,7,10,22,4], threshold=10),
answer 10 deterministic binary questions about intermediate states and output
values.  All ground-truth answers pre-verified by executing the function.

Usage:
    PYTHONPATH=src python src/rcwt_task2.py
    PYTHONPATH=src python src/rcwt_task2.py --model gemini-2.5-flash --n-trials 20
    PYTHONPATH=src python src/rcwt_task2.py --model gpt-4.1-mini --proportions 0,0.90
    PYTHONPATH=src python src/rcwt_task2.py --model claude-haiku-4-5-20251001 \\
        --output-dir results/task2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken

from rcwt_statistics import stable_seed, stratified_bootstrap_mean_ci

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rcwt_task2")

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def build_text_to_tokens(text: str, target_tokens: int) -> str:
    """Repeat or truncate text to hit approximately target_tokens."""
    if target_tokens <= 0:
        return ""
    tokens = _ENCODER.encode(text)
    if len(tokens) >= target_tokens:
        return _ENCODER.decode(tokens[:target_tokens])
    repeated = tokens * ((target_tokens // len(tokens)) + 1)
    return _ENCODER.decode(repeated[:target_tokens])


# ---------------------------------------------------------------------------
# Task 2: Algorithmic trace reasoning
# ---------------------------------------------------------------------------

TASK_DESCRIPTION = """You are analyzing the following Python function and input values.
Study the code carefully, then answer ALL questions below. Trace the function
step-by-step with the given inputs to determine the exact values.

def process_batch(items: list[int], threshold: int = 10) -> dict:
    result = {"above": [], "below": [], "equal": [], "sum": 0}

    for i, val in enumerate(items):
        result["sum"] += val
        if val > threshold:
            result["above"].append(i)
        elif val < threshold:
            result["below"].append(i)
        else:
            result["equal"].append(i)

    result["mean"] = result["sum"] / len(items) if items else 0.0
    result["max_above"] = max((items[i] for i in result["above"]), default=None)
    return result

Input: items=[3, 10, 15, 7, 10, 22, 4], threshold=10

For each question below, answer YES or NO based on your trace.
Format your answers as:
sum_correct: YES/NO
above_count: YES/NO
below_count: YES/NO
equal_count: YES/NO
above_indices: YES/NO
equal_indices: YES/NO
mean_correct: YES/NO
max_above_correct: YES/NO
below_has_zero: YES/NO
no_negatives: YES/NO

Questions:
1. sum_correct: Is result['sum'] equal to 71?
2. above_count: Does result['above'] have exactly 2 elements?
3. below_count: Does result['below'] have exactly 3 elements?
4. equal_count: Does result['equal'] have exactly 2 elements?
5. above_indices: Are indices 2 and 5 in result['above']?
6. equal_indices: Are indices 1 and 4 in result['equal']?
7. mean_correct: Is result['mean'] approximately 10.14 (i.e. 71/7)?
8. max_above_correct: Is result['max_above'] equal to 22?
9. below_has_zero: Is index 0 in result['below']?
10. no_negatives: Does result['above'] contain index 3 (value=7)?"""

# Ground truth: True = correct answer is YES, False = correct answer is NO
# Verified by executing: items=[3,10,15,7,10,22,4], threshold=10
#   sum = 3+10+15+7+10+22+4 = 71
#   above (val>10): idx 2 (15), idx 5 (22)  → 2 elements, indices [2,5]
#   below (val<10): idx 0 (3), idx 3 (7), idx 6 (4) → 3 elements
#   equal (val==10): idx 1 (10), idx 4 (10) → 2 elements, indices [1,4]
#   mean = 71/7 ≈ 10.1429
#   max_above = max(15, 22) = 22
#   idx 0 → value 3 < 10 → in below ✓
#   idx 3 → value 7 < 10 → NOT in above → question "does above contain idx 3?" → False
JUDGE_ITEMS_DICT: dict[str, tuple[str, bool]] = {
    "sum_correct":       ("Is result['sum'] equal to 71?", True),
    "above_count":       ("Does result['above'] have exactly 2 elements?", True),
    "below_count":       ("Does result['below'] have exactly 3 elements?", True),
    "equal_count":       ("Does result['equal'] have exactly 2 elements?", True),
    "above_indices":     ("Are indices 2 and 5 in result['above']?", True),
    "equal_indices":     ("Are indices 1 and 4 in result['equal']?", True),
    "mean_correct":      ("Is result['mean'] approximately 10.14 (71/7)?", True),
    "max_above_correct": ("Is result['max_above'] equal to 22?", True),
    "below_has_zero":    ("Is index 0 in result['below']?", True),
    "no_negatives":      ("Does result['above'] contain index 3 (value=7)?", False),
}

JUDGE_ITEMS: list[str] = list(JUDGE_ITEMS_DICT.keys())

# No noise items in task 2 — all questions are about concrete computed values
# that a model can answer correctly if it has access to the reasoning task.
# We keep the same raw/effective split for structural parity; effective == raw here.
NOISE_ITEMS: set[str] = set()
EFFECTIVE_ITEMS: list[str] = JUDGE_ITEMS[:]

# ---------------------------------------------------------------------------
# Coordination block (reused from rcwt_controlled.py verbatim)
# ---------------------------------------------------------------------------

COORDINATION_TEMPLATE = """## Agent Role & Protocol
You are Agent-3 (Technical Analyst) in a 5-agent system. Your role is to provide
technical analysis grounded in the shared state. Follow the Structured Response Protocol:
all outputs must include [ANALYSIS], [RISKS], and [RECOMMENDATION] sections.

## Communication History
Agent-1 (Product Lead) [Turn 4]: "We need to prioritize the real-time sync engine.
Customer feedback shows 73% of churn happens when collaboration breaks."

Agent-2 (Security Analyst) [Turn 5]: "E2E encryption with CRDT is an unsolved problem
at scale. Signal Protocol handles 1:1 but not N:N collaborative editing. We need to
evaluate MLS (Messaging Layer Security) RFC 9420 as a foundation."

Agent-4 (Frontend Lead) [Turn 6]: "CodeMirror 6 with y-codemirror.next gives us
CRDT-based real-time for code. But the plugin system for custom doc types needs a
different approach — each doc type needs its own CRDT schema."

Agent-5 (DevOps) [Turn 7]: "Current infra is a single PostgreSQL instance. For sub-100ms
we need either Redis pub/sub or WebSocket fan-out with horizontal scaling. I estimate
3 weeks just for the infra migration."

## Shared State (from shared-state snapshot)
[PROPOSITIONS — PROJECT]
- [decision, 95%] architecture: Event-sourced CRDT with WebSocket transport
- [constraint, 100%] deadline: 8 weeks hard deadline, no extensions
- [constraint, 100%] latency: Sub-100ms for character operations
- [fact, 90%] team_size: 3 backend, 2 frontend, 1 security
- [tension, 80%] encryption_vs_crdt: E2E encryption conflicts with CRDT merge
- [decision, 85%] offline: Use CRDT log for offline, merge on reconnect
- [inference, 70%] risk_infra: Infrastructure migration is critical path (3 weeks)
[/PROPOSITIONS — PROJECT]

## Tool Schemas Available
- analyze_dependency_graph(components: list[str]) -> DependencyGraph
- estimate_effort(task: str, team: list[str]) -> EffortEstimate
- query_shared_state(scope: str, truth_class: str) -> list[Proposition]

## Response Format Requirements
Your response MUST follow this structure:
[ANALYSIS] ... [/ANALYSIS]
[RISKS] ... [/RISKS]
[RECOMMENDATION] ... [/RECOMMENDATION]

Previous coordination tokens consumed: {consumed} of {budget} budget."""


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are evaluating whether a model correctly traced a Python function.
The model was asked to answer YES/NO to 10 questions about the function's output.

The CORRECT answers are:
1. sum_correct: YES (result['sum'] = 71)
2. above_count: YES (result['above'] has 2 elements: indices 2 and 5)
3. below_count: YES (result['below'] has 3 elements: indices 0, 3, 6)
4. equal_count: YES (result['equal'] has 2 elements: indices 1 and 4)
5. above_indices: YES (indices 2 and 5 are in result['above'])
6. equal_indices: YES (indices 1 and 4 are in result['equal'])
7. mean_correct: YES (mean = 71/7 ≈ 10.14)
8. max_above_correct: YES (max_above = 22)
9. below_has_zero: YES (index 0, value=3, is in result['below'])
10. no_negatives: NO (index 3, value=7, is NOT in result['above'] — 7 < 10)

For each item, score 1 if the model's answer matches the correct answer, 0 otherwise.
If the model did not answer a question (absent or unclear), score 0.

Respond ONLY with valid JSON (no other text):
{"sum_correct": 0, "above_count": 0, "below_count": 0, "equal_count": 0, "above_indices": 0, "equal_indices": 0, "mean_correct": 0, "max_above_correct": 0, "below_has_zero": 0, "no_negatives": 0}

Replace each 0 with 1 if the model answered correctly for that item."""


# ---------------------------------------------------------------------------
# Model registry (same as rcwt_controlled.py)
# ---------------------------------------------------------------------------

MODELS: dict[str, dict[str, str]] = {
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "tier": "cheap",
    },
    "gpt-4.1-mini": {
        "provider": "openai",
        "tier": "cheap",
    },
    "gemini-2.0-flash": {
        "provider": "google",
        "tier": "legacy",
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "tier": "cheap",
    },
}

JUDGE_MODEL = "claude-haiku-4-5-20251001"
THINKING_MODELS = {"gemini-2.5-flash"}


def redacted_metering(model: str, input_tokens: int, output_tokens: int) -> float:
    """Provider billing is omitted from public artifacts; this placeholder stays zero."""
    _ = (model, input_tokens, output_tokens)
    return 0.0


# ---------------------------------------------------------------------------
# Provider clients (identical to rcwt_controlled.py)
# ---------------------------------------------------------------------------

_clients: dict[str, object] = {}


def _get_anthropic_client():
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic()
    return _clients["anthropic"]


def _get_openai_client():
    if "openai" not in _clients:
        import openai
        _clients["openai"] = openai.OpenAI()
    return _clients["openai"]


def _get_google_client():
    if "google" not in _clients:
        from google import genai
        _clients["google"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _clients["google"]


def call_model(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> tuple[str, int, int]:
    if model in THINKING_MODELS:
        max_tokens = max(max_tokens, 8192)
    provider = MODELS.get(model, {}).get("provider", "anthropic")
    if provider == "anthropic":
        return _call_anthropic(model, system, user, max_tokens, temperature)
    elif provider == "openai":
        return _call_openai(model, system, user, max_tokens, temperature)
    elif provider == "google":
        return _call_google(model, system, user, max_tokens, temperature)
    raise ValueError(f"Unknown provider: {provider}")


def _call_anthropic(model: str, system: str, user: str, max_tokens: int, temperature: float) -> tuple[str, int, int]:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=system, messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens


def _call_openai(model: str, system: str, user: str, max_tokens: int, temperature: float) -> tuple[str, int, int]:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def _call_google(model: str, system: str, user: str, max_tokens: int, temperature: float) -> tuple[str, int, int]:
    from google.genai import types
    client = _get_google_client()
    response = client.models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    text = None
    try:
        text = response.text
    except (ValueError, AttributeError):
        pass
    if text is None:
        candidates = response.candidates or []
        if candidates and candidates[0].content and candidates[0].content.parts:
            parts = [p.text for p in candidates[0].content.parts
                     if hasattr(p, "text") and p.text]
            text = "\n".join(parts) if parts else ""
        else:
            text = ""
            logger.warning("gemini_empty_response model=%s", model)
    in_tok = (response.usage_metadata.prompt_token_count or 0) if response.usage_metadata else 0
    out_tok = (response.usage_metadata.candidates_token_count or 0) if response.usage_metadata else 0
    return text, in_tok, out_tok


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def build_coordination_context(target_tokens: int) -> str:
    base = COORDINATION_TEMPLATE.format(consumed=target_tokens, budget=target_tokens * 2)
    return build_text_to_tokens(base, target_tokens)


def assemble_system(
    coord_tokens: int,
    reason_tokens: int,
    order: str,
) -> str:
    """Assemble system prompt with explicit order control.

    For task 2 the 'reasoning block' is a brief framing header — the actual
    task content (function + questions) is in the user turn.  We still allocate
    reason_tokens to a filler block so total context size stays controlled.

    order='coord_first'  → [coordination block] then [task framing]
    order='reason_first' → [task framing] then [coordination block]
    """
    task_framing = (
        "You are a precise computational assistant. "
        "When given Python code and inputs, trace execution step-by-step "
        "to produce exact, deterministic answers. "
        "Do not guess — compute."
    )

    parts: list[str] = []
    coord_block = build_coordination_context(coord_tokens) if coord_tokens > 10 else ""
    reason_block = (
        build_text_to_tokens(task_framing, reason_tokens)
        if reason_tokens > 10
        else task_framing
    )

    if order == "coord_first":
        if coord_block:
            parts.append(coord_block)
        parts.append(reason_block)
    else:  # reason_first
        parts.append(reason_block)
        if coord_block:
            parts.append(coord_block)

    return "\n\n".join(parts) if parts else task_framing


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def judge_response(task: str, response: str) -> dict[str, int]:
    user_msg = f"## Original Task\n{task}\n\n## Model Response to Evaluate\n{response}"
    text, _, _ = call_model(JUDGE_MODEL, JUDGE_PROMPT, user_msg, max_tokens=256, temperature=0.0)
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        scores = json.loads(text[start:end])
        return {k: min(1, max(0, int(scores.get(k, 0)))) for k in JUDGE_ITEMS}
    except (ValueError, json.JSONDecodeError):
        logger.warning("judge_parse_failed raw=%s", text[:200])
        return {k: 0 for k in JUDGE_ITEMS}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    model: str
    provider: str
    proportion: float
    order: str
    trial_index: int
    coordination_tokens: int
    reasoning_tokens: int
    total_context_tokens: int
    response: str
    scores: dict[str, int]
    mean_score_raw: float
    mean_score_effective: float
    input_tokens_used: int
    output_tokens_used: int
    metering_redacted: float
    elapsed_ms: float


@dataclass
class ConditionAggregate:
    proportion: float
    order: str
    n_trials: int
    mean_raw: float
    std_raw: float
    ci95_raw: tuple[float, float]
    mean_effective: float
    std_effective: float
    ci95_effective: tuple[float, float]
    scores_by_dimension: dict[str, float]
    mean_metering_redacted: float


@dataclass
class ExperimentResult:
    model: str
    provider: str
    total_budget: int
    n_trials_per_cell: int
    aggregates: list[ConditionAggregate] = field(default_factory=list)
    trials: list[TrialResult] = field(default_factory=list)
    total_metering_redacted: float = 0.0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Statistics helpers (identical to rcwt_controlled.py)
# ---------------------------------------------------------------------------


def aggregate_trials(
    trials: list[TrialResult],
    proportion: float,
    order: str,
) -> ConditionAggregate:
    cell = [t for t in trials if t.proportion == proportion and t.order == order]
    n = len(cell)
    if n == 0:
        return ConditionAggregate(
            proportion=proportion, order=order, n_trials=0,
            mean_raw=0.0, std_raw=0.0, ci95_raw=(0.0, 0.0),
            mean_effective=0.0, std_effective=0.0, ci95_effective=(0.0, 0.0),
            scores_by_dimension={}, mean_metering_redacted=0.0,
        )

    raw_scores = [t.mean_score_raw for t in cell]
    eff_scores = [t.mean_score_effective for t in cell]
    mean_raw = sum(raw_scores) / n
    mean_eff = sum(eff_scores) / n
    std_raw = math.sqrt(sum((s - mean_raw) ** 2 for s in raw_scores) / n)
    std_eff = math.sqrt(sum((s - mean_eff) ** 2 for s in eff_scores) / n)

    _, ci95_eff_low, ci95_eff_high = stratified_bootstrap_mean_ci(
        {"calls": eff_scores},
        seed=stable_seed("task2-cell-effective", proportion, order),
    )
    _, ci95_raw_low, ci95_raw_high = stratified_bootstrap_mean_ci(
        {"calls": raw_scores},
        seed=stable_seed("task2-cell-raw", proportion, order),
    )
    ci95_eff = (ci95_eff_low, ci95_eff_high)
    ci95_raw = (ci95_raw_low, ci95_raw_high)

    scores_by_dim: dict[str, float] = {}
    for item in JUDGE_ITEMS:
        scores_by_dim[item] = sum(t.scores.get(item, 0) for t in cell) / n

    mean_metering_redacted = sum(t.metering_redacted for t in cell) / n

    return ConditionAggregate(
        proportion=proportion, order=order, n_trials=n,
        mean_raw=round(mean_raw, 4), std_raw=round(std_raw, 4), ci95_raw=ci95_raw,
        mean_effective=round(mean_eff, 4), std_effective=round(std_eff, 4),
        ci95_effective=ci95_eff,
        scores_by_dimension={k: round(v, 4) for k, v in scores_by_dim.items()},
        mean_metering_redacted=round(mean_metering_redacted, 6),
    )


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_trial(
    model: str,
    proportion: float,
    order: str,
    trial_index: int,
    total_budget: int,
) -> TrialResult:
    provider = MODELS[model]["provider"]
    task_tokens = count_tokens(TASK_DESCRIPTION)
    available = total_budget - task_tokens

    coord_tokens = int(available * proportion)
    reason_tokens = available - coord_tokens

    system = assemble_system(coord_tokens, reason_tokens, order)

    t0 = time.monotonic()
    response, in_tok, out_tok = call_model(model, system, TASK_DESCRIPTION)
    elapsed_ms = (time.monotonic() - t0) * 1000

    scores = judge_response(TASK_DESCRIPTION, response)
    metering_value = redacted_metering(model, in_tok, out_tok)

    mean_raw = sum(scores.values()) / len(JUDGE_ITEMS)
    mean_eff = sum(scores[i] for i in EFFECTIVE_ITEMS) / len(EFFECTIVE_ITEMS)

    logger.info(
        "trial model=%s prop=%.0f%% order=%s trial=%d "
        "raw=%.2f eff=%.2f metering=%.4f",
        model, proportion * 100, order, trial_index,
        mean_raw, mean_eff, metering_value,
    )

    return TrialResult(
        model=model, provider=provider, proportion=proportion,
        order=order, trial_index=trial_index,
        coordination_tokens=coord_tokens, reasoning_tokens=reason_tokens,
        total_context_tokens=total_budget, response=response, scores=scores,
        mean_score_raw=mean_raw, mean_score_effective=mean_eff,
        input_tokens_used=in_tok, output_tokens_used=out_tok,
        metering_redacted=metering_value, elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------


def run_experiment(
    models: list[str],
    proportions: list[float],
    n_trials: int,
    total_budget: int,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    orders = ["coord_first", "reason_first"]

    for model in models:
        provider = MODELS[model]["provider"]
        logger.info("=== experiment_start model=%s provider=%s task=task2 ===", model, provider)
        t0 = time.monotonic()

        exp = ExperimentResult(
            model=model, provider=provider,
            total_budget=total_budget, n_trials_per_cell=n_trials,
        )

        # Build trial schedule: (proportion, order) pairs, then shuffle
        schedule: list[tuple[float, str]] = []
        for prop in proportions:
            for order in orders:
                for _ in range(n_trials):
                    schedule.append((prop, order))
        random.shuffle(schedule)

        cell_idx: dict[tuple[float, str], int] = {}

        for prop, order in schedule:
            idx = cell_idx.get((prop, order), 0)
            cell_idx[(prop, order)] = idx + 1
            trial = run_trial(model, prop, order, idx, total_budget)
            exp.trials.append(trial)

        for prop in proportions:
            for order in orders:
                agg = aggregate_trials(exp.trials, prop, order)
                exp.aggregates.append(agg)
                logger.info(
                    "cell_done model=%s prop=%.0f%% order=%s n=%d "
                    "eff=%.3f±%.3f CI95=[%.3f,%.3f]",
                    model, prop * 100, order, agg.n_trials,
                    agg.mean_effective, agg.std_effective,
                    agg.ci95_effective[0], agg.ci95_effective[1],
                )

        exp.total_metering_redacted = sum(t.metering_redacted for t in exp.trials)
        exp.elapsed_seconds = time.monotonic() - t0
        results.append(exp)
        logger.info(
            "=== experiment_done model=%s metering=%.3f time=%.0fs ===",
            model, exp.total_metering_redacted, exp.elapsed_seconds,
        )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_csv(results: list[ExperimentResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "provider", "proportion", "order", "trial_index",
        "coordination_tokens", "reasoning_tokens", "total_context_tokens",
        "mean_score_raw", "mean_score_effective",
        "input_tokens_used", "output_tokens_used", "metering_redacted", "elapsed_ms",
    ] + JUDGE_ITEMS

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in results:
            for t in exp.trials:
                row = {
                    "model": t.model, "provider": t.provider,
                    "proportion": t.proportion, "order": t.order,
                    "trial_index": t.trial_index,
                    "coordination_tokens": t.coordination_tokens,
                    "reasoning_tokens": t.reasoning_tokens,
                    "total_context_tokens": t.total_context_tokens,
                    "mean_score_raw": t.mean_score_raw,
                    "mean_score_effective": t.mean_score_effective,
                    "input_tokens_used": t.input_tokens_used,
                    "output_tokens_used": t.output_tokens_used,
                    "metering_redacted": t.metering_redacted, "elapsed_ms": t.elapsed_ms,
                }
                row.update(t.scores)
                writer.writerow(row)

    responses_path = output_path.parent / "rcwt_task2_responses.jsonl"
    with open(responses_path, "w") as f:
        for exp in results:
            for t in exp.trials:
                json.dump({
                    "model": t.model, "proportion": t.proportion,
                    "order": t.order, "trial_index": t.trial_index,
                    "response": t.response,
                }, f)
                f.write("\n")
    logger.info("csv_saved path=%s", output_path)


def save_aggregates(results: list[ExperimentResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for exp in results:
        data.append({
            "model": exp.model, "provider": exp.provider,
            "task": "task2_algorithmic_trace",
            "total_budget": exp.total_budget,
            "n_trials_per_cell": exp.n_trials_per_cell,
            "total_metering_redacted": round(exp.total_metering_redacted, 4),
            "elapsed_seconds": round(exp.elapsed_seconds, 1),
            "noise_items_excluded": list(NOISE_ITEMS),
            "effective_items": EFFECTIVE_ITEMS,
            "aggregates": [
                {
                    "proportion": a.proportion,
                    "order": a.order,
                    "n_trials": a.n_trials,
                    "mean_effective": a.mean_effective,
                    "std_effective": a.std_effective,
                    "ci95_effective": list(a.ci95_effective),
                    "mean_raw": a.mean_raw,
                    "std_raw": a.std_raw,
                    "ci95_raw": list(a.ci95_raw),
                    "scores_by_dimension": a.scores_by_dimension,
                    "mean_metering_redacted": a.mean_metering_redacted,
                }
                for a in exp.aggregates
            ],
        })
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("aggregates_saved path=%s", output_path)


def print_summary(results: list[ExperimentResult]) -> None:
    total_metering_redacted = sum(r.total_metering_redacted for r in results)
    print("\n" + "=" * 72)
    print("R(c, W, T) TASK 2 RESULTS — ALGORITHMIC TRACE REASONING")
    print(f"Task: process_batch([3,10,15,7,10,22,4], threshold=10)")
    print(f"Items: {JUDGE_ITEMS}")
    print(f"Total metering redacted: {total_metering_redacted:.3f}")
    print("=" * 72)

    for exp in results:
        print(f"\n  {exp.model} ({exp.provider}) — metering_redacted={exp.total_metering_redacted:.3f} "
              f"in {exp.elapsed_seconds:.0f}s")
        print(f"  {'Prop':>5} {'Order':>12} {'N':>3} {'Eff':>6} {'±CI':>8} "
              f"{'Raw':>6} {'Δpos':>7}")
        print("  " + "-" * 58)

        proportions = sorted({a.proportion for a in exp.aggregates})
        for prop in proportions:
            cf = next((a for a in exp.aggregates
                       if a.proportion == prop and a.order == "coord_first"), None)
            rf = next((a for a in exp.aggregates
                       if a.proportion == prop and a.order == "reason_first"), None)
            for agg, tag in [(cf, "coord_first"), (rf, "reason_first")]:
                if not agg:
                    continue
                ci_half = (agg.ci95_effective[1] - agg.ci95_effective[0]) / 2
                delta = ""
                if cf and rf and tag == "reason_first":
                    d = cf.mean_effective - rf.mean_effective
                    delta = f"{d:+.3f}"
                print(f"  {prop:>4.0%} {tag:>12} {agg.n_trials:>3} "
                      f"{agg.mean_effective:>6.3f} ±{ci_half:.3f}  "
                      f"{agg.mean_raw:>6.3f} {delta:>7}")

    print()
    print("  Position-controlled degradation (pooled across orders):")
    for exp in results:
        baseline_trials = [t for t in exp.trials if t.proportion == 0.0]
        high_trials = [t for t in exp.trials if t.proportion == 0.90]
        if not baseline_trials or not high_trials:
            continue
        base_hits = sum(sum(t.scores[i] for i in EFFECTIVE_ITEMS) for t in baseline_trials)
        base_total = len(baseline_trials) * len(EFFECTIVE_ITEMS)
        high_hits = sum(sum(t.scores[i] for i in EFFECTIVE_ITEMS) for t in high_trials)
        high_total = len(high_trials) * len(EFFECTIVE_ITEMS)
        base_rate = base_hits / base_total if base_total > 0 else 0
        high_rate = high_hits / high_total if high_total > 0 else 0
        n = base_total + high_total
        k = base_hits + high_hits
        p_pool = k / n if n > 0 else 0
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / base_total + 1 / high_total)) if p_pool > 0 else 1
        z = (base_rate - high_rate) / se if se > 0 else 0
        sig = ("p<0.001" if abs(z) > 3.29 else
               ("p<0.01" if abs(z) > 2.58 else
                ("p<0.05" if abs(z) > 1.96 else "n.s.")))
        print(f"    {exp.model}: 0%={base_rate:.3f} vs 90%={high_rate:.3f} "
              f"Δ={base_rate - high_rate:.3f} z={z:.2f} {sig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCWT Task 2: algorithmic trace reasoning")
    p.add_argument(
        "--model", default=None,
        help="Single model to test (default: haiku + gpt-4.1-mini cheap tier)",
    )
    p.add_argument(
        "--n-trials", type=int, default=20,
        help="Trials per (proportion, order) cell (default: 20)",
    )
    p.add_argument(
        "--proportions", default="0,0.25,0.50,0.75,0.90",
        help="Comma-separated coordination proportions (default: 0,0.25,0.50,0.75,0.90)",
    )
    p.add_argument(
        "--budget", type=int, default=4096,
        help="Total context budget in tokens (default: 4096)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: results/task2/)",
    )
    p.add_argument(
        "--judge-model", default=None,
        help="Model to use as judge (default: claude-haiku-4-5-20251001)",
    )
    return p.parse_args()


def check_env() -> list[str]:
    available = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        available.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        available.append("openai")
    if os.environ.get("GEMINI_API_KEY"):
        available.append("google")
    return available


if __name__ == "__main__":
    args = parse_args()
    available = check_env()
    if not available:
        logger.error("No API keys set. Need at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY")
        sys.exit(1)

    logger.info("available_providers=%s", available)

    if args.model:
        selected_models = [args.model]
    else:
        selected_models = [
            m for m, info in MODELS.items()
            if info["provider"] in available and info["tier"] == "cheap"
        ]

    if args.judge_model:
        JUDGE_MODEL = args.judge_model  # type: ignore[assignment]
    logger.info("judge_model=%s", JUDGE_MODEL)

    judge_provider = MODELS.get(JUDGE_MODEL, {}).get("provider", "anthropic")
    if judge_provider not in available:
        fallback = next(
            (m for m in selected_models if MODELS[m]["provider"] in available), None
        )
        if fallback:
            logger.warning("judge_model_unavailable fallback=%s", fallback)
            JUDGE_MODEL = fallback  # type: ignore[assignment]
        else:
            logger.error("No judge model available")
            sys.exit(1)

    proportions = [float(p) for p in args.proportions.split(",")]

    logger.info(
        "config models=%s proportions=%s n_trials=%d budget=%d task=task2",
        selected_models, proportions, args.n_trials, args.budget,
    )

    n_cells = len(proportions) * 2
    n_total_trials = n_cells * args.n_trials * len(selected_models)
    metering_per_trial_redacted = redacted_metering(
        "claude-haiku-4-5-20251001", 4096 + 4096, 512
    )
    logger.info(
        "metering_estimate n_trials=%d est_total_redacted=%.2f",
        n_total_trials, n_total_trials * metering_per_trial_redacted,
    )

    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = Path(__file__).resolve().parents[1] / "results" / "task2"

    results = run_experiment(selected_models, proportions, args.n_trials, args.budget)

    safe_model = selected_models[0].replace("/", "_") if len(selected_models) == 1 else "multi"
    save_csv(results, results_dir / f"rcwt_task2_{safe_model}.csv")
    save_aggregates(results, results_dir / f"rcwt_task2_{safe_model}_aggregates.json")

    print_summary(results)
