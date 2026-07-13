"""R(c, W, T) controlled experiment — position-randomized replication.

Fixes three methodological problems in rcwt_baseline.py:

1. POSITION CONFOUND: Original always put [coordination]+[reasoning] order.
   This confounds token-quantity effect with positional disadvantage
   (lost-in-the-middle). Fix: randomize order across trials — half
   run [coord]+[reason], half run [reason]+[coord].

2. FLOOR EFFECTS: crdt_throughput and mls_rfc score 0 across ALL proportions
   including 0% baseline — those facts are not in the reference. They inflate
   the denominator without contributing signal. Fix: compute both raw score
   (10 items) and effective score (8 items, dropping known noise items).

3. SAMPLE SIZE: N=5 with zero variance at baseline makes t-tests undefined.
   Fix: N=20 per condition. Enables proper statistical testing.

Design: 2×5 factorial
  - Order: coord_first | reason_first
  - Proportion: 0%, 25%, 50%, 75%, 90%
  - N=20 trials per cell
  - Models: Haiku 4.5 + GPT-4.1-mini (cheap tier for volume)

Statistical analysis:
  - Per proportion: mean with call-level bootstrap intervals in release analysis
  - Position effect: two-way ANOVA or at minimum compare order conditions
  - Key test: does 90% still degrade significantly vs 0% after controlling for order?

Usage:
    PYTHONPATH=src python src/rcwt_controlled.py
    PYTHONPATH=src python src/rcwt_controlled.py --model gemini-2.5-flash --n-trials 30
    PYTHONPATH=src python src/rcwt_controlled.py --model gpt-4.1-mini --proportions 0,0.90
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
logger = logging.getLogger("rcwt_controlled")

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
# Content templates (unchanged from rcwt_baseline.py)
# ---------------------------------------------------------------------------

REASONING_TASK = """You are analyzing a software project. Using ONLY the technical reference
provided in your context, answer ALL of the following. Do not invent facts — cite only
what the reference provides. If the reference contradicts itself, identify the contradiction.

Scenario: A team is building a real-time collaborative editing system requiring:
- 50 concurrent editors, sub-100ms latency, offline editing with conflict resolution
- End-to-end encryption (server never sees plaintext)
- Plugin system for custom document types

Team: 3 backend, 2 frontend, 1 security. Deadline: 8 weeks. Current stack: REST + PostgreSQL.

Required analysis:
1. DEPENDENCY GRAPH: Which components block which? Identify the critical path with week estimates.
2. THE ENCRYPTION-CRDT TENSION: The reference describes a specific conflict between E2E encryption
   and CRDT merge. What are the three options listed? Which does the reference recommend and why?
3. INFRASTRUCTURE TRADEOFF: The reference compares Redis Pub/Sub vs Redis Streams vs PostgreSQL
   LISTEN/NOTIFY. State the specific latency and throughput numbers for each.
4. TEAM RISK: Based on the capability assessment in the reference, which team member(s) need
   ramp-up time? How many weeks does the reference estimate?
5. PLUGIN ARCHITECTURE: The reference names three editor frameworks. Which one does it recommend
   for rich text and why? What is the proposed unification strategy?
6. FEASIBILITY VERDICT: Given the 8-week constraint and the reference's infrastructure migration
   estimate, is this project feasible? Show your math."""

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

REASONING_CONTEXT_TEMPLATE = """## Technical Reference
The following technical facts are verified and relevant to this analysis:

### CRDT Literature
- Conflict-free Replicated Data Types (Shapiro et al., 2011) guarantee eventual
  consistency without coordination. Y.js implements Yjs CRDT which handles text,
  arrays, maps with O(n) memory per operation.
- Performance: Y.js benchmarks show 10K ops/sec on commodity hardware.
  With 50 concurrent editors at ~2 chars/sec = 100 ops/sec — well within limits.
- The challenge is schema evolution: changing CRDT types after deployment requires
  migration strategies not covered by the core algorithm.

### Encryption Constraints
- MLS (RFC 9420) provides group key agreement for N:N messaging. Each member
  maintains a ratchet tree. Adding/removing members requires O(log N) messages.
- Problem: MLS assumes atomic messages. CRDT operations are character-level
  (sub-message granularity). Options: (a) batch ops into MLS messages at 50ms
  intervals, (b) use per-document symmetric key rotated via MLS, (c) trust server
  for CRDT merge but encrypt at rest.
- Option (b) is most promising: symmetric key per document, rotated when membership
  changes. CRDT merge happens on client before encryption. Server stores ciphertext.

### Infrastructure Reality
- PostgreSQL LISTEN/NOTIFY has 8KB payload limit and no persistence guarantee.
  Not suitable for real-time sync.
- Redis Pub/Sub: ~100K messages/sec per instance. With 50 users at 2 ops/sec =
  100 messages/sec — trivial load. But Redis Pub/Sub has no persistence.
- Redis Streams: Persistent, ordered, consumer groups. Better for offline sync.
  Trade-off: slightly higher latency (~5ms vs ~1ms for Pub/Sub).
- WebSocket horizontal scaling requires sticky sessions or a broadcast layer
  (Redis, NATS, or custom). Estimated migration: 2-3 weeks.

### Plugin Architecture Patterns
- CodeMirror 6 extensions: composable, tree-shakeable. Each doc type = extension.
- Monaco Editor: heavier but has built-in diff, intellisense. Not CRDT-native.
- ProseMirror: rich text CRDT support via y-prosemirror. More flexible than CM6
  for non-code doc types.
- Recommended: CM6 for code, ProseMirror for rich text, custom for diagrams.
  Unified state layer via Y.js document with typed sub-documents.

### Team Capability Assessment
- Backend team (3): Strong in REST/PostgreSQL, moderate WebSocket experience.
  No CRDT production experience. Will need 1 week ramp-up.
- Frontend team (2): React + TypeScript proficient. One has CM6 experience.
  No ProseMirror experience.
- Security (1): Strong in TLS/OAuth. No MLS or E2E collaborative editing
  experience. Will need significant research time."""


# ---------------------------------------------------------------------------
# Judge items — raw (10) vs effective (8, dropping known noise)
# ---------------------------------------------------------------------------

JUDGE_ITEMS = [
    "crdt_throughput",        # NOISE: Y.js 10K ops/sec — consistently 0
    "mls_rfc",                # NOISE: MLS RFC 9420 — consistently 0 at low coord
    "three_encryption_options",
    "recommends_option_b",
    "pg_limit",
    "redis_pubsub_rate",
    "redis_streams_latency",
    "prosemirror_richtext",
    "backend_rampup",
    "infra_migration_weeks",
]

# Items consistently 0 across all conditions in rcwt_baseline (floor effects)
NOISE_ITEMS = {"crdt_throughput", "mls_rfc"}
EFFECTIVE_ITEMS = [i for i in JUDGE_ITEMS if i not in NOISE_ITEMS]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    model: str
    provider: str
    proportion: float
    order: str          # "coord_first" | "reason_first"
    trial_index: int
    coordination_tokens: int
    reasoning_tokens: int
    total_context_tokens: int
    response: str
    scores: dict[str, int]
    mean_score_raw: float       # 10-item denominator
    mean_score_effective: float  # 8-item denominator (drops noise)
    input_tokens_used: int
    output_tokens_used: int
    metering_redacted: float
    elapsed_ms: float


@dataclass
class ConditionAggregate:
    """Aggregate for a (proportion, order) cell."""
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
# Model registry
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
# Provider clients
# ---------------------------------------------------------------------------

_clients: dict[str, object] = {}


def _get_anthropic_client():
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic(timeout=60.0)
    return _clients["anthropic"]


def _get_openai_client():
    if "openai" not in _clients:
        import openai
        _clients["openai"] = openai.OpenAI(timeout=60.0)
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


def _call_anthropic(model, system, user, max_tokens, temperature):
    client = _get_anthropic_client()
    response = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=system, messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens


def _call_openai(model, system, user, max_tokens, temperature):
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (response.choices[0].message.content,
            response.usage.prompt_tokens, response.usage.completion_tokens)


def _call_google(model, system, user, max_tokens, temperature):
    from google.genai import types
    client = _get_google_client()
    thinking_budget = os.environ.get("RCWT_GOOGLE_THINKING_BUDGET")
    thinking_config = None
    if thinking_budget is not None:
        thinking_config = types.ThinkingConfig(thinking_budget=int(thinking_budget))
    response = client.models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens,
            temperature=temperature,
            thinking_config=thinking_config,
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


def build_reasoning_context(target_tokens: int) -> str:
    return build_text_to_tokens(REASONING_CONTEXT_TEMPLATE, target_tokens)


def assemble_system(
    coord_tokens: int,
    reason_tokens: int,
    order: str,
) -> str:
    """Assemble system prompt with explicit order control.

    order='coord_first'  → [coordination block] then [reasoning block]
    order='reason_first' → [reasoning block] then [coordination block]
    """
    parts: list[str] = []
    coord_block = build_coordination_context(coord_tokens) if coord_tokens > 10 else ""
    reason_block = build_reasoning_context(reason_tokens) if reason_tokens > 10 else ""

    if order == "coord_first":
        if coord_block:
            parts.append(coord_block)
        if reason_block:
            parts.append(reason_block)
    else:  # reason_first
        if reason_block:
            parts.append(reason_block)
        if coord_block:
            parts.append(coord_block)

    return "\n\n".join(parts) if parts else "You are a technical analyst."


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are a factual evaluator. The response was generated from a technical
reference document. Check whether each specific fact appears in the response.

Score each item 1 (present and correct) or 0 (absent or wrong):

1. crdt_throughput: Mentions Y.js benchmark of 10K ops/sec
2. mls_rfc: References MLS RFC 9420 for group key agreement
3. three_encryption_options: Lists the three encryption options (batch ops / per-doc symmetric key / trust server)
4. recommends_option_b: Identifies per-document symmetric key as the recommended option
5. pg_limit: States PostgreSQL LISTEN/NOTIFY 8KB payload limit
6. redis_pubsub_rate: States Redis Pub/Sub ~100K messages/sec
7. redis_streams_latency: Mentions Redis Streams ~5ms latency vs Pub/Sub ~1ms
8. prosemirror_richtext: Names ProseMirror as the rich text recommendation
9. backend_rampup: Mentions backend team needs 1 week CRDT ramp-up
10. infra_migration_weeks: States 2-3 weeks for WebSocket/infrastructure migration

Respond ONLY with valid JSON:
{"crdt_throughput": 0, "mls_rfc": 0, "three_encryption_options": 0, "recommends_option_b": 0, "pg_limit": 0, "redis_pubsub_rate": 0, "redis_streams_latency": 0, "prosemirror_richtext": 0, "backend_rampup": 0, "infra_migration_weeks": 0}

Replace 0 with 1 for each fact that IS present. No other text."""


def judge_response(task: str, response: str) -> dict[str, int]:
    user_msg = f"## Original Task\n{task}\n\n## Response to Evaluate\n{response}"
    text, _, _ = call_model(JUDGE_MODEL, JUDGE_PROMPT, user_msg, max_tokens=200, temperature=0.0)
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        scores = json.loads(text[start:end])
        return {k: min(1, max(0, int(scores.get(k, 0)))) for k in JUDGE_ITEMS}
    except (ValueError, json.JSONDecodeError):
        logger.warning("judge_parse_failed raw=%s", text[:200])
        return {k: 0 for k in JUDGE_ITEMS}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def aggregate_trials(
    trials: list[TrialResult],
    proportion: float,
    order: str,
) -> ConditionAggregate:
    """Compute aggregate stats for a (proportion, order) cell."""
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
        seed=stable_seed("controlled-cell-effective", proportion, order),
    )
    _, ci95_raw_low, ci95_raw_high = stratified_bootstrap_mean_ci(
        {"calls": raw_scores},
        seed=stable_seed("controlled-cell-raw", proportion, order),
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
    task_tokens = count_tokens(REASONING_TASK)
    available = total_budget - task_tokens

    coord_tokens = int(available * proportion)
    reason_tokens = available - coord_tokens

    system = assemble_system(coord_tokens, reason_tokens, order)

    t0 = time.monotonic()
    response, in_tok, out_tok = call_model(model, system, REASONING_TASK)
    elapsed_ms = (time.monotonic() - t0) * 1000

    scores = judge_response(REASONING_TASK, response)
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
        logger.info("=== experiment_start model=%s provider=%s ===", model, provider)
        t0 = time.monotonic()

        exp = ExperimentResult(
            model=model, provider=provider,
            total_budget=total_budget, n_trials_per_cell=n_trials,
        )

        # Build trial schedule: (proportion, order) pairs, then shuffle
        schedule: list[tuple[float, str]] = []
        for prop in proportions:
            for order in orders:
                for i in range(n_trials):
                    schedule.append((prop, order))
        random.shuffle(schedule)

        # Track trial index per (proportion, order) cell
        cell_idx: dict[tuple[float, str], int] = {}

        for prop, order in schedule:
            idx = cell_idx.get((prop, order), 0)
            cell_idx[(prop, order)] = idx + 1
            trial = run_trial(model, prop, order, idx, total_budget)
            exp.trials.append(trial)

        # Aggregate all cells
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

    # Save responses for re-judging
    responses_path = output_path.parent / "rcwt_controlled_responses.jsonl"
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


def plot_results(results: list[ExperimentResult], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    colors = {"coord_first": "#E07B53", "reason_first": "#5B8BD4"}
    markers = {"coord_first": "o", "reason_first": "s"}

    for exp in results:
        proportions = sorted({a.proportion for a in exp.aggregates})

        # Panel 1: effective score by order (position effect check)
        ax = axes[0]
        for order in ["coord_first", "reason_first"]:
            cells = sorted(
                [a for a in exp.aggregates if a.order == order],
                key=lambda a: a.proportion,
            )
            xs = [c.proportion * 100 for c in cells]
            ys = [c.mean_effective for c in cells]
            errs = [max(0.0, c.mean_effective - c.ci95_effective[0]) for c in cells]
            label = f"{exp.model.split('-')[0]} [{order.replace('_', ' ')}]"
            ax.errorbar(xs, ys, yerr=errs, marker=markers[order],
                        color=colors[order], label=label, capsize=4, linewidth=2)

        ax.set_xlabel("Coordination Token Proportion (%)")
        ax.set_ylabel("Fact Recall Rate (0–1)")  # Fixed: was "1-5" in baseline
        ax.set_title("Position Effect Check\n(coord_first vs reason_first)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # Panel 2: pooled effective score across both orders + 95% CI
        ax = axes[1]
        pooled_means, pooled_lo, pooled_hi = [], [], []
        for prop in proportions:
            cells = [a for a in exp.aggregates if a.proportion == prop]
            all_trials = [t for t in exp.trials if t.proportion == prop]
            strata = {
                order: [
                    trial.mean_score_effective
                    for trial in all_trials
                    if trial.order == order
                ]
                for order in ["coord_first", "reason_first"]
            }
            mean, lo, hi = stratified_bootstrap_mean_ci(
                strata,
                seed=stable_seed("controlled-plot", exp.model, prop),
            )
            pooled_means.append(mean)
            pooled_lo.append(max(0.0, mean - lo))
            pooled_hi.append(max(0.0, hi - mean))

        xs = [p * 100 for p in proportions]
        ax.errorbar(xs, pooled_means,
                    yerr=[pooled_lo, pooled_hi],
                    marker="D", color="#2D6A2E", capsize=5, linewidth=2,
                    label=f"{exp.model.split('-')[0]} (pooled)")
        ax.set_xlabel("Coordination Token Proportion (%)")
        ax.set_ylabel("Fact Recall Rate (0–1, effective 8 items)")
        ax.set_title("Pooled Effective Score\n(both orders, 95% CI)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # Panel 3: position delta (coord_first minus reason_first per proportion)
        ax = axes[2]
        deltas, xs_d = [], []
        for prop in proportions:
            cf = next((a for a in exp.aggregates
                       if a.proportion == prop and a.order == "coord_first"), None)
            rf = next((a for a in exp.aggregates
                       if a.proportion == prop and a.order == "reason_first"), None)
            if cf and rf:
                deltas.append(cf.mean_effective - rf.mean_effective)
                xs_d.append(prop * 100)

        bar_colors = ["#E07B53" if d > 0 else "#5B8BD4" for d in deltas]
        ax.bar(xs_d, deltas, color=bar_colors, alpha=0.8, width=6)
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax.set_xlabel("Coordination Token Proportion (%)")
        ax.set_ylabel("Δ Recall (coord_first − reason_first)")
        ax.set_title("Position Effect Magnitude\n(>0 = coord_first hurts more)")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("plot_saved path=%s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCWT controlled experiment")
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
        help="Output directory (default: results/controlled/)",
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


def print_summary(results: list[ExperimentResult]) -> None:
    total_metering_redacted = sum(r.total_metering_redacted for r in results)
    print("\n" + "=" * 72)
    print("R(c, W, T) CONTROLLED RESULTS — POSITION-RANDOMIZED")
    print(f"Effective items: {EFFECTIVE_ITEMS}")
    print(f"Noise items excluded: {list(NOISE_ITEMS)}")
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

    # Position effect test: is the degradation at 90% significant?
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
        # Simple chi-squared test of independence
        n = base_total + high_total
        k = base_hits + high_hits
        # Two-proportion z-test
        p_pool = k / n if n > 0 else 0
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / base_total + 1 / high_total)) if p_pool > 0 else 1
        z = (base_rate - high_rate) / se if se > 0 else 0
        sig = "p<0.001" if abs(z) > 3.29 else ("p<0.01" if abs(z) > 2.58 else
              ("p<0.05" if abs(z) > 1.96 else "n.s."))
        print(f"    {exp.model}: 0%={base_rate:.3f} vs 90%={high_rate:.3f} "
              f"Δ={base_rate - high_rate:.3f} z={z:.2f} {sig}")


if __name__ == "__main__":
    args = parse_args()
    available = check_env()
    if not available:
        logger.error("No API keys set. Need at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY")
        sys.exit(1)

    logger.info("available_providers=%s", available)

    # Select models
    if args.model:
        selected_models = [args.model]
    else:
        selected_models = [
            m for m, info in MODELS.items()
            if info["provider"] in available and info["tier"] == "cheap"
        ]

    # Override judge model if specified
    if args.judge_model:
        JUDGE_MODEL = args.judge_model  # type: ignore[assignment]
    logger.info("judge_model=%s", JUDGE_MODEL)

    # Judge availability check
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
        "config models=%s proportions=%s n_trials=%d budget=%d",
        selected_models, proportions, args.n_trials, args.budget,
    )

    # Provider billing omitted from public artifact
    n_cells = len(proportions) * 2  # 2 orders
    n_total_trials = n_cells * args.n_trials * len(selected_models)
    # Rough: ~4096 tokens input, ~512 output per trial + judge call
    metering_per_trial_redacted = redacted_metering(
        "claude-haiku-4-5-20251001", 4096 + 4096, 512
    )
    logger.info(
        "metering_estimate n_trials=%d est_total_redacted=%.2f",
        n_total_trials, n_total_trials * metering_per_trial_redacted,
    )

    results_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "results" / "controlled"
    )

    results = run_experiment(selected_models, proportions, args.n_trials, args.budget)

    save_csv(results, results_dir / "rcwt_controlled.csv")
    save_aggregates(results, results_dir / "rcwt_controlled_aggregates.json")

    try:
        plot_results(results, results_dir / "rcwt_controlled.png")
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")

    print_summary(results)
