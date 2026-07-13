"""R(c, W, T) baseline measurement — first empirical test of the Medium-Identity Axiom.

Tests whether coordination tokens and reasoning tokens compete within a single
agent's fixed context window. Varies coordination proportion from 0% to 90%
while keeping total context budget fixed.

Multi-model: tests across Anthropic, OpenAI, and Google to determine if the
tradeoff is a universal transformer property, not provider-specific.

Novel contribution: Liu, Kong, Pei (arXiv:2601.17311) formalize inter-agent
budget allocation but not the intra-agent competition measured here.

Usage:
    pip install -r requirements.txt
    PYTHONPATH=src python src/rcwt_baseline.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rcwt")

# ---------------------------------------------------------------------------
# Token counting (approximate — cl100k for all providers)
# ---------------------------------------------------------------------------

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def build_text_to_tokens(text: str, target_tokens: int) -> str:
    """Repeat or truncate text to hit approximately target_tokens."""
    tokens = _ENCODER.encode(text)
    if len(tokens) >= target_tokens:
        return _ENCODER.decode(tokens[:target_tokens])
    repeated = tokens * ((target_tokens // len(tokens)) + 1)
    return _ENCODER.decode(repeated[:target_tokens])


# ---------------------------------------------------------------------------
# Context templates — drawn from real multi-agent system prompts
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
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    model: str
    provider: str
    proportion: float
    trial_index: int
    coordination_tokens: int
    reasoning_tokens: int
    total_context_tokens: int
    response: str
    scores: dict[str, int]  # item -> 0/1
    mean_score: float  # sum / len
    input_tokens_used: int
    output_tokens_used: int
    metering_redacted: float
    elapsed_ms: float


@dataclass
class ProportionAggregate:
    proportion: float
    n_trials: int
    mean_score: float
    std_score: float
    scores_by_dimension: dict[str, float]
    mean_metering_redacted: float


@dataclass
class ExperimentResult:
    model: str
    provider: str
    total_budget: int
    aggregates: list[ProportionAggregate] = field(default_factory=list)
    trials: list[TrialResult] = field(default_factory=list)
    total_metering_redacted: float = 0.0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Model registry for the public artifact
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


def redacted_metering(model: str, input_tokens: int, output_tokens: int) -> float:
    """Provider billing is omitted from public artifacts; this placeholder stays zero."""
    _ = (model, input_tokens, output_tokens)
    return 0.0


# ---------------------------------------------------------------------------
# Provider clients — lazy singletons
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


# ---------------------------------------------------------------------------
# Unified API call layer
# ---------------------------------------------------------------------------


THINKING_MODELS = {"gemini-2.5-flash"}


def call_model(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> tuple[str, int, int]:
    """Call any provider's API. Returns (response_text, input_tokens, output_tokens)."""
    # Thinking models need more output budget (thinking tokens + actual output)
    if model in THINKING_MODELS:
        max_tokens = max(max_tokens, 8192)
    provider = MODELS.get(model, {}).get("provider", "anthropic")

    if provider == "anthropic":
        return _call_anthropic(model, system, user, max_tokens, temperature)
    elif provider == "openai":
        return _call_openai(model, system, user, max_tokens, temperature)
    elif provider == "google":
        return _call_google(model, system, user, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _call_anthropic(
    model: str, system: str, user: str, max_tokens: int, temperature: float,
) -> tuple[str, int, int]:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = response.content[0].text
    return text, response.usage.input_tokens, response.usage.output_tokens


def _call_openai(
    model: str, system: str, user: str, max_tokens: int, temperature: float,
) -> tuple[str, int, int]:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = response.choices[0].message.content
    in_tok = response.usage.prompt_tokens
    out_tok = response.usage.completion_tokens
    return text, in_tok, out_tok


def _call_google(
    model: str, system: str, user: str, max_tokens: int, temperature: float,
) -> tuple[str, int, int]:
    from google.genai import types
    client = _get_google_client()
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    # Gemini thinking models may return None for .text — extract from parts
    text = None
    try:
        text = response.text
    except (ValueError, AttributeError):
        pass
    if text is None:
        candidates = response.candidates or []
        if candidates and candidates[0].content and candidates[0].content.parts:
            text_parts = [p.text for p in candidates[0].content.parts
                          if hasattr(p, "text") and p.text]
            text = "\n".join(text_parts) if text_parts else ""
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
    """Build realistic coordination content at target token count."""
    base = COORDINATION_TEMPLATE.format(consumed=target_tokens, budget=target_tokens * 2)
    return build_text_to_tokens(base, target_tokens)


def build_reasoning_context(target_tokens: int) -> str:
    """Build domain-specific reasoning content at target token count."""
    return build_text_to_tokens(REASONING_CONTEXT_TEMPLATE, target_tokens)


# ---------------------------------------------------------------------------
# Judge (always Sonnet — single evaluator for cross-model consistency)
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

JUDGE_ITEMS = [
    "crdt_throughput", "mls_rfc", "three_encryption_options", "recommends_option_b",
    "pg_limit", "redis_pubsub_rate", "redis_streams_latency",
    "prosemirror_richtext", "backend_rampup", "infra_migration_weeks",
]


def judge_response(task: str, response: str) -> dict[str, int]:
    """Factual checklist evaluation. Returns dict of item -> 0 or 1."""
    user_msg = f"## Original Task\n{task}\n\n## Response to Evaluate\n{response}"
    text, _, _ = call_model(
        JUDGE_MODEL,
        JUDGE_PROMPT,
        user_msg,
        max_tokens=200,
        temperature=0.0,
    )
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        scores = json.loads(text[start:end])
        # Clamp to 0/1
        return {k: min(1, max(0, int(scores.get(k, 0)))) for k in JUDGE_ITEMS}
    except (ValueError, json.JSONDecodeError):
        logger.warning("judge_parse_failed raw=%s", text[:200])
        return {k: 0 for k in JUDGE_ITEMS}


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_trial(
    model: str,
    proportion: float,
    trial_index: int,
    total_budget: int,
) -> TrialResult:
    """Run a single trial at the given coordination proportion."""
    provider = MODELS[model]["provider"]
    task_tokens = count_tokens(REASONING_TASK)
    available = total_budget - task_tokens

    coord_tokens = int(available * proportion)
    reason_tokens = available - coord_tokens

    system_parts: list[str] = []
    if coord_tokens > 10:
        system_parts.append(build_coordination_context(coord_tokens))
    if reason_tokens > 10:
        system_parts.append(build_reasoning_context(reason_tokens))

    system = "\n\n".join(system_parts) if system_parts else "You are a technical analyst."

    start = time.monotonic()
    response, in_tok, out_tok = call_model(model, system, REASONING_TASK)
    elapsed_ms = (time.monotonic() - start) * 1000

    scores = judge_response(REASONING_TASK, response)
    metering_value = redacted_metering(model, in_tok, out_tok)

    n_items = len(scores) or 1
    mean = sum(scores.values()) / n_items
    hits = sum(scores.values())

    logger.info(
        "trial model=%s provider=%s prop=%.0f%% trial=%d hits=%d/%d score=%.2f metering=%.4f",
        model, provider, proportion * 100, trial_index, hits, n_items, mean, metering_value,
    )

    return TrialResult(
        model=model,
        provider=provider,
        proportion=proportion,
        trial_index=trial_index,
        coordination_tokens=coord_tokens,
        reasoning_tokens=reason_tokens,
        total_context_tokens=total_budget,
        response=response,
        scores=scores,
        mean_score=mean,
        input_tokens_used=in_tok,
        output_tokens_used=out_tok,
        metering_redacted=metering_value,
        elapsed_ms=elapsed_ms,
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
    """Run full experiment across models and proportions."""
    results: list[ExperimentResult] = []

    for model in models:
        provider = MODELS[model]["provider"]
        logger.info("=== experiment_start model=%s provider=%s ===", model, provider)
        exp = ExperimentResult(model=model, provider=provider, total_budget=total_budget)
        start = time.monotonic()

        for proportion in proportions:
            trials: list[TrialResult] = []
            for i in range(n_trials):
                trial = run_trial(model, proportion, i, total_budget)
                trials.append(trial)
                exp.trials.append(trial)

            scores = [t.mean_score for t in trials]
            mean_s = sum(scores) / len(scores)
            std_s = (sum((s - mean_s) ** 2 for s in scores) / len(scores)) ** 0.5

            dim_means = {}
            for item in JUDGE_ITEMS:
                vals = [t.scores.get(item, 0) for t in trials]
                dim_means[item] = sum(vals) / len(vals)

            agg = ProportionAggregate(
                proportion=proportion,
                n_trials=n_trials,
                mean_score=mean_s,
                std_score=std_s,
                scores_by_dimension=dim_means,
                mean_metering_redacted=sum(t.metering_redacted for t in trials) / len(trials),
            )
            exp.aggregates.append(agg)
            logger.info(
                "proportion_done model=%s prop=%.0f%% mean=%.2f std=%.2f",
                model, proportion * 100, mean_s, std_s,
            )

        exp.total_metering_redacted = sum(t.metering_redacted for t in exp.trials)
        exp.elapsed_seconds = time.monotonic() - start
        results.append(exp)
        logger.info(
            "=== experiment_done model=%s metering=%.2f time=%.0fs ===",
            model, exp.total_metering_redacted, exp.elapsed_seconds,
        )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_csv(results: list[ExperimentResult], output_path: Path) -> None:
    """Save raw trial data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "provider", "proportion", "trial_index",
        "coordination_tokens", "reasoning_tokens", "total_context_tokens",
        "mean_score", "input_tokens_used", "output_tokens_used", "metering_redacted",
        "elapsed_ms",
    ] + JUDGE_ITEMS
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in results:
            for t in exp.trials:
                row = {
                    "model": t.model, "provider": t.provider,
                    "proportion": t.proportion, "trial_index": t.trial_index,
                    "coordination_tokens": t.coordination_tokens,
                    "reasoning_tokens": t.reasoning_tokens,
                    "total_context_tokens": t.total_context_tokens,
                    "mean_score": t.mean_score,
                    "input_tokens_used": t.input_tokens_used,
                    "output_tokens_used": t.output_tokens_used,
                    "metering_redacted": t.metering_redacted, "elapsed_ms": t.elapsed_ms,
                }
                row.update(t.scores)
                writer.writerow(row)
    logger.info("csv_saved path=%s", output_path)

    # Save full responses for re-judging
    responses_path = output_path.parent / "rcwt_responses.jsonl"
    with open(responses_path, "w") as f:
        for exp in results:
            for t in exp.trials:
                json.dump({
                    "model": t.model, "proportion": t.proportion,
                    "trial_index": t.trial_index, "response": t.response,
                }, f)
                f.write("\n")
    logger.info("responses_saved path=%s", responses_path)


def save_aggregates(results: list[ExperimentResult], output_path: Path) -> None:
    """Save aggregate results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for exp in results:
        data.append({
            "model": exp.model,
            "provider": exp.provider,
            "total_budget": exp.total_budget,
            "total_metering_redacted": exp.total_metering_redacted,
            "elapsed_seconds": exp.elapsed_seconds,
            "aggregates": [
                {
                    "proportion": a.proportion,
                    "n_trials": a.n_trials,
                    "mean_score": round(a.mean_score, 3),
                    "std_score": round(a.std_score, 3),
                    "scores_by_dimension": {
                        k: round(v, 3) for k, v in a.scores_by_dimension.items()
                    },
                    "mean_metering_redacted": round(a.mean_metering_redacted, 5),
                }
                for a in exp.aggregates
            ],
        })
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("aggregates_saved path=%s", output_path)


def plot_results(results: list[ExperimentResult], output_path: Path) -> None:
    """Plot quality vs coordination proportion — multi-model comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Color scheme by provider
    provider_colors = {
        "anthropic": ("#E07B53", "#B5432A"),  # cheap, strong
        "openai": ("#6BA368", "#2D6A2E"),
        "google": ("#5B8BD4", "#2856A3"),
    }
    tier_markers = {"cheap": "o", "strong": "s"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Panel 1: All models, mean quality ---
    ax1 = axes[0]
    for exp in results:
        info = MODELS[exp.model]
        colors = provider_colors.get(info["provider"], ("#888", "#444"))
        color = colors[0] if info["tier"] == "cheap" else colors[1]
        marker = tier_markers[info["tier"]]

        proportions = [a.proportion * 100 for a in exp.aggregates]
        means = [a.mean_score for a in exp.aggregates]
        stds = [a.std_score for a in exp.aggregates]

        label = f"{exp.model.split('-')[0]}-{info['tier']}"
        ax1.errorbar(
            proportions, means, yerr=stds,
            marker=marker, capsize=4, label=label, color=color, linewidth=2,
        )

    ax1.set_xlabel("Coordination Token Proportion (%)", fontsize=12)
    ax1.set_ylabel("Mean Quality Score (1-5)", fontsize=12)
    ax1.set_title("R(c, W, T): Quality vs Coordination\nAll Models", fontsize=13)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1, 5.3)

    # --- Panel 2: Normalized curves (% of baseline at 0%) ---
    ax2 = axes[1]
    for exp in results:
        info = MODELS[exp.model]
        colors = provider_colors.get(info["provider"], ("#888", "#444"))
        color = colors[0] if info["tier"] == "cheap" else colors[1]
        marker = tier_markers[info["tier"]]

        proportions = [a.proportion * 100 for a in exp.aggregates]
        means = [a.mean_score for a in exp.aggregates]
        baseline = means[0] if means[0] > 0 else 1
        normalized = [(m / baseline) * 100 for m in means]

        label = f"{exp.model.split('-')[0]}-{info['tier']}"
        ax2.plot(
            proportions, normalized,
            marker=marker, label=label, color=color, linewidth=2,
        )

    ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="baseline")
    ax2.set_xlabel("Coordination Token Proportion (%)", fontsize=12)
    ax2.set_ylabel("Quality (% of 0% baseline)", fontsize=12)
    ax2.set_title("Normalized Degradation Curve\n(100% = pure reasoning baseline)", fontsize=13)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Per-item heatmap (which facts survive at each proportion) ---
    ax3 = axes[2]
    # Pick the model with the clearest signal (most variance in mean_score)
    variances = []
    for exp in results:
        scores = [a.mean_score for a in exp.aggregates]
        v = max(scores) - min(scores) if scores else 0
        variances.append((v, exp))
    variances.sort(reverse=True)
    if variances:
        best_exp = variances[0][1]
        proportions = [a.proportion * 100 for a in best_exp.aggregates]
        # Group items into categories for readability
        item_categories = {
            "crdt_throughput": "CRDT", "mls_rfc": "Crypto",
            "three_encryption_options": "Crypto", "recommends_option_b": "Crypto",
            "pg_limit": "Infra", "redis_pubsub_rate": "Infra",
            "redis_streams_latency": "Infra", "prosemirror_richtext": "Arch",
            "backend_rampup": "Team", "infra_migration_weeks": "Infra",
        }
        for item in JUDGE_ITEMS:
            vals = [a.scores_by_dimension.get(item, 0) for a in best_exp.aggregates]
            cat = item_categories.get(item, "")
            ax3.plot(
                proportions, vals,
                marker=".", label=f"{cat}/{item.split('_')[0]}",
                alpha=0.7,
            )

        ax3.set_xlabel("Coordination Token Proportion (%)", fontsize=12)
        ax3.set_ylabel("Recall Rate (0-1)", fontsize=12)
        short_name = best_exp.model.split("-")[0]
        ax3.set_title(f"Fact Recall by Item\n({short_name} — highest variance)", fontsize=13)
        ax3.legend(fontsize=6, loc="lower left", ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("plot_saved path=%s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_env() -> list[str]:
    """Check which providers have API keys. Returns list of available providers."""
    available = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        available.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        available.append("openai")
    if os.environ.get("GEMINI_API_KEY"):
        available.append("google")
    return available


if __name__ == "__main__":
    available_providers = check_env()
    if not available_providers:
        logger.error("No API keys set. Need at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY")
        sys.exit(1)

    logger.info("available_providers=%s", available_providers)

    # Select models based on available keys and flags
    if "--strong-only" in sys.argv:
        selected_models = [
            model_id for model_id, info in MODELS.items()
            if info["provider"] in available_providers and info["tier"] == "strong"
        ]
    elif "--strong" in sys.argv:
        selected_models = [
            model_id for model_id, info in MODELS.items()
            if info["provider"] in available_providers
        ]
    elif "--model" in sys.argv:
        idx = sys.argv.index("--model")
        selected_models = [sys.argv[idx + 1]]
    else:
        selected_models = [
            model_id for model_id, info in MODELS.items()
            if info["provider"] in available_providers and info["tier"] == "cheap"
        ]

    if "anthropic" not in available_providers:
        logger.warning("ANTHROPIC_API_KEY missing — judge model unavailable, falling back to first available model")
        JUDGE_MODEL = selected_models[0]

    logger.info("selected_models=%s judge=%s", selected_models, JUDGE_MODEL)

    PROPORTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]
    N_TRIALS = 5
    TOTAL_BUDGET = 4096

    results_dir = Path(__file__).parent / "results"

    results = run_experiment(selected_models, PROPORTIONS, N_TRIALS, TOTAL_BUDGET)

    save_csv(results, results_dir / "rcwt_baseline.csv")
    save_aggregates(results, results_dir / "rcwt_aggregates.json")

    try:
        plot_results(results, results_dir / "rcwt_baseline.png")
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")

    # Summary
    total_metering_redacted = sum(r.total_metering_redacted for r in results)
    print("\n" + "=" * 70)
    print("R(c, W, T) MULTI-MODEL BASELINE RESULTS")
    print(f"Judge: {JUDGE_MODEL}")
    print(f"Total metering redacted: {total_metering_redacted:.2f}")
    print("=" * 70)
    for exp in results:
        print(f"\n  {exp.model} ({exp.provider}) — metering_redacted={exp.total_metering_redacted:.2f} in {exp.elapsed_seconds:.0f}s")
        header = f"  {'Prop':>6} {'Hits':>5} {'Std':>5}"
        for item in JUDGE_ITEMS[:5]:
            header += f" {item[:6]:>6}"
        print(header)
        header2 = f"  {'':>6} {'':>5} {'':>5}"
        for item in JUDGE_ITEMS[5:]:
            header2 += f" {item[:6]:>6}"
        print(f"  {'-' * 70}")
        for a in exp.aggregates:
            d = a.scores_by_dimension
            line = f"  {a.proportion:>5.0%} {a.mean_score:>5.1%} {a.std_score:>5.2f}"
            for item in JUDGE_ITEMS[:5]:
                line += f" {d.get(item, 0):>6.0%}"
            print(line)
            line2 = f"  {'':>6} {'':>5} {'':>5}"
            for item in JUDGE_ITEMS[5:]:
                line2 += f" {d.get(item, 0):>6.0%}"
            print(line2)
    print()
