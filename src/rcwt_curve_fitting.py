"""R(c, W, T) curve fitting — parametric model selection for coordination-reasoning frontier.

Data: RCWT controlled experiment results (2026-03-30)
- 3 models × 9 runner target ratios × 2 orders
- Mean effective score per target, analyzed at the realized share p=c/W
- Goal: compare descriptive functional forms for R(c, W, T)

Candidate models:
1. Power law decay: R(p) = R0 * (1 - p)^alpha
2. Logistic decay:  R(p) = R0 / (1 + exp(k * (p - p0)))
3. Exponential:     R(p) = R0 * exp(-lambda * p)
4. Quadratic:       R(p) = R0 - a*p - b*p^2
5. Piecewise linear: R(p) = R0 for p <= p*; R0 * (1 - beta*(p-p*)/(1-p*)) for p > p*

Usage:
    PYTHONPATH=src python src/rcwt_curve_fitting.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from rcwt_token_accounting import fixed_budget_allocation

# ─── Load empirical data from JSON files ──────────────────────────────────────

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
_AGGREGATE_PATH = _RESULTS_DIR / "rcwt_controlled_aggregates.json"

_MODEL_KEYS = {
    "gemini-2.0-flash": "gemini-2.0-flash",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "gpt-4.1-mini": "gpt-4.1-mini",
}

_PROVIDER_LABELS = {
    "google": "Google",
    "anthropic": "Anthropic",
    "openai": "OpenAI",
}


def _load_empirical() -> dict[str, dict]:
    """Load mean effective score per proportion from the release aggregate file."""
    data: dict[str, dict] = {}
    raw = json.loads(_AGGREGATE_PATH.read_text())
    for experiment in raw:
        model_id = _MODEL_KEYS.get(experiment["model"], experiment["model"])
        provider = _PROVIDER_LABELS.get(experiment["provider"], experiment["provider"])
        aggs = experiment["aggregates"]
        target_ratios = sorted({float(a["proportion"]) for a in aggs})
        total_budget = int(experiment["total_budget"])
        coordination_shares: list[float] = []
        scores: list[float] = []

        for target_ratio in target_ratios:
            cells = [a for a in aggs if float(a["proportion"]) == target_ratio]
            mean = float(np.mean([c["mean_effective"] for c in cells]))
            allocation = fixed_budget_allocation(total_budget, target_ratio)
            coordination_shares.append(allocation.coordination_share)
            scores.append(mean)

        data[model_id] = {
            "provider": provider,
            "total_budget": total_budget,
            "target_ratios": target_ratios,
            "coordination_shares": coordination_shares,
            "scores": scores,
        }
    return data


# ─── Functional forms ─────────────────────────────────────────────────────────

def _power(p: np.ndarray, R0: float, alpha: float) -> np.ndarray:
    """R0 * (1 - p)^alpha  — power law decay."""
    return R0 * np.maximum(0.0, 1.0 - np.asarray(p)) ** alpha


def _logistic(p: np.ndarray, R0: float, k: float, p0: float) -> np.ndarray:
    """R0 / (1 + exp(k*(p-p0)))  — sigmoid decay."""
    return R0 / (1.0 + np.exp(k * (np.asarray(p) - p0)))


def _exponential(p: np.ndarray, R0: float, lam: float) -> np.ndarray:
    """R0 * exp(-lambda * p)  — exponential decay."""
    return R0 * np.exp(-lam * np.asarray(p))


def _quadratic(p: np.ndarray, R0: float, a: float, b: float) -> np.ndarray:
    """R0 - a*p - b*p^2  — quadratic decay."""
    p = np.asarray(p)
    return R0 - a * p - b * p ** 2


def _piecewise(p: np.ndarray, R0: float, p_star: float, beta: float) -> np.ndarray:
    """R0 for p<=p*; R0*(1-beta*(p-p*)/(1-p*)) for p>p*  — threshold + linear drop."""
    p = np.asarray(p, dtype=float)
    denom = max(1.0 - p_star, 1e-9)
    return np.where(
        p <= p_star,
        R0,
        R0 * np.maximum(0.0, 1.0 - beta * (p - p_star) / denom),
    )


# name → (function, param_names, initial_guess, (lower_bounds, upper_bounds))
FUNCTIONAL_FORMS: dict[str, tuple] = {
    "power":      (_power,      ["R₀", "α"],        [1.0, 2.0],           ([0.5, 0.01], [1.2, 50.0])),
    "logistic":   (_logistic,   ["R₀", "k", "p₀"],  [1.0, 15.0, 0.85],   ([0.5, 0.1, 0.0], [1.2, 200.0, 1.0])),
    "exponential":(_exponential,["R₀", "λ"],         [1.0, 0.8],           ([0.5, 0.001], [1.2, 50.0])),
    "quadratic":  (_quadratic,  ["R₀", "a", "b"],   [1.0, 0.1, 1.5],     ([0.5, -2.0, -5.0], [1.2, 5.0, 30.0])),
    "piecewise":  (_piecewise,  ["R₀", "p*", "β"],  [1.0, 0.78, 3.0],    ([0.5, 0.3, 0.1], [1.2, 0.99, 30.0])),
}


# ─── Fitting utilities ─────────────────────────────────────────────────────────

def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _aic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> float:
    n = len(y_true)
    residuals = y_true - y_pred
    sigma2 = float(np.mean(residuals ** 2))
    if sigma2 <= 0:
        return float("inf")
    ll = -n / 2 * np.log(2 * np.pi * sigma2) - np.sum(residuals ** 2) / (2 * sigma2)
    return float(2 * n_params - 2 * ll)


def _c_star(form_name: str, params: list[float]) -> float:
    """Estimate c* as the proportion where |dR/dp| is maximised (steepest drop)."""
    p_grid = np.linspace(0.0, 0.99, 2000)
    fn = FUNCTIONAL_FORMS[form_name][0]
    r_vals = fn(p_grid, *params)
    grad = np.gradient(r_vals, p_grid)
    return float(p_grid[np.argmin(grad)])


# ─── Main fitting loop ─────────────────────────────────────────────────────────

def fit_all(empirical: dict[str, dict]) -> dict[str, dict]:
    results: dict[str, dict] = {}

    for model_id, d in empirical.items():
        p = np.array(d["coordination_shares"])
        r = np.array(d["scores"])
        fits: dict[str, dict] = {}

        for form_name, (fn, param_names, p0, bounds) in FUNCTIONAL_FORMS.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(
                        fn, p, r,
                        p0=p0,
                        bounds=bounds,
                        maxfev=20_000,
                        method="trf",
                    )
                y_pred = fn(p, *popt)
                r2 = _r_squared(r, y_pred)
                aic = _aic(r, y_pred, len(popt))
                c_star_val = _c_star(form_name, list(popt))

                fits[form_name] = {
                    "params": {k: round(float(v), 5) for k, v in zip(param_names, popt)},
                    "r_squared": round(r2, 5),
                    "aic": round(aic, 4),
                    "c_star": round(c_star_val, 4),
                    "predicted": [round(float(v), 5) for v in y_pred],
                    "rmse": round(float(np.sqrt(np.mean((r - y_pred) ** 2))), 5),
                }
            except Exception as exc:
                fits[form_name] = {"error": str(exc)}

        results[model_id] = {
            "provider": d["provider"],
            "total_budget": d["total_budget"],
            "empirical": {
                "target_ratios": d["target_ratios"],
                "coordination_shares": d["coordination_shares"],
                "scores": d["scores"],
            },
            "fits": fits,
        }

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_fits(results: dict[str, dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    p_fine = np.linspace(0.0, 0.99, 400)

    _colors = {
        "power": "#E07B53",
        "logistic": "#5B8BD4",
        "exponential": "#2D9E6B",
        "quadratic": "#9B59B6",
        "piecewise": "#E67E22",
    }
    _labels = {
        "power": "Power: R₀(1−p)^α",
        "logistic": "Logistic: R₀/(1+eᵏ⁽ᵖ⁻ᵖ⁰⁾)",
        "exponential": "Exp: R₀·e^{−λp}",
        "quadratic": "Quadratic: R₀−ap−bp²",
        "piecewise": "Piecewise linear",
    }
    _titles = {
        "gemini-2.0-flash": "Gemini 2.0 Flash (Google)",
        "claude-haiku-4-5": "Claude Haiku 4.5 (Anthropic)",
        "gpt-4.1-mini": "GPT-4.1-mini (OpenAI)",
    }

    for ax, (model_id, res) in zip(axes, results.items()):
        emp = res["empirical"]
        p_emp = np.array(emp["coordination_shares"])
        r_emp = np.array(emp["scores"])

        ax.scatter(p_emp * 100, r_emp, color="black", s=64, zorder=10, label="Empirical mean")

        # Identify best form by R²
        valid_fits = {f: d for f, d in res["fits"].items() if "r_squared" in d}
        best_form = max(valid_fits, key=lambda f: valid_fits[f]["r_squared"])

        for form_name, fit in valid_fits.items():
            fn = FUNCTIONAL_FORMS[form_name][0]
            params = list(fit["params"].values())
            r_vals = fn(p_fine, *params)
            is_best = form_name == best_form
            lw = 2.5 if is_best else 1.2
            ls = "-" if is_best else "--"
            alpha = 1.0 if is_best else 0.5
            r2_str = f"R²={fit['r_squared']:.4f}"
            c_str = f"c*={fit['c_star']*100:.0f}%"
            marker = "★ " if is_best else ""
            label = f"{marker}{_labels[form_name]}\n  [{r2_str}, {c_str}]"
            ax.plot(
                p_fine * 100, r_vals,
                color=_colors[form_name], linewidth=lw, linestyle=ls,
                alpha=alpha, label=label,
            )

        ax.set_title(_titles.get(model_id, model_id), fontsize=12, fontweight="bold")
        ax.set_xlabel("Coordination Proportion c/W (%)", fontsize=10)
        ax.set_ylabel("Effective Recall Rate R(c/W)", fontsize=10)
        ax.set_ylim(-0.02, 1.12)
        ax.set_xlim(-2, 100)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "R(c, W, T): Parametric Model Selection — Coordination-Reasoning Frontier\n"
        "RCWT Controlled Experiment · N=20/cell · 3 providers · 2026-03-30",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_path}")


# ─── Summary printing ──────────────────────────────────────────────────────────

def print_summary(results: dict[str, dict]) -> None:
    form_names = list(FUNCTIONAL_FORMS.keys())

    print("\n" + "=" * 90)
    print("MODEL COMPARISON — R² by functional form")
    print("=" * 90)
    header = f"{'Model ID':<28}" + "".join(f"{f:>14}" for f in form_names) + f"  {'Best'}"
    print(header)
    print("-" * 90)

    for model_id, res in results.items():
        row = f"{model_id:<28}"
        best_form, best_r2 = "", -np.inf
        for f in form_names:
            fit = res["fits"].get(f, {})
            if "r_squared" in fit:
                r2 = fit["r_squared"]
                row += f"{r2:>14.4f}"
                if r2 > best_r2:
                    best_r2 = r2
                    best_form = f
            else:
                row += f"{'err':>14}"
        print(row + f"  {best_form}")

    print("\n" + "=" * 90)
    print("c* ESTIMATES (inflection of steepest drop, per best-fit model)")
    print("=" * 90)
    for model_id, res in results.items():
        valid = {f: d for f, d in res["fits"].items() if "r_squared" in d}
        best_form = max(valid, key=lambda f: valid[f]["r_squared"])
        fdata = valid[best_form]
        c_star_pct = fdata["c_star"] * 100
        print(f"\n  {model_id} ({res['provider']})")
        print(f"    Best form:  {best_form}  (R²={fdata['r_squared']:.5f}, RMSE={fdata['rmse']:.5f})")
        print(f"    c*:         {c_star_pct:.1f}%  ({c_star_pct/100 * 4096:.0f} tokens in W=4096)")
        print(f"    Params:     {fdata['params']}")

    print("\n" + "=" * 90)
    print("AIC COMPARISON (lower = better fit, penalised for parameter count)")
    print("=" * 90)
    header2 = f"{'Model ID':<28}" + "".join(f"{f:>14}" for f in form_names)
    print(header2)
    print("-" * 90)
    for model_id, res in results.items():
        row = f"{model_id:<28}"
        aics = {}
        for f in form_names:
            fit = res["fits"].get(f, {})
            if "aic" in fit:
                aics[f] = fit["aic"]
                row += f"{fit['aic']:>14.2f}"
            else:
                row += f"{'err':>14}"
        print(row)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading empirical data...")
    empirical = _load_empirical()
    for mid, d in empirical.items():
        print(
            f"  {mid}: {len(d['coordination_shares'])} realized shares, "
            f"scores={[round(s, 3) for s in d['scores']]}"
        )

    print("\nFitting parametric models...")
    results = fit_all(empirical)

    out_dir = _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "rcwt_curve_fits.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_json}")

    print_summary(results)

    print("\nGenerating plot...")
    try:
        plot_fits(results, out_dir / "rcwt_curve_fits.png")
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("Plot skipped: matplotlib is not installed in this environment.")
    print("\nDone.")
