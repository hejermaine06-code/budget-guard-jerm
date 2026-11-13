"""budget_guard/run_coverage.py — Week 3 headline: within-run coverage, all methods, all envs.

Within-run evaluation (split by rollout, same model claude-opus-4.7 everywhere):
split-absres, split-CQR, ACI-CQR (gamma tuned on CAL only), windowed-CQR, plus the BAGEN
agent-self-report baseline from the official records. Online methods stream over
the test rows in trajectory order with rollout-safe label revelation (a prefix's
remaining-to-completion label enters the update state only after its rollout ends).

Usage: python -m budget_guard.run_coverage
Writes figures/coverage_main.png and tables/coverage.csv.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .conformal import (
    DEFAULT_ALPHA,
    aci_cqr,
    constant_quantile,
    cqr_split,
    cqr_split_rollout,
    dtaci,
    split_conformal_absres,
    total_minus_spent,
    tune_gamma,
    windowed_cqr,
)
from .conformal import _finite_sample_quantile
from .data import ENVS, load_agent_estimates, load_env
from .features import build_xy
from .metrics import (
    frac_unbounded,
    marginal_coverage,
    median_width,
    normalized_width,
    pooled_bootstrap_ci,
    rollout_bootstrap_draws,
    rollout_coverage,
)
from .plotting import (
    UNBOUNDED_NOTE,
    apply_figure_title,
    figure_title,
    group_bar_width,
    mark_unbounded,
    nominal_line,
)
from .split import group_split

BLUE, AQUA, YELLOW, VIOLET, GRAY = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#8B93A1"
GREEN, RED, MAGENTA, ORANGE = "#008300", "#e34948", "#e87ba4", "#eb6834"
METHOD_COLORS = {
    "agent self-report": GRAY,
    "constant-quantile": RED,
    "conformalized-agent": MAGENTA,
    "split-absres": BLUE,
    "split-CQR": AQUA,
    "guard+self-est": ORANGE,
    "ACI-CQR": YELLOW,
    "windowed-CQR": VIOLET,
    "DtACI": GREEN,
}


def run_env(env: str, seed: int = 0, alpha: float = DEFAULT_ALPHA) -> list[dict]:
    df, baseline = load_env(env)
    est = load_agent_estimates(env)
    X, y, groups = build_xy(df)
    masks = group_split(df, seed=seed)
    tr, ca, te = masks["train"], masks["cal"], masks["test"]
    Xtr, ytr = X[tr], y[tr]
    Xca, yca = X[ca], y[ca]
    Xte, yte = X[te], y[te]

    gamma = tune_gamma(Xtr, ytr, Xca, yca, groups[ca], alpha=alpha, seed=seed)

    intervals = {}
    intervals["constant-quantile"] = constant_quantile(yca, len(yte), alpha=alpha)[:2]
    # No-learning arithmetic baseline reviewers asked for. Not in METHOD_COLORS,
    # so the headline figure is unchanged; table rows only.
    spent = X["tokens_so_far"].to_numpy(dtype=float)
    intervals["total-minus-spent"] = total_minus_spent(
        spent[tr], ytr, groups[tr], spent[ca], yca, spent[te], alpha=alpha)[:2]
    intervals["split-absres"] = split_conformal_absres(Xtr, ytr, Xca, yca, Xte, alpha=alpha)[:2]
    intervals["split-CQR"] = cqr_split(Xtr, ytr, Xca, yca, Xte, alpha=alpha)[:2]
    # One score per calibration rollout (its worst prefix): the guaranteed object.
    # Not in METHOD_COLORS, so the headline figure is unchanged; table rows only.
    intervals["rollout-CQR"] = cqr_split_rollout(
        Xtr, ytr, Xca, yca, groups[ca], Xte, alpha=alpha)[:2]
    gte = groups[te]  # rollout-safe revelation: labels enter only at rollout boundaries
    intervals["ACI-CQR"] = aci_cqr(Xtr, ytr, Xca, yca, Xte, yte, alpha=alpha, gamma=gamma, groups=gte)[:2]
    intervals["windowed-CQR"] = windowed_cqr(Xtr, ytr, Xca, yca, Xte, yte, alpha=alpha, groups=gte)[:2]
    intervals["DtACI"] = dtaci(Xtr, ytr, Xca, yca, Xte, yte, alpha=alpha, groups=gte)[:2]

    if est is not None:
        # (a) agent's own midpoint as the point predictor — zero training at all
        Xa, _, _ = build_xy(df, agent_est=est)
        mid = Xa["agent_mid"].to_numpy(dtype=float)
        mid = np.where(np.isfinite(mid), mid, np.nanmedian(mid[ca]))
        qhat = _finite_sample_quantile(np.abs(y[ca] - mid[ca]), alpha)
        intervals["conformalized-agent"] = (np.maximum(mid[te] - qhat, 0.0), mid[te] + qhat)
        # (b) guard features augmented with the agent's self-estimate
        intervals["guard+self-est"] = cqr_split(Xa[tr], ytr, Xa[ca], yca, Xa[te], alpha=alpha)[:2]

    rows = []
    if baseline is not None:
        rows.append(
            {"env": env, "method": "agent self-report", "coverage": baseline,
             "cov_ci90_lo": np.nan, "cov_ci90_hi": np.nan,
             "median_width": np.nan, "normalized_width": np.nan,
             "frac_unbounded": np.nan, "n_test": np.nan, "gamma": np.nan,
             "run_coverage": np.nan,
             "_draws": np.empty(0)}
        )
    for name, (lo, hi) in intervals.items():
        draws = rollout_bootstrap_draws(yte, lo, hi, groups[te])
        rows.append(
            {
                "env": env,
                "method": name,
                "coverage": marginal_coverage(yte, lo, hi),
                # per-seed 90% cluster-bootstrap CI (quantiles of this seed's draws)
                "cov_ci90_lo": float(np.quantile(draws, 0.05)),
                "cov_ci90_hi": float(np.quantile(draws, 0.95)),
                "_draws": draws,
                "median_width": median_width(lo, hi),
                "normalized_width": normalized_width(yte, lo, hi, agg="median"),
                "frac_unbounded": frac_unbounded(hi),
                "n_test": int(te.sum()),
                "gamma": gamma if name == "ACI-CQR" else np.nan,
                # share of test rollouts covered at EVERY prefix
                "run_coverage": rollout_coverage(yte, lo, hi, groups[te]),
            }
        )
    return rows


def main(seeds: tuple[int, ...] = (0, 1, 2, 3, 4), alpha: float = DEFAULT_ALPHA):
    rows = []
    for seed in seeds:
        for env in ENVS:
            for row in run_env(env, seed=seed, alpha=alpha):
                rows.append({**row, "seed": seed})
    per_seed = pd.DataFrame(rows).drop_duplicates(
        subset=["env", "method", "seed"]  # baseline row repeats identically per seed
    )
    agg = (
        per_seed.groupby(["env", "method"], sort=False)
        .agg(
            coverage=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            median_width=("median_width", "mean"),
            normalized_width=("normalized_width", "mean"),
            frac_unbounded=("frac_unbounded", "mean"),
            n_test=("n_test", "mean"),
            run_coverage=("run_coverage", "mean"),
        )
        .reset_index()
    )
    # cov_ci90 in the aggregate is a POOLED seed×rollout bootstrap interval — an
    # actual 90% interval, unlike the pre-2026-08-01 mean of per-seed CI endpoints
    # (which was not a CI at any level; see metrics.rollout_bootstrap_mean_draws).
    # Baseline rows have no draws and get NaN.
    pooled = per_seed.groupby(["env", "method"], sort=False)["_draws"].apply(
        lambda s: pooled_bootstrap_ci(list(s))
    )
    assert len(pooled) == len(agg)
    agg.insert(4, "cov_ci90_lo", [c[0] for c in pooled])
    agg.insert(5, "cov_ci90_hi", [c[1] for c in pooled])
    table = agg

    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    per_seed.drop(columns=["_draws"]).to_csv("tables/coverage_per_seed.csv", index=False)
    table.to_csv("tables/coverage.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    envs = list(ENVS)
    methods = list(METHOD_COLORS)
    # Width MUST scale with the method count: a fixed width silently overflows into
    # the neighbouring env group once methods are added (was 9 x 0.16 = 1.44 > 1.0).
    width = group_bar_width(len(methods))
    fig, ax = plt.subplots(figsize=(13.5, 5.2), constrained_layout=True)
    xs = np.arange(len(envs))
    any_unbounded = False
    for i, method in enumerate(methods):
        vals, errs, fracs = [], [], []
        for env in envs:
            sel = table[(table["env"] == env) & (table["method"] == method)]
            vals.append(sel["coverage"].item() if len(sel) else np.nan)
            err = sel["coverage_std"].item() if len(sel) else np.nan
            errs.append(err if np.isfinite(err) else 0.0)
            frac = sel["frac_unbounded"].item() if len(sel) else np.nan
            fracs.append(frac)
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(xs + offset, vals, width=width * 0.88,
                      color=METHOD_COLORS[method], label=method,
                      yerr=errs, error_kw={"ecolor": "#4A5361", "lw": 1, "capsize": 2})
        any_unbounded |= mark_unbounded(ax, bars, fracs, vals)
        for bar, v, e in zip(bars, vals, errs):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.018, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90)
    nominal_line(ax, alpha)
    ax.set_xticks(xs, envs)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("marginal coverage")
    title = figure_title(
        "Remaining-budget interval coverage: BAGEN agent vs conformal guard",
        f"α={alpha}, claude-opus-4.7, mean±std over {len(seeds)} split seeds",
    )
    if any_unbounded:
        title += f"\n{UNBOUNDED_NOTE}"
    apply_figure_title(ax, title, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    # ncol capped so the legend wraps instead of being clipped at the canvas edge.
    ax.legend(frameon=False, fontsize=8.5, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.06))
    fig.savefig("figures/coverage_main.png", dpi=200, bbox_inches="tight")

    print(table.to_string(index=False))
    print("\nwrote figures/coverage_main.png, tables/coverage.csv")
    return table


if __name__ == "__main__":
    main()
