"""budget_guard/run_economic.py — headline: the abort frontier.

Sweeps the false-abort tolerance delta and plots, on held-out test rollouts,
(realized false-abort rate) vs (fraction of doomed-run spend saved), against the
hard-cap heuristic frontier. Mean over 5 split seeds.

Usage: python -m budget_guard.run_economic [env]   (default: sokoban)
Writes figures/abort_frontier_<env>.png and tables/abort_frontier_<env>.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from .data import ENVS, load_env
from .economic import (
    calibrate_lower_bound,
    calibrate_threshold,
    evaluate_interval_stopping,
    evaluate_stopping,
    fit_feasibility,
    fit_remaining_regressor,
    hard_cap_baseline,
)
from .features import build_xy
from .plotting import annotate_param_clusters, apply_figure_title, figure_title
from .split import group_split

BLUE, YELLOW, GRAY, GREEN = "#2a78d6", "#eda100", "#8B93A1", "#008300"
DELTAS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
_LABEL_BBOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.92)


def _annotate_swebench_combined(ax, combined: pd.DataFrame) -> None:
    """Hand-placed δ labels — the steep early segment crowds auto-clustering."""
    row30 = combined.loc[combined["delta"] == 0.30].iloc[0]
    row40 = combined.loc[combined["delta"] == 0.40].iloc[0]
    ax.annotate(
        "δ=0.02–0.20", (0, 0), xytext=(0.11, 0.58),
        textcoords="data", fontsize=7.5, color=GREEN, ha="left", va="center",
        bbox=_LABEL_BBOX,
        arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8,
                        connectionstyle="arc3,rad=0.4"),
        zorder=5,
    )
    ax.annotate(
        "δ=0.30",
        (row30["false_abort_rate"], row30["doomed_savings_frac"]),
        xytext=(-18, 4), textcoords="offset points", fontsize=7.5, color=GREEN,
        ha="right", va="center", bbox=_LABEL_BBOX, zorder=5,
    )
    ax.annotate(
        "δ=0.40",
        (row40["false_abort_rate"], row40["doomed_savings_frac"]),
        xytext=(14, 12), textcoords="offset points", fontsize=7.5, color=GREEN,
        ha="left", va="center", bbox=_LABEL_BBOX, zorder=5,
    )


def run(env: str = "sokoban", seeds: tuple[int, ...] = (0, 1, 2, 3, 4)):
    df, _ = load_env(env)
    X, y, groups = build_xy(df)
    budget = float(ENVS[env]["max_ctx"])
    cap_grid = np.quantile(df["total_rollout_tokens"].unique(), [0.2, 0.35, 0.5, 0.65, 0.8])

    guard_rows, half_rows, combined_rows, cap_rows = [], [], [], []
    for seed in seeds:
        masks = group_split(df, seed=seed)
        tr, ca, te = masks["train"], masks["cal"], masks["test"]
        model = fit_feasibility(X[tr], df.loc[tr, "can_finish"])
        s_cal = model.predict_proba(X[ca])[:, 1]
        s_te = model.predict_proba(X[te])[:, 1]
        reg = fit_remaining_regressor(X[tr], y[tr])
        pred_cal, pred_te = reg.predict(X[ca]), reg.predict(X[te])
        for delta in DELTAS:
            tau = calibrate_threshold(df[ca], s_cal, delta)
            guard_rows.append({"seed": seed, "delta": delta,
                               **evaluate_stopping(df[te], s_te, tau)})
            # combined rule: split the false-abort budget across the two triggers
            tau2 = calibrate_threshold(df[ca], s_cal, delta / 2)
            # same gate at delta/2 with NO interval trigger: separates "stricter gate"
            # from "added interval trigger" in the combined rule's savings drop
            half_rows.append({"seed": seed, "delta": delta,
                              **evaluate_stopping(df[te], s_te, tau2)})
            qhat = calibrate_lower_bound(df[ca], pred_cal, delta / 2)
            combined_rows.append({"seed": seed, "delta": delta,
                                  **evaluate_interval_stopping(
                                      df[te], pred_te, qhat, budget,
                                      scores_test=s_te, tau=tau2)})
        for cap in cap_grid:
            cap_rows.append({"seed": seed, **hard_cap_baseline(df[te], cap)})

    def _agg(rows):
        return (pd.DataFrame(rows).groupby("delta", sort=True)
                .agg(false_abort_rate=("false_abort_rate", "mean"),
                     doomed_savings_frac=("doomed_savings_frac", "mean")).reset_index())

    guard = _agg(guard_rows)
    half = _agg(half_rows)
    combined = _agg(combined_rows)
    caps = (pd.DataFrame(cap_rows).groupby("cap_tokens", sort=True)
            .agg(false_abort_rate=("false_abort_rate", "mean"),
                 doomed_savings_frac=("doomed_savings_frac", "mean")).reset_index())

    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    guard["rule"] = "guard (feasibility)"
    half["rule"] = "guard (feasibility at delta/2)"
    combined["rule"] = "guard (feasibility+interval)"
    caps["rule"] = "hard cap"
    pd.concat([guard, half, combined, caps], ignore_index=True).to_csv(
        f"tables/abort_frontier_{env}.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    ax.plot(guard["false_abort_rate"], guard["doomed_savings_frac"], marker="o", ms=5,
            lw=2, color=BLUE, label="guard: feasibility only")
    ax.plot(combined["false_abort_rate"], combined["doomed_savings_frac"], marker="D",
            ms=5, lw=2, color=GREEN, label="guard: feasibility + interval")
    if env == "swebench":
        _annotate_swebench_combined(ax, combined)
    else:
        annotate_param_clusters(
            ax, combined["false_abort_rate"], combined["doomed_savings_frac"],
            combined["delta"], color=GREEN, x_tol=0.05, y_tol=0.08,
            xytext=(10, 12), ha="left",
        )
    ax.plot(caps["false_abort_rate"], caps["doomed_savings_frac"], marker="s", ms=5,
            lw=1.6, ls=":", color=YELLOW, label="hard token cap (sweep cap)")
    ax.set_xlabel("false-abort rate on feasible runs (test)")
    ax.set_ylabel("fraction of doomed-run spend saved")
    ax.set_xlim(-0.01, max(0.5, guard["false_abort_rate"].max() + 0.05))
    ax.set_ylim(0, 1.0)
    apply_figure_title(ax, figure_title(
        f"Abort frontier — {env}",
        f"Mean over {len(seeds)} split seeds",
    ))
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(f"figures/abort_frontier_{env}.png", dpi=200)

    print("guard (feasibility):\n" + guard.to_string(index=False))
    print("\nguard (feasibility at delta/2, no interval):\n" + half.to_string(index=False))
    print("\nguard (feasibility+interval):\n" + combined.to_string(index=False))
    print("\nhard caps:\n" + caps.to_string(index=False))
    print(f"\nwrote figures/abort_frontier_{env}.png, tables/abort_frontier_{env}.csv")
    return guard, combined, caps


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "sokoban")
