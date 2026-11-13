"""budget_guard/run_gated.py — feasibility-gated intervals: predict-feasible-then-bound.

``run_censored`` showed the operative resource can't be tightly bounded from censored
data (a doomed run's lower bound budget−spent is too loose). The fix is selective: gate
on the feasibility classifier and emit a calibrated interval ONLY for runs predicted
feasible-on-r; runs flagged doomed-on-r are aborted, not bounded. The interval covers
y_r conditional on passing the gate, where censoring is filtered out so exchangeability
is ~restored.

Usage: python -m budget_guard.run_gated
Writes tables/gated_intervals.csv.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .censored import gated_upper_qhat
from .economic import fit_feasibility, fit_remaining_regressor
from .multiresource import RESOURCES, build_features, load_warehouse
from .plotting import apply_figure_title, figure_title
from .run_multiresource import PATH
from .split import group_split

ALPHA = 0.1
DELTAS = (0.02, 0.05, 0.10, 0.20, 0.30)  # gate false-abort budgets swept for the frontier
AQUA, YELLOW, BLUE = "#1baf7a", "#eda100", "#2a78d6"


def _cov(reg, qhat, X, y, obs, mask):
    """Coverage y <= pred + qhat over the observed-label rows in ``mask``."""
    mm = mask & obs
    if not mm.any():
        return float("nan")
    return float(np.mean(y[mm] <= reg.predict(X[mm]) + qhat))


def main(seeds: tuple[int, ...] = (0, 1, 2, 3, 4), outdir: str = "figures"):
    df = load_warehouse(PATH, keep_censored=True)
    X = build_features(df)
    cens = df["censored"].to_numpy(dtype=bool)
    print(f"{len(df)} prefix samples, {df['rollout_index'].nunique()} rollouts; "
          f"censored = {cens.mean():.0%}; sweeping gate false-abort budgets {DELTAS}")

    rows = []
    for res in RESOURCES:
        y = df[f"y_{res}"].to_numpy(dtype=float)
        budget = df[f"budget_{res}"].to_numpy(dtype=float)
        total = df[f"total_{res}"].to_numpy(dtype=float)
        obs = np.isfinite(y)
        doomed = cens & np.isfinite(total) & (total > budget)  # doomed-on-r (operative)
        feas_r = ~doomed                                       # r-feasible label

        for seed in seeds:
            m = group_split(df, seed=seed)
            tr, ca, te = m["train"], m["cal"], m["test"]
            clf = fit_feasibility(X[tr], feas_r[tr])
            reg = fit_remaining_regressor(X[tr & ~cens], y[tr & ~cens])
            score = clf.predict_proba(X)[:, 1]                 # P(feasible-on-r | x)
            feas_ca = ca & feas_r
            te_doomed = te & doomed

            for delta in DELTAS:
                # tau: keep scores >= the delta-quantile of FEASIBLE cal rows (~delta false-abort)
                tau = float(np.quantile(score[feas_ca], delta)) if feas_ca.any() else 0.0
                keep = score >= tau
                ca_fit = ca & ~cens & keep                     # kept, observed cal scores
                qhat = gated_upper_qhat(y[ca_fit] - reg.predict(X[ca_fit]), ALPHA)
                rows.append({
                    "resource": res, "delta": delta, "seed": seed, "qhat": qhat,
                    "kept_frac": float(np.mean(keep[te])),
                    "gate_catch_doomed": float(np.mean(~keep[te_doomed])) if te_doomed.any() else float("nan"),
                    "false_abort_feasible": float(np.mean(~keep[te & feas_r & ~cens])),
                    "cov_kept_feasible": _cov(reg, qhat, X, y, obs, te & keep & feas_r),
                    # every admitted prefix, doomed ones included (a live guard cannot
                    # tell them apart); censored rows use RECOVERED labels, exact for
                    # time, interpolated (approximate) for capacity and cost
                    "cov_kept_all": _cov(reg, qhat, X, y, obs, te & keep),
                })

    table = (pd.DataFrame(rows).groupby(["resource", "delta"], sort=False)
             .agg(kept_frac=("kept_frac", "mean"),
                  gate_catch_doomed=("gate_catch_doomed", "mean"),
                  false_abort_feasible=("false_abort_feasible", "mean"),
                  cov_kept_feasible=("cov_kept_feasible", "mean"),
                  cov_kept_all=("cov_kept_all", "mean"),
                  qhat_median=("qhat", "median")).reset_index())
    os.makedirs("tables", exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    table.to_csv("tables/gated_intervals.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"time_weeks": AQUA, "warehouse_item_weeks": YELLOW, "cost_usd": BLUE}
    names = {"time_weeks": "time", "warehouse_item_weeks": "warehouse", "cost_usd": "cost"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for res in RESOURCES:
        sel = table[table["resource"] == res].sort_values("kept_frac")
        ax.plot(sel["kept_frac"], sel["cov_kept_feasible"], marker="o", ms=5, lw=2,
                color=colors[res], label=names[res])
    ax.axhline(1 - ALPHA, color="#4A5361", lw=1.2, ls="--")
    ax.text(0.02, 1 - ALPHA + 0.008, f"nominal {1 - ALPHA:.2f}", fontsize=9, color="#4A5361")
    ax.set_xlabel("fraction of runs kept (gate abstains on predicted doom)")
    ax.set_ylabel("coverage on kept feasible runs")
    apply_figure_title(ax, figure_title(
        "Feasibility-gated selective frontier",
        f"Warehouse, {len(seeds)} seeds; keeping fewer runs (higher δ) lifts coverage on the kept set",
    ))
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, title="resource")
    fig.savefig(f"{outdir}/selective_frontier.png", dpi=200)

    print(table.to_string(index=False))
    print(f"\nwrote {outdir}/selective_frontier.png, tables/gated_intervals.csv")
    return table


if __name__ == "__main__":
    main()
