"""budget_guard/run_multiresource.py — headline: joint multi-resource coverage.

Warehouse env, claude-opus-4.7 run. Compares the BAGEN agent's own joint coverage
against the guard with independent margins (each at alpha) and with the Bonferroni
correction (each at alpha/3), mean over 5 split seeds.

Usage: python -m budget_guard.run_multiresource
Writes figures/joint_vs_marginal.png and tables/multiresource.csv.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .conformal import DEFAULT_ALPHA
from .multiresource import (
    RESOURCES,
    agent_joint_baseline,
    joint_coverage,
    joint_intervals,
    load_warehouse,
)
from .plotting import bar_value_labels, apply_figure_title, figure_title, nominal_line
from .split import group_split

PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "bagen_hf", "estimation",
    "newwarehouse-opus4.7-low-thinking-main", "newwarehouse-opus4.7-low-thinking-main.json",
)
BLUE, AQUA, YELLOW, GRAY = "#2a78d6", "#1baf7a", "#eda100", "#8B93A1"


def main(seeds: tuple[int, ...] = (0, 1, 2, 3, 4), alpha: float = DEFAULT_ALPHA,
         outdir: str = "figures"):
    df = load_warehouse(PATH)
    summary = agent_joint_baseline(PATH)  # BAGEN's rate over ALL 1280 samples (not like-for-like)
    print(f"{len(df)} feasible prefix samples, {df['rollout_index'].nunique()} rollouts; "
          f"agent joint: BAGEN summary (all samples) = {summary['joint']:.3f}; "
          f"like-for-like (feasible test rows) below")

    rows = []
    for seed in seeds:
        masks = group_split(df, seed=seed)
        te = masks["test"]
        # like-for-like agent baseline: the SAME feasible test rows the guard is scored on,
        # from the record's per-sample agent coverage flags (joint = all three contain).
        # None = the agent declared the run impossible and emitted no interval;
        # BAGEN's convention excludes those rows from its coverage rates, so we do
        # too (float-mean skips NaN per-resource; the joint masks to fully-defined
        # rows — .all() would count NaN as covered).
        sub = df.loc[te, [f"agent_cov_{r}" for r in RESOURCES]].astype("float")
        acov = {res: float(sub[f"agent_cov_{res}"].mean()) for res in RESOURCES}
        defined = sub.notna().all(axis=1)
        acov["joint"] = float(sub[defined].astype(bool).all(axis=1).mean())
        rows.append({"scheme": "agent self-report", "seed": seed, **acov})
        for scheme in ("independent", "bonferroni", "copula", "bonferroni-rollout"):
            result = joint_intervals(df, masks, alpha=alpha, scheme=scheme)
            cov = joint_coverage(df, masks["test"], result["intervals"])
            rows.append({"scheme": scheme, "seed": seed, **cov})
    per_seed = pd.DataFrame(rows)
    table = per_seed.groupby("scheme", sort=False).agg(
        **{c: (c, "mean") for c in list(RESOURCES) + ["joint"]},
        joint_std=("joint", "std"),
        joint_run=("joint_run", "mean"),
    ).reset_index()

    os.makedirs("tables", exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    table.to_csv("tables/multiresource.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["agent self-report", "independent", "bonferroni", "copula"]
    colors = {"agent self-report": GRAY, "independent": AQUA, "bonferroni": YELLOW,
              "copula": BLUE}
    cats = list(RESOURCES) + ["joint"]
    xs = np.arange(len(cats))
    width = 0.21

    fig, ax = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    bar_groups: list[tuple] = []
    for i, scheme in enumerate(labels):
        sel = table[table["scheme"] == scheme].iloc[0]
        vals = [sel[c] for c in cats]
        offset = (i - (len(labels) - 1) / 2) * width
        bars = ax.bar(xs + offset, vals, width=width - 0.03, color=colors[scheme],
                      label=scheme, zorder=1)
        for patch in bars:
            patch.set_zorder(1)
        bar_groups.append((bars, vals))
    ax.set_xticks(xs, ["time", "warehouse", "cost", "JOINT (all 3)"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("coverage")
    apply_figure_title(ax, figure_title(
        "Per-resource and joint coverage on Warehouse (opus-4.7)",
        f"Like-for-like comparison, mean over {len(seeds)} split seeds",
    ))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    nominal_line(ax, alpha, zorder=4)
    for bars, vals in bar_groups:
        bar_value_labels(ax, bars, vals, fmt="{:.2f}", fontsize=8,
                         nominal=1 - alpha, nominal_nudge_for={"0.87"})
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    fig.savefig(f"{outdir}/joint_vs_marginal.png", dpi=200)

    print(table.to_string(index=False))
    print(f"\nwrote {outdir}/joint_vs_marginal.png, tables/multiresource.csv")
    return table


if __name__ == "__main__":
    main()
