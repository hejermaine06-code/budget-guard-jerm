"""budget_guard/run_multiresource_overlap.py — why Table 2's joint coverage sits exactly
where it does.

Reviewer V41W: in Table 2, joint coverage equals 1 - (sum of per-resource misses) for
three of four rows — exactly the floor that holds only if the resources NEVER miss on
the same snapshot. Either the column was computed rather than measured, or misses are
disjoint; report the overlap so readers can tell. (It is measured:
multiresource.joint_coverage checks all three resources on each test snapshot.)

Re-runs run_multiresource's exact loop (Warehouse, Claude Opus 4.7, seeds 0-4, same
schemes) and reports, per scheme: per-resource miss rates, the share of snapshots
missing on at least one / at least two resources, and the joint coverage next to its
no-overlap floor (1 - sum of misses) and its ceiling (the lowest per-resource coverage).
The agent row uses the record's own per-resource flags on rows where all three exist.

Nothing existing is changed. Writes tables/multiresource_overlap.csv.
Usage: python -m budget_guard.run_multiresource_overlap
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .conformal import DEFAULT_ALPHA
from .multiresource import RESOURCES, joint_intervals, load_warehouse
from .run_multiresource import PATH
from .split import group_split


def overlap_stats(covered: np.ndarray) -> dict:
    """covered: (n_snapshots, n_resources) booleans. Miss structure of the joint event."""
    covered = np.asarray(covered, dtype=bool)
    miss = ~covered
    per = miss.mean(axis=0)
    n_miss = miss.sum(axis=1)
    return {
        **{f"miss_{r}": float(m) for r, m in zip(RESOURCES, per)},
        "miss_any": float((n_miss >= 1).mean()),
        "miss_two_plus": float((n_miss >= 2).mean()),
        "joint": float((n_miss == 0).mean()),
        "floor_if_disjoint": float(max(0.0, 1.0 - per.sum())),
        "ceiling": float(covered.mean(axis=0).min()),
    }


def main(seeds=(0, 1, 2, 3, 4), alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    df = load_warehouse(PATH)
    rows = []
    for seed in seeds:
        masks = group_split(df, seed=seed)
        te = masks["test"]
        sub = df.loc[te, [f"agent_cov_{r}" for r in RESOURCES]]
        defined = sub.notna().all(axis=1).to_numpy()
        rows.append({"scheme": "agent self-report", "seed": seed,
                     "n": int(defined.sum()),
                     **overlap_stats(sub[defined].astype(bool).to_numpy())})
        for scheme in ("independent", "bonferroni", "copula", "bonferroni-rollout"):
            ivals = joint_intervals(df, masks, alpha=alpha, scheme=scheme)["intervals"]
            cov = np.column_stack([
                (lo <= df.loc[te, f"y_{r}"].to_numpy(float)) & (df.loc[te, f"y_{r}"].to_numpy(float) <= hi)
                for r, (lo, hi) in ((r, ivals[r]) for r in RESOURCES)
            ])
            rows.append({"scheme": scheme, "seed": seed, "n": int(te.sum()), **overlap_stats(cov)})
    per = pd.DataFrame(rows)
    table = per.drop(columns="seed").groupby("scheme", sort=False).mean().reset_index()
    os.makedirs("tables", exist_ok=True)
    table.to_csv("tables/multiresource_overlap.csv", index=False)
    with pd.option_context("display.width", 220):
        print(table.round(4).to_string(index=False))
    print("\nwrote tables/multiresource_overlap.csv")
    return table


if __name__ == "__main__":
    main()
