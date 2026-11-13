"""budget_guard/run_allruns_seeds.py — the 19-run breadth table over all five split seeds.

Reviewers (Hut2, nwZq) noted the "windowed-CQR >= 0.85 on 18 of 19 runs" breadth claim
rests on a single split (seed 0). This repeats run_allruns' exact per-run protocol
(same discovery, same per-run context window, same skip rules, same methods) for
seeds 0-4 and reports, per run, the mean coverage over seeds, and per seed, how many
of the runs clear 0.85 and 0.90. It also adds the no-learning total-minus-spent
baseline so breadth covers that comparison too.

run_allruns.py and tables/allruns.csv are untouched (the paper's seed-0 numbers and
their checker stay as they are). Writes tables/allruns_seeds.csv (per run, mean over
seeds), tables/allruns_seeds_per_seed.csv, tables/allruns_seeds_counts.csv, and
tables/allruns_seeds_avg_counts.csv (runs whose seed-average clears 0.85 / 0.90).

Usage: python -m budget_guard.run_allruns_seeds
"""
from __future__ import annotations

import os

import pandas as pd

from .conformal import DEFAULT_ALPHA, cqr_split, dtaci, total_minus_spent, windowed_cqr
from .extract import flatten_to_samples, load_rollouts, samples_to_dataframe
from .features import build_xy
from .metrics import marginal_coverage
from .run_allruns import ENV_MAX_CTX, _official_meta, discover_runs
from .split import group_split

SEEDS = (0, 1, 2, 3, 4)
METHODS = ("split-CQR", "windowed-CQR", "DtACI", "total-minus-spent")


def main(seeds: tuple[int, ...] = SEEDS, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    rows = []
    for run in discover_runs():
        _, official_ctx = _official_meta(run["run_dir"])
        max_ctx = official_ctx or ENV_MAX_CTX[run["env"]]
        df = samples_to_dataframe(flatten_to_samples(load_rollouts(run["raw"]), max_ctx=max_ctx))
        if df.empty or df["rollout_index"].nunique() < 12:
            continue
        X, y, groups = build_xy(df)
        spent = X["tokens_so_far"].to_numpy(dtype=float)
        for seed in seeds:
            masks = group_split(df, seed=seed)
            tr, ca, te = masks["train"], masks["cal"], masks["test"]
            if te.sum() < 10:
                continue
            gte = groups[te]
            ivals = {
                "split-CQR": cqr_split(X[tr], y[tr], X[ca], y[ca], X[te], alpha=alpha)[:2],
                "windowed-CQR": windowed_cqr(X[tr], y[tr], X[ca], y[ca], X[te], y[te],
                                             alpha=alpha, groups=gte)[:2],
                "DtACI": dtaci(X[tr], y[tr], X[ca], y[ca], X[te], y[te],
                               alpha=alpha, groups=gte)[:2],
                "total-minus-spent": total_minus_spent(spent[tr], y[tr], groups[tr], spent[ca],
                                                       y[ca], spent[te], alpha=alpha)[:2],
            }
            rows.append({"env": run["env"], "model": run["model"], "seed": seed,
                         "n_test": int(te.sum()),
                         **{m: marginal_coverage(y[te], *iv) for m, iv in ivals.items()}})

    per_seed = pd.DataFrame(rows)
    per_run = (per_seed.groupby(["env", "model"], sort=True)[list(METHODS)]
               .agg(["mean", "min"]))
    per_run.columns = [f"{m}_{stat}" for m, stat in per_run.columns]
    per_run = per_run.reset_index()
    counts = (per_seed.groupby("seed")[list(METHODS)]
              .agg([lambda s: int((s >= 0.85).sum()), lambda s: int((s >= 0.90).sum()),
                    "size"]))
    counts.columns = [f"{m}_{name}" for m, name in
                      [(m, n) for m in METHODS for n in ("ge085", "ge090", "n_runs")]]
    counts = counts.reset_index()

    os.makedirs("tables", exist_ok=True)
    per_seed.to_csv("tables/allruns_seeds_per_seed.csv", index=False)
    per_run.to_csv("tables/allruns_seeds.csv", index=False)
    counts.to_csv("tables/allruns_seeds_counts.csv", index=False)

    # Per-run average over seeds: how many of the runs clear 0.85 / 0.90 on average.
    avg_counts = pd.DataFrame([
        {"method": m, "n_runs": len(per_run),
         "mean_ge085": int((per_run[f"{m}_mean"] >= 0.85).sum()),
         "mean_ge090": int((per_run[f"{m}_mean"] >= 0.90).sum())}
        for m in METHODS
    ])
    avg_counts.to_csv("tables/allruns_seeds_avg_counts.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 50):
        print(per_run.round(3).to_string(index=False))
        print("\nruns clearing 0.85 / 0.90, per seed:")
        print(counts.to_string(index=False))
        print("\nruns whose 5-seed AVERAGE clears 0.85 / 0.90:")
        print(avg_counts.to_string(index=False))
    print("\nwrote tables/allruns_seeds.csv, tables/allruns_seeds_per_seed.csv, "
          "tables/allruns_seeds_counts.csv")
    return per_run


if __name__ == "__main__":
    main()
