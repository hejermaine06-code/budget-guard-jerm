"""budget_guard/run_window_runs.py — windowed-CQR with a window of RUNS vs of prefixes.

Code review: ``windowed_cqr`` keeps the last 100 prefix scores, so on long runs (Qwen
SWE-bench ~62 prefixes per run) the window holds only one or two runs. This reruns the
three settings windowed-CQR appears in — within-run (Table 5), cross-environment
(Figure 5a) and model swap (Figure 5b) — with the paper's windowed-CQR next to
``windowed_cqr_runs`` (window = last 25 finished runs), on the same data, splits and
seeds as run_coverage / run_transfer_env / run_transfer. Reported beside, not instead
of, the paper's numbers.

Nothing existing is changed. Writes tables/window_runs.csv.
Usage: python -m budget_guard.run_window_runs
"""
from __future__ import annotations

import os

import pandas as pd

from .conformal import DEFAULT_ALPHA, windowed_cqr, windowed_cqr_runs
from .data import ENVS, load_env
from .features import build_xy
from .metrics import frac_unbounded, marginal_coverage, normalized_width, rollout_coverage
from .split import group_split

SEEDS = (0, 1, 2, 3, 4)
WINDOW_RUNS = 25


def _score(setting, target, seed, method, y, lo, hi, g):
    return {"setting": setting, "target": target, "seed": seed, "method": method,
            "coverage": marginal_coverage(y, lo, hi),
            "run_coverage": rollout_coverage(y, lo, hi, g),
            "normalized_width": normalized_width(y, lo, hi, agg="median"),
            "frac_unbounded": frac_unbounded(hi)}


def _both(setting, target, seed, Xtr, ytr, Xca, yca, gca, Xte, yte, gte, alpha):
    a = windowed_cqr(Xtr, ytr, Xca, yca, Xte, yte, alpha=alpha, groups=gte)[:2]
    b = windowed_cqr_runs(Xtr, ytr, Xca, yca, Xte, yte, gca, gte, alpha=alpha,
                          window_runs=WINDOW_RUNS)[:2]
    return [_score(setting, target, seed, "windowed-CQR (100 prefixes)", yte, *a, gte),
            _score(setting, target, seed, f"windowed-CQR ({WINDOW_RUNS} runs)", yte, *b, gte)]


def main(seeds=SEEDS, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    from .run_transfer import SOURCE as SWE_SOURCE, _load, target_runs

    rows = []
    data = {}
    for env in ENVS:
        df = load_env(env)[0]
        data[env] = (*build_xy(df), df)
    for seed in seeds:
        # 1. within-run (run_coverage protocol)
        for env, (X, y, g, df) in data.items():
            m = group_split(df, seed=seed)
            tr, ca, te = m["train"], m["cal"], m["test"]
            rows += _both("within-run", env, seed, X[tr], y[tr], X[ca], y[ca], g[ca],
                          X[te], y[te], g[te], alpha)
        # 2. cross-environment (run_transfer_env protocol: sokoban 70/30 -> others)
        Xs, ys, gs, sdf = data["sokoban"]
        m = group_split(sdf, fracs=(0.7, 0.3, 0.0), seed=seed)
        for env in (e for e in ENVS if e != "sokoban"):
            Xt, yt, gt, _ = data[env]
            rows += _both("cross-env from sokoban", env, seed, Xs[m["train"]], ys[m["train"]],
                          Xs[m["cal"]], ys[m["cal"]], gs[m["cal"]], Xt, yt, gt, alpha)

    # 3. model swap on SWE-bench (run_transfer protocol: Opus 4.7 70/30 -> other agents)
    src = _load(SWE_SOURCE)
    Xs, ys, gs = build_xy(src)
    tgts = {mname: build_xy(_load(rd)) for mname, rd in target_runs().items()}
    for seed in seeds:
        m = group_split(src, fracs=(0.7, 0.3, 0.0), seed=seed)
        for mname, (Xt, yt, gt) in tgts.items():
            rows += _both("model swap (SWE-bench)", mname, seed, Xs[m["train"]], ys[m["train"]],
                          Xs[m["cal"]], ys[m["cal"]], gs[m["cal"]], Xt, yt, gt, alpha)

    per = pd.DataFrame(rows)
    table = (per.groupby(["setting", "target", "method"], sort=False)
             [["coverage", "run_coverage", "normalized_width", "frac_unbounded"]]
             .mean().reset_index())
    os.makedirs("tables", exist_ok=True)
    per.to_csv("tables/window_runs_per_seed.csv", index=False)
    table.to_csv("tables/window_runs.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.round(3).to_string(index=False))
    print("\nwrote tables/window_runs.csv, tables/window_runs_per_seed.csv")
    return table


if __name__ == "__main__":
    main()
