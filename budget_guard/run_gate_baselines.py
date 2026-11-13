"""budget_guard/run_gate_baselines.py — the learned gate vs the simple stopping rules nwZq named.

Same environments, splits, seeds and rollout-level delta calibration as run_economic /
run_economic_warehouse. For each delta, every SCORE-based rule gets its own threshold
tau from the calibration rollouts (false aborts <= delta per feasible run), so rules are
compared at the same promised false-abort level:

- ``gate (LightGBM)``       the paper's feasibility gate (sanity check: must match Table 3)
- ``gate (logistic)``       logistic regression on the same features
- ``burn-rate (calibrated)``projected-overrun slack, thresholded like the gate

and two rules a deployer would run as-is (no delta; one row):

- ``burn-rate (raw)``       abort as soon as the projection says the run will overrun
- ``agent says impossible`` abort at the agent's own first "is_impossible" verdict

Also reports each scored rule's AUC on test prefixes (how well it separates feasible
from doomed, independent of any threshold) because savings at one delta jump around
with ~8-16 feasible test runs per split.

Nothing existing is changed. Writes tables/gate_baselines.csv (+ _per_seed).
Usage: python -m budget_guard.run_gate_baselines
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import ENVS, load_env
from .economic import _run_rule, calibrate_threshold, fit_feasibility
from .features import build_xy
from .gate_baselines import (
    agent_trigger,
    auc,
    burn_rate_slack_tokens,
    burn_rate_slack_warehouse,
    load_agent_impossible,
    logistic_gate,
    typical_turns,
)
from .multiresource import build_features, load_warehouse
from .run_multiresource import PATH as WAREHOUSE_PATH
from .split import group_split

DELTAS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
SEEDS = (0, 1, 2, 3, 4)
TOKEN_ENVS = ("sokoban", "searchr1", "swebench")
WAREHOUSE_COLS = dict(spent_col="spent_cost_usd", total_col="total_cost_usd",
                      order_col="completed_turns")


def _env_rows(env, df, X, seed, cols, slack_fn, agent_flags) -> list[dict]:
    masks = group_split(df, seed=seed)
    tr, ca, te = masks["train"], masks["cal"], masks["test"]
    dft, feas_te = df[te], df.loc[te, "can_finish"].to_numpy()

    lgbm = fit_feasibility(X[tr], df.loc[tr, "can_finish"])
    logit = logistic_gate(X[tr], df.loc[tr, "can_finish"], seed=seed)
    slack = slack_fn(df, tr)
    scores = {
        "gate (LightGBM)": lgbm.predict_proba(X)[:, 1],
        "gate (logistic)": logit.predict_proba(X)[:, 1],
        "burn-rate (calibrated)": slack,
    }

    rows = []
    for rule, s in scores.items():
        a = auc(feas_te, s[te])
        for delta in DELTAS:
            tau = calibrate_threshold(df[ca], s[ca], delta)
            rows.append({"env": env, "seed": seed, "rule": rule, "delta": delta,
                         "auc": a, **_run_rule(dft, s[te] < tau, **cols)})

    raw = {"burn-rate (raw)": slack[te] < 0}
    if agent_flags is not None:
        raw["agent says impossible"] = agent_trigger(dft, agent_flags)
    for rule, trig in raw.items():
        rows.append({"env": env, "seed": seed, "rule": rule, "delta": np.nan,
                     "auc": np.nan, **_run_rule(dft, trig, **cols)})
    return rows


def main(seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    rows: list[dict] = []
    for env in TOKEN_ENVS:
        df, _ = load_env(env)
        X, _, _ = build_xy(df)
        budget = float(ENVS[env]["max_ctx"])
        official = ENVS[env]["official"]
        flags = load_agent_impossible(official) if os.path.exists(official) else None

        def slack_fn(d, tr, budget=budget):
            return burn_rate_slack_tokens(d, budget, typical_turns(d[tr]))

        for seed in seeds:
            rows += _env_rows(env, df, X, seed, {}, slack_fn, flags)

    wh = load_warehouse(WAREHOUSE_PATH, keep_censored=True)
    Xw = build_features(wh)
    flags_w = load_agent_impossible(WAREHOUSE_PATH)
    for seed in seeds:
        rows += _env_rows("warehouse", wh, Xw, seed, WAREHOUSE_COLS,
                          lambda d, tr: burn_rate_slack_warehouse(d), flags_w)

    per = pd.DataFrame(rows)
    table = (per.groupby(["env", "rule", "delta"], dropna=False, sort=False)
             .agg(false_abort_rate=("false_abort_rate", "mean"),
                  doomed_savings_frac=("doomed_savings_frac", "mean"),
                  auc=("auc", "mean"),
                  n_feasible=("n_feasible", "mean"), n_doomed=("n_doomed", "mean"))
             .reset_index())
    os.makedirs("tables", exist_ok=True)
    per.to_csv("tables/gate_baselines_per_seed.csv", index=False)
    table.to_csv("tables/gate_baselines.csv", index=False)

    show = table[(table["delta"].isna()) | (table["delta"] == 0.10)]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("At delta = 0.10 (scored rules) and as-is rules (delta = NaN):")
        print(show.round(4).to_string(index=False))
    print("\nfull delta sweep in tables/gate_baselines.csv")
    return table


if __name__ == "__main__":
    main()
