"""budget_guard/run_budget_stop.py — the "stop at budget" baseline, and what the gate adds on top.

Reviewer question (nwZq, Kz1h): the gate is only compared against hard caps picked from
quantiles of realised totals. The policy a deployer would actually run is simpler: stop a
run the moment its spend so far passes its budget. In the token environments a run only
counts as feasible if ``total_rollout_tokens <= max_ctx`` (see extract.py), so this rule
can never abort a feasible run, by construction. It needs no data and no calibration.

Because a deployer would run it ANYWAY, the fair question for the learned gate is not
"gate vs cap" but "how much does the gate add on top of stop-at-budget?". This module
reports, per environment and per delta, on the same test rollouts and seeds as
run_economic / run_economic_warehouse:

- ``stop at budget``            the no-learning rule alone (one row; no delta)
- ``gate``                      the feasibility gate alone (as in the paper)
- ``budget + gate``             abort at the first prefix where EITHER fires
- ``added_savings``             (budget + gate) savings minus stop-at-budget savings
- doomed-run anatomy: the share of doomed rollouts that ever pass the budget (the only
  ones stop-at-budget can catch) vs. those that fail while still under budget (only a
  learned gate can catch those).

Warehouse: the budget rule aborts once ANY resource's spend so far exceeds that task's
budget. Warehouse feasibility is BAGEN's own label (``actual_can_finish``), not defined
here, so whether the rule is false-abort-free there is an empirical check, printed as
``false_abort_rate`` rather than assumed. (Warehouse spend-so-far for censored rows is
interpolated — see multiresource.load_warehouse — so its trigger point is approximate.)

Nothing here changes any existing table or figure. Writes tables/budget_stop.csv.

Usage: python -m budget_guard.run_budget_stop
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import ENVS, load_env
from .economic import _run_rule, calibrate_threshold, fit_feasibility
from .features import build_xy
from .multiresource import RESOURCES, build_features, load_warehouse
from .run_multiresource import PATH as WAREHOUSE_PATH
from .split import group_split

DELTAS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
SEEDS = (0, 1, 2, 3, 4)
TOKEN_ENVS = ("sokoban", "searchr1", "swebench")
WAREHOUSE_COLS = dict(spent_col="spent_cost_usd", total_col="total_cost_usd",
                      order_col="completed_turns")


def budget_trigger_tokens(df: pd.DataFrame, budget: float) -> np.ndarray:
    """True at every prefix whose cumulative tokens already exceed the budget."""
    return df["tokens_so_far"].to_numpy(dtype=float) > budget


def budget_trigger_warehouse(df: pd.DataFrame) -> np.ndarray:
    """True at every prefix where ANY resource's spend so far exceeds its task budget."""
    over = np.zeros(len(df), dtype=bool)
    for res in RESOURCES:
        spent = df[f"spent_{res}"].to_numpy(dtype=float)
        budget = df[f"budget_{res}"].to_numpy(dtype=float)
        over |= np.nan_to_num(spent, nan=-np.inf) > budget
    return over


def doomed_anatomy(df_test: pd.DataFrame, trigger: np.ndarray) -> dict:
    """Of the doomed test rollouts, how many does the budget rule ever flag?"""
    tmp = df_test[["rollout_index", "can_finish"]].copy()
    tmp["trigger"] = np.asarray(trigger)
    per = tmp.groupby("rollout_index").agg(feasible=("can_finish", "first"),
                                           flagged=("trigger", "any"))
    doomed = per[~per["feasible"]]
    if len(doomed) == 0:
        return {"n_doomed": 0, "doomed_over_budget_frac": np.nan}
    return {"n_doomed": int(len(doomed)),
            "doomed_over_budget_frac": float(doomed["flagged"].mean())}


def _rows_for(env: str, df: pd.DataFrame, X: pd.DataFrame, budget_trig: np.ndarray,
              seed: int, cols: dict) -> list[dict]:
    masks = group_split(df, seed=seed)
    tr, ca, te = masks["train"], masks["cal"], masks["test"]
    dft = df[te]
    model = fit_feasibility(X[tr], df.loc[tr, "can_finish"])
    s_cal = model.predict_proba(X[ca])[:, 1]
    s_te = model.predict_proba(X[te])[:, 1]
    b_te = budget_trig[te]

    base = _run_rule(dft, b_te, **cols)
    rows = [{"env": env, "seed": seed, "rule": "stop at budget", "delta": np.nan,
             **base, "added_savings": 0.0, **doomed_anatomy(dft, b_te)}]
    for delta in DELTAS:
        tau = calibrate_threshold(df[ca], s_cal, delta)
        gate = s_te < tau
        g = _run_rule(dft, gate, **cols)
        both = _run_rule(dft, gate | b_te, **cols)
        rows.append({"env": env, "seed": seed, "rule": "gate", "delta": delta, **g,
                     "added_savings": np.nan})
        rows.append({"env": env, "seed": seed, "rule": "budget + gate", "delta": delta,
                     **both,
                     "added_savings": both["doomed_savings_frac"] - base["doomed_savings_frac"]})
    return rows


def main(seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    rows: list[dict] = []
    for env in TOKEN_ENVS:
        df, _ = load_env(env)
        X, _, _ = build_xy(df)
        trig = budget_trigger_tokens(df, float(ENVS[env]["max_ctx"]))
        for seed in seeds:
            rows += _rows_for(env, df, X, trig, seed, cols={})

    wh = load_warehouse(WAREHOUSE_PATH, keep_censored=True)
    Xw = build_features(wh)
    trig_w = budget_trigger_warehouse(wh)
    for seed in seeds:
        rows += _rows_for("warehouse", wh, Xw, trig_w, seed, cols=WAREHOUSE_COLS)

    per = pd.DataFrame(rows)
    table = (per.groupby(["env", "rule", "delta"], dropna=False, sort=False)
             .agg(false_abort_rate=("false_abort_rate", "mean"),
                  doomed_savings_frac=("doomed_savings_frac", "mean"),
                  added_savings=("added_savings", "mean"),
                  doomed_over_budget_frac=("doomed_over_budget_frac", "mean"),
                  n_doomed=("n_doomed", "mean"),
                  n_feasible=("n_feasible", "mean"))
             .reset_index())

    os.makedirs("tables", exist_ok=True)
    per.to_csv("tables/budget_stop_per_seed.csv", index=False)
    table.to_csv("tables/budget_stop.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.round(4).to_string(index=False))
    print("\nwrote tables/budget_stop.csv, tables/budget_stop_per_seed.csv")
    return table


if __name__ == "__main__":
    main()
