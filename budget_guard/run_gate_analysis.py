"""budget_guard/run_gate_analysis.py — what the learned gate relies on, when it fires, and
whether it survives a model swap.

Reviewer nwZq: "A feature ablation, plus features describing the agent's actions ...
would strengthen the gate" and "Running the shift experiments on the classifier would
close the largest gap." Reviewer Zm2B / the IAB workshop: limited behavioral
interpretation. Three analyses of the paper's LightGBM gate (fit_feasibility, rollout-
level tau at delta = 0.10), same splits and seeds as run_economic*:

1. ablation   drop one feature group at a time; report test-prefix AUC and savings/
              false aborts at delta = 0.10, next to the all-features gate
2. behavior   gain-based feature-group importance of the full gate, and for doomed test
              runs it catches: at which turn it fires, as a share of the run's length,
              and how much of the run's eventual spend was already gone
3. shift      train + calibrate the gate on Claude Opus 4.7, deploy it frozen on the other
              agents' SWE-bench and Warehouse runs (the interval shift study, now for
              the gate): false aborts, savings and AUC on each target

Feature groups (token envs): progress = k; spend = tokens_so_far, burn_rate;
recent usage = per-turn last/mean/std/min/max, last-3 mean, trend.
Warehouse: progress = turns, weeks, weeks left; cash = current, target, gap;
budgets = the three task budgets. (Warehouse features carry no spend at all.)

Nothing existing is changed. Writes tables/gate_ablation.csv, gate_importance.csv,
gate_timing.csv, gate_shift.csv.  Usage: python -m budget_guard.run_gate_analysis
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import load_env
from .economic import _run_rule, calibrate_threshold, fit_feasibility
from .features import build_xy
from .gate_baselines import auc
from .multiresource import build_features, load_warehouse
from .run_multiresource import PATH as WAREHOUSE_PATH
from .split import group_split

SEEDS = (0, 1, 2, 3, 4)
DELTA = 0.10
TOKEN_ENVS = ("sokoban", "searchr1", "swebench")
WAREHOUSE_COLS = dict(spent_col="spent_cost_usd", total_col="total_cost_usd",
                      order_col="completed_turns")
TOKEN_GROUPS = {
    "progress": ["k"],
    "spend": ["tokens_so_far", "burn_rate"],
    "recent usage": ["last_turn_tokens", "mean_turn_tokens", "std_turn_tokens",
                     "min_turn_tokens", "max_turn_tokens", "last3_mean_tokens", "turn_trend"],
}
WAREHOUSE_GROUPS = {
    "progress": ["completed_turns", "completed_weeks", "weeks_left_of_budget"],
    "cash": ["current_cash_usd", "target_cash_usd", "cash_gap"],
    "budgets": ["budget_time_weeks", "budget_warehouse_item_weeks", "budget_cost_usd"],
}


def _fit_eval(df, X, tr, ca, te, cols):
    model = fit_feasibility(X[tr], df.loc[tr, "can_finish"])
    s = model.predict_proba(X)[:, 1]
    tau = calibrate_threshold(df[ca], s[ca], DELTA)
    out = _run_rule(df[te], s[te] < tau, **cols)
    out["auc"] = auc(df.loc[te, "can_finish"].to_numpy(), s[te])
    return model, s, tau, out


def _timing(df_test, score_te, tau, order_col, spent_col, total_col):
    """For doomed test runs: caught?, turn of first abort, share of run length and of
    eventual spend already used at that point."""
    tmp = df_test[["rollout_index", order_col, "can_finish", spent_col, total_col]].copy()
    tmp["flag"] = score_te < tau
    out = []
    for _, g in tmp.sort_values(order_col).groupby("rollout_index"):
        if bool(g["can_finish"].iloc[0]):
            continue
        n_turns = float(g[order_col].max() + 1)
        hit = g[g["flag"]]
        if len(hit):
            k = float(hit[order_col].iloc[0])
            total = float(g[total_col].iloc[0])
            out.append({"caught": True, "abort_turn": k, "turn_share": k / n_turns,
                        "spend_share": float(hit[spent_col].iloc[0]) / total if total else np.nan})
        else:
            out.append({"caught": False, "abort_turn": np.nan, "turn_share": np.nan,
                        "spend_share": np.nan})
    return out


def _importance(model, groups):
    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                     index=model.booster_.feature_name())
    total = gain.sum() or 1.0
    return {g: float(gain.reindex(cols).fillna(0).sum() / total) for g, cols in groups.items()}


def _env_analysis(env, df, X, groups, cols, order_col, spent_col, total_col):
    abl, imp, tim = [], [], []
    for seed in SEEDS:
        m = group_split(df, seed=seed)
        tr, ca, te = m["train"], m["cal"], m["test"]
        model, s, tau, full = _fit_eval(df, X, tr, ca, te, cols)
        abl.append({"env": env, "seed": seed, "features": "all", **full})
        for g, gcols in groups.items():
            keep = [c for c in X.columns if c not in gcols]
            _, _, _, out = _fit_eval(df, X[keep], tr, ca, te, cols)
            abl.append({"env": env, "seed": seed, "features": f"without {g}", **out})
        imp.append({"env": env, "seed": seed, **_importance(model, groups)})
        for r in _timing(df[te], s[te], tau, order_col, spent_col, total_col):
            tim.append({"env": env, "seed": seed, **r})
    return abl, imp, tim


def _shift_rows():
    from .run_transfer import SOURCE, _load, target_runs
    from .run_transfer_warehouse import SOURCE_PATH, TARGETS

    rows = []
    src = _load(SOURCE)
    Xs, _, _ = build_xy(src)
    tgts = {m: _load(rd) for m, rd in target_runs().items()}
    tgt_x = {m: build_xy(d)[0] for m, d in tgts.items()}
    wsrc = load_warehouse(SOURCE_PATH, keep_censored=True)
    Xws = build_features(wsrc)
    wtgts = {m: load_warehouse(p, keep_censored=True) for m, p in TARGETS.items()}

    for seed in SEEDS:
        for env, sdf, sX, tdfs, tXs, cols in (
            ("swebench", src, Xs, tgts, tgt_x, {}),
            ("warehouse", wsrc, Xws, wtgts, {m: build_features(d) for m, d in wtgts.items()},
             WAREHOUSE_COLS),
        ):
            m = group_split(sdf, seed=seed)
            model, s, tau, own = _fit_eval(sdf, sX, m["train"], m["cal"], m["test"], cols)
            rows.append({"env": env, "seed": seed, "target": "same agent (Opus 4.7)", **own})
            for name, tdf in tdfs.items():
                st = model.predict_proba(tXs[name])[:, 1]
                rows.append({"env": env, "seed": seed, "target": name,
                             **_run_rule(tdf, st < tau, **cols),
                             "auc": auc(tdf["can_finish"].to_numpy(), st)})
    return rows


def main() -> None:
    abl, imp, tim = [], [], []
    for env in TOKEN_ENVS:
        df, _ = load_env(env)
        X, _, _ = build_xy(df)
        a, i, t = _env_analysis(env, df, X, TOKEN_GROUPS, {}, "k",
                                "tokens_so_far", "total_rollout_tokens")
        abl, imp, tim = abl + a, imp + i, tim + t
    wh = load_warehouse(WAREHOUSE_PATH, keep_censored=True)
    a, i, t = _env_analysis("warehouse", wh, build_features(wh), WAREHOUSE_GROUPS,
                            WAREHOUSE_COLS, "completed_turns", "spent_cost_usd",
                            "total_cost_usd")
    abl, imp, tim = abl + a, imp + i, tim + t

    ablation = (pd.DataFrame(abl).groupby(["env", "features"], sort=False)
                [["auc", "false_abort_rate", "doomed_savings_frac"]].mean().reset_index())
    importance = pd.DataFrame(imp).drop(columns="seed").groupby("env", sort=False).mean()
    t = pd.DataFrame(tim)
    timing = t.groupby("env", sort=False).agg(
        doomed_runs=("caught", "size"), caught_frac=("caught", "mean"),
        median_abort_turn=("abort_turn", "median"),
        median_turn_share=("turn_share", "median"),
        median_spend_share=("spend_share", "median")).reset_index()
    shift = (pd.DataFrame(_shift_rows()).groupby(["env", "target"], sort=False)
             [["auc", "false_abort_rate", "doomed_savings_frac"]].mean().reset_index())

    os.makedirs("tables", exist_ok=True)
    ablation.to_csv("tables/gate_ablation.csv", index=False)
    importance.to_csv("tables/gate_importance.csv")
    timing.to_csv("tables/gate_timing.csv", index=False)
    shift.to_csv("tables/gate_shift.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"1. ABLATION (delta = {DELTA}, mean over {len(SEEDS)} seeds)")
        print(ablation.round(3).to_string(index=False))
        print("\n2a. FEATURE-GROUP IMPORTANCE (share of LightGBM gain)")
        print(importance.round(3).to_string())
        print(f"\n2b. WHEN THE GATE FIRES ON DOOMED TEST RUNS (delta = {DELTA})")
        print(timing.round(3).to_string(index=False))
        print(f"\n3. MODEL-SWAP SHIFT (gate frozen from Opus 4.7, delta = {DELTA})")
        print(shift.round(3).to_string(index=False))
    print("\nwrote tables/gate_ablation.csv, gate_importance.csv, gate_timing.csv, gate_shift.csv")


if __name__ == "__main__":
    main()
