"""budget_guard/gate_baselines.py — simple stopping rules to compare the learned gate against.

Reviewer nwZq: "The gate has one baseline, and it is a weak one ... the agent's own
feasibility judgment ..., a logistic regression, and the burn-rate arithmetic rule
would make the improvement easier to attribute." This module holds those rules, each
as a per-prefix SCORE where low = "looks doomed", so every rule can be calibrated with
the exact same rollout-level threshold as the paper's gate
(``economic.calibrate_threshold``: false aborts <= delta per feasible run) and compared
at the same delta. Two rules are also offered uncalibrated, as a deployer would run them:

- ``logistic_gate``          logistic regression on the same features as the LightGBM gate
- ``burn_rate_slack_*``      projected spend at the end vs budget, as a fraction of budget
                             (abort uncalibrated when < 0, i.e. projected to overrun)
- ``load_agent_impossible``  the agent's own per-turn "is_impossible" verdict from the
                             BAGEN estimation record (abort at the first True; no delta)

Leakage: every score uses only turn-k-legal quantities, with one inherited caveat —
Warehouse spend-so-far on censored rows is interpolated from the run's final total
(see multiresource.load_warehouse); the paper's Warehouse caps already use it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .multiresource import RESOURCES


def logistic_gate(X_train: pd.DataFrame, feasible_train, seed: int = 0):
    """Standardized logistic regression; ``predict_proba(X)[:, 1]`` = P(feasible)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=seed),
    )
    model.fit(X_train, np.asarray(feasible_train).astype(int))
    return model


def typical_turns(df_train: pd.DataFrame) -> float:
    """Median number of usable turns per TRAIN rollout (a rollout with N turns yields
    prefixes k = 1..N-1, so N = max k + 1)."""
    return float(df_train.groupby("rollout_index")["k"].max().median() + 1)


def burn_rate_slack_tokens(df: pd.DataFrame, budget: float, n_turns: float) -> np.ndarray:
    """(budget - projected total) / budget, projecting the current burn rate over the
    turns a typical run still has left (at least one more turn)."""
    k = df["k"].to_numpy(dtype=float)
    spent = df["tokens_so_far"].to_numpy(dtype=float)
    burn = spent / k
    projected = spent + burn * np.maximum(n_turns - k, 1.0)
    return (budget - projected) / budget


def burn_rate_slack_warehouse(df: pd.DataFrame) -> np.ndarray:
    """Worst resource's (budget - projected) / budget. Time is projected to the week
    budget; capacity and cost are projected at their weekly burn rate over the weeks
    left in the time budget."""
    weeks = np.maximum(df["completed_weeks"].to_numpy(dtype=float), 1e-9)
    weeks_left = np.maximum(df["budget_time_weeks"].to_numpy(dtype=float) - weeks, 0.0)
    slacks = []
    for res in RESOURCES:
        budget = df[f"budget_{res}"].to_numpy(dtype=float)
        spent = df[f"spent_{res}"].to_numpy(dtype=float)
        if res == "time_weeks":
            projected = spent  # the time budget is the horizon itself
        else:
            projected = spent + spent / weeks * weeks_left
        slacks.append((budget - projected) / budget)
    worst = np.nanmin(np.vstack(slacks), axis=0)
    return np.where(np.isfinite(worst), worst, np.inf)  # unknown spend -> never trigger


def load_agent_impossible(record_path: str) -> pd.DataFrame:
    """sample_id -> agent_impossible (bool) from a BAGEN estimation record.

    ``prediction.is_impossible`` is the agent's own per-turn verdict that the task can
    no longer be finished within budget. Missing or null verdicts count as "continue"
    (False), which is how a deployer would have to treat an unparseable answer.
    """
    with open(record_path, encoding="utf-8") as handle:
        results = json.load(handle)["results"]
    rows = [{"sample_id": r["sample_id"],
             "agent_impossible": bool((r.get("prediction") or {}).get("is_impossible") or False)}
            for r in results]
    return pd.DataFrame(rows)


def agent_trigger(df: pd.DataFrame, flags: pd.DataFrame) -> np.ndarray:
    """Align agent verdicts to df rows by sample_id; rows without a verdict -> False."""
    merged = df[["sample_id"]].merge(flags, on="sample_id", how="left")
    return merged["agent_impossible"].astype("boolean").fillna(False).to_numpy(dtype=bool)


def auc(feasible, score) -> float:
    """Threshold-free separation of feasible vs doomed prefixes (NaN if one class)."""
    from sklearn.metrics import roc_auc_score

    feasible = np.asarray(feasible).astype(int)
    if feasible.min() == feasible.max():
        return float("nan")
    s = np.asarray(score, dtype=float)
    s = np.where(np.isfinite(s), s, np.nanmax(s[np.isfinite(s)]) if np.isfinite(s).any() else 0.0)
    return float(roc_auc_score(feasible, s))
