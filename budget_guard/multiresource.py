"""budget_guard/multiresource.py — joint intervals over {time, warehouse, cost}.

Data comes straight from the official Warehouse estimation records: each per-prefix
``ground_truth`` carries the true remaining amount of all three resources, so no
dialogue extraction is needed. Legal features are quantities known at turn k
(progress so far + the budgets fixed at task start); ``actual_*`` totals are leaks.

Joint methods:
- ``independent``: split-CQR per resource at alpha — each margin covers ~1-alpha,
  but the *joint* event (all three covered at once) does not.
- ``bonferroni_joint``: split-CQR per resource at alpha/3 — union bound guarantees
  joint coverage ≥ 1-alpha, at the price of wider margins.
- ``adaptive_joint_intervals``: keep a recent window of three-resource error vectors,
    updating only after a whole rollout finishes; recompute both resource margins and
    their shared empirical-copula rank as the deployment stream changes.
"""
from __future__ import annotations

from collections import deque
import json

import numpy as np
import pandas as pd

from .conformal import DEFAULT_ALPHA, cqr_split, _require_contiguous_groups

RESOURCES = ("time_weeks", "warehouse_item_weeks", "cost_usd")

# Per-sample agent interval-coverage flags in the record's ``metrics`` (BAGEN's own
# per-resource "did the agent's interval contain the truth" indicators). Averaging
# these over the guard's feasible test rows gives a like-for-like agent baseline.
_AGENT_COV_KEY = {
    "time_weeks": "time_interval_contains_actual",
    "warehouse_item_weeks": "warehouse_interval_contains_actual",
    "cost_usd": "cost_interval_contains_actual",
}

_FEATURES = (
    "completed_turns",
    "completed_weeks",
    "current_cash_usd",
    "target_cash_usd",
    "budget_time_weeks",
    "budget_warehouse_item_weeks",
    "budget_cost_usd",
)


def load_warehouse(path: str, keep_censored: bool = False) -> pd.DataFrame:
    """One row per prefix sample: legal features + the three targets.

    Samples with ``can_finish=False`` carry no remaining-resource labels (the run
    cannot complete within budget, so "remaining to completion" is undefined by
    BAGEN's convention) — but their realized totals ARE recorded, so with
    ``keep_censored=True`` the rows come back flagged ``censored`` with
    ``y_*`` recovered from ``actual_total_* − spent_so_far``. That recoverability
    is what makes Warehouse the mask-and-measure testbed for censored conformal.

    Spend-so-far: exact for time (= completed_weeks); for cost/warehouse it is
    total − actual_remaining (exact) on uncensored rows and the
    completed_weeks/total_weeks × total interpolation on censored rows — which
    also means censored rows' RECOVERED y_* labels carry that interpolation's
    error (only time_weeks is exact there). Downstream "measured coverage on
    censored runs" for cost/warehouse is therefore coverage against an
    approximate label, not ground truth.
    """
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for record in payload["results"]:
        gt = record["ground_truth"]
        is_censored = gt.get(f"actual_remaining_{RESOURCES[0]}") is None
        if is_censored and not keep_censored:
            continue
        row = {
            "rollout_index": record["rollout_index"],
            "sample_id": record["sample_id"],
            "censored": is_censored,
            "can_finish": bool(gt.get("actual_can_finish")),
        }
        for feat in _FEATURES:
            row[feat] = gt[feat]
        total_weeks = gt.get("actual_total_time_weeks")
        frac = (gt["completed_weeks"] / total_weeks) if total_weeks else np.nan
        for res in RESOURCES:
            total = gt.get(f"actual_total_{res}")
            remaining = gt.get(f"actual_remaining_{res}")
            if res == "time_weeks":
                spent = gt["completed_weeks"]
            elif remaining is not None and total is not None:
                # Exact spend-so-far from the recorded remaining. The old
                # week-proportional interpolation carried median 26.5% / 12.2%
                # relative error (item-weeks / cost) on uncensored rows and fed
                # every spent_-based trigger and loop label with it.
                spent = float(total) - float(remaining)
            else:
                spent = frac * total if total is not None else np.nan
            row[f"total_{res}"] = total
            row[f"spent_{res}"] = spent
            if remaining is None and total is not None and np.isfinite(spent):
                # Recovered label for censored rows: total - interpolated spend.
                # NOT exact (the interpolation's error applies) — see the module
                # docstring; only time_weeks is exact here.
                remaining = max(0.0, total - spent)
            row[f"y_{res}"] = remaining
        metrics = record.get("metrics") or {}
        for res in RESOURCES:
            # Preserve None (agent declared the task impossible and emitted no
            # interval). bool(None) -> False silently converted "undefined" into
            # "failed", deflating the agent baseline 10-20 points vs BAGEN's own
            # summary convention, which EXCLUDES these rows. Consumers must
            # average over non-null flags only.
            flag = metrics.get(_AGENT_COV_KEY[res])
            row[f"agent_cov_{res}"] = bool(flag) if flag is not None else None
        rows.append(row)
    df = pd.DataFrame(rows)
    if not keep_censored:
        assert df[[f"y_{r}" for r in RESOURCES]].notna().all().all()
    return df


def agent_joint_baseline(path: str) -> dict:
    """The BAGEN agent's own per-resource and joint coverage, from the summary."""
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)["summary"]
    return {
        "time_weeks": summary["time_interval_coverage_rate"],
        "warehouse_item_weeks": summary["warehouse_interval_coverage_rate"],
        "cost_usd": summary["cost_interval_coverage_rate"],
        "joint": summary["all_interval_coverage_rate"],
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[list(_FEATURES)].astype(float).copy()
    X["cash_gap"] = X["current_cash_usd"] - X["target_cash_usd"]
    X["weeks_left_of_budget"] = X["budget_time_weeks"] - X["completed_weeks"]
    return X


def _joint_rank_qhats(score_mat: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Smallest shared marginal rank whose thresholds cover enough score vectors.

    Known finite-sample optimism (decision 2026-08-02): the shared rank is
    SELECTED on the same calibration scores it then thresholds, so the copula
    scheme's joint coverage carries selection bias and is reported as an
    empirical result, not a guarantee. The guarantee-carrying headline (Bonferroni
    ~0.95 joint coverage) involves no data-driven selection. A split-
    selection variant is future work.

    Each row is one observation and each column is one resource's CQR conformity
    score. We move one *shared* rank upward until at least
    ``ceil((n + 1) * (1 - alpha))`` rows fall below all resource thresholds at once.
    This is the empirical-copula combination step with the finite-sample correction.

    Returns ``(resource_qhats, shared_rank_level)``. With too few observations for a
    finite corrected quantile, all qhats are ``+inf`` — a deliberate fail-safe rather
    than a falsely narrow joint interval.
    """
    scores = np.asarray(score_mat, dtype=float)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("score_mat must have shape (observations, resources)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    n, n_resources = scores.shape
    if n == 0:
        return np.full(n_resources, np.inf), 1.0
    required = int(np.ceil((n + 1) * (1 - alpha)))
    if required > n:
        return np.full(n_resources, np.inf), 1.0

    ordered = np.sort(scores, axis=0)
    for rank_idx in range(required - 1, n):
        qhats = ordered[rank_idx]
        joint_ok = int(np.all(scores <= qhats, axis=1).sum())
        if joint_ok >= required:
            return qhats.copy(), float((rank_idx + 1) / n)
    return ordered[-1].copy(), 1.0


def adaptive_joint_qhats(
    cal_scores: np.ndarray,
    stream_scores: np.ndarray,
    stream_groups: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    window: int = 100,
) -> tuple[np.ndarray, dict]:
    """Rollout-safe online calibration of all resource margins *and* their coupling.

    The window starts with source-environment calibration score vectors. For each new
    rollout, one joint set of resource qhats is computed from the current window and
    used for every prefix in that rollout. Only after all those predictions are made
    are the rollout's true score vectors added to the window. This prevents a subtle
    future leak: a completed rollout's final resource totals are not available while
    its earlier prefixes are still being predicted.

    Recomputing the shared empirical-copula rank adapts both the three margins and how
    their errors move together. ``window`` counts recent prefix score vectors, but all
    vectors enter atomically by completed rollout. This is an empirical online method,
    not a proof of finite-sample validity under arbitrary distribution shift; report
    its rollout-cluster confidence interval rather than claiming a universal guarantee.
    """
    cal = np.asarray(cal_scores, dtype=float)
    stream = np.asarray(stream_scores, dtype=float)
    groups = np.asarray(stream_groups)
    if cal.ndim != 2 or stream.ndim != 2 or cal.shape[1] != stream.shape[1]:
        raise ValueError("cal_scores and stream_scores must be 2-D with equal columns")
    if len(stream) != len(groups):
        raise ValueError("stream_groups must have one value per stream score row")
    if window < 1:
        raise ValueError("window must be positive")
    # Same temporal-leak guard as conformal.aci_cqr/spci/dtaci: the rollout-atomic
    # update below is only atomic when each rollout's rows are contiguous — an
    # interleaved stream would let later rollouts' labels reach earlier predictions.
    _require_contiguous_groups(groups)

    recent = deque((row.copy() for row in cal[-window:]), maxlen=window)
    qhat_path = np.empty_like(stream, dtype=float)
    shared_level_path = np.empty(len(stream), dtype=float)
    rollout_order = list(dict.fromkeys(groups.tolist()))

    for rollout in rollout_order:
        idx = np.flatnonzero(groups == rollout)
        current = np.asarray(recent, dtype=float)
        if current.size == 0:
            current = np.empty((0, stream.shape[1]), dtype=float)
        qhats, shared_level = _joint_rank_qhats(current, alpha)
        qhat_path[idx] = qhats
        shared_level_path[idx] = shared_level
        # Atomic update: no prefix in this rollout can influence another prefix from
        # the same rollout. Labels become available only when the rollout finishes.
        recent.extend(row.copy() for row in stream[idx])

    return qhat_path, {
        "window": window,
        "shared_level_path": shared_level_path,
        "rollout_order": rollout_order,
    }


def adaptive_joint_intervals(
    source_df: pd.DataFrame,
    source_masks: dict[str, np.ndarray],
    stream_df: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    window: int = 100,
) -> dict:
    """Frozen CQR models + rollout-safe adaptive joint calibration on a new stream.

    Models are fit only on the source TRAIN split. The online window is warm-started
    from source CAL score vectors. Target labels are used strictly *after* each target
    rollout's predictions, via :func:`adaptive_joint_qhats`; they are never used to
    refit the models. This is the multi-resource counterpart of windowed-CQR,
    with the empirical-copula combination recalibrated at every rollout.

    As with the joint coverage comparison, ``stream_df`` must contain feasible rows with
    observed remaining-resource labels. Censored/doomed rollouts need the separate gate
    in ``censored.py`` and are not silently treated as fully observed here.
    """
    from .conformal import _cqr_scores, _fit_cqr

    X_source = build_features(source_df)
    X_stream = build_features(stream_df)
    tr, ca = source_masks["train"], source_masks["cal"]

    cal_columns = []
    stream_columns = []
    stream_preds = {}
    for res in RESOURCES:
        y_source = source_df[f"y_{res}"].to_numpy(dtype=float)
        y_stream = stream_df[f"y_{res}"].to_numpy(dtype=float)
        mlo, mhi = _fit_cqr(X_source[tr], y_source[tr], alpha)
        cal_columns.append(
            _cqr_scores(mlo.predict(X_source[ca]), mhi.predict(X_source[ca]), y_source[ca])
        )
        qlo_stream, qhi_stream = mlo.predict(X_stream), mhi.predict(X_stream)
        stream_preds[res] = (qlo_stream, qhi_stream)
        stream_columns.append(_cqr_scores(qlo_stream, qhi_stream, y_stream))

    cal_score_mat = np.column_stack(cal_columns)
    stream_score_mat = np.column_stack(stream_columns)
    frozen_qhats, frozen_shared_level = _joint_rank_qhats(cal_score_mat, alpha)
    qhat_path, metadata = adaptive_joint_qhats(
        cal_score_mat,
        stream_score_mat,
        stream_df["rollout_index"].to_numpy(),
        alpha=alpha,
        window=window,
    )
    intervals = {}
    frozen_intervals = {}
    for i, res in enumerate(RESOURCES):
        qlo_stream, qhi_stream = stream_preds[res]
        intervals[res] = (
            np.maximum(qlo_stream - qhat_path[:, i], 0.0),
            qhi_stream + qhat_path[:, i],
        )
        frozen_intervals[res] = (
            np.maximum(qlo_stream - frozen_qhats[i], 0.0),
            qhi_stream + frozen_qhats[i],
        )
    return {
        "intervals": intervals,
        "frozen_intervals": frozen_intervals,
        "qhat_path": qhat_path,
        "frozen_qhats": frozen_qhats,
        "frozen_shared_level": frozen_shared_level,
        "cal_score_mat": cal_score_mat,
        "stream_score_mat": stream_score_mat,
        **metadata,
    }


def joint_intervals(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    alpha: float = DEFAULT_ALPHA,
    scheme: str = "bonferroni",
) -> dict:
    """Per-resource CQR intervals under the given correction scheme.

    - ``independent``: each margin conformalized at alpha (joint under-covers).
    - ``bonferroni``: each margin at alpha/3 (union bound; conservative).
    - ``copula``: one shared rank level u*, the smallest for which the CAL samples'
      *joint* event {all three scores within their u-quantiles} holds with frequency
      >= 1-alpha (+1 corrected). Adapts to the actual dependence between resources,
      so it sits between independent and Bonferroni in width while keeping >= 1-alpha
      joint coverage on CAL.

    Returns {resource: (lo, hi)} on the test rows plus the per-margin level used.
    """
    from .conformal import _cqr_scores, _fit_cqr

    X = build_features(df)
    tr, ca, te = masks["train"], masks["cal"], masks["test"]

    if scheme == "bonferroni-rollout":
        # Bonferroni with ONE calibration score per rollout (its worst prefix), per
        # resource at alpha/3: if rollouts are exchangeable, a new rollout is covered
        # on all three resources at EVERY prefix with probability >= 1 - alpha.
        from .conformal import cqr_split_rollout

        out = {}
        g = df["rollout_index"].to_numpy()
        for res in RESOURCES:
            y = df[f"y_{res}"].to_numpy(dtype=float)
            lo, hi, _ = cqr_split_rollout(X[tr], y[tr], X[ca], y[ca], g[ca], X[te],
                                          alpha=alpha / len(RESOURCES))
            out[res] = (lo, hi)
        return {"intervals": out, "per_margin_level": 1 - alpha / len(RESOURCES)}

    if scheme in ("independent", "bonferroni"):
        per_margin_alpha = alpha / len(RESOURCES) if scheme == "bonferroni" else alpha
        out = {}
        for res in RESOURCES:
            y = df[f"y_{res}"].to_numpy(dtype=float)
            lo, hi, _ = cqr_split(X[tr], y[tr], X[ca], y[ca], X[te], alpha=per_margin_alpha)
            out[res] = (lo, hi)
        return {"intervals": out, "per_margin_level": 1 - per_margin_alpha}

    assert scheme == "copula", scheme
    cal_scores, test_preds = {}, {}
    for res in RESOURCES:
        y = df[f"y_{res}"].to_numpy(dtype=float)
        mlo, mhi = _fit_cqr(X[tr], y[tr], alpha)
        cal_scores[res] = _cqr_scores(mlo.predict(X[ca]), mhi.predict(X[ca]), y[ca])
        test_preds[res] = (mlo.predict(X[te]), mhi.predict(X[te]))

    score_mat = np.column_stack([cal_scores[res] for res in RESOURCES])
    # Same rule as the adaptive copula (``_joint_rank_qhats``): the smallest shared rank
    # with ceil((n+1)(1-alpha)) CAL vectors jointly inside, +inf when n is too small.
    # The earlier grid used (joint_ok+1)/(n+1) >= 1-alpha (one point looser) and fell
    # back to a finite maximum, so the in-distribution and adaptive copulas disagreed.
    qhats, u_star = _joint_rank_qhats(score_mat, alpha)
    out = {
        res: (np.maximum(test_preds[res][0] - qhats[i], 0.0), test_preds[res][1] + qhats[i])
        for i, res in enumerate(RESOURCES)
    }
    return {"intervals": out, "per_margin_level": u_star}


def joint_coverage(df: pd.DataFrame, test_mask: np.ndarray, intervals: dict) -> dict:
    """Per-resource and all-at-once coverage on the test rows, plus ``joint_run``: the
    share of test rollouts covered on all resources at EVERY prefix."""
    covered_all = np.ones(int(test_mask.sum()), dtype=bool)
    out = {}
    for res, (lo, hi) in intervals.items():
        y = df.loc[test_mask, f"y_{res}"].to_numpy(dtype=float)
        covered = (lo <= y) & (y <= hi)
        out[res] = float(covered.mean())
        covered_all &= covered
    out["joint"] = float(covered_all.mean())
    per_run = pd.Series(covered_all).groupby(df.loc[test_mask, "rollout_index"].to_numpy()).all()
    out["joint_run"] = float(per_run.mean())
    return out
