"""budget_guard/conformal.py — interval predictors for remaining budget.

Offline (Week 2): ``split_conformal_absres`` and ``cqr_split`` — valid when
calibration and test scores are exchangeable.

Online (Week 3): ``aci_cqr`` (Gibbs & Candès 2021) and ``windowed_cqr`` (windowed
variant inspired by Xu & Xie 2023 SPCI) — track drift along the test stream by adapting the
miscoverage level / score window as true labels arrive. The stream order is the
natural trajectory order (rollout by rollout, turn by turn), which mimics
deployment. Label availability: a prefix's label y (remaining-to-completion) is
only observable when its ROLLOUT finishes, so the online methods take a
``groups`` array and reveal a rollout's labels atomically at its boundary —
within a rollout the update state is frozen (same discipline as ``guard.py``'s
observe() and ``multiresource.adaptive_joint_qhats``). ``groups=None`` reveals
each label immediately after prediction; use it only when labels genuinely
arrive per step, never for remaining-budget evaluation.

All intervals are clipped below at 0 (remaining tokens are non-negative).
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

DEFAULT_ALPHA = 0.1

_LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=10,
    verbose=-1,
)


def _fit_point(X, y):
    from ._deps import require_lightgbm

    LGBMRegressor = require_lightgbm().LGBMRegressor

    model = LGBMRegressor(objective="regression", **_LGBM_PARAMS)
    model.fit(X, y)
    return model


def _fit_quantile(X, y, q: float):
    from ._deps import require_lightgbm

    LGBMRegressor = require_lightgbm().LGBMRegressor

    model = LGBMRegressor(objective="quantile", alpha=q, **_LGBM_PARAMS)
    model.fit(X, y)
    return model


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conformal quantile with the (n+1) finite-sample correction.

    Returns +inf when the corrected level exceeds 1 — the guarantee then requires
    an unbounded interval. ACI relies on this: under persistent miscoverage its
    working level sinks until the interval opens up, which is what lets it
    re-achieve coverage after a hard distribution shift.
    """
    n = len(scores)
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


def _clip_nonneg(lo: np.ndarray) -> np.ndarray:
    return np.maximum(lo, 0.0)


def _require_contiguous_groups(groups: np.ndarray) -> None:
    """Reject test streams whose rollout ids are interleaved or reordered.

    The rollout-safe buffering in aci_cqr / spci / dtaci absorbs a rollout's
    buffered labels when the ADJACENT group id changes. That is only equivalent to
    "absorb when the rollout finishes" if each rollout occupies one unbroken block.
    If an id reappears later (e.g. someone sorts the test rows by turn depth before
    calling), every id transition silently absorbs a still-running rollout's
    labels — qhat/alpha_t then depend on remaining-to-completion labels that do
    not exist yet, and coverage is inflated by temporal leakage with no visible
    symptom. All streams built by extract.flatten_to_samples are contiguous, so
    this check never fires on the shipped pipelines; it exists to make the failure
    loud instead of silent for future callers.
    """
    boundaries = np.flatnonzero(groups[1:] != groups[:-1])
    block_ids = groups[np.r_[0, boundaries + 1]]
    if len(block_ids) != len(np.unique(block_ids)):
        raise ValueError(
            "groups must be contiguous per rollout (each rollout id in one "
            "unbroken block): at least one id reappears after a boundary, which "
            "would leak a still-running rollout's labels into predictions made "
            "before that rollout finished. Sort/keep the test rows in original "
            "rollout order, or pass groups=None only for genuinely per-step-"
            "labeled streams."
        )


def split_conformal_absres(X_train, y_train, X_cal, y_cal, X_test, alpha=DEFAULT_ALPHA):
    """Symmetric intervals: point prediction ± calibrated |residual| quantile."""
    model = _fit_point(X_train, y_train)
    qhat = _finite_sample_quantile(np.abs(y_cal - model.predict(X_cal)), alpha)
    pred = model.predict(X_test)
    return _clip_nonneg(pred - qhat), pred + qhat, {"qhat": qhat}


def _cqr_scores(qlo_pred, qhi_pred, y):
    """CQR conformity score: how far y falls outside the quantile band."""
    return np.maximum(qlo_pred - y, y - qhi_pred)


def _fit_cqr(X_train, y_train, alpha):
    return _fit_quantile(X_train, y_train, alpha / 2), _fit_quantile(X_train, y_train, 1 - alpha / 2)


def cqr_split(X_train, y_train, X_cal, y_cal, X_test, alpha=DEFAULT_ALPHA):
    """Conformalized quantile regression (Romano et al. 2019)."""
    mlo, mhi = _fit_cqr(X_train, y_train, alpha)
    qhat = _finite_sample_quantile(
        _cqr_scores(mlo.predict(X_cal), mhi.predict(X_cal), y_cal), alpha
    )
    lo = mlo.predict(X_test) - qhat
    hi = mhi.predict(X_test) + qhat
    return _clip_nonneg(lo), hi, {"qhat": qhat}


def cqr_split_rollout(X_train, y_train, X_cal, y_cal, groups_cal, X_test, alpha=DEFAULT_ALPHA):
    """Rollout-level split-CQR: ONE calibration score per rollout (its worst prefix).

    ``cqr_split`` pools every calibration prefix into one quantile, but prefixes of
    one rollout share a final total and are dependent, so the pooled quantile is not
    the object the finite-sample guarantee is about. Here each calibration rollout
    contributes only its MAX CQR score. If rollouts are exchangeable, then with
    probability >= 1-alpha a new rollout's intervals contain y at EVERY one of its
    prefixes (simultaneous within-run coverage; cf. Dunn, Wasserman & Ramdas 2023,
    two-layer hierarchical conformal). Same construction as
    ``economic.calibrate_lower_bound``. Wider than pooled split-CQR by design; with
    fewer than ceil(1/alpha) - 1 calibration rollouts qhat is +inf (fail-safe).
    """
    groups_cal = np.asarray(groups_cal)
    y_cal = np.asarray(y_cal, dtype=float)
    if len(groups_cal) != len(y_cal):
        raise ValueError("groups_cal must have one rollout id per calibration row")
    mlo, mhi = _fit_cqr(X_train, y_train, alpha)
    scores = _cqr_scores(mlo.predict(X_cal), mhi.predict(X_cal), y_cal)
    rollout_ids, inverse = np.unique(groups_cal, return_inverse=True)
    rollout_max = np.full(len(rollout_ids), -np.inf)
    np.maximum.at(rollout_max, inverse, scores)
    qhat = _finite_sample_quantile(rollout_max, alpha)
    lo = mlo.predict(X_test) - qhat
    hi = mhi.predict(X_test) + qhat
    return _clip_nonneg(lo), hi, {"qhat": qhat, "n_cal_rollouts": len(rollout_ids)}


def aci_cqr(
    X_train, y_train, X_cal, y_cal, X_test, y_test,
    alpha=DEFAULT_ALPHA, gamma: float = 0.01,
    groups=None,
):
    """Adaptive Conformal Inference on top of CQR.

    Streams over the test rows in order; the working miscoverage level moves by
    ``gamma`` toward the target on each revealed error: alpha_t+1 = alpha_t +
    gamma*(alpha - err_t). With ``groups`` (rollout ids aligned to the test rows),
    a rollout's errors are buffered and applied only at the next rollout boundary
    — the level is frozen within a rollout, because remaining-to-completion labels
    do not exist until the rollout ends. ``groups=None`` = per-step revelation.

    Deviation from canonical ACI (decision 2026-08-02): when alpha_t >= 1 the
    canonical algorithm emits an EMPTY interval; we clamp the quantile lookup at
    0.999 and emit a finite interval instead. This is safe-side only (coverage
    and the error feedback can only be more conservative); state it in the paper
    rather than silently diverging.
    """
    mlo, mhi = _fit_cqr(X_train, y_train, alpha)
    cal_scores = np.sort(_cqr_scores(mlo.predict(X_cal), mhi.predict(X_cal), y_cal))
    qlo_test, qhi_test = mlo.predict(X_test), mhi.predict(X_test)
    groups = None if groups is None else np.asarray(groups)
    if groups is not None:
        _require_contiguous_groups(groups)

    alpha_t = alpha
    lo = np.empty(len(y_test))
    hi = np.empty(len(y_test))
    alphas = np.empty(len(y_test))
    pending: list[float] = []
    for t in range(len(y_test)):
        if groups is not None and t > 0 and groups[t] != groups[t - 1]:
            for err in pending:  # previous rollout finished: its labels are now real
                alpha_t = alpha_t + gamma * (alpha - err)
            pending.clear()
        if alpha_t <= 0:
            qhat = float("inf")  # the level demands an unbounded interval
        else:
            qhat = _finite_sample_quantile(cal_scores, min(alpha_t, 0.999))
        lo[t] = qlo_test[t] - qhat if np.isfinite(qhat) else 0.0
        hi[t] = qhi_test[t] + qhat
        err = float(not (lo[t] <= y_test[t] <= hi[t]))
        alphas[t] = alpha_t
        if groups is None:
            alpha_t = alpha_t + gamma * (alpha - err)
        else:
            pending.append(err)
    frac_inf = float(np.mean(~np.isfinite(hi)))
    return _clip_nonneg(lo), hi, {"gamma": gamma, "alpha_path": alphas, "frac_inf": frac_inf}


def windowed_cqr(
    X_train, y_train, X_cal, y_cal, X_test, y_test,
    alpha=DEFAULT_ALPHA, window: int = 100,
    groups=None,
):
    """Windowed online conformal on CQR scores.

    Naming (decision 2026-08-02): earlier notes and tables called this "SPCI".
    That overclaims — SPCI (Xu & Xie 2023) fits a quantile regression on the
    residual series; this method keeps a rolling window of conformity scores and
    reads the empirical quantile off it. Public label is now "windowed-CQR";
    cite SPCI as inspiration, never as the implemented method.

    Keeps the most recent ``window`` conformity scores — warm-started from CAL —
    and reads the interval half-width off that window's empirical quantile, so the
    score distribution itself tracks drift (unlike ACI, which reweights a frozen
    CAL distribution). With ``groups`` (rollout ids aligned to the test rows), a
    rollout's scores enter the window atomically at its boundary — no prefix sees
    a same-rollout label, which is unobservable until the rollout ends.
    """
    mlo, mhi = _fit_cqr(X_train, y_train, alpha)
    recent = deque(
        _cqr_scores(mlo.predict(X_cal), mhi.predict(X_cal), y_cal).tolist(),
        maxlen=window,
    )
    qlo_test, qhi_test = mlo.predict(X_test), mhi.predict(X_test)
    groups = None if groups is None else np.asarray(groups)
    if groups is not None:
        _require_contiguous_groups(groups)

    lo = np.empty(len(y_test))
    hi = np.empty(len(y_test))
    pending: list[float] = []
    for t in range(len(y_test)):
        if groups is not None and t > 0 and groups[t] != groups[t - 1]:
            recent.extend(pending)  # previous rollout finished: labels became real
            pending.clear()
        qhat = _finite_sample_quantile(np.asarray(recent), alpha)
        lo[t] = qlo_test[t] - qhat
        hi[t] = qhi_test[t] + qhat
        score_t = float(_cqr_scores(qlo_test[t : t + 1], qhi_test[t : t + 1], y_test[t : t + 1])[0])
        if groups is None:
            recent.append(score_t)
        else:
            pending.append(score_t)
    return _clip_nonneg(lo), hi, {"window": window}


spci = windowed_cqr  # backward-compatible alias (pre-2026-08-02 name)


def mondrian_cqr(
    X_train, y_train, X_cal, y_cal, k_cal, X_test, k_test,
    alpha=DEFAULT_ALPHA,
    bucket_edges: tuple[int, ...] = (2, 3),
    min_bucket: int = 10,
):
    """Group-conditional (Mondrian) CQR bucketed by turn depth k.

    One qhat per depth bucket (k=1 / k=2 / k>=3 by default) instead of a global one —
    the fix for early-turn under-coverage: validity holds per bucket, not just
    marginally. Buckets with fewer than ``min_bucket`` CAL scores fall back to the
    global qhat.
    """
    mlo, mhi = _fit_cqr(X_train, y_train, alpha)
    cal_scores = _cqr_scores(mlo.predict(X_cal), mhi.predict(X_cal), y_cal)
    k_cal = np.asarray(k_cal)
    k_test = np.asarray(k_test)
    cal_bucket = np.digitize(k_cal, bucket_edges)
    test_bucket = np.digitize(k_test, bucket_edges)

    global_qhat = _finite_sample_quantile(cal_scores, alpha)
    qhat_by_bucket = {}
    for b in np.unique(cal_bucket):
        scores = cal_scores[cal_bucket == b]
        qhat_by_bucket[b] = (
            _finite_sample_quantile(scores, alpha) if len(scores) >= min_bucket else global_qhat
        )

    qlo_test, qhi_test = mlo.predict(X_test), mhi.predict(X_test)
    qhats = np.array([qhat_by_bucket.get(b, global_qhat) for b in test_bucket])
    lo = qlo_test - qhats
    hi = qhi_test + qhats
    return _clip_nonneg(lo), hi, {"qhat_by_bucket": qhat_by_bucket}


def constant_quantile(y_cal, n_test: int, alpha=DEFAULT_ALPHA):
    """No-model baseline: the same empirical [alpha/2, 1-alpha/2] interval of the
    calibration y's for every test point. If the guard barely beats this, the
    features/model add nothing — reviewers will ask."""
    y_cal = np.asarray(y_cal, dtype=float)
    lo = float(np.quantile(y_cal, alpha / 2))
    hi = float(np.quantile(y_cal, 1 - alpha / 2))
    return np.full(n_test, max(lo, 0.0)), np.full(n_test, hi), {}


def windowed_cqr_runs(
    X_train, y_train, X_cal, y_cal, X_test, y_test, groups_cal, groups_test,
    alpha=DEFAULT_ALPHA, window_runs: int = 25,
):
    """windowed-CQR with a window of the last ``window_runs`` FINISHED ROLLOUTS.

    ``windowed_cqr`` keeps the last 100 prefix scores, so a long run (about 62 prefixes
    for Qwen on SWE-bench) fills most of the window by itself and the cutoff is set by
    one or two runs; its warm start is also the last 100 calibration rows in table
    order. Here the window holds whole rollouts: warm-started from the last
    ``window_runs`` calibration rollouts, and each finished test rollout enters as one
    block (rollout-safe: labels only after the rollout ends). The quantile is still
    read from the pooled scores inside the window. Reported beside, not instead of,
    the paper's windowed-CQR.
    """
    mlo, mhi = _fit_cqr(X_train, y_train, alpha)
    cal_scores = _cqr_scores(mlo.predict(X_cal), mhi.predict(X_cal), np.asarray(y_cal, dtype=float))
    groups_cal = np.asarray(groups_cal)
    order = list(dict.fromkeys(groups_cal.tolist()))
    runs = deque((cal_scores[groups_cal == g] for g in order), maxlen=window_runs)

    groups_test = np.asarray(groups_test)
    _require_contiguous_groups(groups_test)
    qlo_test, qhi_test = mlo.predict(X_test), mhi.predict(X_test)
    test_scores = _cqr_scores(qlo_test, qhi_test, np.asarray(y_test, dtype=float))
    lo = np.empty(len(test_scores))
    hi = np.empty(len(test_scores))
    for g in dict.fromkeys(groups_test.tolist()):
        idx = np.flatnonzero(groups_test == g)
        pooled = np.concatenate(list(runs)) if runs else np.empty(0)
        qhat = _finite_sample_quantile(pooled, alpha)
        lo[idx] = qlo_test[idx] - qhat
        hi[idx] = qhi_test[idx] + qhat
        runs.append(test_scores[idx])  # the rollout has finished: its labels are real
    return _clip_nonneg(lo), hi, {"window_runs": window_runs}


def total_minus_spent(spent_train, y_train, groups_train, spent_cal, y_cal, spent_test,
                      alpha=DEFAULT_ALPHA):
    """No-learning arithmetic baseline: "a calibrated total, minus what's spent so far".

    Reviewers (2NSH, Hut2, V41W, Kz1h, nwZq) asked whether the learned features earn
    their place against this rule. It takes the [alpha/2, 1-alpha/2] quantiles of
    run TOTALS on the train split (one total per rollout, so long runs are not
    over-counted), subtracts spend so far, then conformalizes on the calibration
    split exactly like split-CQR (pooled, finite-sample corrected) so the comparison
    is apples to apples. Unlike ``constant_quantile`` it uses spend so far; unlike
    split-CQR it learns nothing. Once a run has spent past the typical total, its
    upper end can fall to 0: that is the rule's real weakness, not a bug.
    """
    spent_train = np.asarray(spent_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    groups_train = np.asarray(groups_train)
    _, first = np.unique(groups_train, return_index=True)
    totals = spent_train[first] + y_train[first]          # remaining = total - spent
    t_lo = float(np.quantile(totals, alpha / 2))
    t_hi = float(np.quantile(totals, 1 - alpha / 2))

    spent_cal = np.asarray(spent_cal, dtype=float)
    scores = _cqr_scores(t_lo - spent_cal, t_hi - spent_cal, np.asarray(y_cal, dtype=float))
    qhat = _finite_sample_quantile(scores, alpha)

    spent_test = np.asarray(spent_test, dtype=float)
    lo = _clip_nonneg(t_lo - spent_test - qhat)
    hi = np.maximum(t_hi - spent_test + qhat, lo)  # never below lo (or below 0)
    return lo, hi, {"qhat": qhat, "total_lo": t_lo, "total_hi": t_hi}


def dtaci(
    X_train, y_train, X_cal, y_cal, X_test, y_test,
    alpha=DEFAULT_ALPHA,
    gammas: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    eta: float = 2.0,
    sigma: float = 0.01,
    groups=None,
):
    """DtACI (Gibbs & Candès 2022, simplified constants): expert aggregation over
    ACI copies with different gammas — no gamma tuning step at all.

    Each expert k tracks its own alpha_t^k; the working level is the weight-averaged
    alpha_t. After each label, beta_t (the largest level that would still have
    covered) scores every expert via pinball loss; weights update exponentially with
    a small uniform mixing sigma to keep experts alive. With ``groups`` (rollout ids
    aligned to the test rows), a rollout's scores are buffered and absorbed only at
    the next rollout boundary — expert weights and levels are frozen within a
    rollout, matching when remaining-to-completion labels actually exist.
    """
    mlo, mhi = _fit_cqr(X_train, y_train, alpha)
    cal_scores = np.sort(_cqr_scores(mlo.predict(X_cal), mhi.predict(X_cal), y_cal))
    qlo_test, qhi_test = mlo.predict(X_test), mhi.predict(X_test)
    n_cal = len(cal_scores)
    groups = None if groups is None else np.asarray(groups)
    if groups is not None:
        _require_contiguous_groups(groups)

    K = len(gammas)
    expert_alpha = np.full(K, alpha)
    weights = np.full(K, 1.0 / K)

    def _absorb(score_t: float) -> None:
        nonlocal weights, expert_alpha
        # beta_t: largest alpha whose (finite-sample) quantile still covers score_t
        beta_t = float(np.sum(cal_scores >= score_t) + 1) / (n_cal + 1)
        # pinball loss of each expert's level against beta_t
        u = beta_t - expert_alpha
        losses = u * (alpha - (u < 0).astype(float))
        weights = weights * np.exp(-eta * losses)
        weights = (1 - sigma) * weights / weights.sum() + sigma / K
        # each expert runs its own ACI update
        expert_err = (expert_alpha > beta_t).astype(float)
        expert_alpha = expert_alpha + np.asarray(gammas) * (alpha - expert_err)

    lo = np.empty(len(y_test))
    hi = np.empty(len(y_test))
    pending: list[float] = []
    for t in range(len(y_test)):
        if groups is not None and t > 0 and groups[t] != groups[t - 1]:
            for score in pending:  # previous rollout finished: labels became real
                _absorb(score)
            pending.clear()
        alpha_t = float(np.clip(np.dot(weights, expert_alpha), 0.0, 0.999))
        qhat = float("inf") if alpha_t <= 0 else _finite_sample_quantile(cal_scores, alpha_t)
        lo[t] = qlo_test[t] - qhat if np.isfinite(qhat) else 0.0
        hi[t] = qhi_test[t] + qhat

        score_t = float(_cqr_scores(qlo_test[t : t + 1], qhi_test[t : t + 1], y_test[t : t + 1])[0])
        if groups is None:
            _absorb(score_t)
        else:
            pending.append(score_t)

    frac_inf = float(np.mean(~np.isfinite(hi)))
    return _clip_nonneg(lo), hi, {"frac_inf": frac_inf, "final_weights": weights}


def tune_gamma(
    X_train, y_train, X_cal, y_cal, cal_groups,
    alpha=DEFAULT_ALPHA,
    gammas: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05),
    seed: int = 0,
) -> float:
    """Pick gamma on CAL only: split CAL's rollouts in half, calibrate on one half,
    stream ACI over the other, keep the gamma whose coverage lands closest to 1-alpha."""
    rng = np.random.default_rng(seed)
    cal_groups = np.asarray(cal_groups)
    rollouts = rng.permutation(np.unique(cal_groups))
    half = set(rollouts[: len(rollouts) // 2])
    in_a = np.isin(cal_groups, list(half))
    Xa, ya = X_cal[in_a], y_cal[in_a]
    Xb, yb = X_cal[~in_a], y_cal[~in_a]
    gb = cal_groups[~in_a]  # tune under the same rollout-safe revelation as deployment

    best_gamma, best_dist = gammas[0], float("inf")
    for gamma in gammas:
        lo, hi, _ = aci_cqr(X_train, y_train, Xa, ya, Xb, yb, alpha=alpha, gamma=gamma, groups=gb)
        coverage = float(np.mean((lo <= yb) & (yb <= hi)))
        dist = abs(coverage - (1 - alpha))
        if dist < best_dist:
            best_gamma, best_dist = gamma, dist
    return best_gamma
