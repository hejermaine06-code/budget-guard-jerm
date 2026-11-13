"""budget_guard/censored.py — conformal upper bounds when doomed runs censor the label.

Setting: remaining-cost y is observed only for feasible runs; infeasible runs are
censored — we know only that y exceeded the remaining budget b = budget − spent
(else the run would have finished inside it). Censoring bites the UPPER endpoint:
censored y's are systematically LARGE, so calibrating hi = pred + qhat on
uncensored data alone under-covers the very runs a budget guard most needs to flag.

Five calibrations of the one-sided upper bound, P(y <= hi) >= 1 − alpha:
- ``drop``      : uncensored scores only (the naive thing; expected to under-cover).
- ``infinity``  : censored scores treated as +inf — always valid, but qhat = inf
                  whenever the censored fraction exceeds alpha (useless at 50%).
- ``bound``     : censored scores replaced by their known lower bound b − pred —
                  informed middle ground; validity depends on how tight b is.
- ``ipcw``      : inverse-probability-of-censoring-weighted quantile of uncensored
                  scores, p̂(uncensored | x) from the feasibility classifier —
                  approximately valid, informative (the survival-analysis route).
- ``resource_bound`` : the multi-resource fix. Doom is cross-resource — a run
                  censored by its *time* budget says nothing about its *cost*
                  remaining — so a censored run bounds resource r's score ONLY when
                  r is the operative (binding) constraint; runs censored by another
                  resource are uninformative for r and dropped. Removes ``bound``'s
                  false cross-resource inflation and ``ipcw``'s positivity failure.
                  Guarantee: coverage of y_r conditional on r being operative, on
                  top of the feasible subpopulation.
- ``km`` (diagnostic): the Kaplan-Meier (1-alpha) score quantile, treating each
                  binding-censored score as right-censored at its lower bound.
                  Returns +inf when the upper tail is all censored — the rigorous
                  statement that no finite valid bound exists (see ``km_upper_qhat``).

Warehouse validation: censored rows secretly retain their totals, so a label is
recoverable — mask it, calibrate as if censored, then measure coverage against
the recovered label. Caveat (2026-08-02 audit): the recovery is EXACT only for
time_weeks; for cost/warehouse it inherits load_warehouse's week-proportional
interpolation (median 12-27% relative error on comparable uncensored rows), so
censored-stratum coverage for those two resources is measured against an
approximate label, not ground truth. Frame claims accordingly.
"""
from __future__ import annotations

import numpy as np


def _weighted_quantile(scores: np.ndarray, weights: np.ndarray, level: float) -> float:
    order = np.argsort(scores)
    scores, weights = scores[order], weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    idx = int(np.searchsorted(cdf, level, side="left"))
    return float(scores[min(idx, len(scores) - 1)])


def upper_qhat(
    scores_uncensored: np.ndarray,
    censored_lower_bounds: np.ndarray,
    alpha: float,
    method: str,
    p_uncensored: np.ndarray | None = None,
    censored_binding: np.ndarray | None = None,
) -> float:
    """qhat for hi = pred + qhat under the five censoring treatments.

    ``censored_lower_bounds``: per censored CAL sample, the known lower bound on its
    score (b_i − pred_i). ``p_uncensored``: P(uncensored | x_i) for the UNCENSORED
    cal samples (ipcw only). ``censored_binding``: bool per censored CAL sample, True
    iff THIS resource is the operative (binding) constraint for that run
    (resource_bound only; None ⇒ treat every censored sample as binding, i.e. bound).
    """
    s_unc = np.asarray(scores_uncensored, dtype=float)
    s_cens = np.asarray(censored_lower_bounds, dtype=float)

    if method == "drop":
        return _corrected_quantile(s_unc, alpha)
    if method == "infinity":
        return _corrected_quantile(np.concatenate([s_unc, np.full(len(s_cens), np.inf)]), alpha)
    if method == "bound":
        return _corrected_quantile(np.concatenate([s_unc, s_cens]), alpha)
    if method == "ipcw":
        assert p_uncensored is not None
        w = 1.0 / np.clip(np.asarray(p_uncensored, dtype=float), 0.05, 1.0)
        return _weighted_quantile(s_unc, w, 1 - alpha)
    if method == "resource_bound":
        # A censored run informs resource r's bound ONLY where r is the operative
        # constraint; runs censored by a *different* resource say nothing about y_r
        # and are dropped — this removes bound's false cross-resource inflation.
        binding = (
            np.ones(len(s_cens), dtype=bool)
            if censored_binding is None
            else np.asarray(censored_binding, dtype=bool)
        )
        return _corrected_quantile(np.concatenate([s_unc, s_cens[binding]]), alpha)
    if method == "km":
        binding = (
            np.ones(len(s_cens), dtype=bool)
            if censored_binding is None
            else np.asarray(censored_binding, dtype=bool)
        )
        return km_upper_qhat(s_unc, s_cens[binding], alpha)
    raise ValueError(method)


def _corrected_quantile(scores: np.ndarray, alpha: float) -> float:
    """(n+1)-corrected conformal quantile; +inf when there are too few scores.

    Previously each treatment clamped the level with ``min(1.0, ...)``, which silently
    returned the LARGEST score (a finite, invalid bound) on small samples, and ``drop``
    crashed on zero uncensored scores. This matches ``conformal._finite_sample_quantile``.
    """
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(s, level, method="higher"))


def gated_upper_qhat(scores_kept: np.ndarray, alpha: float) -> float:
    """Split-conformal upper qhat on the KEPT (predicted-feasible) uncensored scores.

    The complement to ``resource_bound``: instead of trying to *bound* the operative
    resource of a doomed run (impossible from a loose lower bound), a feasibility gate
    *abstains* on runs predicted doomed-on-r and emits an interval only for the rest.
    On that kept set the censored/unbounded runs are filtered out, so the score
    y − pred is observed and exchangeability is ~restored; ``hi = pred + qhat`` covers
    y at 1 − alpha **conditional on the run passing the gate** (selective coverage).
    """
    return _corrected_quantile(np.asarray(scores_kept, dtype=float), alpha)


def km_upper_qhat(
    scores_uncensored: np.ndarray, censored_lower_bounds: np.ndarray, alpha: float
) -> float:
    """Kaplan-Meier (1 - alpha) score quantile under right-censoring.

    Each (binding-)censored score is treated as RIGHT-censored at its lower bound l_i:
    the true score lies in [l_i, inf). The KM survival curve S(t) = P(score > t) is
    estimated from the exact (uncensored) scores as events and the censored scores as
    withdrawals; the (1 - alpha) quantile is the first event time with S(t) <= alpha.

    Returns +inf when S(t) never falls to <= alpha — i.e. the upper tail is dominated
    by censoring, so **no finite valid (1 - alpha) upper bound exists**. This is the
    rigorous form of the cost/warehouse limitation: doomed runs, which carry the
    largest scores, are exactly the censored ones, so the tail is unidentifiable.
    """
    s_e = np.asarray(scores_uncensored, dtype=float)   # events (exact scores)
    s_c = np.asarray(censored_lower_bounds, dtype=float)  # right-censored at l
    if len(s_e) == 0:
        return float("inf")
    surv = 1.0
    for t in np.unique(s_e):
        at_risk = int(np.sum(s_e >= t) + np.sum(s_c >= t))  # uncensored + censored still >= t
        deaths = int(np.sum(s_e == t))
        if at_risk > 0:
            surv *= 1.0 - deaths / at_risk
        if surv <= alpha:
            return float(t)
    return float("inf")
