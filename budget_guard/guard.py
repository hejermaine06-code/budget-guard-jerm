"""budget_guard/guard.py — ``BudgetGuard``: the drop-in wrapper API.

The paper's artifact: wrap any agent loop with a training-free monitor that, each
turn, emits a **calibrated interval on remaining budget** and an **abort
recommendation** — no LLM fine-tuning. It is calibrated purely from completed runs
(CAL logs) and, in online mode, self-recalibrates from the deployment stream's own
finished runs.

    from budget_guard.extract import samples_to_dataframe, flatten_to_samples, load_rollouts
    from budget_guard.guard import BudgetGuard

    cal_df = samples_to_dataframe(flatten_to_samples(load_rollouts(path), max_ctx=7060))
    guard = BudgetGuard.from_logs(cal_df, budget=7060)   # completed rollouts

    guard.reset()                                        # a new live run begins
    for turn in agent_loop():
        step = guard.step(turn.total_tokens)             # one call per agent turn
        print(step.lo, step.hi)                          # calibrated remaining-budget interval
        if step.abort:                                   # doomed: spend will exceed budget
            break
    if not step.abort:
        guard.observe()                                  # feed the finished run back (online)

Coverage semantics (online mode = windowed-CQR, the transfer-robust default): the
conformity-score window is warm-started from CAL and updated once per FINISHED run
(a turn's true remaining-to-completion is only known when its run ends), so ``qhat``
is fixed within a run and tracks drift across runs — exactly the deployment setting.
Set ``online=False`` for the static split-CQR baseline (frozen CAL quantile).

Abort semantics (changed for the camera-ready, to match the paper): the paper's
finding is that the per-turn interval is a measurement, not a stop signal, and a rule
that checks ``spent + lo > budget`` with the alpha-level ``lo`` every turn has no
per-run false-abort bound (the per-turn chances add up). ``step.abort`` therefore
uses a separately calibrated RUN-LEVEL lower bound: a point prediction minus a margin
set from each feasible calibration run's WORST over-prediction
(``economic.calibrate_lower_bound``), so P(abort a run that would have finished within
budget) <= ``abort_delta`` when runs are exchangeable. It needs ``can_finish`` in the
logs; without it the guard falls back to the old per-turn rule (no per-run bound).

Known limitation of online mode: :meth:`observe` only learns from runs that finished
naturally, so stopped (doomed) runs never enter the window. The paper's censoring
results show that dropping such runs under-covers the expensive failures, so a guard
that aborts often may drift toward intervals that are too narrow over time.

``step`` expects the turn's **history-excluded** total tokens — the same
``turn_excluding_history`` accounting the extractor calibrated on (see
``data_dictionary.md``). Feature construction reuses ``features.build_xy`` verbatim,
so a live row is featurized identically to the training rows.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conformal import (
    DEFAULT_ALPHA,
    _cqr_scores,
    _fit_cqr,
    _finite_sample_quantile,
)
from .economic import calibrate_lower_bound, fit_remaining_regressor
from .features import build_xy
from .split import group_split


@dataclass(frozen=True)
class GuardStep:
    """One monitored turn's verdict."""

    k: int  # prefix length (turns completed so far)
    spent: float  # cumulative budget spent through turn k
    lo: float  # calibrated lower bound on remaining budget (>= 0)
    hi: float  # calibrated upper bound on remaining budget (may be +inf)
    abort: bool  # True iff spent + the RUN-LEVEL calibrated lower bound exceeds the budget

    @property
    def projected_total_lo(self) -> float:
        """Optimistic projected total spend = spent + lower-bound remaining."""
        return self.spent + self.lo


class BudgetGuard:
    """Training-free, online-calibrated budget monitor for a live agent run.

    Construct with :meth:`from_logs` from completed rollouts, then per live run:
    :meth:`reset`, one :meth:`step` per turn, and (for runs that finish naturally)
    :meth:`observe` to fold the run's labels back into the online calibration.
    """

    def __init__(
        self,
        mlo,
        mhi,
        cal_scores: np.ndarray,
        *,
        budget: float | None = None,
        alpha: float = DEFAULT_ALPHA,
        window: int = 100,
        online: bool = True,
        point_model=None,
        abort_qhat: float | None = None,
    ):
        self._mlo = mlo
        self._mhi = mhi
        self._point = point_model
        self.abort_qhat = abort_qhat
        self.budget = budget
        self.alpha = alpha
        self.online = online
        self._window: deque[float] = deque(
            np.asarray(cal_scores, dtype=float).tolist(),
            maxlen=window if online else None,
        )
        self._per_turn: list[float] = []

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_logs(
        cls,
        df: pd.DataFrame,
        *,
        budget: float | None = None,
        alpha: float = DEFAULT_ALPHA,
        window: int = 100,
        train_frac: float = 0.5,
        seed: int = 0,
        online: bool = True,
        abort_delta: float = 0.10,
    ) -> "BudgetGuard":
        """Fit the CQR band on TRAIN rollouts and cache CAL conformity scores.

        ``df`` is the extractor table (one row per prefix). The split is by rollout
        (never by prefix) via :func:`split.group_split`; the held-out TEST fraction
        is unused here and folded into calibration.
        """
        X, y, _ = build_xy(df)
        masks = group_split(df, fracs=(train_frac, 1.0 - train_frac, 0.0), seed=seed)
        tr, ca = masks["train"], masks["cal"]
        mlo, mhi = _fit_cqr(X[tr], y[tr], alpha)
        cal_scores = _cqr_scores(mlo.predict(X[ca]), mhi.predict(X[ca]), y[ca])
        point, abort_qhat = None, None
        if "can_finish" in df.columns:
            # run-level abort bound: false aborts <= abort_delta per feasible run
            point = fit_remaining_regressor(X[tr], y[tr])
            abort_qhat = calibrate_lower_bound(df[ca], point.predict(X[ca]), abort_delta)
        return cls(
            mlo, mhi, cal_scores,
            budget=budget, alpha=alpha, window=window, online=online,
            point_model=point, abort_qhat=abort_qhat,
        )

    # -- live monitoring ------------------------------------------------------
    def reset(self) -> None:
        """Begin a new live run (clear per-turn state; keep calibration)."""
        self._per_turn = []

    def step(self, turn_tokens: float) -> GuardStep:
        """Record one agent turn and return its calibrated verdict.

        ``turn_tokens`` = the just-finished turn's history-excluded total tokens.
        """
        self._per_turn.append(float(turn_tokens))
        X_row = self._featurize(self._per_turn)
        qlo = float(self._mlo.predict(X_row)[0])
        qhi = float(self._mhi.predict(X_row)[0])
        qhat = _finite_sample_quantile(np.asarray(self._window, dtype=float), self.alpha)
        lo = max(qlo - qhat, 0.0) if np.isfinite(qhat) else 0.0
        hi = qhi + qhat
        spent = float(sum(self._per_turn))
        if self.budget is None:
            abort = False
        elif self._point is not None and self.abort_qhat is not None:
            lo_run = max(float(self._point.predict(X_row)[0]) - self.abort_qhat, 0.0)
            abort = spent + lo_run > self.budget
        else:  # legacy per-turn rule (no per-run false-abort bound)
            abort = spent + lo > self.budget
        return GuardStep(k=len(self._per_turn), spent=spent, lo=lo, hi=hi, abort=abort)

    def observe(self, per_turn_totals: list[float] | None = None) -> int:
        """Fold a finished run's labels into the online window; return #scores added.

        Call only for runs that completed naturally (an aborted run's total is not a
        valid completion label). Consequence: stopped runs never inform the window, so
        under frequent aborts the online intervals can drift too narrow (see the module
        docstring). With no argument, uses the turns fed via
        :meth:`step`. A no-op when ``online=False``.
        """
        if not self.online:
            return 0
        per = list(per_turn_totals if per_turn_totals is not None else self._per_turn)
        if per_turn_totals is None:
            # Consume the live buffer: the run is over, and leaving it in place made a
            # second observe() silently fold the same conformity scores in twice,
            # double-weighting that run in every subsequent interval.
            self._per_turn = []
        if len(per) < 2:
            return 0
        total = float(sum(per))
        added = 0
        for k in range(1, len(per)):  # prefixes k = 1 .. N-1, mirroring the extractor
            y_k = total - float(sum(per[:k]))
            X_row = self._featurize(per[:k])
            score = float(
                _cqr_scores(self._mlo.predict(X_row), self._mhi.predict(X_row), np.array([y_k]))[0]
            )
            self._window.append(score)
            added += 1
        return added

    # -- internals ------------------------------------------------------------
    def _featurize(self, per: list[float]) -> pd.DataFrame:
        """Featurize a prefix of per-turn totals exactly as ``features.build_xy`` does."""
        row = pd.DataFrame(
            {
                "rollout_index": [0],
                "k": [len(per)],
                "tokens_so_far": [float(sum(per))],
                "per_turn_total": [list(per)],
                "y": [0.0],  # dummy; unused for X
                "sample_id": ["live"],
            }
        )
        X, _, _ = build_xy(row)
        return X
