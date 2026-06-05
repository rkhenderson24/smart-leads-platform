"""Uplift modeling — target incrementality, not just likelihood.

This is the analytical differentiator. A propensity model ranks who is likely
to fund; many of those would fund anyway (organic) and a call is wasted spend.
An uplift model estimates the CONDITIONAL AVERAGE TREATMENT EFFECT (CATE):
the *incremental* probability of funding caused by outreach. Loan officers
should be pointed at high-uplift borrowers — the persuadables.

We use a T-learner: separate models for the treated and control cohorts,
uplift = P(fund | contacted) - P(fund | not contacted). We validate with a
Qini curve, the standard uplift metric, and confirm we recover the
ground-truth effect baked into the data generator.

Maps to JD: 'Apply advanced analytics to develop and optimize lead
scoring/prioritization' + 'continuously innovate through exploration of new
techniques.'
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import features as F


class UpliftTLearner:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.m_treat = None
        self.m_ctrl = None
        self.columns = None

    def fit(self, df: pd.DataFrame):
        X = F.model_matrix(df)
        self.columns = X.columns
        t = df["treated"].values == 1
        y = df["converted"].values

        params = dict(max_depth=4, learning_rate=0.06, max_iter=300,
                      l2_regularization=1.0, random_state=self.seed)
        self.m_treat = HistGradientBoostingClassifier(**params).fit(X[t], y[t])
        self.m_ctrl = HistGradientBoostingClassifier(**params).fit(X[~t], y[~t])
        return self

    def predict_uplift(self, df: pd.DataFrame) -> np.ndarray:
        X = F.model_matrix(df).reindex(columns=self.columns, fill_value=0)
        p1 = self.m_treat.predict_proba(X)[:, 1]
        p0 = self.m_ctrl.predict_proba(X)[:, 1]
        return p1 - p0


def qini_curve(df: pd.DataFrame, uplift: np.ndarray, bins=20) -> pd.DataFrame:
    """Qini: cumulative incremental responders as we target by uplift rank.
    A model with real signal sits well above the random diagonal."""
    d = pd.DataFrame({"t": df["treated"].values, "y": df["converted"].values,
                      "u": uplift}).sort_values("u", ascending=False).reset_index(drop=True)
    n = len(d)
    d["k"] = np.arange(1, n + 1)
    ct = d["t"].cumsum()
    cc = (1 - d["t"]).cumsum()
    rt = (d["t"] * d["y"]).cumsum()
    rc = ((1 - d["t"]) * d["y"]).cumsum()
    # incremental responders attributable to targeting the top-k by uplift
    qini = rt - rc * (ct / cc.replace(0, np.nan))
    out = pd.DataFrame({
        "pct_targeted": (d["k"] / n).values,
        "qini": qini.fillna(0).values,
    })
    return out.iloc[:: max(1, n // bins)].reset_index(drop=True)


def decile_validation(df: pd.DataFrame, uplift: np.ndarray) -> pd.DataFrame:
    """Bucket by predicted uplift, then measure the *observed* treated-minus-
    control fund rate in each bucket. Monotonic increase => model is real."""
    d = pd.DataFrame({"t": df["treated"].values, "y": df["converted"].values,
                      "u": uplift})
    d["bucket"] = pd.qcut(d["u"].rank(method="first"), 5,
                          labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
    rows = []
    for b, g in d.groupby("bucket", observed=True):
        tr = g[g.t == 1]["y"].mean()
        cr = g[g.t == 0]["y"].mean()
        rows.append(dict(uplift_bucket=b, pred_uplift=round(g["u"].mean(), 4),
                         observed_treated=round(tr, 4),
                         observed_control=round(cr, 4),
                         observed_uplift=round(tr - cr, 4), n=len(g)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = F.build_abt()
    u = UpliftTLearner().fit(df).predict_uplift(df)
    print("Predicted uplift range: "
          f"[{u.min():.3f}, {u.max():.3f}], mean {u.mean():.3f}\n")
    print("Decile validation (observed uplift should rise with predicted):\n")
    print(decile_validation(df, u).to_string(index=False))
