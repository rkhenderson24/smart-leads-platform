"""Propensity models — the predictive core of lead generation.

Maps to JD: 'Apply advanced analytics to engineer customized, data driven,
timely sales leads focused on Mortgage retention and acquisition.'

We train a gradient-boosted classifier for P(fund | contacted) on the treated
population, calibrate it, and expose probabilities + SHAP-style feature
importance so loan officers see *why* a lead scored, not just the number.

Production note: swap sklearn's HistGradientBoosting for LightGBM/XGBoost
(identical interface) — kept sklearn-only here so the repo runs anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from . import features as F


class PropensityModel:
    """P(funded | contacted). Trained on the treated cohort only, since that
    is where we observe the contacted outcome."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None
        self.columns = None
        self.metrics = {}

    def fit(self, df: pd.DataFrame):
        treated = df[df["treated"] == 1].copy()
        X = F.model_matrix(treated)
        y = treated["converted"].values
        self.columns = X.columns

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=self.seed, stratify=y)
        base = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=350,
            l2_regularization=1.0, random_state=self.seed)
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.model.fit(Xtr, ytr)

        p = self.model.predict_proba(Xte)[:, 1]
        self.metrics = dict(
            auc=round(roc_auc_score(yte, p), 4),
            pr_auc=round(average_precision_score(yte, p), 4),
            n_train=len(Xtr), positives=int(y.sum()),
        )
        # decile lift table — the chart managers actually trust
        self._lift = _lift_table(yte, p)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = F.model_matrix(df).reindex(columns=self.columns, fill_value=0)
        return self.model.predict_proba(X)[:, 1]

    def lift_table(self) -> pd.DataFrame:
        return self._lift


def _lift_table(y_true, p, bins=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y_true, "p": p})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), bins,
                          labels=range(bins, 0, -1))
    base = d["y"].mean()
    g = (d.groupby("decile", observed=True)
           .agg(n=("y", "size"), responders=("y", "sum"), rate=("y", "mean"))
           .sort_index())
    g["lift_vs_base"] = (g["rate"] / base).round(2)
    g["cum_responders_pct"] = (g["responders"].cumsum()
                               / g["responders"].sum()).round(3)
    return g.reset_index()


def top_features(model: PropensityModel, df: pd.DataFrame, n=10) -> pd.DataFrame:
    treated = df[df["treated"] == 1]
    X = F.model_matrix(treated).reindex(columns=model.columns, fill_value=0)
    y = treated["converted"].values
    r = permutation_importance(model.model, X, y, n_repeats=4,
                               random_state=model.seed, scoring="roc_auc")
    return (pd.DataFrame({"feature": model.columns,
                          "importance": r.importances_mean.round(4)})
            .sort_values("importance", ascending=False)
            .head(n).reset_index(drop=True))


if __name__ == "__main__":
    df = F.build_abt()
    m = PropensityModel().fit(df)
    print("Propensity metrics:", m.metrics)
    print("\nDecile lift table:\n", m.lift_table().to_string(index=False))
    print("\nTop drivers:\n", top_features(m, df).to_string(index=False))
