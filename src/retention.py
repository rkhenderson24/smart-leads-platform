"""Retention / attrition model — the bank's first priority.

At a bank, a mortgage payoff or refi-away is a RELATIONSHIP attrition event:
it risks the deposits, the cards, the servicing income — the whole household.
This module predicts which Citizens mortgage holders are likely to leave
(absent intervention) and sizes the relationship at risk, so high-value,
high-risk households become the top retention leads.

Trained on the natural (control) attrition observed among our own holders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import features as F


class AttritionModel:
    """P(attrite | no successful retention contact) for Citizens holders."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None
        self.columns = None
        self.metrics = {}

    def fit(self, df: pd.DataFrame):
        # Train on the control cohort of our own holders (natural attrition)
        pop = df[(df["has_citizens_mortgage"] == 1) & (df["treated"] == 0)].copy()
        X = F.model_matrix(pop)
        y = pop["attrited"].values
        self.columns = X.columns
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=self.seed, stratify=y)
        self.model = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.07, max_iter=300,
            l2_regularization=1.0, random_state=self.seed).fit(Xtr, ytr)
        p = self.model.predict_proba(Xte)[:, 1]
        self.metrics = dict(auc=round(roc_auc_score(yte, p), 4),
                            base_attrition=round(float(y.mean()), 4),
                            n_holders=int(len(pop)))
        return self

    def attrition_risk(self, df: pd.DataFrame) -> np.ndarray:
        X = F.model_matrix(df).reindex(columns=self.columns, fill_value=0)
        risk = self.model.predict_proba(X)[:, 1]
        # only meaningful for our own holders
        return np.where(df["has_citizens_mortgage"].values == 1, risk, 0.0)


if __name__ == "__main__":
    df = F.build_abt()
    m = AttritionModel().fit(df)
    print("Attrition model:", m.metrics)
    df = df.assign(attrition_risk=m.attrition_risk(df))
    atrisk = df[df["has_citizens_mortgage"] == 1].nlargest(5, "attrition_risk")
    print("\nTop at-risk relationships:")
    print(atrisk[["customer_id", "attrition_risk", "relationship_value",
                  "refi_incentive_bps", "num_products"]].to_string(index=False))
