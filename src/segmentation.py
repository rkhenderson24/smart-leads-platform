"""Customer segmentation — supports lead-strategy development & scoping.

Maps to JD: 'Conduct customer segmentation analyses and scoping to support
Mortgage lead strategy development, assessment and revision.'

KMeans on standardized relationship/financial features (k by silhouette),
profiled into bank-readable personas with a recommended outreach play.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import features as F

SEG_FEATURES = ["annual_income", "credit_score", "relationship_tenure_months",
                "num_products", "deposit_balance", "refi_incentive_bps",
                "equity_pct"]


def fit_segments(df, k_range=range(4, 8), seed=42):
    X = StandardScaler().fit_transform(df[SEG_FEATURES].fillna(0))
    best_k, best_score, best_labels = None, -1, None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        idx = np.random.default_rng(seed).choice(len(X), min(5000, len(X)), replace=False)
        s = silhouette_score(X[idx], km.labels_[idx])
        if s > best_score:
            best_k, best_score, best_labels = k, s, km.labels_
    df = df.copy()
    df["segment"] = best_labels
    return df, best_k, round(best_score, 3)


def profile_segments(df) -> pd.DataFrame:
    agg = (df.groupby("segment")
           .agg(customers=("customer_id", "count"),
                avg_income=("annual_income", "mean"),
                avg_deposit=("deposit_balance", "mean"),
                avg_products=("num_products", "mean"),
                pct_citizens_mtg=("has_citizens_mortgage", "mean"),
                pct_competitor_mtg=("has_competitor_mortgage", "mean"),
                avg_refi_bps=("refi_incentive_bps", "mean"),
                avg_equity=("equity_pct", "mean"),
                avg_rel_value=("relationship_value", "mean"))
           .round(2))
    agg["persona"] = agg.apply(_persona, axis=1)
    agg["recommended_play"] = agg.apply(_play, axis=1)
    return agg.sort_values("avg_rel_value", ascending=False)


def _persona(r) -> str:
    if r["pct_competitor_mtg"] > 0.4:
        return "Banked, Mortgage Elsewhere (acquire)"
    if r["avg_refi_bps"] > 60 and r["pct_citizens_mtg"] > 0.5:
        return "At-Risk Refi (retain)"
    if r["avg_equity"] > 0.4 and r["pct_citizens_mtg"] > 0.5:
        return "Equity-Rich Homeowners (HELOC)"
    if r["avg_deposit"] > 150_000:
        return "High-Value Deposit Households (deepen)"
    if r["pct_citizens_mtg"] + r["pct_competitor_mtg"] < 0.3:
        return "Deposit-Only / Purchase Prospects"
    return "Stable Relationship Core"


def _play(r) -> str:
    if r["pct_competitor_mtg"] > 0.4:
        return "Consolidate outside mortgage in-house"
    if r["avg_refi_bps"] > 60 and r["pct_citizens_mtg"] > 0.5:
        return "Proactive in-house refi (retention)"
    if r["avg_equity"] > 0.4:
        return "HELOC / cash-out positioning"
    if r["avg_deposit"] > 150_000:
        return "Wealth + mortgage cross-sell"
    return "Pre-approval nurture / annual review"


if __name__ == "__main__":
    df = F.build_abt()
    df, k, sil = fit_segments(df)
    print(f"Selected k={k} (silhouette {sil})\n")
    print(profile_segments(df).to_string())
