"""Call disposition analytics — closing the feedback loop.

Maps to JD: 'Manage and analyze call disposition data to optimize lead
selection and assess performance.'

Disposition outcomes are the ground truth that tells us whether scoring works.
Terminal favorable outcomes differ by lead type: 'Funded' (acquisition / new
origination) and 'Retained' (a saved at-risk relationship). We compute the
funnel, the production validation of the score, and per-type performance.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def disposition_funnel(contacts: pd.DataFrame) -> pd.DataFrame:
    treated = contacts[contacts["treated"] == 1]
    f = treated["last_disposition"].value_counts().rename("count").to_frame()
    order = ["No Answer", "Callback Requested", "Not Interested",
             "Application Started", "Funded", "Retained"]
    f = f.reindex([o for o in order if o in f.index])
    f["share"] = (f["count"] / f["count"].sum()).round(4)
    return f


def conversion_by_score(leads: pd.DataFrame, contacts: pd.DataFrame) -> pd.DataFrame:
    """Did higher-scored leads actually convert more among the contacted?"""
    merged = leads.merge(contacts[["customer_id", "treated", "converted"]],
                         on="customer_id", how="inner")
    treated = merged[merged["treated"] == 1].copy()
    if treated.empty:
        return pd.DataFrame()
    treated["score_decile"] = pd.qcut(
        treated["lead_score"].rank(method="first"), 10, labels=range(10, 0, -1))
    g = (treated.groupby("score_decile", observed=True)
         .agg(leads=("customer_id", "size"), converted=("converted", "sum"),
              conv_rate=("converted", "mean")).sort_index())
    g["lift_vs_avg"] = (g["conv_rate"] / treated["converted"].mean()).round(2)
    return g.reset_index().round(4)


def performance_by_lead_type(leads, contacts) -> pd.DataFrame:
    merged = leads.merge(contacts[["customer_id", "treated", "converted"]],
                         on="customer_id", how="inner")
    treated = merged[merged["treated"] == 1]
    if treated.empty:
        return pd.DataFrame()
    g = (treated.groupby("lead_type")
         .agg(leads=("customer_id", "size"), conv_rate=("converted", "mean"),
              avg_value=("priority_value", "mean"))
         .sort_values("conv_rate", ascending=False))
    return g.round(4).reset_index()


if __name__ == "__main__":
    contacts = pd.read_parquet(C.CONTACTS)
    leads = pd.read_parquet(C.LEADS)
    print("Disposition funnel:\n", disposition_funnel(contacts).to_string())
    print("\nConversion by lead-score decile:\n",
          conversion_by_score(leads, contacts).to_string(index=False))
    print("\nPerformance by lead type:\n",
          performance_by_lead_type(leads, contacts).to_string(index=False))
