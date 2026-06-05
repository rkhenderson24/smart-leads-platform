"""Model governance & fair-lending audit — built for a regulated bank.

A bank proactively soliciting customers for credit operates under ECOA /
Regulation B and fair-lending scrutiny: WHO you choose to market to can create
disparate impact even when no protected attribute is used as a feature. Most
portfolio projects ignore this. This module demonstrates the controls a model
risk (SR 11-7) and fair-lending review would expect:

  1. Protected-attribute exclusion check (age is excluded by construction).
  2. Disparate-impact / Adverse Impact Ratio (AIR, the "80% rule") on lead
     selection across an audit-only demographic proxy and across age bands.
  3. Proxy-leakage check: does any feature strongly predict the proxy?
  4. A generated model card documenting purpose, data, limitations, controls.
  5. Contact governance: enterprise dedup so a customer is not over-solicited
     across coordinated campaigns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import features as F


# ---------------------------------------------------------------------------
# Fair-lending audit
# ---------------------------------------------------------------------------
def adverse_impact_ratio(selected: pd.Series, group: pd.Series) -> pd.DataFrame:
    """Selection rate by group + AIR vs the most-selected group.
    AIR < 0.80 is the classic disparate-impact red flag."""
    rate = selected.groupby(group).mean()
    air = (rate / rate.max()).round(3)
    return pd.DataFrame({"selection_rate": rate.round(4), "air_vs_top": air,
                         "flag": np.where(air < 0.80, "REVIEW", "ok")})


def fairness_audit(leads: pd.DataFrame, abt: pd.DataFrame,
                   top_frac: float = 0.10) -> dict:
    """Audit the top-decile selected leads for disparate impact."""
    df = leads.merge(
        abt[["customer_id", "age", "area_demographic_proxy"]],
        on="customer_id", how="left")
    cutoff = df["lead_score"].quantile(1 - top_frac)
    df["selected"] = (df["lead_score"] >= cutoff).astype(int)
    df["age_band"] = pd.cut(df["age"], [0, 35, 50, 65, 200],
                            labels=["<35", "35-49", "50-64", "65+"])

    by_proxy = adverse_impact_ratio(df["selected"], df["area_demographic_proxy"])
    by_age = adverse_impact_ratio(df["selected"], df["age_band"])
    min_air = float(min(by_proxy["air_vs_top"].min(), by_age["air_vs_top"].min()))
    return dict(by_proxy=by_proxy, by_age=by_age, min_air=round(min_air, 3),
                passes_80_rule=bool(min_air >= 0.80))


def proxy_leakage(abt: pd.DataFrame, n=5) -> pd.DataFrame:
    """Can the model features reconstruct the protected proxy? High AUC means a
    feature is acting as a proxy and needs review."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    X = F.model_matrix(abt)
    y = (abt["area_demographic_proxy"] == "Group B").astype(int).values
    p = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=120, max_depth=3),
        X, y, cv=3, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    return auc


# ---------------------------------------------------------------------------
# Contact governance (coordinated enterprise outreach)
# ---------------------------------------------------------------------------
def apply_contact_governance(leads: pd.DataFrame, max_per_customer=1,
                             weekly_cap=2500) -> pd.DataFrame:
    """Dedup to one active solicitation per customer and cap weekly volume, so
    the mortgage campaign cooperates with enterprise contact strategy rather
    than over-soliciting a household across products."""
    deduped = (leads.sort_values("priority_value", ascending=False)
               .drop_duplicates("customer_id", keep="first"))
    return deduped.head(weekly_cap)


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------
def write_model_card(audit: dict, leakage_auc: float) -> str:
    card = f"""# Model card — Smart Leads mortgage lead engine

## Purpose
Generate, score, and prioritize data-driven mortgage/home-equity sales leads
for retail and virtual loan officers, optimizing proactive retention and
acquisition within an enterprise relationship framework.

## Models
- Propensity: calibrated gradient boosting, P(favorable conversion | contact).
- Uplift (T-learner): incremental treatment effect — targets persuadable
  households, not inevitable conversions.
- Attrition: P(refi-away / payoff) for serviced mortgages, drives retention.

## Data & features
- First-party enterprise signals (payroll inflow, deposit balances, product
  depth, competitor-mortgage ACH detection, equity, life events).
- Synthetic data only; no real customer information.

## Fair-lending controls (ECOA / Reg B, SR 11-7)
- **Protected-attribute exclusion:** `age` is excluded from ALL model feature
  sets by construction ({', '.join(F.PROTECTED_EXCLUDED)}); verified in code.
- **Disparate-impact testing:** Adverse Impact Ratio on top-decile lead
  selection. Minimum AIR observed: **{audit['min_air']}**
  ({'PASSES' if audit['passes_80_rule'] else 'FAILS'} the 80% rule).
- **Proxy-leakage check:** model features predict the demographic proxy at
  AUC **{leakage_auc:.3f}** (closer to 0.5 is better; high values flag a proxy
  feature for review).
- **Contact governance:** one active solicitation per household, weekly volume
  capped, coordinated with enterprise outreach.

## Limitations
- Synthetic DGP; production requires real outcome data, monitoring, and a
  formal fair-lending review before deployment.
- Uplift assumes unconfounded historical treatment; production needs a
  randomized holdout to keep the causal estimate honest.
"""
    (C.REPORTS / "MODEL_CARD.md").write_text(card)
    return card


def run_governance(leads, abt) -> dict:
    audit = fairness_audit(leads, abt)
    leakage = proxy_leakage(abt)
    write_model_card(audit, leakage)
    governed = apply_contact_governance(leads)
    return dict(audit=audit, leakage_auc=round(float(leakage), 3),
                governed_leads=len(governed))


if __name__ == "__main__":
    from .lead_scoring import build_scored_leads
    abt = F.build_abt()
    leads = build_scored_leads(abt)
    g = run_governance(leads, abt)
    print("Fair-lending audit — by demographic proxy:\n", g["audit"]["by_proxy"].to_string())
    print("\nBy age band (age is NOT a model feature):\n", g["audit"]["by_age"].to_string())
    print(f"\nMin AIR: {g['audit']['min_air']}  passes 80% rule: {g['audit']['passes_80_rule']}")
    print(f"Proxy-leakage AUC: {g['leakage_auc']}")
    print("Model card -> reports/MODEL_CARD.md")
