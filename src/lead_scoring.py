"""Lead scoring & prioritization — the bank's coordinated lead engine.

A lead's priority is expressed in DOLLARS AT STAKE for the household
relationship, not a per-loan fee. Two economic frames:

  Retention  (our at-risk mortgage holders):
     priority = attrition_risk x P(save|contact) x relationship_value x HORIZON
     -> protecting a high-value, multi-year household relationship.

  Acquisition / refi / HELOC / cross-sell:
     priority = uplift x (new_business_value + deepened relationship)
     -> incremental product economics from a timely, warm, first-party lead.

Retention naturally floats high because losing a relationship loses years of
value, which is exactly how a bank prioritizes vs a monoline lender.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from .propensity import PropensityModel
from .uplift import UpliftTLearner
from .retention import AttritionModel

RETENTION_HORIZON_YEARS = 5     # relationship value protected by a save
DEEPEN_HORIZON_YEARS = 2        # relationship value added by new business
ATTRITION_FLAG = 0.25           # risk above which a holder is a retention lead


def _lead_type(r) -> str:
    if r["has_citizens_mortgage"] and r["attrition_risk"] >= ATTRITION_FLAG:
        return "Retention — at-risk relationship"
    if r["has_competitor_mortgage"]:
        return "Acquisition — competitor payoff"
    if r["deposit_only"] and (r["large_deposit_inflow"] or r["life_event"]):
        return "Acquisition — purchase intent"
    if r["has_citizens_mortgage"] and (r["refi_incentive_bps"] >= C.REFI_INCENTIVE_BPS
                                       or r["high_equity"]):
        return "Refi / HELOC"
    if r["deposit_only"]:
        return "Cross-sell / deepen"
    return "Portfolio review"


def _signal(r) -> str:
    if r["has_competitor_mortgage"]:
        return "Competitor mortgage detected via ACH"
    if r["large_deposit_inflow"]:
        return "Large deposit inflow (home-sale / down payment)"
    if r["refi_incentive_bps"] >= C.REFI_INCENTIVE_BPS:
        return f"Rate gap {r['refi_incentive_bps']:.0f} bps vs market"
    if r["high_equity"]:
        return "Tappable home equity"
    if r["life_event"]:
        return "Life event (move / household growth)"
    if r["monthly_payroll_inflow"] > 0:
        return "Active payroll relationship"
    return "Relationship review"


def _timing(r) -> float:
    m = 1.0
    if r["refi_incentive_bps"] >= 75:
        m += 0.5
    if r["large_deposit_inflow"] or r["life_event"]:
        m += 0.3
    return m


def build_scored_leads(df=None, propensity=None, uplift=None, attrition=None):
    if df is None:
        df = F.build_abt()
    propensity = propensity or PropensityModel().fit(df)
    uplift = uplift or UpliftTLearner().fit(df)
    attrition = attrition or AttritionModel().fit(df)

    out = df.copy()
    out["p_convert"] = propensity.score(out)
    out["uplift"] = np.clip(uplift.predict_uplift(out), 0, None)
    out["attrition_risk"] = attrition.attrition_risk(out)
    out["lead_type"] = out.apply(_lead_type, axis=1)
    out["signal"] = out.apply(_signal, axis=1)
    out["timing_mult"] = out.apply(_timing, axis=1)

    is_retention = out["lead_type"].str.startswith("Retention")
    retention_val = (out["attrition_risk"] * out["uplift"]
                     * out["relationship_value"] * RETENTION_HORIZON_YEARS)
    acquisition_val = (out["uplift"]
                       * (out["new_business_value"].fillna(C.NEW_ORIGINATION_VALUE)
                          + out["relationship_value"] * DEEPEN_HORIZON_YEARS))
    out["value_at_stake"] = np.where(is_retention, retention_val, acquisition_val)
    out["priority_value"] = (out["value_at_stake"] * out["timing_mult"]).round(0)

    out["lead_score"] = (out["priority_value"].rank(pct=True) * 100).round(1)
    # Strategic ordering (bank prioritizes protecting relationships first)
    tier_map = {"Retention — at-risk relationship": 1,
                "Acquisition — competitor payoff": 2,
                "Acquisition — purchase intent": 2,
                "Refi / HELOC": 3, "Cross-sell / deepen": 4,
                "Portfolio review": 5}
    out["priority_tier"] = out["lead_type"].map(tier_map)
    out["rationale"] = out.apply(_rationale, axis=1)

    cols = ["customer_id", "region", "channel_pref", "lead_score",
            "priority_value", "priority_tier", "value_at_stake", "uplift",
            "attrition_risk", "p_convert", "relationship_value",
            "new_business_value", "lead_type", "signal", "timing_mult",
            "rationale", "mortgage_status", "refi_incentive_bps", "equity_pct",
            "credit_score", "annual_income", "num_products"]
    leads = out[cols].sort_values("priority_value", ascending=False).reset_index(drop=True)
    leads["lead_id"] = np.arange(1, len(leads) + 1)
    leads.to_parquet(C.LEADS, index=False)
    return leads


def _rationale(r) -> str:
    if r["lead_type"].startswith("Retention"):
        return (f"{r['signal']}. {r['attrition_risk']*100:.0f}% refi-away risk on a "
                f"~${r['relationship_value']:,.0f}/yr relationship; "
                f"+{r['uplift']*100:.1f}pp save rate if contacted.")
    return (f"{r['signal']}. +{r['uplift']*100:.1f}pp conversion lift if contacted; "
            f"~${r['new_business_value'] if pd.notna(r['new_business_value']) else C.NEW_ORIGINATION_VALUE:,.0f} "
            f"new business + relationship deepening.")


if __name__ == "__main__":
    leads = build_scored_leads()
    print(f"Scored {len(leads):,} leads\n")
    print("Lead volume by type:\n", leads["lead_type"].value_counts().to_string())
    print("\nValue at stake by type ($M):")
    print((leads.groupby("lead_type")["priority_value"].sum() / 1e6).round(1).to_string())
    print("\nTop 8 priority leads:")
    print(leads.head(8)[["lead_id", "lead_score", "priority_value", "lead_type",
                         "signal"]].to_string(index=False))
