"""Feature engineering — the analytic base table (ABT) for a bank.

Builds one row per customer joining the deposit relationship, the serviced
mortgage (if any), and outcomes. Engineers the first-party signals and the
RELATIONSHIP VALUE that drive lead generation and scoring.

Fair-lending discipline: `age` is an ECOA-protected basis and is NEVER placed
in a model feature set (see NUMERIC/CATEGORICAL). It is retained on the ABT
only for the fair-lending audit in governance.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_raw() -> dict:
    return dict(
        customers=pd.read_parquet(C.CUSTOMERS),
        mortgages=pd.read_parquet(C.LOANS),
        rates=pd.read_parquet(C.RATES),
        officers=pd.read_parquet(C.OFFICERS),
        contacts=pd.read_parquet(C.CONTACTS),
    )


def build_abt() -> pd.DataFrame:
    raw = load_raw()
    cust, mtg, contacts = raw["customers"], raw["mortgages"], raw["contacts"]
    market_now = raw["rates"]["market_rate_30yr"].iloc[-1]

    df = cust.merge(mtg, on="customer_id", how="left")
    df["has_citizens_mortgage"] = (df["mortgage_status"] == "citizens").astype(int)
    df["has_competitor_mortgage"] = (df["mortgage_status"] == "competitor").astype(int)
    df["deposit_only"] = (df["mortgage_status"] == "none").astype(int)

    # --- engineered features --------------------------------------------------
    df["refi_incentive_bps"] = ((df["note_rate"] - market_now) * 100).clip(lower=0).fillna(0)
    df["equity_pct"] = (1 - df["ltv"]).clip(lower=0).fillna(0)
    df["high_equity"] = (df["equity_pct"] >= (1 - C.HIGH_EQUITY_LTV)).astype(int)
    df["tenure_milestone"] = (df["relationship_tenure_months"] >= C.TENURE_MILESTONE_MONTHS).astype(int)
    df["life_event"] = ((df["recent_move"] == 1) | (df["growing_household"] == 1)).astype(int)
    df["balance_per_income"] = (df["current_balance"].fillna(0) / df["annual_income"]).round(3)
    df["deposit_to_income"] = (df["deposit_balance"] / df["annual_income"]).round(3)

    # --- relationship value (annual $ to the bank) ----------------------------
    # This is the bank's lens: protect/grow the household, not just a loan.
    deposit_value = df["deposit_balance"] * C.DEPOSIT_NIM
    product_value = df["num_products"] * C.PRODUCT_ANNUAL_VALUE
    mortgage_value = df["current_balance"].fillna(0) * C.MORTGAGE_SERVICING_MARGIN
    wealth_value = df["has_wealth"] * 400
    df["relationship_value"] = (deposit_value + product_value + mortgage_value
                                + wealth_value).round(0)

    # --- value at stake per lead type ----------------------------------------
    # Retention protects the whole relationship; acquisition/refi adds product value.
    df["new_business_value"] = (
        C.NEW_ORIGINATION_VALUE
        * (1 + 0.4 * (df["annual_income"] > 200_000))
        * (1 + 0.25 * (df["loan_type"] == "Jumbo").fillna(False))).round(0)

    # --- outcomes -------------------------------------------------------------
    df = df.merge(contacts[["customer_id", "treated", "converted", "attrited",
                            "last_disposition", "n_calls"]],
                  on="customer_id", how="left")
    for col in ["treated", "converted", "attrited"]:
        df[col] = df[col].fillna(0).astype(int)
    return df


# Model features. Deliberately EXCLUDES age (ECOA-protected) and any direct
# protected attribute. First-party bank signals do the heavy lifting.
NUMERIC = [
    "annual_income", "monthly_payroll_inflow", "has_payroll", "credit_score",
    "relationship_tenure_months", "num_products", "deposit_balance",
    "has_wealth", "deposit_to_income", "large_deposit_inflow", "life_event",
    "has_citizens_mortgage", "has_competitor_mortgage", "deposit_only",
    "refi_incentive_bps", "equity_pct", "high_equity", "tenure_milestone",
    "balance_per_income",
]
CATEGORICAL = ["region", "channel_pref", "loan_type"]

# Explicitly forbidden from any model (auditable list)
PROTECTED_EXCLUDED = ["age"]


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC].copy().fillna(0)
    cat = pd.get_dummies(df[CATEGORICAL].astype("object").fillna("NA"),
                         prefix=CATEGORICAL)
    return pd.concat([X, cat], axis=1)
