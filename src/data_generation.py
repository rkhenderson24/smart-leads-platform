"""Synthetic data generation for the Smart Leads platform (BANK edition).

Produces a realistic-but-fake ENTERPRISE bank customer base. Every record is
a banking customer first (deposits, cards, auto, wealth); a mortgage is one
product in the household relationship. The mortgage relationship splits three
ways:

  * Citizens mortgage holders  -> retention / refi / HELOC plays
  * Competitor mortgage holders (detected via ACH to an outside servicer)
                               -> acquisition plays (win the mortgage)
  * Deposit-only customers     -> purchase prospects / cross-sell

The platform's edge is FIRST-PARTY DATA a monoline lender cannot see:
payroll inflow (income), large deposit inflow (home-sale proceeds / down
payment), competitor-mortgage detection, and held-product depth. These drive
both lead generation and a KNOWN heterogeneous treatment effect on conversion,
so the uplift model has real ground truth to recover.

We also generate attrition for Citizens mortgage holders (refi-away risk) so
the retention model has a real target. No real customer data is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _rng() -> np.random.Generator:
    return np.random.default_rng(C.SEED)


def make_customers(rng) -> pd.DataFrame:
    n = C.N_CUSTOMERS

    # NOTE: age is an ECOA-protected basis. It is generated for fair-lending
    # auditing only and is intentionally EXCLUDED from every model feature set
    # (see features.py). Keep that discipline visible.
    age = np.clip(rng.normal(46, 13, n), 22, 85).round().astype(int)

    annual_income = np.clip(rng.lognormal(11.3, 0.45, n), 28_000, 750_000).round(-2)
    has_payroll = rng.random(n) < 0.74                     # direct deposit on file
    monthly_payroll_inflow = np.where(
        has_payroll, (annual_income / 12 * rng.uniform(0.85, 1.0, n)).round(-1), 0)

    credit_score = np.clip(rng.normal(735, 52, n), 560, 830).round().astype(int)
    relationship_tenure_months = np.clip(rng.exponential(60, n), 1, 420).round().astype(int)
    num_products = rng.choice([1, 2, 3, 4, 5, 6], n,
                              p=[0.20, 0.27, 0.23, 0.16, 0.09, 0.05])
    deposit_balance = np.clip(rng.lognormal(9.6, 1.1, n), 100, 1_200_000).round(-2)
    has_wealth = (deposit_balance > 250_000) | (rng.random(n) < 0.06)

    region = rng.choice(C.REGIONS, n, p=[0.34, 0.40, 0.26])
    channel_pref = rng.choice(C.CHANNELS, n, p=[0.62, 0.38])

    # Audit-only synthetic proxy for the demographic composition of a customer's
    # area (e.g. a majority-minority census tract flag). Used EXCLUSIVELY by the
    # fair-lending audit in governance.py to test for disparate impact. It is
    # NEVER a model feature. Mild income correlation makes the audit non-trivial.
    p_grp_b = np.clip(0.42 - (annual_income - annual_income.mean())
                      / annual_income.std() * 0.08, 0.1, 0.8)
    area_demographic_proxy = np.where(rng.random(n) < p_grp_b, "Group B", "Group A")

    # First-party intent signals (the data moat)
    large_deposit_inflow = rng.random(n) < 0.07            # home sale / DP saved
    recent_move = rng.random(n) < 0.08
    growing_household = rng.random(n) < 0.10

    # Mortgage relationship status
    u = rng.random(n)
    mortgage_status = np.where(
        u < C.P_CITIZENS_MORTGAGE, "citizens",
        np.where(u < C.P_CITIZENS_MORTGAGE + C.P_COMPETITOR_MORTGAGE,
                 "competitor", "none"))

    return pd.DataFrame(dict(
        customer_id=np.arange(1, n + 1),
        age=age,
        annual_income=annual_income.astype(int),
        monthly_payroll_inflow=monthly_payroll_inflow.astype(int),
        has_payroll=has_payroll.astype(int),
        credit_score=credit_score,
        relationship_tenure_months=relationship_tenure_months,
        num_products=num_products,
        deposit_balance=deposit_balance.astype(int),
        has_wealth=has_wealth.astype(int),
        region=region,
        channel_pref=channel_pref,
        area_demographic_proxy=area_demographic_proxy,
        large_deposit_inflow=large_deposit_inflow.astype(int),
        recent_move=recent_move.astype(int),
        growing_household=growing_household.astype(int),
        mortgage_status=mortgage_status,
    ))


def make_mortgages(rng, customers) -> pd.DataFrame:
    """Detail rows for the mortgages we SERVICE (Citizens holders only)."""
    c = customers[customers["mortgage_status"] == "citizens"].copy()
    n = len(c)
    property_value = np.clip(c["annual_income"].values * rng.uniform(2.5, 5.5, n),
                             90_000, 3_000_000).round(-3)
    orig_amount = (property_value * rng.uniform(0.55, 0.97, n)).round(-3)
    age_of_loan = rng.integers(1, 84, n)
    note_rate = np.clip(rng.normal(5.4, 1.3, n), 2.4, 9.0).round(3)
    loan_type = rng.choice(C.LOAN_TYPES, n, p=[0.55, 0.18, 0.12, 0.08, 0.07])

    mr = note_rate / 100 / 12
    pmt = orig_amount * mr / (1 - (1 + mr) ** -360)
    k = age_of_loan
    cur = (orig_amount * (1 + mr) ** k - pmt * ((1 + mr) ** k - 1) / mr)
    cur = np.clip(cur, 0, orig_amount).round(-2)
    ltv = (cur / property_value).round(3)

    return pd.DataFrame(dict(
        customer_id=c["customer_id"].values,
        property_value=property_value.astype(int),
        orig_amount=orig_amount.astype(int),
        current_balance=cur.astype(int),
        note_rate=note_rate,
        ltv=ltv,
        loan_type=loan_type,
        age_of_loan_months=age_of_loan,
    ))


def make_market_rates(rng) -> pd.DataFrame:
    weeks = pd.date_range("2025-01-06", periods=C.N_WEEKS_HISTORY, freq="W-MON")
    walk = np.cumsum(rng.normal(0, 0.04, len(weeks)))
    rate = np.clip(C.MARKET_RATE_BASELINE + walk - walk.mean(), 5.0, 7.5).round(3)
    return pd.DataFrame(dict(week=weeks, market_rate_30yr=rate))


def make_officers(rng) -> pd.DataFrame:
    n = C.N_OFFICERS
    channel = np.where(np.arange(n) < int(n * 0.66), "Retail", "Virtual")
    region = rng.choice(C.REGIONS, n)
    capacity = np.where(channel == "Virtual",
                        rng.integers(45, 70, n), rng.integers(25, 45, n))
    skill = np.clip(rng.normal(0.5, 0.16, n), 0.1, 0.95).round(3)
    return pd.DataFrame(dict(
        lo_id=np.arange(1, n + 1),
        lo_name=[f"LO_{i:03d}" for i in range(1, n + 1)],
        channel=channel, region=region,
        weekly_capacity=capacity, skill_score=skill))


def _attrition_risk(df, market_now) -> np.ndarray:
    """True refi-away / payoff risk for Citizens mortgage holders, absent any
    retention outreach. Rises with a positive note-rate-minus-market gap (they
    can save by refinancing — possibly with a competitor) and falls with
    relationship depth (more products = stickier household)."""
    gap = (df["note_rate"].fillna(0) - market_now).clip(lower=0)
    risk = (0.04
            + 0.22 * np.tanh(gap / 0.75)
            + 0.05 * (df["num_products"] <= 1)
            - 0.03 * (df["num_products"] >= 4)
            + 0.04 * (df["has_payroll"] == 0))
    risk = np.where(df["mortgage_status"] == "citizens", risk, 0.0)
    return np.clip(risk, 0, 0.6)


def _true_uplift(df, market_now) -> np.ndarray:
    """Ground-truth incremental P(favorable conversion) from outreach.

    Heterogeneous by design and tied to first-party signals:
      * competitor-mortgage + payroll + good credit  -> high (win their loan)
      * deposit-only + large inflow / life event      -> high (purchase)
      * Citizens holder with refi incentive           -> retention save
      * high equity                                    -> HELOC
    Already-deep relationships and no-signal customers barely move.
    """
    cit = df["mortgage_status"] == "citizens"
    comp = df["mortgage_status"] == "competitor"
    none = df["mortgage_status"] == "none"
    gap = (df["note_rate"].fillna(0) - market_now).clip(lower=0)
    equity = (1 - df["ltv"]).clip(lower=0).fillna(0)

    u = np.full(len(df), 0.02)
    # acquisition: take a competitor's mortgage
    u += np.where(comp, 0.14 + 0.05 * (df["has_payroll"] == 1)
                  + 0.04 * (df["credit_score"] > 700), 0)
    # purchase acquisition from deposit-only base
    u += np.where(none, 0.05 + 0.12 * df["large_deposit_inflow"]
                  + 0.06 * df["growing_household"], 0)
    # retention / refi-in-house for our own at-risk holders
    u += np.where(cit, 0.16 * np.tanh(gap / 0.75) + 0.06 * equity, 0)
    # deeper relationships are harder to move incrementally
    u -= 0.03 * (df["num_products"] >= 5)
    return np.clip(u, 0, 0.5)


def make_contacts(rng, customers, mortgages, rates) -> pd.DataFrame:
    """Vectorized one-year outreach simulation governed by the DGP above.

    `converted` = favorable relationship outcome: an origination (acquisition /
    refi / HELOC) OR a retained-and-refinanced-in-house save for at-risk
    holders. `attrited` is the natural loss event for Citizens holders absent a
    successful retention contact.
    """
    market_now = rates["market_rate_30yr"].iloc[-1]
    df = customers.merge(mortgages[["customer_id", "note_rate", "ltv"]],
                         on="customer_id", how="left")

    attrition_risk = _attrition_risk(df, market_now)
    uplift = _true_uplift(df, market_now)
    n = len(df)

    treated = rng.random(n) < 0.35
    baseline = 0.015 + 0.01 * (df["num_products"].values >= 2)
    p_conv = baseline + np.where(treated, uplift, 0.0)
    converted = rng.random(n) < p_conv

    # Natural attrition for Citizens holders; a successful retention contact
    # (treated AND converted) keeps them in-house.
    cit = (df["mortgage_status"] == "citizens").values
    attrite_p = np.where(cit, attrition_risk, 0.0)
    attrited = (rng.random(n) < attrite_p) & ~(treated & converted) & cit

    # disposition labels
    DISPO_TREATED = ["No Answer", "Callback Requested", "Not Interested",
                     "Application Started"]
    dispo = np.array(["Not Contacted"] * n, dtype=object)
    rnd = rng.random(n)
    tmask = treated & ~converted
    dispo[tmask] = np.select(
        [rnd[tmask] < 0.34, rnd[tmask] < 0.54, rnd[tmask] < 0.84],
        ["No Answer", "Callback Requested", "Not Interested"],
        default="Application Started")
    dispo[treated & converted & cit] = "Retained"
    dispo[treated & converted & ~cit] = "Funded"

    n_calls = np.where(treated, rng.integers(1, 5, n), 0)

    return pd.DataFrame(dict(
        customer_id=df["customer_id"].values,
        treated=treated.astype(int),
        n_calls=n_calls,
        last_disposition=dispo,
        converted=converted.astype(int),
        attrited=attrited.astype(int),
        channel=df["channel_pref"].values,
        contact_week=np.where(treated, rng.integers(1, C.N_WEEKS_HISTORY, n), 0),
    ))


def generate_all(verbose: bool = True) -> dict:
    rng = _rng()
    customers = make_customers(rng)
    mortgages = make_mortgages(rng, customers)
    rates = make_market_rates(rng)
    officers = make_officers(rng)
    contacts = make_contacts(rng, customers, mortgages, rates)

    customers.to_parquet(C.CUSTOMERS, index=False)
    mortgages.to_parquet(C.LOANS, index=False)
    rates.to_parquet(C.RATES, index=False)
    officers.to_parquet(C.OFFICERS, index=False)
    contacts.to_parquet(C.CONTACTS, index=False)
    customers.head(500).to_csv(C.DATA / "sample_customers.csv", index=False)
    mortgages.head(500).to_csv(C.DATA / "sample_mortgages.csv", index=False)

    if verbose:
        print(f"customers : {len(customers):>7,}")
        ms = customers["mortgage_status"].value_counts()
        for k in ["citizens", "competitor", "none"]:
            print(f"   {k:<11}: {ms.get(k, 0):>7,}")
        print(f"mortgages : {len(mortgages):>7,}  (serviced by us)")
        print(f"officers  : {len(officers):>7,}")
        tr, ct = contacts[contacts.treated == 1], contacts[contacts.treated == 0]
        print(f"conversion | treated {tr.converted.mean():.3%} | "
              f"control {ct.converted.mean():.3%} | "
              f"naive ATE {tr.converted.mean()-ct.converted.mean():+.3%}")
        cit = customers["mortgage_status"] == "citizens"
        print(f"attrition  | citizens-holder rate "
              f"{contacts.loc[cit.values, 'attrited'].mean():.3%}")
    return dict(customers=customers, mortgages=mortgages, rates=rates,
                officers=officers, contacts=contacts)


if __name__ == "__main__":
    generate_all()
