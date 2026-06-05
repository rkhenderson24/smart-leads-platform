# Model card — Smart Leads mortgage lead engine

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
  sets by construction (age); verified in code.
- **Disparate-impact testing:** Adverse Impact Ratio on top-decile lead
  selection. Minimum AIR observed: **0.942**
  (PASSES the 80% rule).
- **Proxy-leakage check:** model features predict the demographic proxy at
  AUC **0.588** (closer to 0.5 is better; high values flag a proxy
  feature for review).
- **Contact governance:** one active solicitation per household, weekly volume
  capped, coordinated with enterprise outreach.

## Limitations
- Synthetic DGP; production requires real outcome data, monitoring, and a
  formal fair-lending review before deployment.
- Uplift assumes unconfounded historical treatment; production needs a
  randomized holdout to keep the causal estimate honest.
