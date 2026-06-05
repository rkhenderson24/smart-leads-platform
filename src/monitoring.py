"""Monitoring, drift detection & data quality control.

Maps to JD: 'Automate various standard procedures including data flows,
monitoring, & quality control.'

Production models silently rot when the incoming population drifts. This
module provides:
  - Population Stability Index (PSI) for feature/score drift,
  - automated data-quality assertions (nulls, ranges, referential integrity),
  - a single health-check entrypoint suitable for a scheduled job / CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate, >0.25 alert."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        return 0.0
    e = np.histogram(expected, bins=cuts)[0] / len(expected)
    a = np.histogram(actual, bins=cuts)[0] / len(actual)
    e = np.clip(e, 1e-4, None)
    a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_report(reference: pd.DataFrame, current: pd.DataFrame,
               cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        val = psi(reference[c].values, current[c].values)
        flag = ("stable" if val < 0.1 else "monitor" if val < 0.25 else "ALERT")
        rows.append(dict(feature=c, psi=round(val, 4), status=flag))
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def data_quality_checks(customers, loans, contacts) -> pd.DataFrame:
    checks = []

    def add(name, ok, detail=""):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))

    add("customers: unique ids", customers["customer_id"].is_unique)
    add("customers: no null credit", customers["credit_score"].notna().all())
    add("customers: credit in [300,850]",
        customers["credit_score"].between(300, 850).all())
    add("loans: ltv in [0,1.2]", loans["ltv"].between(0, 1.2).all(),
        f"max ltv {loans['ltv'].max():.2f}")
    add("loans: balance <= property value",
        (loans["current_balance"] <= loans["property_value"] * 1.01).all())
    add("loans: referential integrity",
        loans["customer_id"].isin(customers["customer_id"]).all())
    add("contacts: funded is binary",
        contacts["converted"].isin([0, 1]).all())
    add("contacts: treated coverage",
        0.2 < contacts["treated"].mean() < 0.5,
        f"treated share {contacts['treated'].mean():.2%}")
    return pd.DataFrame(checks)


def health_check() -> dict:
    customers = pd.read_parquet(C.CUSTOMERS)
    loans = pd.read_parquet(C.LOANS)
    contacts = pd.read_parquet(C.CONTACTS)

    dq = data_quality_checks(customers, loans, contacts)
    # simulate "current week" drift by reweighting toward higher-rate borrowers
    ref = customers.sample(frac=0.5, random_state=1)
    cur = customers.sample(frac=0.5, random_state=2, weights=customers["credit_score"])
    drift = psi_report(ref, cur, ["annual_income", "credit_score",
                                   "relationship_tenure_months", "deposit_balance"])
    return dict(
        all_dq_passed=bool(dq["passed"].all()),
        failed_checks=int((~dq["passed"]).sum()),
        max_psi=float(drift["psi"].max()),
        dq=dq, drift=drift,
    )


if __name__ == "__main__":
    h = health_check()
    print("Data quality:\n", h["dq"].to_string(index=False))
    print("\nDrift (PSI):\n", h["drift"].to_string(index=False))
    print(f"\nAll DQ passed: {h['all_dq_passed']} | max PSI: {h['max_psi']:.4f}")
