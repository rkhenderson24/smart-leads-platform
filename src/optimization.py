"""Lead distribution & channel-mix optimization.

Maps to JD:
  - 'Apply advanced analytics to continually optimize lead distribution across
     Virtual/Retail loan officers.'
  - 'maximize production via provision of an optimal number and mix of leads
     across channels.'

Two optimization problems, both solved as linear programs with PuLP (CBC):

1) DISTRIBUTION: assign the top-priority leads to loan officers to maximize
   total expected production, subject to per-officer weekly capacity, channel
   matching (virtual leads -> virtual LOs), and regional alignment. Each
   officer's effective yield is scaled by their close-rate skill.

2) CHANNEL MIX: given a finite outreach budget (total weekly call capacity),
   choose how many leads to push to Retail vs Virtual to maximize production,
   accounting for diminishing returns per channel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp

from . import config as C


# ---------------------------------------------------------------------------
# 1) Lead -> Officer assignment
# ---------------------------------------------------------------------------
def assign_leads(leads: pd.DataFrame, officers: pd.DataFrame,
                 max_leads: int = 1500) -> pd.DataFrame:
    """Assign the top `max_leads` to officers to maximize expected production.

    Solved as a capacitated assignment LP. Channel + region eligibility prune
    the variable space so it stays tractable.
    """
    L = leads.nlargest(max_leads, "priority_value").reset_index(drop=True)
    O = officers.reset_index(drop=True)

    prob = pulp.LpProblem("lead_distribution", pulp.LpMaximize)

    # eligibility: same channel preference, same region (relationship fit)
    elig = {}
    for li, lead in L.iterrows():
        for oi, off in O.iterrows():
            if off["channel"] == lead["channel_pref"] and off["region"] == lead["region"]:
                elig[(li, oi)] = pulp.LpVariable(f"x_{li}_{oi}", cat="Binary")

    if not elig:
        return pd.DataFrame()

    # objective: sum of priority_value * officer skill
    prob += pulp.lpSum(
        v * float(L.loc[li, "priority_value"]) * float(O.loc[oi, "skill_score"])
        for (li, oi), v in elig.items()
    )

    # each lead assigned at most once
    for li in L.index:
        vs = [v for (l, o), v in elig.items() if l == li]
        if vs:
            prob += pulp.lpSum(vs) <= 1

    # officer capacity
    for oi in O.index:
        vs = [v for (l, o), v in elig.items() if o == oi]
        if vs:
            prob += pulp.lpSum(vs) <= int(O.loc[oi, "weekly_capacity"])

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    rows = []
    for (li, oi), v in elig.items():
        if v.value() and v.value() > 0.5:
            rows.append(dict(
                lead_id=int(L.loc[li, "lead_id"]),
                customer_id=int(L.loc[li, "customer_id"]),
                lo_id=int(O.loc[oi, "lo_id"]),
                lo_name=O.loc[oi, "lo_name"],
                channel=O.loc[oi, "channel"],
                region=O.loc[oi, "region"],
                priority_value=float(L.loc[li, "priority_value"]),
                expected_production=float(L.loc[li, "priority_value"])
                * float(O.loc[oi, "skill_score"]),
            ))
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_parquet(C.ASSIGNMENTS, index=False)
    return out


def assignment_summary(assignments: pd.DataFrame, leads: pd.DataFrame,
                       officers: pd.DataFrame) -> dict:
    n_top = min(len(assignments), len(leads))
    return dict(
        leads_assigned=len(assignments),
        officers_engaged=assignments["lo_id"].nunique(),
        expected_production=round(assignments["expected_production"].sum()),
        avg_capacity_util=round(
            (assignments.groupby("lo_id").size()
             / officers.set_index("lo_id")["weekly_capacity"]).mean(), 3),
    )


# ---------------------------------------------------------------------------
# 2) Channel mix optimization (number & mix of leads across channels)
# ---------------------------------------------------------------------------
def optimize_channel_mix(total_capacity: int,
                         retail_yield: float = 0.052,
                         virtual_yield: float = 0.041,
                         retail_decay: float = 1.2e-5,
                         virtual_decay: float = 0.8e-5,
                         avg_rev: float = C.NEW_ORIGINATION_VALUE
                         ) -> dict:
    """Maximize production = sum_c leads_c * (yield_c - decay_c * leads_c) * rev.

    Diminishing returns (decay) capture lead fatigue: pushing too many leads
    to one channel lowers per-lead conversion. Solved on a grid (the objective
    is concave; a grid is exact enough and transparent for a reviewer).
    """
    best = None
    for retail in range(0, total_capacity + 1, max(1, total_capacity // 200)):
        virtual = total_capacity - retail
        r_conv = max(0.0, retail_yield - retail_decay * retail)
        v_conv = max(0.0, virtual_yield - virtual_decay * virtual)
        production = (retail * r_conv + virtual * v_conv) * avg_rev
        if best is None or production > best["production"]:
            best = dict(retail_leads=retail, virtual_leads=virtual,
                        retail_conv=round(r_conv, 4), virtual_conv=round(v_conv, 4),
                        expected_funded=round(retail * r_conv + virtual * v_conv, 1),
                        production=round(production))
    return best


if __name__ == "__main__":
    leads = pd.read_parquet(C.LEADS)
    officers = pd.read_parquet(C.OFFICERS)
    a = assign_leads(leads, officers, max_leads=1500)
    print("Assignment summary:", assignment_summary(a, leads, officers))
    print("\nChannel mix @ 2400 weekly capacity:")
    print(optimize_channel_mix(2400))
