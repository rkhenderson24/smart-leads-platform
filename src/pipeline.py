"""End-to-end pipeline — one command runs the whole Smart Leads platform.

    python -m src.pipeline

Stages: generate enterprise data -> features -> segment -> propensity ->
uplift (causal) -> attrition (retention) -> score & prioritize -> optimize
distribution & channel mix -> disposition validation -> fair-lending
governance + health checks. Writes artifacts to data/ and reports/.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from . import config as C
from . import data_generation, features, segmentation, disposition, monitoring, governance
from .propensity import PropensityModel
from .uplift import UpliftTLearner, decile_validation
from .retention import AttritionModel
from .lead_scoring import build_scored_leads
from .optimization import assign_leads, assignment_summary, optimize_channel_mix


def run(verbose: bool = True) -> dict:
    t0 = time.time()
    log = (print if verbose else (lambda *a, **k: None))

    log("\n[1/9] Generating enterprise bank customer base...")
    data_generation.generate_all(verbose=verbose)

    log("\n[2/9] Building analytic base table...")
    abt = features.build_abt()

    log("\n[3/9] Customer segmentation...")
    seg_df, k, sil = segmentation.fit_segments(abt)
    segmentation.profile_segments(seg_df).to_csv(C.REPORTS / "segment_profiles.csv")
    log(f"   -> {k} segments (silhouette {sil})")

    log("\n[4/9] Propensity model...")
    prop = PropensityModel().fit(abt)
    log(f"   -> AUC {prop.metrics['auc']}")

    log("\n[5/9] Uplift (causal) model...")
    upl = UpliftTLearner().fit(abt)
    val = decile_validation(abt, upl.predict_uplift(abt))
    log(f"   -> observed uplift {val.iloc[0]['observed_uplift']:.3f} (low) "
        f"-> {val.iloc[-1]['observed_uplift']:.3f} (high)")

    log("\n[6/9] Attrition / retention model...")
    attr = AttritionModel().fit(abt)
    log(f"   -> AUC {attr.metrics['auc']}  base attrition {attr.metrics['base_attrition']}")

    log("\n[7/9] Scoring & prioritizing leads...")
    leads = build_scored_leads(abt, prop, upl, attr)
    by_type = leads["lead_type"].value_counts().to_dict()
    log(f"   -> {len(leads):,} leads; "
        f"{by_type.get('Retention — at-risk relationship', 0)} retention, "
        f"{by_type.get('Acquisition — competitor payoff', 0)} competitor-payoff")

    log("\n[8/9] Optimizing distribution & channel mix...")
    officers = pd.read_parquet(C.OFFICERS)
    assigns = assign_leads(leads, officers, max_leads=1500)
    asum = assignment_summary(assigns, leads, officers)
    mix = optimize_channel_mix(int(officers["weekly_capacity"].sum()))
    log(f"   -> {asum['leads_assigned']} leads -> {asum['officers_engaged']} officers; "
        f"expected production ${asum['expected_production']:,.0f}")

    log("\n[9/9] Disposition validation + fair-lending governance...")
    contacts = pd.read_parquet(C.CONTACTS)
    conv = disposition.conversion_by_score(leads, contacts)
    gov = governance.run_governance(leads, abt)
    health = monitoring.health_check()
    log(f"   -> min AIR {gov['audit']['min_air']} "
        f"(80% rule: {gov['audit']['passes_80_rule']}); "
        f"DQ passed {health['all_dq_passed']}; max PSI {health['max_psi']:.3f}")

    summary = dict(
        runtime_sec=round(time.time() - t0, 1),
        n_customers=int(len(abt)),
        mortgage_mix=abt["mortgage_status"].value_counts().to_dict(),
        n_leads=int(len(leads)),
        leads_by_type=by_type,
        total_value_at_stake_m=round(float(leads["priority_value"].sum() / 1e6), 1),
        propensity=prop.metrics,
        uplift_top_bucket=float(val.iloc[-1]["observed_uplift"]),
        attrition=attr.metrics,
        assignment=asum,
        channel_mix=mix,
        fair_lending=dict(min_air=gov["audit"]["min_air"],
                          passes_80_rule=gov["audit"]["passes_80_rule"],
                          proxy_leakage_auc=gov["leakage_auc"]),
        data_quality_passed=bool(health["all_dq_passed"]),
        max_psi=round(float(health["max_psi"]), 4),
    )
    (C.REPORTS / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if verbose:
        log("\n" + "=" * 60 + "\nRUN SUMMARY\n" + "=" * 60)
        log(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
