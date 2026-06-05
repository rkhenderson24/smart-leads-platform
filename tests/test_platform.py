"""Smoke + correctness tests. Run with `pytest -q`."""
import numpy as np

from src import features, data_generation, governance
from src.propensity import PropensityModel
from src.uplift import UpliftTLearner, decile_validation
from src.retention import AttritionModel
from src.lead_scoring import build_scored_leads
from src.experiment import required_sample_size, evaluate_ab
from src.monitoring import psi
from src.optimization import optimize_channel_mix


def setup_module(module):
    data_generation.generate_all(verbose=False)
    module.abt = features.build_abt()


def test_abt_and_mortgage_mix():
    assert len(abt) > 0 and "converted" in abt and "relationship_value" in abt
    assert set(abt["mortgage_status"].unique()) <= {"citizens", "competitor", "none"}


def test_age_excluded_from_features():
    # age is ECOA-protected and must never be a model feature
    assert "age" not in features.NUMERIC
    assert "age" not in features.model_matrix(abt).columns


def test_propensity_beats_random():
    assert PropensityModel().fit(abt).metrics["auc"] > 0.55


def test_uplift_is_monotonic():
    u = UpliftTLearner().fit(abt).predict_uplift(abt)
    val = decile_validation(abt, u)
    assert val.iloc[-1]["observed_uplift"] > val.iloc[0]["observed_uplift"]


def test_attrition_model():
    assert AttritionModel().fit(abt).metrics["auc"] > 0.6


def test_lead_scoring_outputs():
    leads = build_scored_leads(abt)
    assert leads["lead_score"].between(0, 100).all()
    assert (leads["priority_value"] >= 0).all()
    assert leads["priority_tier"].min() == 1  # retention tier present


def test_fair_lending_audit_runs():
    leads = build_scored_leads(abt)
    audit = governance.fairness_audit(leads, abt)
    assert 0 <= audit["min_air"] <= 1


def test_power_analysis_monotone():
    assert required_sample_size(0.05, 0.10) > required_sample_size(0.05, 0.30)


def test_ab_detects_real_lift():
    res = evaluate_ab(6000, 300, 6000, 420)
    assert res["significant"] and res["prob_treatment_better"] > 0.95


def test_psi_zero_for_same_dist():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-6


def test_channel_mix_positive():
    mix = optimize_channel_mix(2000)
    assert mix["production"] > 0 and mix["retail_leads"] + mix["virtual_leads"] == 2000
