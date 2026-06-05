"""Experimental design & evaluation.

Maps to JD: 'Develop and implement experimental designs to meet business
objectives.'

Before rolling a new lead strategy to the whole floor, we run a controlled
test. This module provides:
  - power analysis (sample size for a target minimum detectable effect),
  - randomized assignment,
  - frequentist evaluation (two-proportion z-test + confidence interval),
  - a quick Bayesian read (probability the treatment beats control).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def required_sample_size(baseline_rate: float, mde_relative: float,
                         alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm sample size to detect a relative lift `mde_relative` on a
    `baseline_rate` conversion, at given alpha/power."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde_relative)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar))
           + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n = num / (p2 - p1) ** 2
    return int(np.ceil(n))


def evaluate_ab(n_ctrl, conv_ctrl, n_treat, conv_treat) -> dict:
    """Two-proportion z-test + 95% CI on the absolute lift, plus a Bayesian
    posterior probability that treatment > control (Beta-Binomial)."""
    p_c = conv_ctrl / n_ctrl
    p_t = conv_treat / n_treat
    lift_abs = p_t - p_c
    lift_rel = lift_abs / p_c if p_c else np.nan

    p_pool = (conv_ctrl + conv_treat) / (n_ctrl + n_treat)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_ctrl + 1 / n_treat))
    z = lift_abs / se if se else 0.0
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    se_ci = np.sqrt(p_c * (1 - p_c) / n_ctrl + p_t * (1 - p_t) / n_treat)
    ci = (lift_abs - 1.96 * se_ci, lift_abs + 1.96 * se_ci)

    # Bayesian: Beta(1,1) priors, Monte Carlo P(treat > ctrl)
    rng = np.random.default_rng(0)
    sc = rng.beta(1 + conv_ctrl, 1 + n_ctrl - conv_ctrl, 50_000)
    st = rng.beta(1 + conv_treat, 1 + n_treat - conv_treat, 50_000)
    prob_better = float((st > sc).mean())

    return dict(
        control_rate=round(p_c, 4), treat_rate=round(p_t, 4),
        abs_lift=round(lift_abs, 4), rel_lift=round(lift_rel, 4),
        z=round(z, 3), p_value=round(pval, 4),
        ci95=(round(ci[0], 4), round(ci[1], 4)),
        significant=bool(pval < 0.05),
        prob_treatment_better=round(prob_better, 4),
    )


if __name__ == "__main__":
    n = required_sample_size(0.05, 0.15)
    print(f"Per-arm n to detect +15% rel lift on 5% baseline (80% power): {n:,}")
    res = evaluate_ab(n_ctrl=6000, conv_ctrl=300, n_treat=6000, conv_treat=360)
    print("\nA/B evaluation (uplift-ranked leads vs propensity-ranked):")
    for k, v in res.items():
        print(f"  {k:>22}: {v}")
