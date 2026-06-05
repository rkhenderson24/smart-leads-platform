"..."# %% [markdown]
# # Smart Leads — Strategy Analysis
#
# A narrative walkthrough for **non-technical stakeholders**: what the lead
# engine found, and what we recommend the sales floor do about it. Open in
# Jupyter (`jupytext --to notebook analysis/01_strategy_report.py`) or run as
# a script.

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent if (pathlib.Path.cwd().name=="analysis") else pathlib.Path.cwd()))

import pandas as pd
from src import features, segmentation, disposition
from src.propensity import PropensityModel
from src.uplift import UpliftTLearner, decile_validation
from src.lead_scoring import build_scored_leads

pd.options.display.float_format = lambda x: f"{x:,.3f}"

# %% [markdown]
# ## 1. The book of business, segmented
# We group customers into behavioral segments and attach a recommended play.

# %%
abt = features.build_abt()
seg_df, k, sil = segmentation.fit_segments(abt)
profile = segmentation.profile_segments(seg_df)
print(f"{k} segments (silhouette {sil})")
profile[["persona", "customers", "avg_income", "avg_refi_bps",
         "avg_equity", "recommended_play"]]

# %% [markdown]
# ## 2. Who should we actually call?
# Propensity ranks the *likely*. Uplift ranks the *persuadable*. We prioritize
# uplift, because calling a sure-thing wastes officer capacity.

# %%
prop = PropensityModel().fit(abt)
upl = UpliftTLearner().fit(abt)
print("Propensity AUC:", prop.metrics["auc"])
decile_validation(abt, upl.predict_uplift(abt))

# %% [markdown]
# The top uplift bucket converts dramatically better *because of* the call —
# that is where the floor's time should go.

# %% [markdown]
# ## 3. The prioritized lead queue
# Each lead carries a plain-English rationale an officer can act on.

# %%
leads = build_scored_leads(abt, prop, upl)
leads.head(8)[["lead_id", "lead_score", "priority_value", "trigger", "rationale"]]

# %% [markdown]
# ## 4. Is the score working in production?
# We validate against actual call dispositions: higher-scored leads should
# fund at higher rates.

# %%
contacts = pd.read_parquet("data/contacts.parquet") if pathlib.Path("data/contacts.parquet").exists() else None
if contacts is not None:
    print(disposition.conversion_by_score(leads, contacts))

# %% [markdown]
# ## Recommendation
# 1. Route the top-uplift leads to officers daily under capacity constraints.
# 2. Run a controlled test (uplift-ranked vs propensity-ranked) before full
#    rollout — see `src/experiment.py`.
# 3. Monitor PSI weekly; retrain on drift — see `src/monitoring.py`.
