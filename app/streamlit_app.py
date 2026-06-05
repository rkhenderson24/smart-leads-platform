"""Smart Leads — Streamlit command center (BANK edition).

Run:  streamlit run app/streamlit_app.py
(Run `python -m src.pipeline` first to generate artifacts.)

A manager + loan-officer view over the scored relationship book: KPIs, the
priority queue, segment strategy, uplift validation, retention/attrition,
dispositions, channel-mix optimizer, fair-lending governance, and the GenAI
copilot.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C            # noqa: E402
from src.copilot import generate_brief  # noqa: E402
from src.optimization import optimize_channel_mix  # noqa: E402

st.set_page_config(page_title="Smart Leads Command Center", page_icon="📍",
                   layout="wide")

st.markdown("""
<style>
  .stApp { background: #0b1220; color: #e6edf3; }
  section[data-testid="stSidebar"] { background: #0f1830; }
  .kpi { background: #13203b; border:1px solid #1f3a5f; border-radius:14px; padding:16px 18px; }
  .kpi h3 { margin:0; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase;
            color:#7fa6d6; font-weight:600; }
  .kpi .v { font-size:1.7rem; font-weight:700; color:#fff; margin-top:4px; }
  .kpi .d { font-size:.78rem; color:#5fd0a0; margin-top:2px; }
  h1,h2,h3 { color:#fff; }
  .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:.72rem;
           font-weight:600; background:#1b3a2e; color:#5fd0a0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    return (pd.read_parquet(C.LEADS),
            pd.read_parquet(C.ASSIGNMENTS) if C.ASSIGNMENTS.exists() else pd.DataFrame(),
            pd.read_parquet(C.OFFICERS),
            pd.read_parquet(C.CONTACTS))


try:
    leads, assigns, officers, contacts = load()
except FileNotFoundError:
    st.error("No artifacts found. Run `python -m src.pipeline` first.")
    st.stop()


def kpi(col, title, value, delta=""):
    col.markdown(f"<div class='kpi'><h3>{title}</h3><div class='v'>{value}</div>"
                 f"<div class='d'>{delta}</div></div>", unsafe_allow_html=True)


retention = leads[leads["lead_type"].str.startswith("Retention")]
competitor = leads[leads["lead_type"] == "Acquisition — competitor payoff"]

st.title("📍 Smart Leads — Mortgage Sales Command Center")
st.caption("Enterprise relationship leads: retain at-risk households, win "
           "competitor mortgages, deepen the book. Synthetic data; no real "
           "customer information.")

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Scored Relationships", f"{len(leads):,}")
kpi(c2, "Total Value at Stake", f"${leads['priority_value'].sum()/1e6:,.1f}M",
    "incremental + protected relationship value")
kpi(c3, "At-Risk Relationships", f"{len(retention):,}",
    f"${retention['relationship_value'].sum()/1e6:,.1f}M/yr to protect")
kpi(c4, "Competitor Mortgages Detected", f"{len(competitor):,}",
    "acquisition targets via ACH")

tabs = st.tabs(["🎯 Priority Queue", "🛡️ Retention", "🧭 Segments",
                "📈 Uplift", "📞 Dispositions", "⚙️ Channel Mix",
                "⚖️ Fair Lending", "🤖 Copilot"])

with tabs[0]:
    st.subheader("Priority queue")
    f1, f2, f3 = st.columns([1, 1, 1])
    region = f1.selectbox("Region", ["All"] + sorted(leads["region"].unique()))
    ltype = f2.selectbox("Lead type", ["All"] + sorted(leads["lead_type"].unique()))
    order = f3.selectbox("Order by", ["Value at stake", "Retention first (strategic)"])
    q = leads.copy()
    if region != "All": q = q[q["region"] == region]
    if ltype != "All": q = q[q["lead_type"] == ltype]
    if order.startswith("Retention"):
        q = q.sort_values(["priority_tier", "priority_value"], ascending=[True, False])
    st.dataframe(
        q.head(200)[["lead_id", "lead_score", "priority_value", "lead_type",
                     "signal", "uplift", "relationship_value", "region",
                     "channel_pref", "rationale"]],
        use_container_width=True, height=440)

with tabs[1]:
    st.subheader("Retention — protect at-risk relationships first")
    r1, r2, r3 = st.columns(3)
    kpi(r1, "At-risk relationships", f"{len(retention):,}")
    kpi(r2, "Avg refi-away risk", f"{retention['attrition_risk'].mean()*100:.0f}%")
    kpi(r3, "Relationship value/yr", f"${retention['relationship_value'].sum()/1e6:,.1f}M")
    st.markdown("Highest-value households about to refinance away:")
    st.dataframe(
        retention.nlargest(50, "relationship_value")[
            ["lead_id", "attrition_risk", "relationship_value", "uplift",
             "refi_incentive_bps", "num_products", "region"]],
        use_container_width=True, height=320)

with tabs[2]:
    st.subheader("Segments → recommended play")
    prof = C.REPORTS / "segment_profiles.csv"
    if prof.exists():
        st.dataframe(pd.read_csv(prof), use_container_width=True)
    st.markdown("Value at stake by lead type ($M)")
    st.bar_chart((leads.groupby("lead_type")["priority_value"].sum() / 1e6).round(1))

with tabs[3]:
    st.subheader("Uplift — incrementality, not just likelihood")
    from src.uplift import UpliftTLearner, decile_validation
    from src.features import build_abt
    abt = build_abt()
    u = UpliftTLearner().fit(abt).predict_uplift(abt)
    val = decile_validation(abt, u).set_index("uplift_bucket")
    st.dataframe(val, use_container_width=True)
    st.line_chart(val[["pred_uplift", "observed_uplift"]])

with tabs[4]:
    from src.disposition import disposition_funnel, conversion_by_score
    st.subheader("Call disposition funnel")
    st.bar_chart(disposition_funnel(contacts)["count"])
    conv = conversion_by_score(leads, contacts)
    if not conv.empty:
        st.subheader("Conversion by lead-score decile")
        st.line_chart(conv.set_index("score_decile")["conv_rate"])

with tabs[5]:
    st.subheader("Channel mix optimizer")
    cap = st.slider("Total weekly call capacity", 500, 5000,
                    int(officers["weekly_capacity"].sum()), step=100)
    mix = optimize_channel_mix(cap)
    m1, m2, m3 = st.columns(3)
    kpi(m1, "Retail leads", f"{mix['retail_leads']:,}")
    kpi(m2, "Virtual leads", f"{mix['virtual_leads']:,}")
    kpi(m3, "Expected production", f"${mix['production']:,.0f}",
        f"{mix['expected_funded']:.0f} funded")

with tabs[6]:
    st.subheader("⚖️ Fair-lending & model governance")
    from src.governance import fairness_audit, proxy_leakage
    from src.features import build_abt, PROTECTED_EXCLUDED
    abt = build_abt()
    audit = fairness_audit(leads, abt)
    st.markdown(f"Protected attributes excluded from all models: "
                f"`{', '.join(PROTECTED_EXCLUDED)}` · "
                f"Min Adverse Impact Ratio: **{audit['min_air']}** "
                f"({'passes' if audit['passes_80_rule'] else 'FAILS'} the 80% rule)")
    g1, g2 = st.columns(2)
    g1.markdown("Selection rate by demographic proxy")
    g1.dataframe(audit["by_proxy"], use_container_width=True)
    g2.markdown("Selection rate by age band (age not a feature)")
    g2.dataframe(audit["by_age"], use_container_width=True)
    st.caption("Disparate-impact testing on top-decile lead selection (ECOA / "
               "Reg B / SR 11-7). Full model card in reports/MODEL_CARD.md.")

with tabs[7]:
    st.subheader("🤖 GenAI call brief")
    st.caption("Uses Claude when ANTHROPIC_API_KEY is set; deterministic fallback otherwise.")
    lid = st.selectbox("Lead", leads["lead_id"].head(50))
    lead = leads[leads["lead_id"] == lid].iloc[0].to_dict()
    st.markdown(f"<span class='badge'>{lead['lead_type']}</span> "
                f"score {lead['lead_score']} · {lead['signal']}",
                unsafe_allow_html=True)
    if st.button("Generate call brief"):
        st.code(generate_brief(lead), language="text")
