"""Lead Copilot — natural-language next-best-action for loan officers.

Maps to JD: 'Document and communicate analytic insights for both technical and
non-technical audiences' + 'Continuously innovate.'

Given a scored lead, produces a five-second call brief: why this household,
what to say, and the compliance note. Uses the Anthropic Messages API when
ANTHROPIC_API_KEY is set; otherwise a deterministic template keeps the repo
fully functional offline (production-style graceful degradation).
"""
from __future__ import annotations

import os
import textwrap

import pandas as pd

SYSTEM = (
    "You are a retail-bank mortgage sales copilot. Given a structured lead, "
    "write a crisp call brief for a loan officer: (1) one-line why-now, (2) two "
    "talking points tied to the household's banking relationship, (3) one "
    "fair-lending / compliance note. Plain language, under 90 words.")


def _points(lead) -> list:
    t = lead.get("lead_type", "")
    if t.startswith("Retention"):
        return ["Acknowledge the relationship; lead with an in-house refi that "
                "matches or beats the market rate before they shop out.",
                "Reinforce the value of keeping deposits, autopay, and servicing "
                "in one place."]
    if "competitor payoff" in t:
        return ["They already bank with us — position consolidating their "
                "outside mortgage here for a single relationship.",
                "Offer a fast pre-qual using the deposit/payroll history we "
                "already hold."]
    if "purchase" in t:
        return ["Connect to the recent inflow — offer pre-approval and rate-lock.",
                "Bundle with the existing deposit relationship for a rate benefit."]
    if t.startswith("Refi") or "HELOC" in t:
        return ["Lead with the rate gap or tappable equity and the monthly impact.",
                "Frame HELOC as flexible funds against home equity."]
    return ["Open with an annual relationship review.",
            "Listen for refinance, equity, or cross-sell signals."]


def _fallback_brief(lead: dict) -> str:
    up = lead.get("uplift", 0)
    if str(lead.get("lead_type", "")).startswith("Retention"):
        why = (f"At-risk relationship — {lead.get('attrition_risk', 0)*100:.0f}% "
               f"refi-away risk on ~${lead.get('relationship_value', 0):,.0f}/yr.")
    else:
        why = (f"{lead.get('signal', 'Relationship opportunity')} — "
               f"+{up*100:.1f}pp conversion lift if contacted.")
    p = _points(lead)
    return textwrap.dedent(f"""\
        Why now: {why}
        Talking points:
          - {p[0]}
          - {p[1]}
        Compliance: confirm consent to contact; do not quote a rate or terms
        without a documented quote (ECOA/Reg B). Log the disposition after the call.""")


def generate_brief(lead: dict, use_llm: bool = True) -> str:
    if use_llm and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=300,
                system=SYSTEM, messages=[{"role": "user", "content": f"Lead: {lead}"}])
            return "".join(b.text for b in msg.content if b.type == "text")
        except Exception as e:
            return _fallback_brief(lead) + f"\n[copilot fallback: {e.__class__.__name__}]"
    return _fallback_brief(lead)


if __name__ == "__main__":
    from . import config as C
    leads = pd.read_parquet(C.LEADS)
    for _, row in leads.head(2).iterrows():
        d = row.to_dict()
        print(f"=== Lead #{int(d['lead_id'])} · {d['lead_type']} "
              f"(score {d['lead_score']}) ===")
        print(generate_brief(d), "\n")
