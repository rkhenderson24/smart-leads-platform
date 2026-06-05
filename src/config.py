"""Central configuration for the Smart Leads platform.

Keeping paths, constants, and business assumptions in one place so every
module reads from a single source of truth (and so a reviewer can see the
business logic without spelunking).
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DATA.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

# Generated datasets (parquet for speed, csv mirror for browsability on GitHub)
CUSTOMERS = DATA / "customers.parquet"
LOANS = DATA / "loans.parquet"
OFFICERS = DATA / "loan_officers.parquet"
CONTACTS = DATA / "contacts.parquet"
RATES = DATA / "market_rates.parquet"
LEADS = DATA / "leads_scored.parquet"
ASSIGNMENTS = DATA / "lead_assignments.parquet"

SEED = 42

# ----------------------------------------------------------------------------
# Population assumptions (an enterprise retail-BANK customer base)
# ----------------------------------------------------------------------------
# These are bank customers first — they hold deposits, cards, auto, wealth —
# and a mortgage is one product within the household relationship. Only a
# subset hold a mortgage WITH us; others hold one elsewhere (acquisition) or
# none at all (purchase prospects / cross-sell).
N_CUSTOMERS = 40_000
N_OFFICERS = 60          # ~39 retail, ~21 virtual
N_WEEKS_HISTORY = 52
CHANNELS = ["Retail", "Virtual"]
# Citizens' actual footprint: New England / Mid-Atlantic / Midwest
REGIONS = ["New England", "Mid-Atlantic", "Midwest"]
LOAN_TYPES = ["Conventional", "FHA", "VA", "Jumbo", "ARM"]

# Mortgage relationship mix across the customer base
P_CITIZENS_MORTGAGE = 0.40   # hold a mortgage WITH us  -> retention / refi / HELOC
P_COMPETITOR_MORTGAGE = 0.22 # mortgage elsewhere (ACH detected) -> acquisition
# remainder: deposit-only -> purchase prospect / cross-sell

# Economics used both to value leads and protect relationships
MARKET_RATE_BASELINE = 6.25          # current market 30yr, %
DEPOSIT_NIM = 0.025                  # net interest margin on deposit balances
PRODUCT_ANNUAL_VALUE = 120           # annual contribution per held product
MORTGAGE_SERVICING_MARGIN = 0.005    # annual margin on serviced mortgage balance
NEW_ORIGINATION_VALUE = 9_500        # gain-on-sale + fees on a new origination

# Lead trigger logic
REFI_INCENTIVE_BPS = 50      # note_rate >= market + 0.50% => refi/retention flag
HIGH_EQUITY_LTV = 0.70       # LTV below this => cash-out / HELOC opportunity
TENURE_MILESTONE_MONTHS = 60

# ----------------------------------------------------------------------------
# Lead trigger logic — the "why now" behind every lead
# ----------------------------------------------------------------------------
# A customer becomes a candidate lead when one of these conditions fires.
REFI_INCENTIVE_BPS = 50      # note_rate must be >= market + 0.50% to flag refi
HIGH_EQUITY_LTV = 0.70       # LTV below this => cash-out / HELOC opportunity
TENURE_MILESTONE_MONTHS = 60 # relationship-deepening trigger
