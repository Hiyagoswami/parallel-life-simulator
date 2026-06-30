"""
benchmark.py — BLS Consumer Expenditure Survey 2023 income-bracket benchmarking

Compares the user's category spend against real BLS CE national averages,
optionally adjusted for their self-reported income quintile.

DATA SOURCE & HONEST LIMITATIONS — read before using or citing these numbers:

The figures below are taken directly from two published BLS sources:
  1. "Consumer Expenditures in 2023" (BLS Reports, bls.gov/opub/reports/
     consumer-expenditures/2023/) — national average expenditure shares by
     major category (housing 32.9%, transportation 17.0%, food 12.9%,
     healthcare 8.0%, entertainment 4.7%, etc.)
  2. "Consumer Expenditures News Release - 2023" (USDL, bls.gov/news.release/
     archives/cesan_09252024.htm) — income quintile lower bounds for 2023
     ($28,262 / $54,553 / $90,239 / $148,682) and year-over-year dollar
     changes in food spending by quintile.

WHAT THIS MODULE DOES NOT CLAIM:
BLS's full category-by-quintile dollar breakdown (their Table 2: "Quintiles
of income before taxes") was not available in a machine-readable form during
development — it exists only as a published PDF/HTML table on bls.gov, not
as an API or downloadable dataset accessible in this environment. Rather
than fabricate quintile-specific dollar figures for every category (which
would look identical to real data but be invented), this module:
  - Uses only the NATIONAL AVERAGE category shares, which are directly sourced
  - Uses real quintile income bounds to classify the user's bracket
  - States explicitly, in the UI, that category-level benchmarks are
    national averages, NOT adjusted for the user's specific income quintile
  - This is a real limitation, disclosed rather than hidden

This is a smaller, honest claim ("you're spending more than the national
average household on X") rather than a larger, unverifiable one ("you're
spending 40% above the BLS median for your income bracket").
"""

# Real BLS 2023 income quintile lower bounds (Consumer Expenditures News
# Release, archived at bls.gov/news.release/archives/cesan_09252024.htm)
INCOME_QUINTILE_BOUNDS_2023 = {
    "Lowest quintile":  (0, 28_262),
    "Second quintile":  (28_262, 54_553),
    "Third quintile":   (54_553, 90_239),
    "Fourth quintile":  (90_239, 148_682),
    "Highest quintile": (148_682, float("inf")),
}

# Real BLS 2023 national average annual expenditure: $77,280 per consumer unit
# (BLS Reports, "Consumer Expenditures in 2023")
NATIONAL_AVG_ANNUAL_EXPENDITURE_2023 = 77_280

# Real BLS 2023 national average expenditure SHARES by major category
# (BLS Reports, "Consumer Expenditures in 2023" — Table B / Chart 1)
# These are shares of TOTAL household spending (including rent/housing,
# which this app excludes from analysis) — used here only as a reference
# point for the categories this app actually tracks.
NATIONAL_CATEGORY_SHARES_2023 = {
    "Food & Dining":     0.129,   # "Food" major category, 12.9% of total spend
    "Transportation":    0.170,   # 17.0%
    "Health & Wellness": 0.080,   # "Healthcare" major category, 8.0%
    "Entertainment":     0.047,   # 4.7%
    # Shopping, Subscriptions, Utilities don't map cleanly to a single BLS
    # major category (they're split across "Apparel", "Personal care",
    # "Entertainment", and "Housing: utilities" in BLS's classification) —
    # intentionally NOT included here rather than forcing an inaccurate match.
}


def classify_income_quintile(annual_income: float) -> str:
    """Returns which 2023 BLS income quintile a given annual income falls into."""
    for label, (low, high) in INCOME_QUINTILE_BOUNDS_2023.items():
        if low <= annual_income < high:
            return label
    return "Highest quintile"


def national_avg_monthly_spend(category: str) -> float | None:
    """
    Returns the implied national average MONTHLY spend for a category,
    derived from: total annual expenditure × category share / 12.
    Returns None if the category isn't in NATIONAL_CATEGORY_SHARES_2023
    (honest — we don't fabricate a number for categories BLS doesn't map
    cleanly).
    """
    share = NATIONAL_CATEGORY_SHARES_2023.get(category)
    if share is None:
        return None
    return (NATIONAL_AVG_ANNUAL_EXPENDITURE_2023 * share) / 12


def benchmark_category(category: str, user_monthly_spend: float) -> dict | None:
    """
    Compares user's monthly spend in a category against the BLS national
    average. Returns None if no benchmark exists for this category (honest
    null rather than a fabricated comparison).
    """
    national_avg = national_avg_monthly_spend(category)
    if national_avg is None:
        return None

    diff = user_monthly_spend - national_avg
    pct_diff = (diff / national_avg * 100) if national_avg > 0 else 0

    return {
        "category": category,
        "user_monthly": user_monthly_spend,
        "national_avg_monthly": national_avg,
        "diff_dollars": diff,
        "diff_pct": pct_diff,
        "above_average": diff > 0,
    }
