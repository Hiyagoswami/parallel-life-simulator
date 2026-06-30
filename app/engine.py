"""
engine.py — Financial simulation and ranking engine

Monte Carlo model:
  - Log-normal monthly returns, Itô correction applied
  - Annual volatility: 15.6% (S&P 500 historical, Damodaran 2024)
  - Annual mean: user-specified (default 7% = S&P 500 30yr nominal avg, Shiller dataset)

Inflation adjustment:
  - Default 3.1% annual CPI (US 20-year average)

run_monte_carlo is cached via @st.cache_data in app.py (not here, to keep
this module Streamlit-independent and unit-testable in isolation).
"""

import numpy as np
from classifier import REDUCIBILITY, action_label

INFLATION_RATE = 0.031
SP500_VOL      = 0.156


def compute_compound(monthly_saving: float, years: float, rate: float) -> float:
    """Deterministic compound growth of a fixed monthly investment."""
    months    = years * 12
    monthly_r = rate / 12
    if monthly_r == 0:
        return monthly_saving * months
    return monthly_saving * ((1 + monthly_r) ** months - 1) / monthly_r


def inflation_adjust(nominal_value: float, years: int,
                     inflation_rate: float = INFLATION_RATE) -> float:
    """Convert a nominal future value to today's purchasing power."""
    return nominal_value / ((1 + inflation_rate) ** years)


def run_monte_carlo(monthly_saving: float, years: int, mean_rate: float,
                    n_simulations: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Simulate n_simulations wealth paths over years * 12 months.
    Returns array of shape (n_simulations, months).
    """
    rng         = np.random.default_rng(seed)
    months      = years * 12
    monthly_mu  = mean_rate / 12
    monthly_sig = SP500_VOL / np.sqrt(12)

    log_returns = rng.normal(
        loc   = monthly_mu - 0.5 * monthly_sig ** 2,
        scale = monthly_sig,
        size  = (n_simulations, months)
    )
    monthly_returns = np.exp(log_returns)

    wealth = np.zeros((n_simulations, months))
    for m in range(months):
        if m == 0:
            wealth[:, m] = monthly_saving * monthly_returns[:, m]
        else:
            wealth[:, m] = (wealth[:, m - 1] + monthly_saving) * monthly_returns[:, m]

    return wealth


def monte_carlo_stats(wealth_matrix: np.ndarray,
                      monthly_saving: float,
                      years: int) -> dict:
    """Compute summary statistics from a Monte Carlo wealth matrix."""
    final             = wealth_matrix[:, -1]
    total_contributed = monthly_saving * years * 12

    return {
        "p10":          float(np.percentile(final, 10)),
        "p25":          float(np.percentile(final, 25)),
        "median":       float(np.percentile(final, 50)),
        "p75":          float(np.percentile(final, 75)),
        "p90":          float(np.percentile(final, 90)),
        "mean":         float(np.mean(final)),
        "std":          float(np.std(final)),
        "prob_positive": float(np.mean(final > 0) * 100),
        "prob_double":   float(np.mean(final > total_contributed * 2) * 100),
        "total_contributed": total_contributed,
    }


def rank_scenarios(by_category: dict, n_months: int, top_n: int = 3) -> list:
    """
    Score each spending category by: avg_monthly_spend × reducibility_weight.
    Categories with reducibility 0 (Fixed/Excluded) are never returned,
    since they can't be a savings lever by definition.
    """
    rows = []
    for cat, total in by_category.items():
        weight = REDUCIBILITY.get(cat, 0.30)
        if weight <= 0:
            continue  # FIX #5 — never recommend cutting rent/payroll/transfers
        monthly = total / n_months
        score   = monthly * weight
        rows.append({
            "category":       cat,
            "monthly_spend":  monthly,
            "reducibility":   weight,
            "score":          score,
            "reduction_pct":  weight,
            "monthly_saving": monthly * weight,
            "action":         action_label(cat, weight),
        })
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_n] if rows else []
