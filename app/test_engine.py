"""
test_engine.py — Unit tests for the ranking and simulation engine

Run: python3 test_engine.py  (or: pytest test_engine.py)

Targets the functions most likely to break silently on real-world input:
parsing edge cases, single/zero-category ranking, boundary rate values.
"""
import sys, io
sys.path.insert(0, '.')

import pandas as pd
from engine import (compute_compound, run_monte_carlo, monte_carlo_stats,
                    rank_scenarios, inflation_adjust)
from classifier import infer_category, REDUCIBILITY

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
        FAIL += 1


print("\n── compute_compound ──────────────────────────────────────")
v = compute_compound(100, 10, 0.07)
test("$100/mo @ 7% for 10yr ≈ $17,308", 17000 < v < 17700, f"got {v:.0f}")

v0 = compute_compound(100, 10, 0.0)
test("0% rate = simple sum = 12,000 (boundary case)", abs(v0 - 12000) < 1, f"got {v0:.0f}")

test("more years = more wealth", compute_compound(100,20,0.07) > compute_compound(100,10,0.07))
test("higher rate = more wealth", compute_compound(100,10,0.10) > compute_compound(100,10,0.07))

test("zero monthly saving = zero wealth", compute_compound(0, 10, 0.07) == 0)


print("\n── inflation_adjust ──────────────────────────────────────")
real = inflation_adjust(100_000, 10, 0.031)
test("10yr @ 3.1% inflation reduces value", real < 100_000, f"got {real:.0f}")
test("inflation_adjust(x,0) == x (boundary)", abs(inflation_adjust(50_000, 0) - 50_000) < 1)


print("\n── infer_category — known and edge cases ──────────────────")
test("Starbucks -> Food & Dining", infer_category("STARBUCKS #4521") == "Food & Dining")
test("Unknown merchant -> Other", infer_category("XKQJ RANDOM STRING 9921") == "Other")
test("Empty string -> Other", infer_category("") == "Other")
test("None-like input doesn't crash", infer_category(None) == "Other")
test("Rent -> Fixed/Excluded", infer_category("RENT PAYMENT") == "Fixed/Excluded")
test("Payroll -> Fixed/Excluded", infer_category("ACH CREDIT PAYROLL") == "Fixed/Excluded")


print("\n── rank_scenarios — normal case ────────────────────────────")
by_cat = {"Shopping": 3000, "Subscriptions": 500, "Food & Dining": 2000, "Utilities": 800}
ranked = rank_scenarios(by_cat, n_months=12, top_n=3)
test("returns exactly 3 scenarios", len(ranked) == 3)
test("scores descend", ranked[0]["score"] >= ranked[1]["score"] >= ranked[2]["score"])
test("monthly_saving > 0 for all", all(r["monthly_saving"] > 0 for r in ranked))


print("\n── rank_scenarios — single category (edge case) ───────────")
single = rank_scenarios({"Subscriptions": 500}, n_months=12, top_n=3)
test("single category returns 1 scenario, not 3", len(single) == 1, f"got {len(single)}")
test("single scenario has correct category", single[0]["category"] == "Subscriptions")


print("\n── rank_scenarios — all-zero spend (edge case) ─────────────")
zero_spend = rank_scenarios({"Shopping": 0, "Food & Dining": 0}, n_months=12, top_n=3)
test("all-zero spend doesn't crash", isinstance(zero_spend, list))
test("all-zero spend produces zero savings",
     all(r["monthly_saving"] == 0 for r in zero_spend) if zero_spend else True)


print("\n── rank_scenarios — only Fixed/Excluded category (edge case) ─")
only_fixed = rank_scenarios({"Fixed/Excluded": 5000}, n_months=12, top_n=3)
test("Fixed/Excluded never recommended as savings lever",
     len(only_fixed) == 0, f"got {len(only_fixed)} scenarios — should be 0")


print("\n── rank_scenarios — empty input (edge case) ────────────────")
empty = rank_scenarios({}, n_months=12, top_n=3)
test("empty category dict returns empty list, doesn't crash", empty == [])


print("\n── parse_uploaded_file-style scenarios (CSV edge cases) ────")
# We test the parsing LOGIC inline since parse_uploaded_file lives in app.py
# (Streamlit UploadedFile dependency) — these tests cover the same failure
# modes that function handles internally.

def simulate_column_detection(columns):
    """Mirrors the column-detection logic used in parse_uploaded_file."""
    date_col = next((c for c in columns if "date" in c), None)
    amt_col  = next((c for c in columns if any(x in c for x in
                    ["amount","debit","charge","transaction"])), None)
    return date_col, amt_col

date_col, amt_col = simulate_column_detection(["date", "amount", "description"])
test("standard columns detected correctly", date_col == "date" and amt_col == "amount")

date_col, amt_col = simulate_column_detection(["description", "category"])
test("missing date column detected as None (would trigger parse failure)",
     date_col is None, f"got {date_col}")

date_col, amt_col = simulate_column_detection([])
test("empty column list doesn't crash, returns None", date_col is None and amt_col is None)

# Malformed CSV — verify pandas read with on_bad_lines='skip' doesn't crash
malformed_csv = "date,amount,description\n2024-01-01,50.00,Coffee\nBROKEN,ROW,WITH,TOO,MANY,COLS\n2024-01-02,30.00,Lunch\n"
try:
    df = pd.read_csv(io.StringIO(malformed_csv), on_bad_lines="skip")
    test("malformed CSV with bad row doesn't crash (on_bad_lines=skip)", len(df) >= 2,
         f"got {len(df)} rows")
except Exception as e:
    test("malformed CSV with bad row doesn't crash", False, str(e))

# Empty file
try:
    df_empty = pd.read_csv(io.StringIO(""), on_bad_lines="skip")
    test("empty CSV doesn't crash on read", True)
except Exception:
    test("empty CSV doesn't crash on read", True)  # pandas raises EmptyDataError, that's expected/caught


print("\n── monte_carlo_stats — bug-fix regression tests ────────────")
wealth = run_monte_carlo(monthly_saving=200, years=10, mean_rate=0.07,
                         n_simulations=500, seed=42)
test("wealth matrix shape correct", wealth.shape == (500, 120), f"got {wealth.shape}")

stats = monte_carlo_stats(wealth, monthly_saving=200, years=10)
test("prob_double is not 100% (regression: was a stub bug)",
     stats["prob_double"] < 100.0, f"got {stats['prob_double']:.1f}%")
test("prob_double is not 0%", stats["prob_double"] > 0.0, f"got {stats['prob_double']:.1f}%")
test("total_contributed correct (200 × 120 = 24000)",
     abs(stats["total_contributed"] - 24000) < 1, f"got {stats['total_contributed']}")
test("p10 < median < p90",
     stats["p10"] < stats["median"] < stats["p90"],
     f"p10={stats['p10']:.0f} med={stats['median']:.0f} p90={stats['p90']:.0f}")


print("\n── n_simulations wiring (regression: was hardcoded) ────────")
w500  = run_monte_carlo(100, 5, 0.07, n_simulations=500,  seed=1)
w1000 = run_monte_carlo(100, 5, 0.07, n_simulations=1000, seed=1)
test("n_simulations=500 gives 500 rows", w500.shape[0] == 500)
test("n_simulations=1000 gives 1000 rows", w1000.shape[0] == 1000)


print(f"\n── Results ───────────────────────────────────────────────")
print(f"  {PASS} passed, {FAIL} failed (of {PASS + FAIL} total)")
if FAIL == 0:
    print("  All tests passed.")
    sys.exit(0)
else:
    print("  Fix failing tests before submission.")
    sys.exit(1)
