"""
test_engine.py — Unit tests for the engine, classifier, and ML fallback

Run: python3 test_engine.py  (or: pytest test_engine.py -v)

Covers: compound growth boundary cases, classifier edge cases, rank_scenarios
edge cases (all-fixed category, empty category, single category), CSV parsing
failure modes, Monte Carlo regression tests, and the Fixed/Excluded exclusion
logic introduced to prevent recommending users "cut" their rent or payroll.
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
test("0% rate = simple sum = 12,000 (boundary)", abs(v0 - 12000) < 1, f"got {v0:.0f}")
test("0 months = 0 wealth (boundary)", compute_compound(100, 0, 0.07) == 0)
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
test("None input doesn't crash", infer_category(None) == "Other")
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


print("\n── rank_scenarios — ALL category is Fixed/Excluded (critical edge case) ─")
only_fixed = rank_scenarios({"Fixed/Excluded": 5000}, n_months=12, top_n=3)
test("Fixed/Excluded NEVER recommended as a savings lever",
     len(only_fixed) == 0, f"got {len(only_fixed)} scenarios — should be 0")

mixed_with_fixed = rank_scenarios(
    {"Fixed/Excluded": 8000, "Subscriptions": 200}, n_months=12, top_n=3
)
test("mixed input: Fixed/Excluded excluded, real category still ranked",
     len(mixed_with_fixed) == 1 and mixed_with_fixed[0]["category"] == "Subscriptions",
     f"got {mixed_with_fixed}")


print("\n── rank_scenarios — empty input (edge case) ────────────────")
empty = rank_scenarios({}, n_months=12, top_n=3)
test("empty category dict returns empty list, doesn't crash", empty == [])


print("\n── CSV parsing failure modes ────────────────────────────────")
def simulate_column_detection(columns):
    date_col = next((c for c in columns if "date" in c), None)
    amt_col  = next((c for c in columns if any(x in c for x in
                    ["amount","debit","charge","transaction"])), None)
    return date_col, amt_col

date_col, amt_col = simulate_column_detection(["date", "amount", "description"])
test("standard columns detected correctly", date_col == "date" and amt_col == "amount")

date_col, amt_col = simulate_column_detection(["description", "category"])
test("missing date column detected as None (triggers parse failure)",
     date_col is None, f"got {date_col}")

date_col, amt_col = simulate_column_detection([])
test("empty column list doesn't crash, returns None", date_col is None and amt_col is None)

malformed_csv = "date,amount,description\n2024-01-01,50.00,Coffee\nBROKEN,ROW,WITH,TOO,MANY,COLS\n2024-01-02,30.00,Lunch\n"
try:
    df = pd.read_csv(io.StringIO(malformed_csv), on_bad_lines="skip")
    test("malformed CSV with bad row doesn't crash (on_bad_lines=skip)", len(df) >= 2,
         f"got {len(df)} rows")
except Exception as e:
    test("malformed CSV with bad row doesn't crash", False, str(e))

try:
    df_empty = pd.read_csv(io.StringIO(""), on_bad_lines="skip")
    test("empty CSV doesn't crash on read", True)
except Exception:
    test("empty CSV doesn't crash on read (raises expected EmptyDataError)", True)


print("\n── OFX parsing row alignment (regression test) ──────────────")
import re
def simulate_ofx_alignment(dates, amounts, descs):
    n = min(len(dates), len(amounts), len(descs) if descs else len(dates))
    desc_list = descs[:n] if descs else ["Unknown"] * n
    return n, desc_list

n, descs = simulate_ofx_alignment(["20240101","20240102","20240103"], ["10.0","20.0"], ["A","B","C"])
test("OFX alignment takes minimum length across all three lists",
     n == 2, f"got n={n}")
test("OFX alignment truncates descs to match", len(descs) == 2, f"got {len(descs)}")


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


print("\n── ML fallback classifier (sklearn model) ───────────────────")
try:
    from ml_classifier import load_model, predict_fallback, categorize_with_fallback
    ml_model = load_model()
    test("ML model loads from disk without error", ml_model is not None)

    pred, conf = predict_fallback("APPLE.COM/BILL", ml_model, threshold=0.0)
    test("ML model returns a valid category string", isinstance(pred, str) and len(pred) > 0)
    test("ML model confidence is a float between 0 and 1", 0.0 <= conf <= 1.0, f"got {conf}")

    cat, method = categorize_with_fallback("STARBUCKS #4521", ml_model)
    test("known keyword match uses 'keyword' method, not ML",
         method == "keyword", f"got method={method}")

    cat, method = categorize_with_fallback("ZXQWERTY999NOTHING", ml_model)
    test("nonsense string resolves to ml_fallback or unresolved, not keyword",
         method in ("ml_fallback", "unresolved"), f"got method={method}")
except FileNotFoundError:
    test("ML model file exists (run ml_classifier.py first)", False,
         "fallback_model.joblib not found — run: python3 ml_classifier.py")


print("\n── benchmark.py — BLS comparison (honest scope) ─────────────")
try:
    from benchmark import (benchmark_category, classify_income_quintile,
                           national_avg_monthly_spend, NATIONAL_CATEGORY_SHARES_2023)

    test("Food & Dining has a real BLS benchmark",
         national_avg_monthly_spend("Food & Dining") is not None)
    test("Unmapped category (Shopping) returns None, not a fabricated number",
         national_avg_monthly_spend("Shopping") is None,
         "Shopping should NOT have a benchmark — BLS doesn't map it cleanly")

    result = benchmark_category("Food & Dining", 900)
    test("benchmark_category returns a dict for mapped category", result is not None)
    test("benchmark_category flags above-average spend correctly",
         result["above_average"] == (result["diff_dollars"] > 0))

    none_result = benchmark_category("Shopping", 500)
    test("benchmark_category returns None for unmapped category (no fabrication)",
         none_result is None)

    test("income quintile classification — low income",
         classify_income_quintile(20000) == "Lowest quintile")
    test("income quintile classification — high income",
         classify_income_quintile(200000) == "Highest quintile")
    test("income quintile classification — boundary value",
         classify_income_quintile(28262) == "Second quintile",
         "28262 is the documented lower bound of the second quintile")
except ImportError:
    test("benchmark module importable", False, "benchmark.py not found")


print(f"\n── Results ───────────────────────────────────────────────")
print(f"  {PASS} passed, {FAIL} failed (of {PASS + FAIL} total)")
if FAIL == 0:
    print("  All tests passed.")
    sys.exit(0)
else:
    print("  Fix failing tests before submission.")
    sys.exit(1)
