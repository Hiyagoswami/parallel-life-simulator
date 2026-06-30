# Parallel Life Simulator

> What could your money have built by now?

A personal finance tool that ingests your real bank statement, categorizes every transaction using a keyword-based merchant classifier, ranks your top savings opportunities using a reducibility scoring engine, and projects compound wealth growth across three parallel scenarios using Monte Carlo simulation.

**Live demo:** https://parallel-life-hiya.hub.zerve.cloud
**Built with:** Python · Streamlit · pandas · numpy · matplotlib · Zerve AI

---

## What it does

1. **Ingests** CSV exports from Chase, Amex, Bank of America, Mint, and YNAB — or OFX/QFX files
2. **Cleans** raw transaction data: normalizes mixed date formats, strips currency symbols, removes duplicates
3. **Categorizes** merchants using 200+ keyword mappings across 8 categories (7 spending + Fixed/Excluded)
4. **Excludes** rent, payroll deposits, transfers, and loan payments from spend analysis entirely
5. **Ranks** discretionary categories by savings potential using `monthly_spend × reducibility_weight`
6. **Simulates** 1,000+ Monte Carlo wealth paths per scenario using log-normal returns, with confidence bands and an inflation-adjustment toggle

---

## Project structure

```
parallel-life-simulator/
├── app.py              # Streamlit UI — imports from classifier.py and engine.py
├── classifier.py        # Keyword categorization + reducibility weights
├── engine.py             # Compound growth, Monte Carlo, ranking logic
├── validation.csv        # 100 manually labeled transactions (ground truth)
├── validate.py           # Measures classifier accuracy against validation.csv
├── test_engine.py        # 34 unit tests covering engine + classifier edge cases
├── requirements.txt
└── README.md
```

`app.py` genuinely imports from `classifier.py` and `engine.py` — these are real, separately-importable modules, not inlined code with comment dividers. `engine.py` has no Streamlit dependency, so it can be unit-tested in isolation.

---

## How categorization works

Transactions are categorized using keyword matching against merchant descriptions, checked against keyword lists for 8 categories (the 7 spending categories below, plus `Fixed/Excluded`).

| Category | Example merchants |
|---|---|
| Food & Dining | Starbucks, DoorDash, Chipotle |
| Shopping | Amazon, Target, Costco |
| Subscriptions | Netflix, Spotify, Planet Fitness |
| Transportation | Uber, Shell, Lyft |
| Health & Wellness | CVS, Walgreens, Equinox |
| Entertainment | Ticketmaster, AMC, Steam |
| Utilities | ComEd, Verizon, Comcast |
| **Fixed/Excluded** | **Rent, payroll deposit, internal transfer, loan payment** |

**`Fixed/Excluded` is the most important category.** Rent, mortgage payments, payroll deposits, and account transfers are detected and removed from spend analysis entirely — they are never treated as a discretionary expense and never appear as a "savings opportunity" in the ranking engine. Without this, the tool would give wrong advice (e.g. "cut your rent by 30%") the moment a real bank statement is uploaded instead of demo data.

### Measured accuracy: 89.0%

This number is not asserted — it's measured. `validate.py` runs `infer_category()` against `validation.csv`, a set of 100 manually labeled transactions that deliberately includes hard cases: payment-processor prefixes (`SQ*`, `TST*`, `PAYPAL*`), abbreviated merchant names (`WM SUPERCENTER`, `AMZN MKTP`), and hyphenated variants (`WAL-MART`). Reproduce it yourself:

```bash
python3 validate.py
```

Known tradeoff, documented in `classifier.py`: the keyword `"mobil"` was removed from Transportation because it falsely matched `"T-MOBILE"` as a gas station charge. This fixed a false positive but introduced a false negative for standalone Mobil gas stations (now classified as `Other`) — a deliberate tradeoff measured and accepted because the original collision was more common in the validation set.

---

## Ranking engine

Scenarios are not hardcoded. The app scores every detected discretionary category using:

```
score = avg_monthly_spend × reducibility_weight
```

**Reducibility weights are a heuristic, not a regression.** They were calibrated by eye against the general shape of discretionary-vs-fixed spending categories described in the BLS Consumer Expenditure Survey 2023 (https://www.bls.gov/cex/) — this is a reasoned mapping informed by that data, **not a fitted statistical coefficient**. Stated explicitly to avoid overclaiming precision the project doesn't have.

| Category | Weight | Rationale |
|---|---|---|
| Subscriptions | 0.90 | Fully discretionary, easy to cancel |
| Shopping | 0.60 | Discretionary but habitual |
| Entertainment | 0.55 | Discretionary |
| Food & Dining | 0.35 | Reducible but a necessity |
| Transportation | 0.25 | Semi-fixed (commute, gas) |
| Health & Wellness | 0.20 | Important, harder to reduce |
| Utilities | 0.10 | Largely fixed costs |
| Fixed/Excluded | 0.00 | Never a savings lever |

The top 3 scoring categories become the scenarios — derived from the user's own data, not generic presets.

---

## Monte Carlo simulation

Each scenario's projected outcome is modeled as 1,000+ simulated return paths (slider-adjustable 200–2,000), not a single deterministic line:

- **Model:** log-normal monthly returns with Itô correction
- **Volatility:** 15.6% annual (S&P 500 historical, Damodaran 2024)
- **Mean return:** user-adjustable 4–12%, default 7% (S&P 500 30-year nominal average, Shiller dataset)
- **Output:** 10th/25th/median/75th/90th percentile bands per scenario, plus probability of doubling total contributed capital

`run_monte_carlo()` is cached at the Streamlit layer via `@st.cache_data`, keyed on `(monthly_saving, years, rate, n_sims)`, so identical inputs don't re-run 1,000+ simulation paths on every UI rerun.

---

## Testing

```bash
python3 test_engine.py
```

34 unit tests covering: compound growth boundary cases (0% rate, 0 years), classifier edge cases (empty string, unknown merchant, None input), `rank_scenarios` with a single category / all-zero spend / only-Fixed-category / empty input, CSV parsing failure modes (missing date column, malformed rows, empty file), and explicit regression tests for the three bugs fixed in this version (`prob_double` stub, unwired `n_simulations`, dead `monte_carlo_stats()` call).

---

## Data schema

| Field | Detected by | Example column names |
|---|---|---|
| Date | Contains `date` | `Date`, `Transaction Date`, `Posted Date` |
| Amount | Contains `amount`, `debit`, `charge` | `Amount`, `Debit`, `Transaction Amount` |
| Description | Contains `description`, `merchant`, `name`, `memo` | `Description`, `Merchant Name`, `Details` |
| Category | Exact match `category` | `Category` (optional — inferred if absent) |

---

## Privacy

All processing happens in the user's Streamlit session. No transaction data is stored, logged, or transmitted to any external API.

---

## Run locally

```bash
git clone https://github.com/Hiyagoswami/parallel-life-simulator
cd parallel-life-simulator
pip install -r requirements.txt
streamlit run app.py
```

---

## Built by

Hiya Goswami · Marketing Analytics & Data Science
University of Illinois Chicago, B.S. Marketing + Information & Decision Sciences, 2026
[LinkedIn](https://linkedin.com/in/hiyagoswami) · Built on Zerve AI
