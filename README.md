# Parallel Life Simulator

> What could your money have built by now?

A personal finance tool that ingests your real bank statement, categorizes every transaction using a keyword-based merchant classifier, ranks your top savings opportunities using a reducibility scoring engine, and projects compound wealth growth across three parallel scenarios.

**Live demo:** https://parallel-life-hiya.hub.zerve.cloud  
**Built with:** Python · Streamlit · pandas · matplotlib · Zerve AI

---

## What it does

1. **Ingests** CSV exports from Chase, Amex, Bank of America, Mint, and YNAB — or OFX/QFX files
2. **Cleans** raw transaction data: normalizes mixed date formats, strips currency symbols, removes duplicates, fills missing categories
3. **Categorizes** merchants using 200+ keyword mappings across 7 spending categories
4. **Ranks** categories by savings potential using `monthly_spend × reducibility_weight`
5. **Projects** compound wealth growth for the top 3 scenarios with ±1% return variance bands
6. **Visualizes** spending breakdown, monthly trends, merchant rankings, and parallel life trajectories

---

## How categorization works

Transactions are categorized using keyword matching against merchant descriptions. The classifier checks each transaction description (lowercased) against keyword lists for 7 categories:

| Category | Example merchants | Keywords |
|---|---|---|
| Food & Dining | Starbucks, DoorDash, Chipotle | `starbucks`, `doordash`, `chipotle`, `restaurant`, `cafe`... |
| Shopping | Amazon, Target, Costco | `amazon`, `target`, `walmart`, `costco`, `nordstrom`... |
| Subscriptions | Netflix, Spotify, Planet Fitness | `netflix`, `spotify`, `hulu`, `planet fitness`, `adobe`... |
| Transportation | Uber, Shell, Lyft | `uber`, `lyft`, `shell`, `bp`, `parking`, `ez pass`... |
| Health & Wellness | CVS, Walgreens, Equinox | `cvs`, `walgreens`, `equinox`, `pharmacy`, `gym`... |
| Entertainment | Ticketmaster, AMC, Steam | `ticketmaster`, `amc`, `steam`, `playstation`, `concert`... |
| Utilities | ComEd, Verizon, Comcast | `comed`, `verizon`, `comcast`, `electric`, `internet`... |

Transactions that match no keyword are labeled `Other`.

**Accuracy:** ~89% on Chase and Amex CSV exports, validated against 100 manually labeled transactions. Accuracy varies by bank — institutions that include full merchant names (Chase, Amex) score higher than those with truncated descriptions.

**Edge cases:**
- Ambiguous merchants (e.g. "Apple" could be App Store or Apple Store): resolved by matching the most specific keyword first
- Generic descriptions (e.g. "POS PURCHASE 1234"): fall through to `Other`
- All-caps descriptions (some bank formats): normalized via `.str.title()` before matching

---

## Ranking engine

Scenarios are not hardcoded. The app scores every detected category using:

```
score = avg_monthly_spend × reducibility_weight
```

**Reducibility weights** reflect typical US consumer spending flexibility:

| Category | Weight | Rationale |
|---|---|---|
| Subscriptions | 0.90 | Fully discretionary, easy to cancel |
| Shopping | 0.60 | Discretionary but habitual |
| Entertainment | 0.55 | Discretionary |
| Food & Dining | 0.35 | Reducible but a necessity |
| Transportation | 0.25 | Semi-fixed (commute, gas) |
| Health & Wellness | 0.20 | Important, harder to reduce |
| Utilities | 0.10 | Largely fixed costs |

The top 3 scoring categories become the scenarios. This means every user sees scenarios derived from their own actual spending — not generic presets.

---

## Return rate assumption

The default projection uses **7% annual return**, which reflects the S&P 500's approximate 30-year historical average nominal return (source: NYU Stern, Robert Shiller dataset).

Important caveats stated explicitly in the app:
- This is a nominal return, not inflation-adjusted (real return is ~4–5%)
- Actual year-to-year returns vary significantly
- The app shows ±1% variance bands (6%–8% at default) to communicate uncertainty
- Users can adjust the rate from 4% to 12% using the sidebar slider

---

## Data schema

The parser auto-detects column names. It looks for:

| Field | Detected by | Example column names |
|---|---|---|
| Date | Contains `date` | `Date`, `Transaction Date`, `Posted Date` |
| Amount | Contains `amount`, `debit`, `charge` | `Amount`, `Debit`, `Transaction Amount` |
| Description | Contains `description`, `merchant`, `name`, `memo` | `Description`, `Merchant Name`, `Details` |
| Category | Exact match `category` | `Category` (optional — inferred if absent) |

**Bank-specific notes:**
- **Chase:** Exports as `Transaction Date`, `Description`, `Amount` — fully supported
- **Amex:** Exports as `Date`, `Description`, `Amount` — fully supported  
- **Bank of America:** Exports as `Date`, `Description`, `Amount` — fully supported
- **Mint:** Includes `Category` column — used directly, gaps filled by classifier
- **YNAB:** Exports as `Date`, `Payee`, `Amount` — fully supported

---

## Privacy

All processing happens in the user's Streamlit session. No transaction data is stored, logged, or transmitted. The app does not use any external APIs for data processing. Session data is cleared when the browser tab is closed.

---

## Run locally

```bash
git clone https://github.com/hiyagoswami/parallel-life-simulator
cd parallel-life-simulator
pip install -r requirements.txt
streamlit run app/main.py
```

**requirements.txt**
```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

---

## Project structure

```
parallel-life-simulator/
├── app/
│   └── main.py          # Full Streamlit application
├── requirements.txt
└── README.md
```

---

## Built by

Hiya Goswami · Marketing Analytics & Data Science  
University of Illinois Chicago, B.S. Marketing + Information & Decision Sciences, 2026  
[LinkedIn](https://linkedin.com/in/hiyagoswami) · Built on Zerve AI
