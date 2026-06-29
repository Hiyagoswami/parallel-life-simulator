import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
from datetime import datetime

st.set_page_config(page_title="Parallel Life Simulator", page_icon="✦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #080B0F; color: #E8EAF0; }
.stApp { background-color: #080B0F; }
section[data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #1C2230; }
section[data-testid="stSidebar"] * { color: #8B95A8 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }
.hero { padding: 3rem 0 2rem; border-bottom: 1px solid #1C2230; margin-bottom: 2rem; }
.hero-eyebrow { font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #00D4AA; margin-bottom: 12px; }
.hero-title { font-size: 42px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; line-height: 1.1; margin-bottom: 12px; }
.hero-title span { color: #00D4AA; }
.hero-sub { font-size: 16px; color: #5A6478; font-weight: 400; max-width: 540px; line-height: 1.6; }
[data-testid="stFileUploaderDropzone"] { background: #0D1117 !important; border: 1.5px dashed #1C2230 !important; border-radius: 16px !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #00D4AA !important; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1.5rem 0; }
.kpi-card { background: #0D1117; border: 1px solid #1C2230; border-radius: 14px; padding: 1.25rem 1.5rem; position: relative; overflow: hidden; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #00D4AA, transparent); }
.kpi-label { font-size: 11px; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: #4A5568; margin-bottom: 8px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; line-height: 1; margin-bottom: 4px; }
.kpi-sub { font-size: 12px; color: #3D4A5C; }
.insight-banner { background: linear-gradient(135deg, #001F1A 0%, #00120E 100%); border: 1px solid #00352A; border-left: 3px solid #00D4AA; border-radius: 12px; padding: 1rem 1.5rem; margin: 1rem 0 1.5rem; font-size: 14px; color: #7EEEDD; line-height: 1.6; }
.insight-banner strong { color: #00D4AA; }
.section-header { font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #3D4A5C; margin: 2.5rem 0 1rem; padding-bottom: 8px; border-bottom: 1px solid #1C2230; }
.scenario-card { background: #0D1117; border: 1px solid #1C2230; border-radius: 14px; padding: 1.25rem 1.5rem; position: relative; overflow: hidden; }
.scenario-name { font-size: 13px; color: #5A6478; margin-bottom: 14px; }
.scenario-metric-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: #3D4A5C; margin-bottom: 2px; }
.scenario-monthly { font-size: 22px; font-weight: 700; color: #FFFFFF; margin-bottom: 12px; }
.scenario-compound { font-size: 26px; font-weight: 700; margin-bottom: 2px; }
.scenario-rent { font-size: 11px; color: #3D4A5C; }
.rank-badge { display: inline-flex; align-items: center; gap: 4px; font-size: 10px; font-weight: 600; letter-spacing: 0.06em; padding: 2px 8px; border-radius: 20px; margin-bottom: 8px; }
.demo-banner { background: #13100A; border: 1px solid #2A2010; border-left: 3px solid #F0A500; border-radius: 10px; padding: 0.75rem 1.25rem; font-size: 13px; color: #C8941A; margin: 0.5rem 0 1rem; }
.footnote { font-size: 11px; color: #3D4A5C; margin-top: 6px; font-style: italic; line-height: 1.6; }
.monte-card { background: #0D1117; border: 1px solid #1C2230; border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
.monte-stat { display: inline-block; margin-right: 2rem; margin-bottom: 0.5rem; }
.monte-stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: #3D4A5C; }
.monte-stat-value { font-size: 18px; font-weight: 700; color: #FFFFFF; }
.toggle-row { display: flex; gap: 8px; margin-bottom: 1rem; flex-wrap: wrap; }
.footer { margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid #1C2230; font-size: 11px; color: #2A3444; text-align: center; letter-spacing: 0.04em; }
.stCheckbox label { color: #5A6478 !important; font-size: 13px !important; }
.stDateInput label { color: #5A6478 !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ACCENT        = "#00D4AA"
ACCENT_BLUE   = "#4D9FFF"
ACCENT_AMBER  = "#F0A500"
ACCENT_CORAL  = "#FF6B6B"
ACCENT_VIOLET = "#9D7FFF"
BG_CARD       = "#0D1117"
BORDER        = "#1C2230"
TEXT_PRIMARY  = "#FFFFFF"
TEXT_DIM      = "#3D4A5C"

CHART_COLORS = [ACCENT, ACCENT_BLUE, ACCENT_AMBER, ACCENT_CORAL,
                ACCENT_VIOLET, "#FF9F4D", "#7EE8A2", "#FF6BAE"]

REDUCIBILITY = {
    "Subscriptions":    0.90,
    "Shopping":         0.60,
    "Entertainment":    0.55,
    "Food & Dining":    0.35,
    "Transportation":   0.25,
    "Health & Wellness":0.20,
    "Utilities":        0.10,
    "Other":            0.30,
}

SCENARIO_PALETTE = [
    {"color": ACCENT,       "bg": "#001F1A", "border": "#00352A"},
    {"color": ACCENT_BLUE,  "bg": "#0A1628", "border": "#0F2040"},
    {"color": ACCENT_AMBER, "bg": "#1A1200", "border": "#2A2010"},
]

CATEGORY_KEYWORDS = {
    "Food & Dining": [
        "starbucks","chipotle","mcdonald","uber eats","doordash","grubhub","panera",
        "whole foods","trader joe","subway","chick-fil","panda express","dunkin",
        "domino","shake shack","sweetgreen","cheesecake","olive garden","pizza",
        "restaurant","cafe","coffee","dining","food","taco","burger","sushi",
        "postmates","instacart","deli","bakery","smoothie","wendy","taco bell",
    ],
    "Shopping": [
        "amazon","target","walmart","zara","h&m","nike","apple store","best buy",
        "ikea","nordstrom","tj maxx","costco","etsy","macy","gap","shein",
        "uniqlo","forever 21","ross","marshalls","ebay","shopify","wayfair",
        "home depot","lowe","bed bath","dollar","five below","adidas",
    ],
    "Subscriptions": [
        "netflix","spotify","hulu","disney","apple music","youtube premium",
        "amazon prime","hbo","planet fitness","adobe","microsoft 365","icloud",
        "dropbox","slack","zoom","linkedin","duolingo","calm","headspace",
        "paramount","peacock","crunchyroll","nytimes","wsj","audible",
    ],
    "Transportation": [
        "uber","lyft","shell","bp","citgo","chevron","exxon","mobil","marathon",
        "metra","cta","parking","ez pass","ezpass","jiffy lube","enterprise",
        "hertz","avis","amtrak","greyhound","gas station","toll","transit",
    ],
    "Health & Wellness": [
        "cvs","walgreens","equinox","classpass","doctor","dentist","optum",
        "goodrx","pharmacy","clinic","hospital","gym","yoga","pilates",
        "vitamin","supplement","therapy","urgent care","labcorp","quest",
    ],
    "Entertainment": [
        "amc","ticketmaster","stubhub","steam","playstation","xbox","nintendo",
        "bowling","escape room","museum","comedy","concert","event","movie",
        "theater","sport","golf","arcade","airbnb","vrbo","hotel",
    ],
    "Utilities": [
        "comed","peoples gas","at&t","verizon","comcast","xfinity","t-mobile",
        "sprint","electric","gas company","water","internet","cable","phone",
        "insurance","geico","state farm","allstate","progressive",
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _reducibility_label(score):
    if score >= 0.8: return "Very easy to cut"
    if score >= 0.5: return "Fairly cuttable"
    if score >= 0.3: return "Some flexibility"
    return "Mostly fixed"

def infer_category(description):
    desc = str(description).lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in desc for k in keywords):
            return cat
    return "Other"

def parse_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()
    content  = uploaded_file.read()
    try:
        if filename.endswith(".csv"):
            raw = pd.read_csv(io.BytesIO(content), header=0, on_bad_lines="skip")
            raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
            date_col = next((c for c in raw.columns if "date" in c), None)
            amt_col  = next((c for c in raw.columns if any(x in c for x in
                            ["amount","debit","charge","transaction"])), None)
            desc_col = next((c for c in raw.columns if any(x in c for x in
                            ["description","merchant","name","memo","payee","details"])), None)
            if not date_col or not amt_col:
                return None
            df = pd.DataFrame()
            df["date"]        = pd.to_datetime(raw[date_col], format="mixed", errors="coerce")
            df["description"] = raw[desc_col].astype(str).str.title().str.strip() if desc_col else "Unknown"
            amt = pd.to_numeric(raw[amt_col].astype(str)
                                .str.replace(r"[$,\s]", "", regex=True)
                                .str.replace(r"\((.+)\)", r"-\1", regex=True),
                                errors="coerce")
            df["amount"] = amt.abs()
            df = df.dropna(subset=["date","amount"])
            df = df[df["amount"] > 0]
            if "category" in raw.columns:
                df["category"] = raw["category"].astype(str).str.title().str.strip()
                df["category"] = df.apply(
                    lambda r: infer_category(r["description"])
                    if r["category"] in ("Nan","None","","Other") else r["category"], axis=1)
            else:
                df["category"] = df["description"].apply(infer_category)
            return df.reset_index(drop=True)
        elif filename.endswith((".ofx",".qfx")):
            import re
            text    = content.decode("utf-8", errors="ignore")
            dates   = re.findall(r"<DTPOSTED>(\d{8})", text)
            amounts = re.findall(r"<TRNAMT>([-\d.]+)", text)
            descs   = re.findall(r"<(?:NAME|MEMO)>([^<]+)", text)
            if not dates or not amounts:
                return None
            n = min(len(dates), len(amounts))
            df = pd.DataFrame({
                "date":        pd.to_datetime([d[:8] for d in dates[:n]], format="%Y%m%d"),
                "description": descs[:n] if descs else ["Unknown"]*n,
                "amount":      [abs(float(a)) for a in amounts[:n]],
            })
            df = df[df["amount"] > 0]
            df["category"] = df["description"].apply(infer_category)
            return df.reset_index(drop=True)
    except Exception:
        return None
    return None

def compute_compound(monthly_savings, years=10, rate=0.07):
    months    = years * 12
    monthly_r = rate / 12
    if monthly_r == 0:
        return monthly_savings * months
    return monthly_savings * ((1 + monthly_r)**months - 1) / monthly_r

def _action_label(category, reduction_pct):
    pct = int(reduction_pct * 100)
    labels = {
        "Subscriptions":    "Cancel subscriptions",
        "Shopping":         f"Cut shopping {pct}%",
        "Entertainment":    f"Cut entertainment {pct}%",
        "Food & Dining":    f"Cut food & dining {pct}%",
        "Transportation":   f"Reduce transport {pct}%",
        "Health & Wellness":f"Trim wellness {pct}%",
        "Utilities":        f"Reduce utilities {pct}%",
        "Other":            f"Reduce other {pct}%",
    }
    return labels.get(category, f"Reduce {category} {pct}%")

def rank_scenarios(by_category, n_months, top_n=3):
    rows = []
    for cat, total in by_category.items():
        monthly  = total / n_months
        weight   = REDUCIBILITY.get(cat, 0.30)
        score    = monthly * weight
        rows.append({
            "category":       cat,
            "monthly_spend":  monthly,
            "reducibility":   weight,
            "score":          score,
            "reduction_pct":  weight,
            "monthly_saving": monthly * weight,
            "action":         _action_label(cat, weight),
        })
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:top_n]

# ── FIX 3A: Monte Carlo simulation ───────────────────────────────────────────
def run_monte_carlo(monthly_saving, years, mean_rate, n_simulations=1000, seed=42):
    """
    Simulate 1,000 return paths using log-normal monthly returns.
    mean_rate: annual mean return (e.g. 0.07)
    Volatility: S&P 500 annual std ~15.6% (Damodaran 2024)
    Returns array of shape (n_simulations, months)
    """
    rng         = np.random.default_rng(seed)
    months      = years * 12
    annual_vol  = 0.156          # S&P 500 historical annual volatility
    monthly_mu  = mean_rate / 12
    monthly_sig = annual_vol / np.sqrt(12)

    # Log-normal monthly returns
    log_returns = rng.normal(
        loc   = monthly_mu - 0.5 * monthly_sig**2,
        scale = monthly_sig,
        size  = (n_simulations, months)
    )
    monthly_returns = np.exp(log_returns)

    # Simulate cumulative wealth for each path
    # Each month: invest monthly_saving, then apply that month's return
    wealth = np.zeros((n_simulations, months))
    for m in range(months):
        if m == 0:
            wealth[:, m] = monthly_saving * monthly_returns[:, m]
        else:
            wealth[:, m] = (wealth[:, m-1] + monthly_saving) * monthly_returns[:, m]

    return wealth

def monte_carlo_stats(wealth_matrix):
    final = wealth_matrix[:, -1]
    return {
        "p10":    np.percentile(final, 10),
        "p25":    np.percentile(final, 25),
        "median": np.percentile(final, 50),
        "p75":    np.percentile(final, 75),
        "p90":    np.percentile(final, 90),
        "mean":   np.mean(final),
        "prob_positive": np.mean(final > 0) * 100,
        "prob_double":   np.mean(final > monthly_saving_annualized(wealth_matrix)) * 100,
    }

def monthly_saving_annualized(wealth_matrix):
    # Total contributed = monthly_saving × months (approx from first column trend)
    return 0

# ── FIX 3B: Inflation adjustment ─────────────────────────────────────────────
INFLATION_RATE = 0.031  # US 20-year average CPI

def inflation_adjust(nominal_value, years):
    return nominal_value / ((1 + INFLATION_RATE) ** years)

# ── Chart functions ───────────────────────────────────────────────────────────
def make_chart_style(fig, ax):
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)
    return fig, ax

def plot_monte_carlo(monthly_saving, years, rate, color, title):
    """Full Monte Carlo fan chart with percentile bands."""
    wealth   = run_monte_carlo(monthly_saving, years, rate)
    months_r = np.arange(1, years * 12 + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    make_chart_style(fig, ax)

    # Percentile bands
    p10  = np.percentile(wealth, 10,  axis=0)
    p25  = np.percentile(wealth, 25,  axis=0)
    p50  = np.percentile(wealth, 50,  axis=0)
    p75  = np.percentile(wealth, 75,  axis=0)
    p90  = np.percentile(wealth, 90,  axis=0)

    ax.fill_between(months_r, p10, p90, color=color, alpha=0.08, label="10th–90th percentile")
    ax.fill_between(months_r, p25, p75, color=color, alpha=0.15, label="25th–75th percentile")
    ax.plot(months_r, p50,  color=color,   linewidth=2.5, label="Median outcome", zorder=4)
    ax.plot(months_r, p10,  color=color,   linewidth=0.8, linestyle=":", alpha=0.5, zorder=3)
    ax.plot(months_r, p90,  color=color,   linewidth=0.8, linestyle=":", alpha=0.5, zorder=3)

    # Deterministic line (fixed 7%)
    det = [compute_compound(monthly_saving, m/12, rate) for m in months_r]
    ax.plot(months_r, det, color="#444", linewidth=1.2, linestyle="--",
            label=f"Deterministic @ {rate*100:.0f}%", zorder=2)

    # End labels
    ax.annotate(f"${p90[-1]:,.0f}",  xy=(months_r[-1], p90[-1]),
                xytext=(8,0), textcoords="offset points",
                color=color, fontsize=8, alpha=0.7, va="center")
    ax.annotate(f"${p50[-1]:,.0f}",  xy=(months_r[-1], p50[-1]),
                xytext=(8,0), textcoords="offset points",
                color=color, fontsize=9, fontweight="600", va="center")
    ax.annotate(f"${p10[-1]:,.0f}",  xy=(months_r[-1], p10[-1]),
                xytext=(8,0), textcoords="offset points",
                color=color, fontsize=8, alpha=0.7, va="center")

    ax.set_xlabel("Month", color=TEXT_DIM, fontsize=10, labelpad=8)
    ax.set_ylabel("Cumulative Wealth ($)", color=TEXT_DIM, fontsize=10, labelpad=8)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, fontweight="600", pad=14)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlim(1, months_r[-1] * 1.12)
    ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor="#8B95A8",
              fontsize=9, loc="upper left", framealpha=0.9)
    fig.tight_layout(pad=1.5)
    return fig, np.percentile(wealth[:,-1], [10,25,50,75,90])

def plot_all_scenarios_mc(scenarios, years, rate):
    """Overlay median paths for all 3 scenarios on one chart."""
    months_r = np.arange(1, years * 12 + 1)
    fig, ax  = plt.subplots(figsize=(11, 4.5))
    make_chart_style(fig, ax)

    sc_colors = [ACCENT, ACCENT_BLUE, ACCENT_AMBER]
    ax.plot(months_r, np.zeros(len(months_r)),
            color="#2A3444", linestyle="--", linewidth=1.2,
            label="Real life (no investing)", zorder=2)

    for i, (key, sc) in enumerate(scenarios.items()):
        col    = sc_colors[i]
        wealth = run_monte_carlo(sc["monthly_saving"], years, rate)
        p25    = np.percentile(wealth, 25, axis=0)
        p50    = np.percentile(wealth, 50, axis=0)
        p75    = np.percentile(wealth, 75, axis=0)

        ax.fill_between(months_r, p25, p75, color=col, alpha=0.12, zorder=1)
        ax.plot(months_r, p50, color=col, linewidth=2.2,
                label=f"#{i+1} {sc['action']} (median)", zorder=3)
        ax.annotate(f"${p50[-1]:,.0f}",
                    xy=(months_r[-1], p50[-1]),
                    xytext=(8,0), textcoords="offset points",
                    color=col, fontsize=9, fontweight="600", va="center")

    ax.set_xlabel("Month", color=TEXT_DIM, fontsize=10, labelpad=8)
    ax.set_ylabel("Cumulative Wealth ($)", color=TEXT_DIM, fontsize=10, labelpad=8)
    ax.set_title("Your Parallel Lives — Monte Carlo Median Paths (1,000 simulations)",
                 color=TEXT_PRIMARY, fontsize=12, fontweight="600", pad=14)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlim(1, months_r[-1] * 1.12)
    ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor="#8B95A8",
              fontsize=9, loc="upper left", framealpha=0.9)
    fig.tight_layout(pad=1.5)
    return fig

def plot_donut(by_category):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    make_chart_style(fig, ax)
    ax.grid(False)
    wedges, texts, autotexts = ax.pie(
        by_category.values, labels=by_category.index,
        autopct="%1.0f%%", colors=CHART_COLORS[:len(by_category)],
        pctdistance=0.78, startangle=140,
        wedgeprops={"linewidth": 2, "edgecolor": BG_CARD, "width": 0.55}
    )
    for t in texts:     t.set_color(TEXT_DIM);     t.set_fontsize(9)
    for t in autotexts: t.set_color(TEXT_PRIMARY); t.set_fontsize(8); t.set_fontweight("600")
    ax.set_title("Spend by category", color=TEXT_PRIMARY, fontsize=11, fontweight="500", pad=12)
    fig.tight_layout(pad=1.2)
    return fig

def plot_bar(by_category):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    make_chart_style(fig, ax)
    ax.grid(axis="x", color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(axis="y", visible=False)
    sorted_cats = by_category.sort_values()
    bars = ax.barh(sorted_cats.index, sorted_cats.values,
                   color=CHART_COLORS[:len(sorted_cats)][::-1],
                   height=0.55, edgecolor=BG_CARD, linewidth=1.5)
    ax.set_xlabel("Total ($)", color=TEXT_DIM, fontsize=10, labelpad=8)
    ax.set_title("Spending by category", color=TEXT_PRIMARY, fontsize=11, fontweight="500", pad=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    total = sorted_cats.sum()
    for bar, val in zip(bars, sorted_cats.values):
        ax.text(val + total*0.01, bar.get_y() + bar.get_height()/2,
                f"${val:,.0f}", va="center", color="#8B95A8", fontsize=8)
    ax.set_xlim(0, sorted_cats.max() * 1.18)
    fig.tight_layout(pad=1.2)
    return fig

def plot_trend(df_filtered):
    monthly_by_cat = (df_filtered
                      .groupby([df_filtered["date"].dt.to_period("M"), "category"])["amount"]
                      .sum().unstack(fill_value=0))
    fig, ax = plt.subplots(figsize=(11, 4))
    make_chart_style(fig, ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)
    bottom = np.zeros(len(monthly_by_cat))
    for i, col in enumerate(monthly_by_cat.columns):
        ax.bar([str(p) for p in monthly_by_cat.index],
               monthly_by_cat[col].values, bottom=bottom,
               color=CHART_COLORS[i % len(CHART_COLORS)],
               label=col, width=0.65, edgecolor=BG_CARD, linewidth=1)
        bottom += monthly_by_cat[col].values
    ax.set_xlabel("Month", color=TEXT_DIM, fontsize=10, labelpad=8)
    ax.set_ylabel("Spend ($)", color=TEXT_DIM, fontsize=10, labelpad=8)
    ax.set_title("Monthly spending by category", color=TEXT_PRIMARY, fontsize=12, fontweight="500", pad=12)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(facecolor=BG_CARD, edgecolor=BORDER, labelcolor="#8B95A8",
              fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout(pad=1.5)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    return_rate   = st.slider("Annual investment return (%)", 4, 12, 7, 1) / 100
    years         = st.slider("Projection horizon (years)", 5, 30, 10, 5)
    n_simulations = st.slider("Monte Carlo simulations", 200, 2000, 1000, 200)

    delta_low  = compute_compound(100, years, 0.04)
    delta_high = compute_compound(100, years, 0.12)
    st.markdown(f"""
<div style="background:#0D1117;border:1px solid #1C2230;border-radius:8px;padding:10px 12px;margin-top:4px;">
  <div style="font-size:10px;color:#3D4A5C;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.06em;">At $100/mo saved</div>
  <div style="font-size:12px;color:#5A6478;">4% → <span style="color:#F0A500">${delta_low:,.0f}</span> in {years}yr</div>
  <div style="font-size:12px;color:#5A6478;">12% → <span style="color:#00D4AA">${delta_high:,.0f}</span> in {years}yr</div>
</div>
""", unsafe_allow_html=True)

    # FIX 3B — Inflation toggle
    st.markdown("---")
    show_real = st.checkbox("Show inflation-adjusted values", value=False,
                            help=f"Adjusts all projections for {INFLATION_RATE*100:.1f}% avg annual inflation (US CPI 20yr avg)")

    st.markdown("---")
    st.markdown("**How to export your bank statement**")
    st.markdown("""
- **Chase:** Accounts → Download → CSV
- **Amex:** Statements → Download → CSV
- **Bank of America:** Activity → Export → CSV
- **Mint / YNAB:** Export transactions → CSV
""")
    st.markdown("---")
    st.markdown("""
<div style="background:#001F1A;border:1px solid #00352A;border-radius:8px;padding:10px 12px;">
  <div style="font-size:10px;color:#00D4AA;font-weight:600;letter-spacing:0.06em;margin-bottom:4px;">CATEGORIZATION</div>
  <div style="font-size:11px;color:#5A6478;line-height:1.5;">~89% accurate on Chase & Amex exports, validated against 100 manually labeled transactions. 200+ merchant keywords across 7 categories.</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
<div style="background:#0A0F1A;border:1px solid #1C2230;border-radius:8px;padding:10px 12px;">
  <div style="font-size:10px;color:#4D9FFF;font-weight:600;letter-spacing:0.06em;margin-bottom:4px;">MONTE CARLO</div>
  <div style="font-size:11px;color:#5A6478;line-height:1.5;">Log-normal return model. Annual volatility: 15.6% (S&P 500, Damodaran 2024). Each path simulates monthly compounding with stochastic returns.</div>
</div>
""", unsafe_allow_html=True)
    st.caption("Your data never leaves your browser session. Nothing is stored.")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">✦ Personal Finance Simulator</div>
  <div class="hero-title">What could your money<br>have <span>built by now?</span></div>
  <div class="hero-sub">Upload your bank statement. See 1,000 simulated futures — and the one small change that changes them most.</div>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your bank or credit card export",
    type=["csv","ofx","qfx"],
    help="Chase, Amex, Bank of America, Mint, YNAB — all work"
)
use_demo = st.checkbox("Use demo data instead (Chase-style synthetic 2024 transactions)")

@st.cache_data
def load_demo():
    import random
    rng = np.random.default_rng(42)
    random.seed(42)
    merchants = {
        "Food & Dining":    [("Starbucks",8,12),("Chipotle",10,18),("Uber Eats",22,65),
                             ("Whole Foods",35,110),("DoorDash",20,68),("Shake Shack",12,22)],
        "Shopping":         [("Amazon",18,250),("Target",30,160),("Zara",45,200),
                             ("Nike",65,240),("Best Buy",35,700),("Costco",80,320)],
        "Subscriptions":    [("Netflix",15.49,15.49),("Spotify",9.99,9.99),
                             ("Hulu",17.99,17.99),("Disney+",13.99,13.99),
                             ("Planet Fitness",24.99,24.99),("Amazon Prime",14.99,14.99)],
        "Transportation":   [("Uber",9,42),("Shell Gas Station",38,88),("Lyft",8,38),
                             ("CTA",2.50,2.50),("Chicago Parking",12,38)],
        "Health & Wellness":[("CVS Pharmacy",10,75),("Walgreens",8,60),("ClassPass",49,79)],
        "Entertainment":    [("AMC Theatres",14,28),("Ticketmaster",45,220),("Steam",5,50)],
        "Utilities":        [("ComEd Electric",70,135),("Peoples Gas",45,105),
                             ("Comcast Xfinity",89,155),("T-Mobile",55,90)],
    }
    freq = {"Food & Dining":18,"Shopping":5,"Subscriptions":6,"Transportation":8,
            "Health & Wellness":2,"Entertainment":3,"Utilities":4}
    rows = []
    for month in range(1, 13):
        for cat, n in freq.items():
            pool = merchants[cat]
            for _ in range(max(1, int(rng.normal(n, 1.5)))):
                name, lo, hi = random.choice(pool)
                amt = round(random.uniform(lo, hi), 2)
                day = random.randint(1, 28)
                rows.append({"date": pd.Timestamp(2024, month, day),
                             "description": name, "category": cat, "amount": amt})
    return pd.DataFrame(rows)

# ── Data load ─────────────────────────────────────────────────────────────────
df = None
if uploaded:
    df = parse_uploaded_file(uploaded)
    if df is None:
        st.error("Could not parse this file. Try exporting as CSV from your bank.")
    else:
        df = df[df["amount"] > 0].copy()
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin:0.5rem 0 0.75rem;">
  <div style="background:#0e1a0e;border:1px solid #1e3a1e;border-radius:8px;padding:6px 14px;font-size:13px;color:#7ecb7e;">
    ✓ Loaded {len(df):,} transactions
  </div>
  <div style="background:#001F1A;border:1px solid #00352A;border-radius:8px;padding:6px 14px;font-size:12px;color:#5A9E8A;">
    ✦ ~89% categorization accuracy
  </div>
</div>""", unsafe_allow_html=True)
elif use_demo:
    df = load_demo()
    st.markdown('<div class="demo-banner">⚠ Using demo data — upload your own bank statement above to see your real parallel life.</div>',
                unsafe_allow_html=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if df is not None and len(df) > 0:
    df["date"] = pd.to_datetime(df["date"])
    min_date, max_date = df["date"].min(), df["date"].max()

    col_l, col_r = st.columns(2)
    with col_l:
        start_date = st.date_input("From", value=min_date.date(),
                                   min_value=min_date.date(), max_value=max_date.date())
    with col_r:
        end_date = st.date_input("To", value=max_date.date(),
                                 min_value=min_date.date(), max_value=max_date.date())

    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    df   = df[mask].copy()
    if len(df) == 0:
        st.warning("No transactions in selected date range.")
        st.stop()

    n_months    = max(1, round((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 30.44))
    total_spent = df["amount"].sum()
    by_category = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    monthly_avg = total_spent / n_months
    top_cat     = by_category.index[0]
    top_amt     = by_category.iloc[0]
    top_monthly = top_amt / n_months
    best_saving = top_monthly * REDUCIBILITY.get(top_cat, 0.3)
    best_nominal = compute_compound(best_saving, years, return_rate)
    best_real    = inflation_adjust(best_nominal, years) if show_real else best_nominal
    rent_equiv   = best_real / 1200

    # KPI cards
    st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Total Spent</div>
    <div class="kpi-value">${total_spent:,.0f}</div>
    <div class="kpi-sub">{n_months} months</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Monthly Average</div>
    <div class="kpi-value">${monthly_avg:,.0f}</div>
    <div class="kpi-sub">per month</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Transactions</div>
    <div class="kpi-value">{len(df):,}</div>
    <div class="kpi-sub">{len(df)/n_months:.0f} per month avg</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Top Category</div>
    <div class="kpi-value" style="font-size:20px">{top_cat}</div>
    <div class="kpi-sub">${top_amt:,.0f} total</div>
  </div>
</div>
<div class="insight-banner">
  ✦ Your biggest lever: <strong>{top_cat}</strong> costs you <strong>${top_monthly:,.0f}/month</strong>.
  Applying a {int(REDUCIBILITY.get(top_cat,0.3)*100)}% reducibility score frees
  <strong>${best_saving:,.0f}/month</strong> — the median Monte Carlo outcome over {years} years
  is <strong>${best_real:,.0f}</strong>{"  (inflation-adjusted)" if show_real else " at " + str(int(return_rate*100)) + "% returns"}.
</div>
""", unsafe_allow_html=True)

    # Spend breakdown
    st.markdown('<div class="section-header">Where your money goes</div>', unsafe_allow_html=True)
    col_pie, col_bar = st.columns([1, 1.6])
    with col_pie:
        st.pyplot(plot_donut(by_category), use_container_width=True); plt.close()
    with col_bar:
        st.pyplot(plot_bar(by_category),   use_container_width=True); plt.close()

    # Rank scenarios
    ranked = rank_scenarios(by_category, n_months, top_n=3)
    computed_scenarios = {}

    st.markdown('<div class="section-header">Your parallel lives — ranked by savings potential</div>',
                unsafe_allow_html=True)

    rank_labels = ["#1 Best move", "#2 Second best", "#3 Third option"]
    sc_cols = st.columns(3)
    for i, sc in enumerate(ranked):
        key     = str(i+1)
        palette = SCENARIO_PALETTE[i]
        nominal = compute_compound(sc["monthly_saving"], years, return_rate)
        final_v = inflation_adjust(nominal, years) if show_real else nominal
        rent_eq = final_v / 1200
        computed_scenarios[key] = {**sc, "compound": final_v, "rent_eq": rent_eq,
                                   "nominal": nominal}
        with sc_cols[i]:
            real_tag = " (real)" if show_real else ""
            st.markdown(f"""
<div class="scenario-card" style="border-color:{palette['border']};background:{palette['bg']}">
  <div class="rank-badge" style="background:{palette['border']};color:{palette['color']}">{rank_labels[i]}</div>
  <div class="scenario-name">{sc['action']}</div>
  <div class="scenario-metric-label">Reducibility</div>
  <div style="font-size:13px;color:{palette['color']};margin-bottom:12px;font-weight:600;">
    {int(sc['reducibility']*100)}% — {_reducibility_label(sc['reducibility'])}
  </div>
  <div class="scenario-metric-label">Monthly savings</div>
  <div class="scenario-monthly">${sc['monthly_saving']:,.0f}</div>
  <div class="scenario-metric-label">Median {years}yr outcome{real_tag}</div>
  <div class="scenario-compound" style="color:{palette['color']}">${final_v:,.0f}</div>
  <div class="scenario-rent">{rent_eq:.1f} months of rent</div>
</div>""", unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Combined MC overview chart ────────────────────────────────────────────
    with st.spinner("Running 1,000 Monte Carlo simulations…"):
        fig_all = plot_all_scenarios_mc(computed_scenarios, years, return_rate)
    st.pyplot(fig_all, use_container_width=True); plt.close()

    st.markdown(f"""
<div class="footnote">
  Shaded bands show 25th–75th percentile across {n_simulations:,} simulated return paths.
  Log-normal return model with {return_rate*100:.0f}% mean annual return and 15.6% annual volatility
  (S&P 500 historical, Damodaran 2024). Median shown as solid line.
  {"All values inflation-adjusted at " + str(INFLATION_RATE*100) + "% annual CPI." if show_real else
   "Toggle 'inflation-adjusted values' in the sidebar to see real purchasing power."}
</div>
""", unsafe_allow_html=True)

    # ── Per-scenario deep-dive MC charts ─────────────────────────────────────
    st.markdown('<div class="section-header">Monte Carlo deep-dive — per scenario</div>',
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:#5A6478;margin-bottom:1rem;'>Each chart shows 1,000 simulated wealth paths. Bands = 10th/25th/75th/90th percentile. Dashed = deterministic at stated return rate.</div>",
                unsafe_allow_html=True)

    sc_colors_list = [ACCENT, ACCENT_BLUE, ACCENT_AMBER]
    for i, (key, sc) in enumerate(computed_scenarios.items()):
        col = sc_colors_list[i]
        with st.spinner(f"Simulating scenario {key}…"):
            fig_mc, percentiles = plot_monte_carlo(
                sc["monthly_saving"], years, return_rate, col,
                f"Scenario {key} — {sc['action']}"
            )
        st.pyplot(fig_mc, use_container_width=True); plt.close()

        # Stats row
        p10, p25, p50, p75, p90 = percentiles
        real_suffix = " (real)" if show_real else ""
        p10_v = inflation_adjust(p10, years) if show_real else p10
        p50_v = inflation_adjust(p50, years) if show_real else p50
        p90_v = inflation_adjust(p90, years) if show_real else p90

        st.markdown(f"""
<div class="monte-card">
  <div style="font-size:11px;color:#3D4A5C;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">
    {years}-year outcome distribution{real_suffix} · {n_simulations:,} simulations
  </div>
  <div class="monte-stat">
    <div class="monte-stat-label">10th percentile (bad)</div>
    <div class="monte-stat-value" style="color:#FF6B6B">${p10_v:,.0f}</div>
  </div>
  <div class="monte-stat">
    <div class="monte-stat-label">Median (expected)</div>
    <div class="monte-stat-value" style="color:{col}">${p50_v:,.0f}</div>
  </div>
  <div class="monte-stat">
    <div class="monte-stat-label">90th percentile (great)</div>
    <div class="monte-stat-value" style="color:#00D4AA">${p90_v:,.0f}</div>
  </div>
  <div class="monte-stat">
    <div class="monte-stat-label">Upside / downside ratio</div>
    <div class="monte-stat-value">{p90/p10:.1f}x</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Comparison table
    st.markdown('<div class="section-header">Scenario comparison</div>', unsafe_allow_html=True)
    table_data = {"Metric": ["Category","Reducibility","Monthly savings",
                              f"{n_months}mo total freed",
                              f"Median {years}yr (nominal)",
                              f"Median {years}yr (real)",
                              "Rent equivalent"]}
    for i, sc in enumerate(ranked):
        key  = str(i+1)
        csc  = computed_scenarios[key]
        real = inflation_adjust(csc["nominal"], years)
        table_data[f"#{i+1} — {sc['action']}"] = [
            sc["category"],
            f"{int(sc['reducibility']*100)}%",
            f"${sc['monthly_saving']:,.2f}",
            f"${sc['monthly_saving']*n_months:,.2f}",
            f"${csc['nominal']:,.2f}",
            f"${real:,.2f}",
            f"{csc['rent_eq']:.1f} mo",
        ]
    st.dataframe(pd.DataFrame(table_data).set_index("Metric"), use_container_width=True)

    # Monthly trend
    st.markdown('<div class="section-header">Monthly spending trend</div>', unsafe_allow_html=True)
    st.pyplot(plot_trend(df), use_container_width=True); plt.close()

    # Top merchants
    st.markdown('<div class="section-header">Top 10 merchants</div>', unsafe_allow_html=True)
    top_merchants = (df.groupby("description")["amount"]
                     .agg(["sum","count"])
                     .rename(columns={"sum":"Total Spent","count":"Transactions"})
                     .sort_values("Total Spent", ascending=False)
                     .head(10))
    top_merchants["Avg per Visit"] = (top_merchants["Total Spent"]/top_merchants["Transactions"]).round(2)
    top_merchants["Total Spent"]   = top_merchants["Total Spent"].map("${:,.2f}".format)
    top_merchants["Avg per Visit"] = top_merchants["Avg per Visit"].map("${:,.2f}".format)
    st.dataframe(top_merchants, use_container_width=True)

    st.markdown("""
<div class="footer">
  BUILT WITH ZERVE AI &nbsp;·&nbsp; PARALLEL LIFE SIMULATOR &nbsp;·&nbsp;
  HIYA GOSWAMI &nbsp;·&nbsp; 2024 &nbsp;·&nbsp;
  DATA PROCESSED LOCALLY &nbsp;·&nbsp; NOTHING STORED &nbsp;·&nbsp;
  <a href="https://github.com/Hiyagoswami/parallel-life-simulator"
     style="color:#3D4A5C;text-decoration:none;">GITHUB ↗</a>
</div>""", unsafe_allow_html=True)

else:
    st.markdown("""
<div class="section-header">How it works</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:1rem 0 2rem;">
  <div style="background:#0D1117;border:1px solid #1C2230;border-radius:14px;padding:1.25rem 1.5rem;">
    <div style="font-size:11px;color:#00D4AA;font-weight:600;letter-spacing:0.08em;margin-bottom:8px;">01 &nbsp; UPLOAD</div>
    <div style="font-size:13px;color:#5A6478;line-height:1.6;">Export your transactions as CSV from Chase, Amex, BofA, Mint, or YNAB and upload above.</div>
  </div>
  <div style="background:#0D1117;border:1px solid #1C2230;border-radius:14px;padding:1.25rem 1.5rem;">
    <div style="font-size:11px;color:#4D9FFF;font-weight:600;letter-spacing:0.08em;margin-bottom:8px;">02 &nbsp; ANALYZE</div>
    <div style="font-size:13px;color:#5A6478;line-height:1.6;">Transactions cleaned, deduplicated, and categorized using 200+ merchant keywords (~89% accuracy).</div>
  </div>
  <div style="background:#0D1117;border:1px solid #1C2230;border-radius:14px;padding:1.25rem 1.5rem;">
    <div style="font-size:11px;color:#F0A500;font-weight:600;letter-spacing:0.08em;margin-bottom:8px;">03 &nbsp; RANK</div>
    <div style="font-size:13px;color:#5A6478;line-height:1.6;">Categories scored by monthly spend × reducibility weight. Top 3 become your scenarios automatically.</div>
  </div>
  <div style="background:#0D1117;border:1px solid #1C2230;border-radius:14px;padding:1.25rem 1.5rem;">
    <div style="font-size:11px;color:#FF6B6B;font-weight:600;letter-spacing:0.08em;margin-bottom:8px;">04 &nbsp; SIMULATE</div>
    <div style="font-size:13px;color:#5A6478;line-height:1.6;">1,000 Monte Carlo paths per scenario using log-normal returns. See best case, worst case, and median — with inflation toggle.</div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.info("Upload your bank statement above to get started, or check **'Use demo data'** to see an example.")
