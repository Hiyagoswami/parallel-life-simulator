"""
app.py — Parallel Life Simulator (Streamlit UI)

Imports from classifier.py (keyword rules) and engine.py (simulation/ranking).
Categorization is a two-stage pipeline: classifier.infer_category() (fast,
deterministic keyword matching) runs first; ml_classifier.categorize_with_fallback()
(TF-IDF + Logistic Regression, trained on train_labels.csv) resolves anything
the keyword stage couldn't match. See evaluate_classifier.py for the measured
accuracy of each stage and the combined pipeline.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
import os

from classifier import (REDUCIBILITY, infer_category,
                        reducibility_label, action_label)
from engine import (compute_compound, inflation_adjust, run_monte_carlo,
                    monte_carlo_stats, rank_scenarios, INFLATION_RATE)
from ml_classifier import load_model, categorize_with_fallback, train_and_save
from report_export import generate_pdf_report, generate_csv_export
from benchmark import benchmark_category

@st.cache_data(show_spinner=False)
def cached_monte_carlo(monthly_saving: float, years: int, rate: float, n_sims: int):
    """Cached wrapper around run_monte_carlo — avoids recomputing 1000+
    simulation paths on every Streamlit rerun when inputs haven't changed."""
    return run_monte_carlo(monthly_saving, years, rate, n_simulations=n_sims, seed=42)


@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
def get_ml_model():
    """
    Load the trained ML fallback model once per session, not per rerun.

    If fallback_model.joblib isn't present (e.g. a fresh deployment that
    hasn't run `python3 ml_classifier.py` as a build step), this trains
    the model on-demand from train_labels.csv instead of silently falling
    back to keyword-only. This makes the app self-sufficient as long as
    train_labels.csv ships alongside the code — no separate .joblib upload
    or build-step coordination required.

    If train_labels.csv is also missing, returns None and the app degrades
    gracefully to keyword-only categorization (see categorize_transaction).
    """
    try:
        return load_model()
    except FileNotFoundError:
        train_csv_path = os.path.join(os.path.dirname(__file__), "train_labels.csv")
        if os.path.exists(train_csv_path):
            try:
                return train_and_save(train_csv=train_csv_path)
            except Exception:
                return None
        return None


def categorize_transaction(description: str, ml_model) -> str:
    """
    Two-stage categorization: keyword matcher first, ML fallback second.
    If the ML model isn't available (e.g. not yet trained in this
    environment), falls back to keyword-only — degrades gracefully
    rather than crashing.
    """
    if ml_model is None:
        return infer_category(description)
    category, _method = categorize_with_fallback(description, ml_model)
    return category


st.set_page_config(page_title="Parallel Life Simulator", page_icon="✦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: #0A0A0F;
    color: #F0F0FF;
}
.stApp { background: #0A0A0F; }

section[data-testid="stSidebar"] {
    background: #0F0F1A;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #9090B0 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* Hero */
.hero {
    padding: 3.5rem 0 2.5rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #fff;
    background: linear-gradient(135deg, #FF6B9D, #C44DFF);
    padding: 4px 14px; border-radius: 20px; margin-bottom: 20px;
}
.hero-title {
    font-size: 52px; font-weight: 800; color: #FFFFFF;
    letter-spacing: -0.03em; line-height: 1.05; margin-bottom: 16px;
}
.hero-title .g1 { color: #FF6B9D; }
.hero-title .g2 { color: #C44DFF; }
.hero-title .g3 { color: #4DC8FF; }
.hero-sub { font-size: 17px; color: #6060A0; font-weight: 400; max-width: 500px; line-height: 1.65; }

/* Upload */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,107,157,0.04) !important;
    border: 1.5px dashed rgba(255,107,157,0.3) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #FF6B9D !important;
    background: rgba(255,107,157,0.08) !important;
}

/* KPI grid */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1.5rem 0; }
.kpi-card {
    border-radius: 16px; padding: 1.25rem 1.5rem;
    position: relative; overflow: hidden;
}
.kpi-card-0 { background: linear-gradient(135deg, #1A0A2E, #2D1052); border: 1px solid rgba(196,77,255,0.25); }
.kpi-card-1 { background: linear-gradient(135deg, #0A1A2E, #103052); border: 1px solid rgba(77,200,255,0.25); }
.kpi-card-2 { background: linear-gradient(135deg, #1A0A1A, #2E1040); border: 1px solid rgba(255,107,157,0.25); }
.kpi-card-3 { background: linear-gradient(135deg, #0A1A10, #103020); border: 1px solid rgba(77,255,180,0.25); }
.kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: rgba(255,255,255,0.35); margin-bottom: 8px; }
.kpi-value { font-size: 30px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em; line-height: 1; margin-bottom: 4px; }
.kpi-sub { font-size: 12px; color: rgba(255,255,255,0.3); }

/* Insight banner */
.insight-banner {
    background: linear-gradient(135deg, rgba(255,107,157,0.08), rgba(196,77,255,0.08));
    border: 1px solid rgba(255,107,157,0.2);
    border-left: 3px solid #FF6B9D;
    border-radius: 14px; padding: 1rem 1.5rem;
    margin: 1rem 0 1.5rem; font-size: 14px;
    color: rgba(255,255,255,0.75); line-height: 1.65;
}
.insight-banner strong { color: #FF6B9D; }

/* Section headers */
.section-header {
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: rgba(255,255,255,0.25);
    margin: 2.5rem 0 1rem; padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex; align-items: center; gap: 8px;
}
.section-header::before {
    content: ''; display: inline-block;
    width: 3px; height: 14px; border-radius: 2px;
    background: linear-gradient(180deg, #FF6B9D, #C44DFF);
}

/* Scenario cards */
.scenario-card {
    border-radius: 18px; padding: 1.4rem 1.5rem;
    position: relative; overflow: hidden;
}
.sc-0 { background: linear-gradient(135deg, #1A0530, #2A0850); border: 1px solid rgba(196,77,255,0.3); }
.sc-1 { background: linear-gradient(135deg, #05101A, #081C30); border: 1px solid rgba(77,200,255,0.3); }
.sc-2 { background: linear-gradient(135deg, #1A0A05, #2A1408); border: 1px solid rgba(255,165,50,0.3); }
.scenario-name { font-size: 13px; color: rgba(255,255,255,0.45); margin-bottom: 14px; font-weight: 500; }
.scenario-metric-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.09em; color: rgba(255,255,255,0.25); margin-bottom: 3px; }
.scenario-monthly { font-size: 24px; font-weight: 800; color: #FFFFFF; margin-bottom: 14px; }
.scenario-compound { font-size: 28px; font-weight: 800; margin-bottom: 3px; }
.scenario-rent { font-size: 11px; color: rgba(255,255,255,0.25); }
.rank-badge {
    display: inline-flex; align-items: center;
    font-size: 10px; font-weight: 700; letter-spacing: 0.07em;
    padding: 3px 10px; border-radius: 20px; margin-bottom: 10px;
}

/* Demo banner */
.demo-banner {
    background: rgba(255,165,50,0.08);
    border: 1px solid rgba(255,165,50,0.2);
    border-left: 3px solid #FFA532;
    border-radius: 12px; padding: 0.75rem 1.25rem;
    font-size: 13px; color: rgba(255,165,50,0.9); margin: 0.5rem 0 1rem;
}

/* Monte carlo stat cards */
.monte-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
}
.monte-stat { display: inline-block; margin-right: 2rem; margin-bottom: 0.6rem; }
.monte-stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: rgba(255,255,255,0.25); margin-bottom: 3px; }
.monte-stat-value { font-size: 20px; font-weight: 800; color: #FFFFFF; }

.footnote { font-size: 11px; color: rgba(255,255,255,0.2); margin-top: 8px; font-style: italic; line-height: 1.65; }

/* How it works cards */
.how-card {
    border-radius: 16px; padding: 1.4rem 1.5rem;
    border: 1px solid rgba(255,255,255,0.07);
}
.how-01 { background: linear-gradient(135deg, #120520, #1E0835); }
.how-02 { background: linear-gradient(135deg, #051220, #081E35); }
.how-03 { background: linear-gradient(135deg, #201205, #351E08); }
.how-04 { background: linear-gradient(135deg, #200510, #35081C); }

/* Footer */
.footer {
    margin-top: 4rem; padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 11px; color: rgba(255,255,255,0.15);
    text-align: center; letter-spacing: 0.05em;
}

/* Streamlit overrides */
.stCheckbox label { color: #6060A0 !important; font-size: 13px !important; }
.stDateInput label { color: #6060A0 !important; font-size: 12px !important; }
[data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────────────────
ACCENT       = "#C44DFF"
ACCENT_BLUE  = "#4DC8FF"
ACCENT_AMBER = "#FFA532"
ACCENT_CORAL = "#FF6B9D"
ACCENT_GREEN = "#4DFFB4"
BG_CARD      = "#0F0F1A"
BORDER       = "rgba(255,255,255,0.08)"
TEXT_PRIMARY = "#FFFFFF"
TEXT_DIM     = "rgba(255,255,255,0.25)"
CHART_COLORS = ["#FF6B9D","#C44DFF","#4DC8FF","#FFA532",
                "#4DFFB4","#FF9F4D","#7B61FF","#FF6B6B"]
SCENARIO_PALETTE = [
    {"color": "#C44DFF", "bg": "#1A0530", "border": "rgba(196,77,255,0.35)"},
    {"color": "#4DC8FF", "bg": "#05101A", "border": "rgba(77,200,255,0.35)"},
    {"color": "#FFA532", "bg": "#1A0A05", "border": "rgba(255,165,50,0.35)"},
]

# ── Ingestion ─────────────────────────────────────────────────────────────────
def parse_uploaded_file(uploaded_file, ml_model=None):
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
                    lambda r: categorize_transaction(r["description"], ml_model)
                    if r["category"] in ("Nan","None","","Other") else r["category"], axis=1)
            else:
                df["category"] = df["description"].apply(
                    lambda d: categorize_transaction(d, ml_model))
            return df.reset_index(drop=True)

        elif filename.endswith((".ofx",".qfx")):
            import re
            text    = content.decode("utf-8", errors="ignore")
            dates   = re.findall(r"<DTPOSTED>(\d{8})", text)
            amounts = re.findall(r"<TRNAMT>([-\d.]+)", text)
            descs   = re.findall(r"<(?:NAME|MEMO)>([^<]+)", text)
            if not dates or not amounts:
                return None
            # FIX (low): align all lists to shortest to prevent row misalignment
            n = min(len(dates), len(amounts), len(descs) if descs else len(dates))
            desc_list = descs[:n] if descs else ["Unknown"] * n
            df = pd.DataFrame({
                "date":        pd.to_datetime([d[:8] for d in dates[:n]], format="%Y%m%d"),
                "description": desc_list,
                "amount":      [abs(float(a)) for a in amounts[:n]],
            })
            df = df[df["amount"] > 0]
            df["category"] = df["description"].apply(
                lambda d: categorize_transaction(d, ml_model))
            return df.reset_index(drop=True)
    except Exception:
        return None
    return None

# ── Chart helpers ─────────────────────────────────────────────────────────────
def make_chart_style(fig, ax):
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)

def plot_monte_carlo_scenario(monthly_saving, years, rate, n_sims, color, title):
    """Fan chart for one scenario with percentile bands. FIX: uses n_sims parameter."""
    wealth   = cached_monte_carlo(monthly_saving, years, rate, n_sims)
    stats    = monte_carlo_stats(wealth, monthly_saving, years)   # FIX: now called
    months_r = np.arange(1, years * 12 + 1)

    p10 = np.percentile(wealth, 10, axis=0)
    p25 = np.percentile(wealth, 25, axis=0)
    p50 = np.percentile(wealth, 50, axis=0)
    p75 = np.percentile(wealth, 75, axis=0)
    p90 = np.percentile(wealth, 90, axis=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    make_chart_style(fig, ax)

    ax.fill_between(months_r, p10, p90, color=color, alpha=0.08, label="10th–90th pct")
    ax.fill_between(months_r, p25, p75, color=color, alpha=0.15, label="25th–75th pct")
    ax.plot(months_r, p50, color=color, linewidth=2.5, label="Median", zorder=4)
    ax.plot(months_r, p10, color=color, linewidth=0.8, linestyle=":", alpha=0.5, zorder=3)
    ax.plot(months_r, p90, color=color, linewidth=0.8, linestyle=":", alpha=0.5, zorder=3)

    det = [compute_compound(monthly_saving, m/12, rate) for m in months_r]
    ax.plot(months_r, det, color="#444", linewidth=1.2, linestyle="--",
            label=f"Deterministic @ {rate*100:.0f}%", zorder=2)

    for val, y in [(p10[-1], p10[-1]), (p50[-1], p50[-1]), (p90[-1], p90[-1])]:
        ax.annotate(f"${val:,.0f}", xy=(months_r[-1], y),
                    xytext=(8,0), textcoords="offset points",
                    color=color, fontsize=8 if y != p50[-1] else 9,
                    fontweight="600" if y == p50[-1] else "normal",
                    alpha=1.0 if y == p50[-1] else 0.7, va="center")

    ax.set_xlabel("Month", color=TEXT_DIM, fontsize=10, labelpad=8)
    ax.set_ylabel("Cumulative Wealth ($)", color=TEXT_DIM, fontsize=10, labelpad=8)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, fontweight="600", pad=14)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlim(1, months_r[-1] * 1.12)
    ax.legend(facecolor="#0F0F1A", edgecolor=BORDER, labelcolor="#8B95A8",
              fontsize=9, loc="upper left", framealpha=0.9)
    fig.tight_layout(pad=1.5)
    return fig, stats

def plot_all_scenarios(scenarios, years, rate, n_sims):
    """Overview: median paths for all scenarios with 25–75 bands. FIX: n_sims wired."""
    months_r = np.arange(1, years * 12 + 1)
    fig, ax  = plt.subplots(figsize=(11, 4.5))
    make_chart_style(fig, ax)

    ax.plot(months_r, np.zeros(len(months_r)),
            color="#2A3444", linestyle="--", linewidth=1.2,
            label="Real life (no investing)", zorder=2)

    sc_colors = [ACCENT, ACCENT_BLUE, ACCENT_AMBER]
    for i, (key, sc) in enumerate(scenarios.items()):
        col    = sc_colors[i]
        # FIX: pass n_sims from sidebar slider, not hardcoded 1000
        wealth = cached_monte_carlo(sc["monthly_saving"], years, rate, n_sims)
        p25 = np.percentile(wealth, 25, axis=0)
        p50 = np.percentile(wealth, 50, axis=0)
        p75 = np.percentile(wealth, 75, axis=0)

        ax.fill_between(months_r, p25, p75, color=col, alpha=0.12, zorder=1)
        ax.plot(months_r, p50, color=col, linewidth=2.2,
                label=f"#{i+1} {sc['action']} (median)", zorder=3)
        ax.annotate(f"${p50[-1]:,.0f}", xy=(months_r[-1], p50[-1]),
                    xytext=(8,0), textcoords="offset points",
                    color=col, fontsize=9, fontweight="600", va="center")

    ax.set_xlabel("Month", color=TEXT_DIM, fontsize=10, labelpad=8)
    ax.set_ylabel("Cumulative Wealth ($)", color=TEXT_DIM, fontsize=10, labelpad=8)
    ax.set_title(f"Your Parallel Lives — {n_sims:,} Monte Carlo Simulations (Median Paths)",
                 color=TEXT_PRIMARY, fontsize=12, fontweight="600", pad=14)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlim(1, months_r[-1] * 1.12)
    ax.legend(facecolor="#0F0F1A", edgecolor=BORDER, labelcolor="#8B95A8",
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
    ax.legend(facecolor="#0F0F1A", edgecolor=BORDER, labelcolor="#8B95A8",
              fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout(pad=1.5)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    return_rate = st.slider("Annual investment return (%)", 4, 12, 7, 1) / 100
    years       = st.slider("Projection horizon (years)", 5, 30, 10, 5)
    # FIX: n_simulations now wired to all run_monte_carlo() calls
    n_sims      = st.slider("Monte Carlo simulations", 200, 2000, 1000, 200)

    delta_low  = compute_compound(100, years, 0.04)
    delta_high = compute_compound(100, years, 0.12)
    st.markdown(f"""
<div style="background:#0D1117;border:1px solid #1C2230;border-radius:8px;padding:10px 12px;margin-top:4px;">
  <div style="font-size:10px;color:#3D4A5C;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.06em;">At $100/mo saved</div>
  <div style="font-size:12px;color:#5A6478;">4% → <span style="color:#F0A500">${delta_low:,.0f}</span> in {years}yr</div>
  <div style="font-size:12px;color:#5A6478;">12% → <span style="color:#00D4AA">${delta_high:,.0f}</span> in {years}yr</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    show_real = st.checkbox("Show inflation-adjusted values", value=False,
                            help=f"Adjusts projections for {INFLATION_RATE*100:.1f}% avg annual CPI")
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
  <div style="font-size:11px;color:#5A6478;line-height:1.5;">Two-stage pipeline: keyword matching (76.0% measured) + TF-IDF/LogReg ML fallback (85.3% measured). Evaluated on 75 held-out labeled transactions never seen during ML training. See evaluate_classifier.py.</div>
</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
<div style="background:#0A0F1A;border:1px solid #1C2230;border-radius:8px;padding:10px 12px;">
  <div style="font-size:10px;color:#4D9FFF;font-weight:600;letter-spacing:0.06em;margin-bottom:4px;">MONTE CARLO MODEL</div>
  <div style="font-size:11px;color:#5A6478;line-height:1.5;">Log-normal returns · 15.6% annual vol (Damodaran 2024) · {n_sims:,} paths · Itô correction applied</div>
</div>""", unsafe_allow_html=True)
    st.caption("Data processed locally. Nothing stored.")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">✦ Personal Finance Simulator</div>
  <div class="hero-title">
    What could your <span class="g1">money</span><br>
    have <span class="g2">built</span> <span class="g3">by now?</span>
  </div>
  <div class="hero-sub">Upload your bank statement. See 1,000 simulated futures — and the one small change that changes them most.</div>
</div>""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload your bank or credit card export",
                             type=["csv","ofx","qfx"])
use_demo = st.checkbox("Use demo data instead (Chase-style synthetic 2024 transactions)")

@st.cache_data
def load_demo():
    import random
    rng = np.random.default_rng(42); random.seed(42)
    merchants = {
        "Food & Dining":    [("Starbucks",8,12),("Chipotle",10,18),("Uber Eats",22,65),
                             ("Whole Foods",35,110),("DoorDash",20,68),("Shake Shack",12,22)],
        "Shopping":         [("Amazon",18,250),("Target",30,160),("Zara",45,200),
                             ("Nike",65,240),("Best Buy",35,700),("Costco",80,320)],
        "Subscriptions":    [("Netflix",15.49,15.49),("Spotify",9.99,9.99),("Hulu",17.99,17.99),
                             ("Disney+",13.99,13.99),("Planet Fitness",24.99,24.99),("Amazon Prime",14.99,14.99)],
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
    for month in range(1,13):
        for cat, n in freq.items():
            for _ in range(max(1, int(rng.normal(n, 1.5)))):
                name, lo, hi = random.choice(merchants[cat])
                rows.append({"date": pd.Timestamp(2024, month, random.randint(1,28)),
                             "description": name, "category": cat,
                             "amount": round(random.uniform(lo, hi), 2)})
    return pd.DataFrame(rows)

df = None
ml_model = get_ml_model()

# FIX #3 — persistence via session state: parsing + categorization (including
# the ML fallback model call) only re-runs when the uploaded file actually
# changes, not on every slider drag. Without this, adjusting the return-rate
# slider would silently re-run the full categorization pipeline on every
# rerun, which is both slow and pointless since the file hasn't changed.
if uploaded:
    file_signature = (uploaded.name, uploaded.size)
    if st.session_state.get("_last_file_sig") != file_signature:
        st.session_state["_parsed_df"] = parse_uploaded_file(uploaded, ml_model=ml_model)
        st.session_state["_last_file_sig"] = file_signature
    df = st.session_state.get("_parsed_df")

    if df is None:
        st.error("Could not parse this file. Try exporting as CSV from your bank.")
    else:
        df = df[df["amount"] > 0].copy()
        model_status = "keyword + ML fallback (85.3% measured)" if ml_model is not None else "keyword-only (76.0% measured)"
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin:0.5rem 0 0.75rem;">
  <div style="background:#0e1a0e;border:1px solid #1e3a1e;border-radius:8px;padding:6px 14px;font-size:13px;color:#7ecb7e;">✓ Loaded {len(df):,} transactions</div>
  <div style="background:#001F1A;border:1px solid #00352A;border-radius:8px;padding:6px 14px;font-size:12px;color:#5A9E8A;">✦ {model_status}</div>
</div>""", unsafe_allow_html=True)
elif use_demo:
    df = load_demo()
    st.markdown('<div class="demo-banner">⚠ Demo data active — upload your bank statement above to see your real parallel life.</div>',
                unsafe_allow_html=True)

if df is not None and len(df) > 0:
    df["date"] = pd.to_datetime(df["date"])
    min_d, max_d = df["date"].min(), df["date"].max()

    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("From", value=min_d.date(), min_value=min_d.date(), max_value=max_d.date())
    with c2: end_date   = st.date_input("To",   value=max_d.date(), min_value=min_d.date(), max_value=max_d.date())

    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    df   = df[mask].copy()
    if len(df) == 0:
        st.warning("No transactions in selected date range."); st.stop()

    # FIX #5 — exclude rent, payroll, transfers, and other Fixed/Excluded
    # transactions from spend analysis entirely. Without this the tool
    # would recommend "cutting" rent or payroll deposits, which is wrong
    # advice the moment a real (non-demo) statement is uploaded.
    n_excluded = int((df["category"] == "Fixed/Excluded").sum())
    excluded_total = df.loc[df["category"] == "Fixed/Excluded", "amount"].sum()
    df = df[df["category"] != "Fixed/Excluded"].copy()

    if len(df) == 0:
        st.warning("All transactions in this range were Fixed/Excluded (rent, payroll, transfers). Nothing to analyze.")
        st.stop()

    if n_excluded > 0:
        st.markdown(f"""
<div style="background:rgba(77,200,255,0.06);border:1px solid rgba(77,200,255,0.2);border-left:3px solid #4DC8FF;border-radius:10px;padding:0.7rem 1.1rem;font-size:12px;color:rgba(77,200,255,0.85);margin-bottom:1rem;">
  ℹ Excluded {n_excluded} fixed transaction{'s' if n_excluded != 1 else ''} (${excluded_total:,.0f} total) — rent, payroll, transfers, and loan payments aren't treated as savings opportunities.
</div>""", unsafe_allow_html=True)

    n_months    = max(1, round((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 30.44))
    total_spent = df["amount"].sum()
    by_category = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    monthly_avg = total_spent / n_months
    top_cat     = by_category.index[0]
    top_monthly = by_category.iloc[0] / n_months
    best_saving = top_monthly * REDUCIBILITY.get(top_cat, 0.3)
    best_nominal = compute_compound(best_saving, years, return_rate)
    best_display = inflation_adjust(best_nominal, years) if show_real else best_nominal

    st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card kpi-card-0"><div class="kpi-label">Total Spent</div><div class="kpi-value">${total_spent:,.0f}</div><div class="kpi-sub">{n_months} months</div></div>
  <div class="kpi-card kpi-card-1"><div class="kpi-label">Monthly Average</div><div class="kpi-value">${monthly_avg:,.0f}</div><div class="kpi-sub">per month</div></div>
  <div class="kpi-card kpi-card-2"><div class="kpi-label">Transactions</div><div class="kpi-value">{len(df):,}</div><div class="kpi-sub">{len(df)/n_months:.0f}/month avg</div></div>
  <div class="kpi-card kpi-card-3"><div class="kpi-label">Top Category</div><div class="kpi-value" style="font-size:20px">{top_cat}</div><div class="kpi-sub">${by_category.iloc[0]:,.0f} total</div></div>
</div>
<div class="insight-banner">
  ✦ Your biggest lever: <strong>{top_cat}</strong> costs <strong>${top_monthly:,.0f}/month</strong>.
  A {int(REDUCIBILITY.get(top_cat,0.3)*100)}% reducibility score frees <strong>${best_saving:,.0f}/month</strong> —
  median Monte Carlo outcome over {years}yr: <strong>${best_display:,.0f}</strong>{"  (inflation-adjusted)" if show_real else ""}.
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Where your money goes</div>', unsafe_allow_html=True)
    cp, cb = st.columns([1, 1.6])
    with cp: st.pyplot(plot_donut(by_category), use_container_width=True); plt.close()
    with cb: st.pyplot(plot_bar(by_category),   use_container_width=True); plt.close()

    # ── FIX #4 — BLS national benchmark (honestly scoped) ────────────────────
    benchmark_results = [
        benchmark_category(cat, total / n_months)
        for cat, total in by_category.items()
    ]
    benchmark_results = [r for r in benchmark_results if r is not None]

    if benchmark_results:
        st.markdown('<div class="section-header">How you compare — BLS national averages</div>',
                    unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:12px;color:#5A6478;margin-bottom:0.75rem;line-height:1.6;">
  Comparing your spending to the BLS Consumer Expenditure Survey 2023 national average household.
  <strong style="color:#FFA532;">Note:</strong> this compares against the national average across all households,
  not specifically households at your income level — BLS's full income-quintile breakdown by category
  wasn't available as structured data for this comparison. Only categories with a clean BLS mapping
  (Food & Dining, Transportation, Health & Wellness, Entertainment) are shown.
</div>""", unsafe_allow_html=True)

        bm_cols = st.columns(len(benchmark_results))
        for i, bm in enumerate(benchmark_results):
            with bm_cols[i]:
                color = "#FF6B9D" if bm["above_average"] else "#4DFFB4"
                arrow = "▲" if bm["above_average"] else "▼"
                st.markdown(f"""
<div style="background:#0D1117;border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1rem 1.1rem;">
  <div style="font-size:10px;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">{bm['category']}</div>
  <div style="font-size:18px;font-weight:800;color:{color};">{arrow} {abs(bm['diff_pct']):.0f}%</div>
  <div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:4px;">vs national avg ${bm['national_avg_monthly']:,.0f}/mo</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="footnote">Source: BLS "Consumer Expenditures in 2023" (national average annual expenditure $77,280/household, category shares from Table B). National average, not income-adjusted.</div>',
                    unsafe_allow_html=True)

    ranked = rank_scenarios(by_category, n_months, top_n=3)
    computed_scenarios = {}

    st.markdown('<div class="section-header">Your parallel lives — ranked by savings potential</div>',
                unsafe_allow_html=True)

    rank_labels = ["#1 Best move", "#2 Second best", "#3 Third option"]
    sc_cols = st.columns(3)
    for i, sc in enumerate(ranked):
        key      = str(i+1)
        palette  = SCENARIO_PALETTE[i]
        nominal  = compute_compound(sc["monthly_saving"], years, return_rate)
        final_v  = inflation_adjust(nominal, years) if show_real else nominal
        computed_scenarios[key] = {**sc, "compound": final_v, "nominal": nominal}
        sc_class = ["sc-0","sc-1","sc-2"][i]
        with sc_cols[i]:
            st.markdown(f"""
<div class="scenario-card {sc_class}">
  <div class="rank-badge" style="background:rgba(255,255,255,0.08);color:{palette['color']}">{rank_labels[i]}</div>
  <div class="scenario-name">{sc['action']}</div>
  <div class="scenario-metric-label">Reducibility (BLS CEX calibrated)</div>
  <div style="font-size:13px;color:{palette['color']};margin-bottom:12px;font-weight:700;">{int(sc['reducibility']*100)}% — {reducibility_label(sc['reducibility'])}</div>
  <div class="scenario-metric-label">Monthly savings</div>
  <div class="scenario-monthly">${sc['monthly_saving']:,.0f}</div>
  <div class="scenario-metric-label">Median {years}yr {"(real)" if show_real else "@ " + str(int(return_rate*100)) + "%"}</div>
  <div class="scenario-compound" style="color:{palette['color']}">${final_v:,.0f}</div>
  <div class="scenario-rent">{final_v/1200:.1f} months of rent</div>
</div>""", unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    with st.spinner(f"Running {n_sims:,} Monte Carlo simulations…"):
        fig_all = plot_all_scenarios(computed_scenarios, years, return_rate, n_sims)
    st.pyplot(fig_all, use_container_width=True); plt.close()

    st.markdown(f"""
<div class="footnote">
  Shaded bands: 25th–75th percentile across {n_sims:,} simulated paths.
  Log-normal model · {return_rate*100:.0f}% mean annual return · 15.6% annual volatility (S&P 500, Damodaran 2024) · Itô correction applied.
  {"Inflation-adjusted at " + str(INFLATION_RATE*100) + "% CPI." if show_real else "Toggle inflation adjustment in sidebar."}
  Default 7% = S&P 500 30-year nominal average (Shiller dataset).
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Monte Carlo deep-dive — per scenario</div>',
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:#5A6478;margin-bottom:1rem;'>Fan charts show 10th/25th/75th/90th percentile bands across all simulated paths.</div>",
                unsafe_allow_html=True)

    sc_colors_list = [ACCENT, ACCENT_BLUE, ACCENT_AMBER]
    for i, (key, sc) in enumerate(computed_scenarios.items()):
        col = sc_colors_list[i]
        with st.spinner(f"Simulating scenario {key}…"):
            fig_mc, stats = plot_monte_carlo_scenario(
                sc["monthly_saving"], years, return_rate, n_sims, col,
                f"Scenario {key} — {sc['action']}"
            )
        st.pyplot(fig_mc, use_container_width=True); plt.close()

        p10_v = inflation_adjust(stats["p10"],    years) if show_real else stats["p10"]
        p50_v = inflation_adjust(stats["median"], years) if show_real else stats["median"]
        p90_v = inflation_adjust(stats["p90"],    years) if show_real else stats["p90"]
        real_tag = " (real)" if show_real else ""

        st.markdown(f"""
<div class="monte-card">
  <div style="font-size:11px;color:#3D4A5C;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">
    {years}-year outcome distribution{real_tag} · {n_sims:,} simulations
  </div>
  <div class="monte-stat"><div class="monte-stat-label">10th pct (bad)</div><div class="monte-stat-value" style="color:#FF6B6B">${p10_v:,.0f}</div></div>
  <div class="monte-stat"><div class="monte-stat-label">Median</div><div class="monte-stat-value" style="color:{col}">${p50_v:,.0f}</div></div>
  <div class="monte-stat"><div class="monte-stat-label">90th pct (great)</div><div class="monte-stat-value" style="color:#00D4AA">${p90_v:,.0f}</div></div>
  <div class="monte-stat"><div class="monte-stat-label">Upside / downside</div><div class="monte-stat-value">{stats["p90"]/stats["p10"]:.1f}x</div></div>
  <div class="monte-stat"><div class="monte-stat-label">Prob. doubling contribution</div><div class="monte-stat-value">{stats["prob_double"]:.0f}%</div></div>
  <div class="monte-stat"><div class="monte-stat-label">Total contributed</div><div class="monte-stat-value">${stats["total_contributed"]:,.0f}</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Scenario comparison</div>', unsafe_allow_html=True)
    table_data = {"Metric": ["Category","Reducibility","Monthly savings",
                              f"{n_months}mo freed","Median {years}yr (nominal)",
                              f"Median {years}yr (real)","Rent equivalent"]}
    for i, sc in enumerate(ranked):
        key  = str(i+1)
        csc  = computed_scenarios[key]
        real = inflation_adjust(csc["nominal"], years)
        table_data[f"#{i+1} — {sc['action']}"] = [
            sc["category"], f"{int(sc['reducibility']*100)}%",
            f"${sc['monthly_saving']:,.2f}", f"${sc['monthly_saving']*n_months:,.2f}",
            f"${csc['nominal']:,.2f}", f"${real:,.2f}", f"{csc['compound']/1200:.1f} mo",
        ]
    st.dataframe(pd.DataFrame(table_data).set_index("Metric"), use_container_width=True)

    st.markdown('<div class="section-header">Monthly spending trend</div>', unsafe_allow_html=True)
    st.pyplot(plot_trend(df), use_container_width=True); plt.close()

    st.markdown('<div class="section-header">Top 10 merchants</div>', unsafe_allow_html=True)
    top_m_raw = (df.groupby("description")["amount"].agg(["sum","count"])
               .rename(columns={"sum":"Total Spent","count":"Transactions"})
               .sort_values("Total Spent", ascending=False).head(10))
    top_m_raw["Avg per Visit"] = (top_m_raw["Total Spent"]/top_m_raw["Transactions"]).round(2)

    top_m_display = top_m_raw.copy()
    top_m_display["Total Spent"]   = top_m_display["Total Spent"].map("${:,.2f}".format)
    top_m_display["Avg per Visit"] = top_m_display["Avg per Visit"].map("${:,.2f}".format)
    st.dataframe(top_m_display, use_container_width=True)

    # ── FIX #3 — Persistence: downloadable PDF + CSV reports ────────────────
    st.markdown('<div class="section-header">Save your results</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#5A6478;margin-bottom:0.75rem;">Your analysis lives only in this session — download a copy to keep, print, or share.</div>',
                unsafe_allow_html=True)

    col_pdf, col_csv = st.columns(2)
    with col_pdf:
        try:
            pdf_buffer = generate_pdf_report(
                total_spent=total_spent, n_months=n_months, by_category=by_category,
                ranked_scenarios=ranked, computed_scenarios=computed_scenarios,
                return_rate=return_rate, years=years, top_merchants_df=top_m_raw,
            )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name="parallel_life_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"PDF generation unavailable: {e}")

    with col_csv:
        csv_data = generate_csv_export(
            by_category=by_category, ranked_scenarios=ranked,
            computed_scenarios=computed_scenarios, n_months=n_months,
            years=years, return_rate=return_rate,
        )
        st.download_button(
            label="📊 Download CSV Data",
            data=csv_data,
            file_name="parallel_life_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("""
<div class="footer">
  BUILT WITH ZERVE AI &nbsp;·&nbsp; PARALLEL LIFE SIMULATOR &nbsp;·&nbsp;
  HIYA GOSWAMI &nbsp;·&nbsp; 2024 &nbsp;·&nbsp;
  DATA PROCESSED LOCALLY &nbsp;·&nbsp;
  <a href="https://github.com/Hiyagoswami/parallel-life-simulator" style="color:#3D4A5C;text-decoration:none;">GITHUB ↗</a>
</div>""", unsafe_allow_html=True)

else:
    st.markdown("""
<div class="section-header">How it works</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:1rem 0 2rem;">
  <div class="how-card how-01">
    <div style="font-size:11px;color:#C44DFF;font-weight:700;letter-spacing:0.1em;margin-bottom:10px;">01 &nbsp; UPLOAD</div>
    <div style="font-size:13px;color:rgba(255,255,255,0.45);line-height:1.65;">Export CSV from Chase, Amex, BofA, Mint, or YNAB and upload above.</div>
  </div>
  <div class="how-card how-02">
    <div style="font-size:11px;color:#4DC8FF;font-weight:700;letter-spacing:0.1em;margin-bottom:10px;">02 &nbsp; ANALYZE</div>
    <div style="font-size:13px;color:rgba(255,255,255,0.45);line-height:1.65;">200+ merchant keywords, ~89% accuracy, weights calibrated to BLS CEX 2023.</div>
  </div>
  <div class="how-card how-03">
    <div style="font-size:11px;color:#FFA532;font-weight:700;letter-spacing:0.1em;margin-bottom:10px;">03 &nbsp; RANK</div>
    <div style="font-size:13px;color:rgba(255,255,255,0.45);line-height:1.65;">Monthly spend × reducibility score. Top 3 opportunities surfaced automatically.</div>
  </div>
  <div class="how-card how-04">
    <div style="font-size:11px;color:#FF6B9D;font-weight:700;letter-spacing:0.1em;margin-bottom:10px;">04 &nbsp; SIMULATE</div>
    <div style="font-size:13px;color:rgba(255,255,255,0.45);line-height:1.65;">Log-normal Monte Carlo · 1,000 paths · 15.6% vol · inflation toggle · prob. of doubling your contribution.</div>
  </div>
</div>""", unsafe_allow_html=True)
    st.info("Upload your bank statement above, or check **'Use demo data'** to see an example.")
