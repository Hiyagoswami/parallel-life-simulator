"""
report_export.py — Generates a downloadable PDF summary report and
CSV data export of the user's analysis results.

This gives the project an "after I close this tab" story: the user's
analysis survives the session as an actual artifact they can keep, print,
or share — not just pixels on a page that vanish on refresh.
"""

import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from datetime import datetime


def generate_pdf_report(total_spent, n_months, by_category, ranked_scenarios,
                        computed_scenarios, return_rate, years, top_merchants_df,
                        username="User"):
    """
    Builds a multi-page PDF report summarizing the analysis:
      Page 1 — Summary KPIs + spending breakdown
      Page 2 — Ranked scenarios + Monte Carlo outcomes
      Page 3 — Top merchants table

    Returns BytesIO buffer ready for st.download_button.
    """
    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # ── Page 1: Summary ──────────────────────────────────────────────
        fig1 = plt.figure(figsize=(8.5, 11))
        fig1.suptitle("Parallel Life Simulator — Analysis Report",
                      fontsize=16, fontweight="bold", y=0.97)
        fig1.text(0.1, 0.93, f"Generated {datetime.now().strftime('%B %d, %Y')}",
                  fontsize=9, color="#666")

        ax_summary = fig1.add_axes([0.1, 0.55, 0.8, 0.3])
        ax_summary.axis("off")
        summary_lines = [
            f"Total Spent:           ${total_spent:,.2f}  ({n_months} months)",
            f"Monthly Average:       ${total_spent/n_months:,.2f}",
            f"Top Category:          {by_category.index[0]}  (${by_category.iloc[0]:,.2f})",
            f"Annual Return Used:    {return_rate*100:.0f}%",
            f"Projection Horizon:    {years} years",
        ]
        for i, line in enumerate(summary_lines):
            ax_summary.text(0, 0.9 - i*0.18, line, fontsize=11, family="monospace")

        ax_pie = fig1.add_axes([0.1, 0.08, 0.8, 0.42])
        ax_pie.pie(by_category.values, labels=by_category.index, autopct="%1.0f%%",
                  colors=plt.cm.Set2.colors, startangle=140)
        ax_pie.set_title("Spending by Category", fontsize=12, pad=10)

        pdf.savefig(fig1)
        plt.close(fig1)

        # ── Page 2: Scenarios ────────────────────────────────────────────
        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.suptitle("Your Ranked Savings Scenarios", fontsize=14, fontweight="bold", y=0.97)

        ax_table = fig2.add_axes([0.05, 0.55, 0.9, 0.35])
        ax_table.axis("off")
        table_rows = []
        for i, sc in enumerate(ranked_scenarios):
            key = str(i+1)
            csc = computed_scenarios[key]
            table_rows.append([
                f"#{i+1}", sc["action"], f"{int(sc['reducibility']*100)}%",
                f"${sc['monthly_saving']:,.0f}", f"${csc['compound']:,.0f}"
            ])
        tbl = ax_table.table(
            cellText=table_rows,
            colLabels=["Rank", "Action", "Reducibility", "Monthly Saving", f"{years}yr Outcome"],
            loc="center", cellLoc="left"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 2)

        pdf.savefig(fig2)
        plt.close(fig2)

        # ── Page 3: Top merchants ────────────────────────────────────────
        if top_merchants_df is not None and len(top_merchants_df) > 0:
            fig3 = plt.figure(figsize=(8.5, 11))
            fig3.suptitle("Top 10 Merchants", fontsize=14, fontweight="bold", y=0.97)
            ax_m = fig3.add_axes([0.05, 0.4, 0.9, 0.5])
            ax_m.axis("off")
            merch_rows = top_merchants_df.reset_index().values.tolist()
            tbl2 = ax_m.table(
                cellText=merch_rows,
                colLabels=list(top_merchants_df.reset_index().columns),
                loc="center", cellLoc="left"
            )
            tbl2.auto_set_font_size(False)
            tbl2.set_fontsize(8)
            tbl2.scale(1, 1.8)
            pdf.savefig(fig3)
            plt.close(fig3)

    buf.seek(0)
    return buf


def generate_csv_export(by_category, ranked_scenarios, computed_scenarios,
                        n_months, years, return_rate):
    """Returns a CSV string summarizing category spend and scenario outcomes."""
    rows = []
    rows.append(["Category", "Total Spent", "Monthly Avg", "Reducibility",
                "Monthly Saving Opportunity", f"{years}yr Outcome @ {return_rate*100:.0f}%"])

    ranked_by_cat = {r["category"]: r for r in ranked_scenarios}
    computed_by_idx = {str(i+1): r["category"] for i, r in enumerate(ranked_scenarios)}

    for cat, total in by_category.items():
        if cat in ranked_by_cat:
            r = ranked_by_cat[cat]
            idx = [k for k, v in computed_by_idx.items() if v == cat]
            outcome = computed_scenarios[idx[0]]["compound"] if idx else ""
            rows.append([cat, f"{total:.2f}", f"{total/n_months:.2f}",
                        f"{int(r['reducibility']*100)}%",
                        f"{r['monthly_saving']:.2f}", f"{outcome:.2f}" if outcome else ""])
        else:
            rows.append([cat, f"{total:.2f}", f"{total/n_months:.2f}", "N/A", "N/A", "N/A"])

    output = io.StringIO()
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(output, index=False)
    return output.getvalue()
