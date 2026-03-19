from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from .models import HistoricalDataset
from .utils import _format_percentage_or_na, format_currency, highlight_changes


def _compound_period_return_pct(return_series: pd.Series) -> float:
    monthly_returns = pd.to_numeric(return_series, errors="coerce").dropna().astype(float) / 100.0
    if monthly_returns.empty:
        return 0.0
    if not (1.0 + monthly_returns).gt(0).all():
        return float("nan")
    return float((np.prod(1.0 + monthly_returns.to_numpy(dtype=float)) - 1.0) * 100.0)


def _aggregate_results_for_display(results_df: pd.DataFrame, view_mode: str) -> pd.DataFrame:
    if str(view_mode) != "Yearly" or results_df.empty:
        return results_df.copy()

    df = results_df.copy()
    df["Period"] = pd.to_datetime(df["Period"])
    yearly_rows: List[Dict[str, object]] = []

    for year, group in df.groupby(df["Period"].dt.year, sort=True):
        yearly_rows.append(
            {
                "Period": str(year),
                "Months Included": int(len(group)),
                "Balance (USD)": float(pd.to_numeric(group["Balance (USD)"], errors="coerce").iloc[-1]),
                "Portfolio Total Return (%)": _compound_period_return_pct(group["Portfolio Total Return (%)"]),
                "Dividend Yield (USD)": float(pd.to_numeric(group["Dividend Yield (USD)"], errors="coerce").sum()),
                "Withdrawal (USD)": float(pd.to_numeric(group["Withdrawal (USD)"], errors="coerce").sum()),
                "Total Withdrawal + Dividend (USD)": float(pd.to_numeric(group["Total Withdrawal + Dividend (USD)"], errors="coerce").sum()),
                "Contribution (USD)": float(pd.to_numeric(group["Contribution (USD)"], errors="coerce").sum()),
                "Dividend Reinvestment (USD)": float(pd.to_numeric(group["Dividend Reinvestment (USD)"], errors="coerce").sum()),
                "Trading Cost (USD)": float(pd.to_numeric(group["Trading Cost (USD)"], errors="coerce").sum()),
                "Fee (USD)": float(pd.to_numeric(group["Fee (USD)"], errors="coerce").sum()),
                "Real Balance (USD)": float(pd.to_numeric(group["Real Balance (USD)"], errors="coerce").iloc[-1]),
                "Rebalanced": bool(group["Rebalanced"].astype(bool).any()),
                "Ending Weights": str(group["Ending Weights"].iloc[-1]),
            }
        )

    return pd.DataFrame(yearly_rows)


def _aggregate_benchmark_for_display(comparison_df: pd.DataFrame, view_mode: str) -> pd.DataFrame:
    if str(view_mode) != "Yearly" or comparison_df.empty:
        return comparison_df.copy()

    df = comparison_df.copy()
    df["Period"] = pd.to_datetime(df["Period"])
    yearly_rows: List[Dict[str, object]] = []

    for year, group in df.groupby(df["Period"].dt.year, sort=True):
        portfolio_total_return = _compound_period_return_pct(group["Portfolio Total Return (%)"])
        benchmark_total_return = _compound_period_return_pct(group["Benchmark Total Return (%)"])
        ending_portfolio_balance = float(pd.to_numeric(group["Balance (USD)"], errors="coerce").iloc[-1])
        ending_benchmark_balance = float(pd.to_numeric(group["Benchmark Balance (USD)"], errors="coerce").iloc[-1])
        relative_wealth = ((ending_portfolio_balance / ending_benchmark_balance) - 1.0) * 100.0 if ending_benchmark_balance != 0 else float("nan")
        excess_return = portfolio_total_return - benchmark_total_return if math.isfinite(portfolio_total_return) and math.isfinite(benchmark_total_return) else float("nan")
        yearly_rows.append(
            {
                "Period": str(year),
                "Months Included": int(len(group)),
                "Balance (USD)": ending_portfolio_balance,
                "Portfolio Total Return (%)": portfolio_total_return,
                "Benchmark Balance (USD)": ending_benchmark_balance,
                "Benchmark Total Return (%)": benchmark_total_return,
                "Excess Return (%)": excess_return,
                "Relative Wealth (%)": relative_wealth,
            }
        )

    return pd.DataFrame(yearly_rows)


def _format_benchmark_display_table(comparison_df: pd.DataFrame, view_mode: str) -> pd.DataFrame:
    display_df = _aggregate_benchmark_for_display(comparison_df, view_mode=view_mode)
    if display_df.empty:
        return display_df

    display_df = display_df.copy()
    if str(view_mode) == "Monthly":
        display_df["Period"] = pd.to_datetime(display_df["Period"]).dt.strftime("%Y-%m")
    else:
        display_df["Period"] = display_df["Period"].astype(str)

    display_cols = [
        "Period",
        "Balance (USD)",
        "Portfolio Total Return (%)",
        "Benchmark Balance (USD)",
        "Benchmark Total Return (%)",
        "Excess Return (%)",
        "Relative Wealth (%)",
    ]
    if "Months Included" in display_df.columns:
        display_cols.insert(1, "Months Included")

    for currency_col in ["Balance (USD)", "Benchmark Balance (USD)"]:
        if currency_col in display_df.columns:
            display_df[currency_col] = display_df[currency_col].apply(format_currency)

    for pct_col in [
        "Portfolio Total Return (%)",
        "Benchmark Total Return (%)",
        "Excess Return (%)",
        "Relative Wealth (%)",
    ]:
        if pct_col in display_df.columns:
            display_df[pct_col] = display_df[pct_col].apply(_format_percentage_or_na)

    return display_df[display_cols]


def render_metric_tabs(metrics: Dict[str, float]) -> None:
    tab1, tab2, tab3 = st.tabs(["Overview", "Risk", "Cashflow"])
    with tab1:
        cols = st.columns(4)
        cols[0].metric("Final Balance", f"${format_currency(metrics['Final Balance'])}")
        cols[1].metric("Real Final Balance", f"${format_currency(metrics['Real Final Balance'])}")
        cols[2].metric("CAGR", f"{metrics['CAGR']:.2f}%" if not math.isnan(metrics["CAGR"]) else "N/A")
        cols[3].metric("Max Drawdown", f"{metrics['Max Drawdown']:.2f}%")
        cols2 = st.columns(4)
        cols2[0].metric("Best Month", f"{metrics['Best Month']:.2f}%")
        cols2[1].metric("Worst Month", f"{metrics['Worst Month']:.2f}%")
        cols2[2].metric("Years Simulated", f"{metrics['Years Simulated']:.1f}")
        cols2[3].metric("Rebalanced Months", f"{int(metrics['Rebalanced Months'])}")
    with tab2:
        cols = st.columns(4)
        cols[0].metric("Volatility", f"{metrics['Volatility']:.2f}%")
        cols[1].metric("Sharpe", f"{metrics['Sharpe Ratio']:.2f}")
        cols[2].metric("Sortino", f"{metrics['Sortino Ratio']:.2f}")
        cols[3].metric("Downside Dev", f"{metrics['Downside Deviation']:.2f}%")
        cols2 = st.columns(4)
        cols2[0].metric("CVaR 95", f"{metrics['CVaR 95']:.2f}%")
        cols2[1].metric("VaR 95", f"{metrics['Parametric VaR 95']:.2f}%")
        cols2[2].metric("Ulcer Index", f"{metrics['Ulcer Index']:.2f}")
        cols2[3].metric("Calmar", f"{metrics['Calmar Ratio']:.2f}")
        st.caption(f"Recovery time: {metrics['Recovery Years']:.1f} years")
    with tab3:
        cols = st.columns(5)
        cols[0].metric("Total Withdrawals", f"${format_currency(metrics['Total Withdrawals'])}")
        cols[1].metric("Total Dividends", f"${format_currency(metrics['Total Dividends'])}")
        cols[2].metric("Total Contributions", f"${format_currency(metrics['Total Contributions'])}")
        cols[3].metric("Trading Costs", f"${format_currency(metrics['Total Trading Costs'])}")
        cols[4].metric("Net Cash Delivered", f"${format_currency(metrics['Net Cash Delivered'])}")
        st.caption(f"Withdrawal shortfall months: {int(metrics['Withdrawal Shortfall Months'])}")


def render_table(results_df: pd.DataFrame, view_mode: str = "Monthly") -> None:
    display_source = _aggregate_results_for_display(results_df, view_mode=view_mode)
    if display_source.empty:
        st.info("No rows available for the selected table view.")
        return

    display_cols = [
        "Period",
        "Balance (USD)",
        "Portfolio Total Return (%)",
        "Dividend Yield (USD)",
        "Withdrawal (USD)",
        "Total Withdrawal + Dividend (USD)",
        "Contribution (USD)",
        "Dividend Reinvestment (USD)",
        "Trading Cost (USD)",
        "Fee (USD)",
        "Real Balance (USD)",
        "Rebalanced",
        "Ending Weights",
    ]
    if "Months Included" in display_source.columns:
        display_cols.insert(1, "Months Included")

    display_df = display_source[display_cols].copy()
    if str(view_mode) == "Monthly":
        display_df["Period"] = pd.to_datetime(display_df["Period"]).dt.strftime("%Y-%m")
    else:
        display_df["Period"] = display_df["Period"].astype(str)

    for currency_col in [
        "Balance (USD)",
        "Dividend Yield (USD)",
        "Withdrawal (USD)",
        "Total Withdrawal + Dividend (USD)",
        "Contribution (USD)",
        "Dividend Reinvestment (USD)",
        "Trading Cost (USD)",
        "Fee (USD)",
        "Real Balance (USD)",
    ]:
        if currency_col in display_df.columns:
            display_df[currency_col] = display_df[currency_col].apply(format_currency)

    display_df["Portfolio Total Return (%)"] = display_df["Portfolio Total Return (%)"].apply(_format_percentage_or_na)
    styled_df = display_df.style.applymap(highlight_changes, subset=["Portfolio Total Return (%)"])
    st.write(styled_df.to_html(index=False), unsafe_allow_html=True)


def render_overlap_alerts(dataset: HistoricalDataset, overlap_warning_lines: Sequence[str]) -> None:
    if not overlap_warning_lines:
        return
    message = "\n\n".join(f"- {line}" for line in overlap_warning_lines)
    st.warning("Overlap / sample-window caution\n\n" + message)


def render_input_change_warning() -> None:
    st.info(
        "Inputs have changed since the last successful run. Results below still reflect the last completed run until you click Run simulation again."
    )
