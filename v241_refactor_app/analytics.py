from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .models import HistoricalDataset
from .utils import _annual_pct_to_monthly_decimal

def add_real_dollar_columns(df: pd.DataFrame, inflation_rate: float) -> pd.DataFrame:
    out = df.copy()
    monthly_inflation_rate = _annual_pct_to_monthly_decimal(inflation_rate) if inflation_rate is not None else 0.0
    real_balance = []
    real_cashflow = []
    for i, (_, row) in enumerate(out.iterrows()):
        discount = (1.0 + monthly_inflation_rate) ** i
        real_balance.append(row["Balance (USD)"] / discount if discount else row["Balance (USD)"])
        real_cashflow.append(row["Total Withdrawal + Dividend (USD)"] / discount if discount else row["Total Withdrawal + Dividend (USD)"])
    out["Real Balance (USD)"] = real_balance
    out["Real Total Withdrawal + Dividend (USD)"] = real_cashflow
    return out

def max_drawdown(balance_series: pd.Series) -> float:
    if balance_series.empty:
        return 0.0
    running_max = balance_series.cummax().replace(0, pd.NA)
    drawdowns = (balance_series - running_max) / running_max
    return float(drawdowns.min() * 100.0) if not drawdowns.empty else 0.0

def compute_risk_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    returns = pd.to_numeric(results_df.get("Portfolio Total Return (%)", pd.Series(dtype=float)), errors="coerce").dropna() / 100.0
    if returns.empty:
        return {
            "Volatility": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Downside Deviation": 0.0,
            "Recovery Years": 0.0,
            "CVaR 95": 0.0,
            "Ulcer Index": 0.0,
            "Parametric VaR 95": 0.0,
            "Calmar Ratio": 0.0,
        }

    mean_monthly = returns.mean()
    std_monthly = returns.std(ddof=0)
    volatility = std_monthly * math.sqrt(12.0) * 100.0
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0) if not downside.empty else 0.0
    downside_dev = downside_std * math.sqrt(12.0) * 100.0 if downside_std > 0 else 0.0
    sharpe = ((mean_monthly * 12.0) / (std_monthly * math.sqrt(12.0))) if std_monthly > 0 else 0.0
    sortino = ((mean_monthly * 12.0) / (downside_std * math.sqrt(12.0))) if downside_std > 0 else 0.0

    balances = pd.to_numeric(results_df["Balance (USD)"], errors="coerce")
    rolling_max = balances.cummax()
    underwater = balances < rolling_max
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)

    drawdown_pct = ((balances / rolling_max) - 1.0).fillna(0.0) * 100.0
    ulcer_index = float(np.sqrt(np.mean(np.square(drawdown_pct.to_numpy(dtype=float))))) if not drawdown_pct.empty else 0.0
    q05 = returns.quantile(0.05)
    tail = returns[returns <= q05]
    cvar_95 = float(tail.mean() * 100.0) if not tail.empty else 0.0

    z_95 = 1.6448536269514722
    parametric_var_95 = -max((z_95 * std_monthly) - mean_monthly, 0.0) * 100.0
    if len(returns) > 0 and (1.0 + returns).gt(0).all():
        cagr = (np.prod(1.0 + returns.to_numpy()) ** (12.0 / len(returns)) - 1.0) * 100.0
    else:
        cagr = float("nan")
    max_dd_abs = abs(max_drawdown(balances))
    calmar = (cagr / max_dd_abs) if max_dd_abs > 0 and not math.isnan(cagr) else 0.0

    return {
        "Volatility": float(volatility),
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Downside Deviation": float(downside_dev),
        "Recovery Years": float(longest / 12.0),
        "CVaR 95": cvar_95,
        "Ulcer Index": ulcer_index,
        "Parametric VaR 95": float(parametric_var_95),
        "Calmar Ratio": float(calmar),
    }

def compute_summary_metrics(results_df: pd.DataFrame, periods: Sequence[pd.Timestamp]) -> Dict[str, float]:
    if results_df.empty:
        return {}
    monthly_returns = pd.to_numeric(results_df["Portfolio Total Return (%)"], errors="coerce").dropna() / 100.0
    n_periods = max(len(monthly_returns), 1)
    years_simulated = len(results_df) / 12.0
    if n_periods > 0 and (1.0 + monthly_returns).gt(0).all():
        cagr = (np.prod(1.0 + monthly_returns.to_numpy()) ** (12.0 / n_periods) - 1.0) * 100.0
    else:
        cagr = float("nan")

    shortfall_months = int((pd.to_numeric(results_df["Withdrawal Shortfall (USD)"], errors="coerce").fillna(0.0) > 1e-9).sum())
    survived = float(results_df["Balance (USD)"].min()) > 0 and shortfall_months == 0
    metrics = {
        "Final Balance": float(results_df.iloc[-1]["Balance (USD)"]),
        "Real Final Balance": float(results_df.iloc[-1].get("Real Balance (USD)", results_df.iloc[-1]["Balance (USD)"])),
        "Total Withdrawals": float(results_df["Withdrawal (USD)"].sum()),
        "Total Dividends": float(results_df["Dividend Yield (USD)"].sum()),
        "Total Contributions": float(results_df["Contribution (USD)"].sum()),
        "Net Cash Delivered": float(results_df["Total Withdrawal + Dividend (USD)"].sum()),
        "Total Trading Costs": float(results_df["Trading Cost (USD)"].sum()),
        "CAGR": float(cagr),
        "Best Month": float(results_df["Portfolio Total Return (%)"].max()),
        "Worst Month": float(results_df["Portfolio Total Return (%)"].min()),
        "Max Drawdown": max_drawdown(results_df["Balance (USD)"]),
        "Months Simulated": float(len(results_df)),
        "Years Simulated": float(years_simulated),
        "Portfolio Survived": 1.0 if survived else 0.0,
        "Rebalanced Months": float(results_df["Rebalanced"].sum()) if "Rebalanced" in results_df.columns else 0.0,
        "Withdrawal Shortfall Months": float(shortfall_months),
    }
    metrics.update(compute_risk_metrics(results_df))
    return metrics

def compute_rolling_returns(results_df: pd.DataFrame, window_years: int) -> pd.DataFrame:
    window_months = window_years * 12
    if len(results_df) < window_months:
        return pd.DataFrame(columns=["End Period", f"Rolling {window_years}Y Return (%)"])
    returns = pd.to_numeric(results_df["Portfolio Total Return (%)"], errors="coerce").fillna(0.0) / 100.0
    periods = pd.to_datetime(results_df["Period"])
    rows = []
    for i in range(window_months - 1, len(returns)):
        window_slice = returns.iloc[i - window_months + 1 : i + 1]
        if (1.0 + window_slice).gt(0).all():
            annualized = (np.prod(1.0 + window_slice.to_numpy()) ** (12.0 / window_months) - 1.0) * 100.0
        else:
            annualized = np.nan
        rows.append({"End Period": periods.iloc[i], f"Rolling {window_years}Y Return (%)": annualized})
    return pd.DataFrame(rows)

def build_overlap_warning_lines(dataset: HistoricalDataset) -> List[str]:
    diagnostics_df = pd.DataFrame(dataset.diagnostics)
    lines: List[str] = [
        f"Common usable sample: {dataset.overlap_start.strftime('%Y-%m')} to {dataset.overlap_end.strftime('%Y-%m')} ({dataset.overlap_months} monthly points)."
    ]
    if dataset.overlap_months < 60:
        lines.append(
            "Warning: the overlapping history is shorter than 5 years, so drawdown, CAGR, and withdrawal conclusions are based on a limited sample."
        )
    if not diagnostics_df.empty and "History Lost To Overlap (Months)" in diagnostics_df.columns:
        reduced = diagnostics_df[diagnostics_df["History Lost To Overlap (Months)"] > 0].copy()
        if not reduced.empty:
            reduced = reduced.sort_values("History Lost To Overlap (Months)", ascending=False)
            top_rows = [
                f"{row['Ticker']} lost {int(row['History Lost To Overlap (Months)'])} months"
                for _, row in reduced.head(3).iterrows()
            ]
            lines.append("Assets shortening the common sample most: " + ", ".join(top_rows) + ".")
    if dataset.overlap_months < 180:
        lines.append(
            "Caution: the common sample may be dominated by a single market regime. Treat historical CAGR and Monte Carlo percentiles as regime-conditional, not as neutral forward expectations."
        )
    return lines


def build_overlap_summary_df(dataset: HistoricalDataset) -> pd.DataFrame:
    diagnostics_df = pd.DataFrame(dataset.diagnostics)
    if diagnostics_df.empty:
        return diagnostics_df
    desired_columns = [
        "Ticker",
        "Type",
        "Allocation (%)",
        "First Month",
        "Last Month",
        "Monthly Points",
        "Overlap Start",
        "Overlap End",
        "Overlap Months",
        "History Lost To Overlap (Months)",
    ]
    existing_columns = [col for col in desired_columns if col in diagnostics_df.columns]
    return diagnostics_df[existing_columns].copy()
