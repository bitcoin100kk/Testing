from __future__ import annotations

import io
from dataclasses import asdict
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from .models import AssetConfig, PortfolioInputs


def simulation_to_csv(results_df: pd.DataFrame) -> bytes:
    return results_df.to_csv(index=False).encode("utf-8")


def _sheet_name(name: str) -> str:
    cleaned = ''.join('_' if ch in "[]:*?/\\" else ch for ch in name)
    return cleaned[:31] if cleaned else "Sheet1"


def _autosize_worksheet(worksheet, dataframe: pd.DataFrame, include_index: bool = False) -> None:
    start_col = 2 if include_index else 1
    if include_index:
        try:
            worksheet.column_dimensions["A"].width = max(12, len(str(dataframe.index.name or "Index")) + 2)
        except Exception:
            pass
    for offset, column in enumerate(dataframe.columns, start=start_col):
        letter = worksheet.cell(row=1, column=offset).column_letter
        max_len = len(str(column))
        if not dataframe.empty:
            series = dataframe[column].astype(str)
            max_len = max(max_len, int(series.map(len).max()))
        worksheet.column_dimensions[letter].width = min(max(max_len + 2, 12), 40)


def _build_run_provenance_df(
    *,
    portfolio_inputs: PortfolioInputs,
    year_range: Tuple[int, int],
    diagnostics_df: Optional[pd.DataFrame],
    forward_audit_df: Optional[pd.DataFrame],
    decision_objective: Optional[str],
    fragility_df: Optional[pd.DataFrame],
    recommendation_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    overlap_start = overlap_end = ""
    overlap_months = 0
    fallback_count = 0
    fallback_tickers = ""
    filtered_dividend_events = 0
    clipped_dividend_months = 0
    data_sources = ""
    if diagnostics_df is not None and not diagnostics_df.empty:
        overlap_start = str(diagnostics_df.get("Overlap Start", pd.Series(dtype=object)).iloc[0]) if "Overlap Start" in diagnostics_df.columns else ""
        overlap_end = str(diagnostics_df.get("Overlap End", pd.Series(dtype=object)).iloc[0]) if "Overlap End" in diagnostics_df.columns else ""
        overlap_months = int(pd.to_numeric(diagnostics_df.get("Overlap Months", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if "Overlap Months" in diagnostics_df.columns else 0
        fallback_mask = diagnostics_df.get("Fallback Used", pd.Series(dtype=bool)).fillna(False).astype(bool) if "Fallback Used" in diagnostics_df.columns else pd.Series(dtype=bool)
        fallback_count = int(fallback_mask.sum())
        if fallback_count and "Ticker" in diagnostics_df.columns:
            fallback_tickers = ", ".join(sorted(diagnostics_df.loc[fallback_mask, "Ticker"].astype(str).tolist()))
        if "Filtered Dividend Events" in diagnostics_df.columns:
            filtered_dividend_events = int(pd.to_numeric(diagnostics_df["Filtered Dividend Events"], errors="coerce").fillna(0).sum())
        if "Clipped Dividend Months" in diagnostics_df.columns:
            clipped_dividend_months = int(pd.to_numeric(diagnostics_df["Clipped Dividend Months"], errors="coerce").fillna(0).sum())
        if "Data Source" in diagnostics_df.columns:
            data_sources = ", ".join(sorted({str(v) for v in diagnostics_df["Data Source"].dropna().tolist()}))

    forward_mode = str(getattr(portfolio_inputs, "monte_carlo_forward_mode", "Historical Base"))
    forward_bucket_summary = ""
    if forward_audit_df is not None and not forward_audit_df.empty and "Bucket" in forward_audit_df.columns:
        forward_bucket_summary = ", ".join(sorted({str(v) for v in forward_audit_df["Bucket"].dropna().tolist()}))

    rows = [
        {"Section": "Run", "Field": "Selected Year Range", "Value": f"{int(year_range[0])}-{int(year_range[1])}"},
        {"Section": "Data", "Field": "Overlap Window", "Value": f"{overlap_start} to {overlap_end}" if overlap_start and overlap_end else "N/A"},
        {"Section": "Data", "Field": "Overlap Months", "Value": int(overlap_months)},
        {"Section": "Data", "Field": "Data Sources Used", "Value": data_sources or "N/A"},
        {"Section": "Data", "Field": "Fallback Assets", "Value": fallback_tickers or "None"},
        {"Section": "Data", "Field": "Fallback Asset Count", "Value": int(fallback_count)},
        {"Section": "Data", "Field": "Filtered Dividend Events", "Value": int(filtered_dividend_events)},
        {"Section": "Data", "Field": "Clipped Dividend Months", "Value": int(clipped_dividend_months)},
        {"Section": "Monte Carlo", "Field": "Bootstrap Method", "Value": str(getattr(portfolio_inputs, "monte_carlo_bootstrap_method", ""))},
        {"Section": "Monte Carlo", "Field": "Regime Mode", "Value": str(getattr(portfolio_inputs, "monte_carlo_regime_mode", ""))},
        {"Section": "Monte Carlo", "Field": "Simulations", "Value": int(getattr(portfolio_inputs, "monte_carlo_sims", 0))},
        {"Section": "Monte Carlo", "Field": "Years", "Value": int(getattr(portfolio_inputs, "monte_carlo_years", 0))},
        {"Section": "Monte Carlo", "Field": "Seed", "Value": int(getattr(portfolio_inputs, "monte_carlo_seed", 0))},
        {"Section": "Monte Carlo", "Field": "Block Size (Months)", "Value": int(getattr(portfolio_inputs, "monte_carlo_block_size_months", 0))},
        {"Section": "Monte Carlo", "Field": "Regime Window (Months)", "Value": int(getattr(portfolio_inputs, "monte_carlo_regime_window_months", 0))},
        {"Section": "Monte Carlo", "Field": "Forward Mode", "Value": forward_mode},
        {"Section": "Monte Carlo", "Field": "Forward Return Haircut (%)", "Value": float(getattr(portfolio_inputs, "monte_carlo_return_haircut_pct", 0.0))},
        {"Section": "Monte Carlo", "Field": "Forward Return Shift (%)", "Value": float(getattr(portfolio_inputs, "monte_carlo_return_shift_pct", 0.0))},
        {"Section": "Monte Carlo", "Field": "Forward Vol Multiplier", "Value": float(getattr(portfolio_inputs, "monte_carlo_vol_multiplier", 1.0))},
        {"Section": "Monte Carlo", "Field": "Forward Dividend Multiplier", "Value": float(getattr(portfolio_inputs, "monte_carlo_dividend_multiplier", 1.0))},
        {"Section": "Monte Carlo", "Field": "Forward Buckets Present", "Value": forward_bucket_summary or "N/A"},
        {"Section": "Decision Lab", "Field": "Decision Objective", "Value": decision_objective or "Not prepared"},
        {"Section": "Decision Lab", "Field": "Fragility Included", "Value": bool(fragility_df is not None and not fragility_df.empty)},
        {"Section": "Decision Lab", "Field": "Policy Recommendation Included", "Value": bool(recommendation_df is not None and not recommendation_df.empty)},
    ]
    return pd.DataFrame(rows)


def _build_decision_settings_df(decision_objective: Optional[str], fragility_settings_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if not decision_objective and (fragility_settings_df is None or fragility_settings_df.empty):
        return None
    payload = {
        "Decision Objective": str(decision_objective) if decision_objective else "Not prepared",
        "Policy Comparison Scope": "Constrained candidate set around current plan",
        "Fragility Grid Notes": "Return-shift axis is forced into explicit custom forward-stress mode.",
    }
    if fragility_settings_df is not None and not fragility_settings_df.empty:
        first = fragility_settings_df.iloc[0]
        payload["Fragility Mode"] = first.get("Fragility Mode", "N/A")
        payload["Fragility Grid Shape"] = first.get("Grid Shape", "N/A")
        payload["Fragility MC Sims Per Run"] = first.get("MC Sims Per Run", "N/A")
    return pd.DataFrame([payload])


def export_full_simulation_workbook(
    *,
    results_df: pd.DataFrame,
    component_df: pd.DataFrame,
    selected_returns_df: pd.DataFrame,
    selected_divs_df: pd.DataFrame,
    weighted_returns: pd.Series,
    weighted_divs: pd.Series,
    metrics: Dict[str, float],
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    year_range: Tuple[int, int],
    diagnostics_df: Optional[pd.DataFrame] = None,
    risk_table: Optional[pd.DataFrame] = None,
    rolling3_df: Optional[pd.DataFrame] = None,
    rolling5_df: Optional[pd.DataFrame] = None,
    benchmark_comparison_df: Optional[pd.DataFrame] = None,
    benchmark_results_df: Optional[pd.DataFrame] = None,
    mc_percentiles_df: Optional[pd.DataFrame] = None,
    mc_summary_df: Optional[pd.DataFrame] = None,
    mc_paths_df: Optional[pd.DataFrame] = None,
    mc_convergence_df: Optional[pd.DataFrame] = None,
    scenario_comparison_df: Optional[pd.DataFrame] = None,
    forward_audit_df: Optional[pd.DataFrame] = None,
    fragility_df: Optional[pd.DataFrame] = None,
    fragility_pivot_df: Optional[pd.DataFrame] = None,
    policy_df: Optional[pd.DataFrame] = None,
    recommendation_df: Optional[pd.DataFrame] = None,
    fragility_settings_df: Optional[pd.DataFrame] = None,
    decision_objective: Optional[str] = None,
) -> bytes:
    buffer = io.BytesIO()
    settings_df = pd.DataFrame([asdict(portfolio_inputs)])
    settings_df.insert(0, "year_range_start", int(year_range[0]))
    settings_df.insert(1, "year_range_end", int(year_range[1]))

    assets_df = pd.DataFrame([asdict(asset) for asset in assets])
    metrics_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
    weighted_returns_df = weighted_returns.rename("Target Weighted Monthly Price Return (%)").rename_axis("Period").reset_index()
    weighted_divs_df = weighted_divs.rename("Target Weighted Monthly Dividend Yield (%)").rename_axis("Period").reset_index()
    run_provenance_df = _build_run_provenance_df(
        portfolio_inputs=portfolio_inputs,
        year_range=year_range,
        diagnostics_df=diagnostics_df,
        forward_audit_df=forward_audit_df,
        decision_objective=decision_objective,
        fragility_df=fragility_df,
        recommendation_df=recommendation_df,
    )
    decision_settings_df = _build_decision_settings_df(decision_objective, fragility_settings_df)

    sheets = [
        ("Summary Metrics", metrics_df, False),
        ("Simulation Settings", settings_df, False),
        ("Run Provenance", run_provenance_df, False),
        ("Assets", assets_df, False),
        ("Results", results_df, False),
        ("Per Asset Data", component_df, False),
        ("Asset Returns Matrix", selected_returns_df, True),
        ("Asset Dividend Matrix", selected_divs_df, True),
        ("Weighted Returns", weighted_returns_df, False),
        ("Weighted Dividends", weighted_divs_df, False),
    ]

    optional_sheets = [
        ("Risk Metrics", risk_table, False),
        ("Rolling 3Y Returns", rolling3_df, False),
        ("Rolling 5Y Returns", rolling5_df, False),
        ("Benchmark Comparison", benchmark_comparison_df, False),
        ("Benchmark Results", benchmark_results_df, False),
        ("Scenario Comparison", scenario_comparison_df, False),
        ("Data Diagnostics", diagnostics_df, False),
        ("Forward Audit", forward_audit_df, False),
        ("Decision Lab Settings", decision_settings_df, False),
        ("Fragility Settings", fragility_settings_df, False),
        ("Fragility Summary", fragility_df, False),
        ("Fragility Matrix", fragility_pivot_df, True),
        ("Policy Comparison", policy_df, False),
        ("Policy Recommendation", recommendation_df, False),
        ("Monte Carlo Percentiles", mc_percentiles_df, False),
        ("Monte Carlo Summary", mc_summary_df, False),
        ("Monte Carlo Paths", mc_paths_df, False),
        ("Monte Carlo Convergence", mc_convergence_df, False),
    ]

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df, include_index in sheets + optional_sheets:
            if df is None:
                continue
            if isinstance(df, pd.Series):
                df = df.to_frame()
            if not isinstance(df, pd.DataFrame):
                continue
            sheet_name = _sheet_name(name)
            df.to_excel(writer, sheet_name=sheet_name, index=include_index)
            worksheet = writer.book[sheet_name]
            _autosize_worksheet(worksheet, df, include_index=include_index)
            worksheet.freeze_panes = "A2"

    buffer.seek(0)
    return buffer.getvalue()
