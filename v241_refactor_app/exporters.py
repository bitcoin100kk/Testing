from __future__ import annotations

import io
from dataclasses import asdict
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from .models import AssetConfig, PortfolioInputs

def simulation_to_csv(results_df: pd.DataFrame) -> bytes:
    return results_df.to_csv(index=False).encode("utf-8")

def _sheet_name(name: str) -> str:
    cleaned = "".join("_" if ch in '[]:*?/\\' else ch for ch in name)
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
) -> bytes:
    buffer = io.BytesIO()
    settings_df = pd.DataFrame([asdict(portfolio_inputs)])
    settings_df.insert(0, "year_range_start", int(year_range[0]))
    settings_df.insert(1, "year_range_end", int(year_range[1]))

    assets_df = pd.DataFrame([asdict(asset) for asset in assets])
    metrics_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
    weighted_returns_df = weighted_returns.rename("Target Weighted Monthly Price Return (%)").rename_axis("Period").reset_index()
    weighted_divs_df = weighted_divs.rename("Target Weighted Monthly Dividend Yield (%)").rename_axis("Period").reset_index()

    sheets = [
        ("Summary Metrics", metrics_df, False),
        ("Simulation Settings", settings_df, False),
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
