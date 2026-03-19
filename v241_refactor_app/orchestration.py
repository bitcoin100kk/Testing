from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from .analytics import (
    build_overlap_summary_df,
    build_overlap_warning_lines,
    compute_rolling_returns,
    compute_summary_metrics,
)
from .data_layer import prepare_historical_dataset
from .engine import run_historical_simulation
from .exporters import export_full_simulation_workbook
from .models import AssetConfig, HistoricalSelection, PortfolioInputs, RunArtifacts, RunSnapshot
from .monte_carlo import simulate_monte_carlo
from .utils import (
    build_core_signature,
    deserialize_assets,
    deserialize_portfolio_inputs,
    serialize_assets,
    serialize_portfolio_inputs,
)


@st.cache_data(ttl=3600, show_spinner=False)
def run_historical_simulation_cached(
    portfolio_input_dict: Dict[str, object],
    asset_specs: Tuple[Tuple[str, float, str], ...],
    token: str,
    selected_range: Tuple[int, int],
) -> Dict[str, object]:
    portfolio_inputs = deserialize_portfolio_inputs(portfolio_input_dict)
    assets = deserialize_assets(asset_specs)
    dataset = prepare_historical_dataset(asset_specs, token)
    selection = run_historical_simulation(
        portfolio_inputs=portfolio_inputs,
        assets=assets,
        dataset=dataset,
        selected_range=selected_range,
    )
    return {"dataset": dataset, "selection": selection}


@st.cache_data(ttl=3600, show_spinner=False)
def simulate_monte_carlo_cached(
    portfolio_input_dict: Dict[str, object],
    asset_specs: Tuple[Tuple[str, float, str], ...],
    selected_returns_df: pd.DataFrame,
    selected_divs_df: pd.DataFrame,
    start_period: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    portfolio_inputs = deserialize_portfolio_inputs(portfolio_input_dict)
    assets = deserialize_assets(asset_specs)
    percentiles_df, summary_df, paths_df, convergence_df = simulate_monte_carlo(
        portfolio_inputs=portfolio_inputs,
        assets=assets,
        historical_returns_df=selected_returns_df,
        historical_dividends_df=selected_divs_df,
        start_period=start_period,
    )
    return {
        "percentiles_df": percentiles_df,
        "summary_df": summary_df,
        "paths_df": paths_df,
        "convergence_df": convergence_df,
    }


def make_run_snapshot(portfolio_inputs: PortfolioInputs, assets: Sequence[AssetConfig], token: str, raw_signature: str) -> RunSnapshot:
    return RunSnapshot(
        portfolio_inputs=serialize_portfolio_inputs(portfolio_inputs),
        assets=[asdict(asset) for asset in assets],
        token=token,
        raw_signature=raw_signature,
    )


def snapshot_assets(snapshot: RunSnapshot) -> List[AssetConfig]:
    return [AssetConfig(**asset) for asset in snapshot.assets]


def snapshot_portfolio_inputs(snapshot: RunSnapshot) -> PortfolioInputs:
    return deserialize_portfolio_inputs(snapshot.portfolio_inputs)


def snapshot_asset_specs(snapshot: RunSnapshot) -> Tuple[Tuple[str, float, str], ...]:
    return serialize_assets(snapshot_assets(snapshot))


def snapshot_core_signature(snapshot: RunSnapshot) -> str:
    return build_core_signature(snapshot.portfolio_inputs, snapshot_asset_specs(snapshot), snapshot.token)


def build_risk_table(metrics: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": [
                "Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Downside Deviation",
                "CVaR 95",
                "Parametric VaR 95",
                "Ulcer Index",
                "Calmar Ratio",
                "Max Drawdown",
                "Recovery Years",
            ],
            "Value": [
                metrics["Volatility"],
                metrics["Sharpe Ratio"],
                metrics["Sortino Ratio"],
                metrics["Downside Deviation"],
                metrics["CVaR 95"],
                metrics["Parametric VaR 95"],
                metrics["Ulcer Index"],
                metrics["Calmar Ratio"],
                metrics["Max Drawdown"],
                metrics["Recovery Years"],
            ],
        }
    )


def build_scenario_comparison(
    *,
    current_metrics: Dict[str, float],
    compare_saved: Sequence[str],
    saved_scenarios: Dict[str, Dict[str, object]],
    token: str,
) -> Optional[pd.DataFrame]:
    comparison_rows = [
        {
            "Scenario": "Current",
            "Final Balance": current_metrics["Final Balance"],
            "Real Final Balance": current_metrics["Real Final Balance"],
            "CAGR": current_metrics["CAGR"],
            "Max Drawdown": current_metrics["Max Drawdown"],
            "Volatility": current_metrics["Volatility"],
        }
    ]
    for name in compare_saved:
        cfg = saved_scenarios.get(name)
        if not cfg:
            continue
        saved_assets = tuple((asset["ticker"], float(asset["allocation"]), asset["asset_type"]) for asset in cfg["assets"])
        saved_inputs = dict(cfg.get("inputs", {}))
        saved_range = tuple(cfg["year_range"])
        saved_bundle = run_historical_simulation_cached(
            portfolio_input_dict=saved_inputs,
            asset_specs=saved_assets,
            token=token,
            selected_range=saved_range,
        )
        saved_selection = saved_bundle["selection"]
        saved_metrics = compute_summary_metrics(saved_selection.results_df, saved_selection.filtered_periods)
        comparison_rows.append(
            {
                "Scenario": name,
                "Final Balance": saved_metrics["Final Balance"],
                "Real Final Balance": saved_metrics["Real Final Balance"],
                "CAGR": saved_metrics["CAGR"],
                "Max Drawdown": saved_metrics["Max Drawdown"],
                "Volatility": saved_metrics["Volatility"],
            }
        )
    return pd.DataFrame(comparison_rows) if comparison_rows else None


def build_benchmark_package(
    *,
    results_df: pd.DataFrame,
    portfolio_inputs: PortfolioInputs,
    metrics: Dict[str, float],
    token: str,
    year_range: Tuple[int, int],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if not portfolio_inputs.benchmark_enabled:
        return None, None, None
    benchmark_assets = ((portfolio_inputs.benchmark_ticker, 100.0, portfolio_inputs.benchmark_type),)
    benchmark_input_dict = {**serialize_portfolio_inputs(portfolio_inputs), "benchmark_enabled": False}
    benchmark_bundle = run_historical_simulation_cached(
        portfolio_input_dict=benchmark_input_dict,
        asset_specs=benchmark_assets,
        token=token,
        selected_range=year_range,
    )
    benchmark_selection = benchmark_bundle["selection"]
    benchmark_results = benchmark_selection.results_df.copy()
    comparison = results_df[["Period", "Balance (USD)", "Portfolio Total Return (%)"]].merge(
        benchmark_results[["Period", "Balance (USD)", "Portfolio Total Return (%)"]].rename(
            columns={
                "Balance (USD)": "Benchmark Balance (USD)",
                "Portfolio Total Return (%)": "Benchmark Total Return (%)",
            }
        ),
        on="Period",
        how="inner",
    )
    if comparison.empty:
        return benchmark_results, None, None
    comparison["Excess Return (%)"] = comparison["Portfolio Total Return (%)"] - comparison["Benchmark Total Return (%)"]
    comparison["Relative Wealth (%)"] = ((comparison["Balance (USD)"] / comparison["Benchmark Balance (USD)"]) - 1.0) * 100.0
    benchmark_metrics = compute_summary_metrics(benchmark_results, benchmark_selection.filtered_periods)
    summary_table = pd.DataFrame(
        {
            "Metric": ["Final Balance", "CAGR", "Max Drawdown", "Volatility", "Sharpe Ratio", "CVaR 95"],
            "Portfolio": [
                metrics["Final Balance"],
                metrics["CAGR"],
                metrics["Max Drawdown"],
                metrics["Volatility"],
                metrics["Sharpe Ratio"],
                metrics["CVaR 95"],
            ],
            "Benchmark": [
                benchmark_metrics["Final Balance"],
                benchmark_metrics["CAGR"],
                benchmark_metrics["Max Drawdown"],
                benchmark_metrics["Volatility"],
                benchmark_metrics["Sharpe Ratio"],
                benchmark_metrics["CVaR 95"],
            ],
        }
    )
    summary_table["Excess"] = summary_table["Portfolio"] - summary_table["Benchmark"]
    return benchmark_results, comparison, summary_table


def build_core_run_artifacts(*, snapshot: RunSnapshot, year_range: Tuple[int, int]) -> RunArtifacts:
    portfolio_inputs = snapshot_portfolio_inputs(snapshot)
    assets = snapshot_assets(snapshot)
    asset_specs = snapshot_asset_specs(snapshot)
    historical_bundle = run_historical_simulation_cached(
        portfolio_input_dict=snapshot.portfolio_inputs,
        asset_specs=asset_specs,
        token=snapshot.token,
        selected_range=year_range,
    )
    dataset = historical_bundle["dataset"]
    selection = historical_bundle["selection"]
    metrics = compute_summary_metrics(selection.results_df, selection.filtered_periods)
    rolling3 = compute_rolling_returns(selection.results_df, 3)
    rolling5 = compute_rolling_returns(selection.results_df, 5)
    risk_table = build_risk_table(metrics)
    diagnostics_df = pd.DataFrame(dataset.diagnostics)
    overlap_warning_lines = build_overlap_warning_lines(dataset)
    overlap_summary_df = build_overlap_summary_df(dataset)
    return RunArtifacts(
        portfolio_inputs=portfolio_inputs,
        assets=list(assets),
        year_range=year_range,
        dataset=dataset,
        selection=selection,
        metrics=metrics,
        rolling3=rolling3,
        rolling5=rolling5,
        risk_table=risk_table,
        diagnostics_df=diagnostics_df,
        overlap_warning_lines=overlap_warning_lines,
        overlap_summary_df=overlap_summary_df,
    )


def build_benchmark_artifacts(
    *,
    snapshot: RunSnapshot,
    year_range: Tuple[int, int],
    results_df: pd.DataFrame,
    metrics: Dict[str, float],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    portfolio_inputs = snapshot_portfolio_inputs(snapshot)
    return build_benchmark_package(
        results_df=results_df,
        portfolio_inputs=portfolio_inputs,
        metrics=metrics,
        token=snapshot.token,
        year_range=year_range,
    )


def build_monte_carlo_artifacts(*, snapshot: RunSnapshot, selection: HistoricalSelection) -> Dict[str, pd.DataFrame]:
    if selection.selected_returns_df.empty or selection.selected_divs_df.empty:
        raise ValueError("Historical monthly data is required before Monte Carlo can run.")
    asset_specs = snapshot_asset_specs(snapshot)
    return simulate_monte_carlo_cached(
        portfolio_input_dict=snapshot.portfolio_inputs,
        asset_specs=asset_specs,
        selected_returns_df=selection.selected_returns_df,
        selected_divs_df=selection.selected_divs_df,
        start_period=selection.selected_returns_df.index[0],
    )


@st.cache_data(ttl=3600, show_spinner=False)
def build_export_bytes_cached(
    portfolio_input_dict: Dict[str, object],
    asset_specs: Tuple[Tuple[str, float, str], ...],
    year_range: Tuple[int, int],
    results_df: pd.DataFrame,
    component_df: pd.DataFrame,
    selected_returns_df: pd.DataFrame,
    selected_divs_df: pd.DataFrame,
    weighted_returns: pd.Series,
    weighted_divs: pd.Series,
    metrics: Dict[str, float],
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
    portfolio_inputs = deserialize_portfolio_inputs(portfolio_input_dict)
    assets = deserialize_assets(asset_specs)
    return export_full_simulation_workbook(
        results_df=results_df,
        component_df=component_df,
        selected_returns_df=selected_returns_df,
        selected_divs_df=selected_divs_df,
        weighted_returns=weighted_returns,
        weighted_divs=weighted_divs,
        metrics=metrics,
        portfolio_inputs=portfolio_inputs,
        assets=assets,
        year_range=year_range,
        diagnostics_df=diagnostics_df if diagnostics_df is not None and not diagnostics_df.empty else None,
        risk_table=risk_table if risk_table is not None and not risk_table.empty else None,
        rolling3_df=rolling3_df if rolling3_df is not None and not rolling3_df.empty else None,
        rolling5_df=rolling5_df if rolling5_df is not None and not rolling5_df.empty else None,
        benchmark_comparison_df=benchmark_comparison_df,
        benchmark_results_df=benchmark_results_df,
        mc_percentiles_df=mc_percentiles_df,
        mc_summary_df=mc_summary_df,
        mc_paths_df=mc_paths_df,
        mc_convergence_df=mc_convergence_df,
        scenario_comparison_df=scenario_comparison_df,
    )


def build_run_artifacts(
    *,
    snapshot: RunSnapshot,
    year_range: Tuple[int, int],
    compare_saved: Sequence[str],
    saved_scenarios: Dict[str, Dict[str, object]],
) -> RunArtifacts:
    artifacts = build_core_run_artifacts(snapshot=snapshot, year_range=year_range)
    scenario_comparison_df = None
    if compare_saved:
        scenario_comparison_df = build_scenario_comparison(
            current_metrics=artifacts.metrics,
            compare_saved=compare_saved,
            saved_scenarios=saved_scenarios,
            token=snapshot.token,
        )
    benchmark_results_df, benchmark_comparison_df, benchmark_summary_table = build_benchmark_artifacts(
        snapshot=snapshot,
        year_range=year_range,
        results_df=artifacts.selection.results_df,
        metrics=artifacts.metrics,
    )
    mc_percentiles = mc_summary = mc_paths = mc_convergence = None
    if artifacts.portfolio_inputs.analysis_mode in ("Monte Carlo", "Both"):
        mc_outputs = build_monte_carlo_artifacts(snapshot=snapshot, selection=artifacts.selection)
        mc_percentiles = mc_outputs["percentiles_df"]
        mc_summary = mc_outputs["summary_df"]
        mc_paths = mc_outputs["paths_df"]
        mc_convergence = mc_outputs["convergence_df"]
    export_bytes = build_export_bytes_cached(
        portfolio_input_dict=snapshot.portfolio_inputs,
        asset_specs=snapshot_asset_specs(snapshot),
        year_range=year_range,
        results_df=artifacts.selection.results_df,
        component_df=artifacts.selection.component_df,
        selected_returns_df=artifacts.selection.selected_returns_df,
        selected_divs_df=artifacts.selection.selected_divs_df,
        weighted_returns=artifacts.selection.weighted_returns,
        weighted_divs=artifacts.selection.weighted_divs,
        metrics=artifacts.metrics,
        diagnostics_df=artifacts.diagnostics_df,
        risk_table=artifacts.risk_table,
        rolling3_df=artifacts.rolling3,
        rolling5_df=artifacts.rolling5,
        benchmark_comparison_df=benchmark_comparison_df,
        benchmark_results_df=benchmark_results_df,
        mc_percentiles_df=mc_percentiles,
        mc_summary_df=mc_summary,
        mc_paths_df=mc_paths,
        mc_convergence_df=mc_convergence,
        scenario_comparison_df=scenario_comparison_df,
    )
    artifacts.benchmark_results_df = benchmark_results_df
    artifacts.benchmark_comparison_df = benchmark_comparison_df
    artifacts.benchmark_summary_table = benchmark_summary_table
    artifacts.scenario_comparison_df = scenario_comparison_df
    artifacts.mc_percentiles = mc_percentiles
    artifacts.mc_summary = mc_summary
    artifacts.mc_paths = mc_paths
    artifacts.mc_convergence = mc_convergence
    artifacts.export_bytes = export_bytes
    return artifacts
