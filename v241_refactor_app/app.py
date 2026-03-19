from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import pandas as pd
import streamlit as st

from . import orchestration as orch
from .analytics import build_overlap_warning_lines
from .config import APP_CAPTION, APP_TITLE, DEFAULT_TIINGO_API_TOKEN, configure_page
from .data_layer import prepare_historical_dataset
from .exporters import simulation_to_csv
from .models import AssetConfig, PortfolioInputs
from .scenario_ui import (
    apply_preset_if_requested,
    collect_assets,
    load_saved_scenario,
    save_current_scenario,
    validate_assets,
)
from .state import (
    _sync_slider_from_year_boxes,
    _sync_year_boxes_from_slider,
    clear_cached_artifacts,
    get_active_run_snapshot,
    get_bucket_artifact,
    get_rendered_artifacts,
    initialize_state,
    mark_run_snapshot,
    normalize_year_state,
    reset_year_state,
    set_latest_error,
    store_bucket_artifact,
    store_rendered_artifacts,
)
from .ui_components import (
    _format_benchmark_display_table,
    render_input_change_warning,
    render_metric_tabs,
    render_overlap_alerts,
    render_table,
)
from .utils import build_raw_signature


def _build_portfolio_inputs(
    *,
    initial_investment: float,
    withdrawal_rate: float,
    reinvest_dividends: bool,
    contribution_amount: float,
    contribution_end_year_input: str,
    annual_fee_rate: float,
    inflation_rate: float,
    enable_real_dollars: bool,
    tax_rate_dividends: float,
    tax_rate_withdrawals: float,
    withdrawal_mode: str,
    show_benchmark_overlay: bool,
    benchmark_ticker: str,
    benchmark_type: str,
    rebalancing_method: str,
    rebalance_band: float,
    rebalance_cost_bps: float,
    cashflow_trade_cost_bps: float,
    asset_class_aware_costs: bool,
    aum_cost_scaling: bool,
    aum_cost_scaling_strength: float,
    crypto_cost_multiplier: float,
    analysis_mode: str,
    monte_carlo_sims: int,
    monte_carlo_years: int,
    monte_carlo_seed: int,
    monte_carlo_block_size_months: int,
    monte_carlo_regime_mode: str,
    monte_carlo_regime_window_months: int,
    monte_carlo_bootstrap_method: str,
    monte_carlo_regime_strength: float,
    monte_carlo_adaptive_convergence: bool,
    monte_carlo_target_stderr_pct: float,
) -> PortfolioInputs:
    contribution_end_year = int(contribution_end_year_input) if str(contribution_end_year_input).strip() else None
    return PortfolioInputs(
        initial_investment=float(initial_investment),
        withdrawal_rate=float(withdrawal_rate),
        reinvest_dividends=bool(reinvest_dividends),
        contribution_amount=float(contribution_amount),
        contribution_end_year=contribution_end_year,
        annual_fee_rate=float(annual_fee_rate),
        inflation_rate=float(inflation_rate if enable_real_dollars else 0.0),
        tax_rate_dividends=float(tax_rate_dividends),
        tax_rate_withdrawals=float(tax_rate_withdrawals),
        withdrawal_mode=withdrawal_mode,
        benchmark_enabled=bool(show_benchmark_overlay),
        benchmark_ticker=benchmark_ticker or "SPY",
        benchmark_type=benchmark_type,
        rebalancing_method=rebalancing_method,
        rebalance_band=float(rebalance_band),
        rebalance_cost_bps=float(rebalance_cost_bps),
        cashflow_trade_cost_bps=float(cashflow_trade_cost_bps),
        asset_class_aware_costs=bool(asset_class_aware_costs),
        aum_cost_scaling=bool(aum_cost_scaling),
        aum_cost_scaling_strength=float(aum_cost_scaling_strength),
        crypto_cost_multiplier=float(crypto_cost_multiplier),
        analysis_mode=analysis_mode,
        monte_carlo_sims=int(monte_carlo_sims),
        monte_carlo_years=int(monte_carlo_years),
        monte_carlo_seed=int(monte_carlo_seed),
        monte_carlo_block_size_months=int(monte_carlo_block_size_months),
        monte_carlo_regime_mode=monte_carlo_regime_mode,
        monte_carlo_regime_window_months=int(monte_carlo_regime_window_months),
        monte_carlo_bootstrap_method=monte_carlo_bootstrap_method,
        monte_carlo_regime_strength=float(monte_carlo_regime_strength),
        monte_carlo_adaptive_convergence=bool(monte_carlo_adaptive_convergence),
        monte_carlo_target_stderr_pct=float(monte_carlo_target_stderr_pct),
    )


def _build_raw_state_payload(assets: List[AssetConfig], widget_values: Dict[str, object], token: str) -> Dict[str, object]:
    return {"assets": [asdict(asset) for asset in assets], "inputs": widget_values, "token": token}


def _core_render_signature(core_signature: str, year_range: tuple[int, int]) -> str:
    return build_raw_signature({"kind": "core", "core": core_signature, "year_range": list(year_range)})


def _benchmark_signature(active_core_signature: str, active_inputs: PortfolioInputs, year_range: tuple[int, int]) -> str:
    return build_raw_signature(
        {
            "kind": "benchmark",
            "core": active_core_signature,
            "year_range": list(year_range),
            "benchmark": {
                "enabled": bool(active_inputs.benchmark_enabled),
                "ticker": active_inputs.benchmark_ticker,
                "asset_type": active_inputs.benchmark_type,
            },
        }
    )


def _mc_signature(active_core_signature: str, active_inputs: PortfolioInputs, year_range: tuple[int, int]) -> str:
    return build_raw_signature(
        {
            "kind": "mc",
            "core": active_core_signature,
            "year_range": list(year_range),
            "mc": {
                "analysis_mode": active_inputs.analysis_mode,
                "sims": active_inputs.monte_carlo_sims,
                "years": active_inputs.monte_carlo_years,
                "seed": active_inputs.monte_carlo_seed,
                "block_size": active_inputs.monte_carlo_block_size_months,
                "regime_mode": active_inputs.monte_carlo_regime_mode,
                "regime_window": active_inputs.monte_carlo_regime_window_months,
                "bootstrap_method": active_inputs.monte_carlo_bootstrap_method,
                "regime_strength": active_inputs.monte_carlo_regime_strength,
                "adaptive_convergence": active_inputs.monte_carlo_adaptive_convergence,
                "target_stderr_pct": active_inputs.monte_carlo_target_stderr_pct,
            },
        }
    )


def _scenario_signature(active_core_signature: str, year_range: tuple[int, int], compare_saved: List[str], saved_scenarios: Dict[str, Dict[str, object]]) -> str:
    selected_saved_payload = {name: saved_scenarios.get(name) for name in compare_saved}
    return build_raw_signature(
        {
            "kind": "scenario",
            "core": active_core_signature,
            "year_range": list(year_range),
            "compare_saved": list(compare_saved),
            "saved_payload": selected_saved_payload,
        }
    )


def _export_signature(
    active_core_signature: str,
    year_range: tuple[int, int],
    benchmark_signature: str | None,
    mc_signature: str | None,
    scenario_signature: str | None,
) -> str:
    return build_raw_signature(
        {
            "kind": "export",
            "core": active_core_signature,
            "year_range": list(year_range),
            "benchmark_signature": benchmark_signature,
            "mc_signature": mc_signature,
            "scenario_signature": scenario_signature,
        }
    )


def main() -> None:
    configure_page()
    initialize_state()

    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    with st.sidebar:
        st.header("Power Features")
        st.selectbox(
            "Scenario preset",
            [
                "Custom",
                "100% SPY",
                "60/40 Classic",
                "Dividend + Bitcoin",
                "Global 3-Fund",
                "Mega Cap Growth + Defensives",
            ],
            key="preset_name",
        )
        apply_preset_if_requested()
        st.divider()
        st.subheader("Scenario Lab")
        scenario_names = [""] + sorted(st.session_state.get("saved_scenarios", {}).keys())
        selected_saved_scenario = st.selectbox("Load saved scenario", scenario_names, key="scenario_to_load")
        if st.button("Load saved scenario") and selected_saved_scenario:
            load_saved_scenario(selected_saved_scenario)
        compare_saved = st.multiselect("Compare saved scenarios", sorted(st.session_state.get("saved_scenarios", {}).keys()), default=[])
        show_component_table = st.checkbox("Show per-asset monthly contribution table", value=False)
        show_benchmark_overlay = st.checkbox("Show benchmark comparison", value=bool(st.session_state.get("benchmark_enabled", True)))
        enable_real_dollars = st.checkbox("Use inflation-adjusted analytics", value=True)
        tiingo_token = st.text_input("Tiingo API Token", value=DEFAULT_TIINGO_API_TOKEN, type="password")

    st.subheader("Core Portfolio Inputs")
    initial_investment = st.number_input("Initial Investment (USD)", value=float(st.session_state.get("initial_investment", 100000.0)), min_value=0.0, step=1000.0, key="initial_investment")
    withdrawal_rate = st.number_input("Annual Withdrawal Rate (%)", value=float(st.session_state.get("withdrawal_rate", 4.0)), min_value=0.0, max_value=100.0, step=0.1, key="withdrawal_rate")
    reinvest_dividends = st.checkbox("Reinvest Dividends", value=bool(st.session_state.get("reinvest_dividends", False)), key="reinvest_dividends")

    controls_col1, controls_col2, controls_col3, controls_col4 = st.columns(4)
    with controls_col1:
        num_assets = int(st.number_input("Number of Assets", min_value=1, max_value=10, value=int(st.session_state.get("num_assets", 1)), step=1, key="num_assets"))
    with controls_col2:
        withdrawal_mode = st.selectbox(
            "Withdrawal Mode",
            ["Percent of Balance", "Fixed Dollar", "Inflation-Adjusted Dollar", "Dividend First"],
            index=["Percent of Balance", "Fixed Dollar", "Inflation-Adjusted Dollar", "Dividend First"].index(st.session_state.get("withdrawal_mode", "Percent of Balance")),
            key="withdrawal_mode",
        )
    with controls_col3:
        annual_fee_rate = st.number_input("Annual Fee Drag (%)", min_value=0.0, max_value=10.0, value=float(st.session_state.get("annual_fee_rate", 0.0)), step=0.01, key="annual_fee_rate")
    with controls_col4:
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Historical Backtest", "Monte Carlo", "Both"],
            index=["Historical Backtest", "Monte Carlo", "Both"].index(st.session_state.get("analysis_mode", "Both")),
            key="analysis_mode",
        )

    advanced_col1, advanced_col2, advanced_col3, advanced_col4 = st.columns(4)
    with advanced_col1:
        contribution_amount = st.number_input("Annual Contribution (USD)", min_value=0.0, value=float(st.session_state.get("contribution_amount", 0.0)), step=1000.0, key="contribution_amount")
    with advanced_col2:
        inflation_rate = st.number_input("Inflation Rate (%)", min_value=0.0, max_value=20.0, value=float(st.session_state.get("inflation_rate", 3.0)), step=0.1, key="inflation_rate")
    with advanced_col3:
        tax_rate_dividends = st.number_input("Dividend Tax Rate (%)", min_value=0.0, max_value=60.0, value=float(st.session_state.get("tax_rate_dividends", 0.0)), step=0.5, key="tax_rate_dividends")
    with advanced_col4:
        tax_rate_withdrawals = st.number_input("Withdrawal Tax Rate (%)", min_value=0.0, max_value=60.0, value=float(st.session_state.get("tax_rate_withdrawals", 0.0)), step=0.5, key="tax_rate_withdrawals")

    with st.expander("Portfolio Management", expanded=True):
        rb1, rb2, rb3, rb4, rb5 = st.columns(5)
        with rb1:
            rebalancing_method = st.selectbox(
                "Rebalancing Method",
                ["None", "Annual", "Monthly", "Threshold Band", "Contributions Only", "Withdrawals Only"],
                index=["None", "Annual", "Monthly", "Threshold Band", "Contributions Only", "Withdrawals Only"].index(st.session_state.get("rebalancing_method", "Annual")),
                key="rebalancing_method",
            )
        with rb2:
            rebalance_band = st.number_input("Rebalance Threshold Band (%)", min_value=0.0, max_value=50.0, value=float(st.session_state.get("rebalance_band", 5.0)), step=0.5, key="rebalance_band")
        with rb3:
            rebalance_cost_bps = st.number_input("Base Rebalance Cost (bps)", min_value=0.0, max_value=500.0, value=float(st.session_state.get("rebalance_cost_bps", 5.0)), step=1.0, key="rebalance_cost_bps")
        with rb4:
            cashflow_trade_cost_bps = st.number_input("Base Cashflow Trade Cost (bps)", min_value=0.0, max_value=500.0, value=float(st.session_state.get("cashflow_trade_cost_bps", 2.0)), step=1.0, key="cashflow_trade_cost_bps")
        with rb5:
            contribution_end_year_input = st.text_input(
                "Stop Annual Contributions After Year (blank = never)",
                value=st.session_state.get("contribution_end_year_text", ""),
                key="contribution_end_year_text",
            )

        cost1, cost2, cost3, cost4 = st.columns(4)
        with cost1:
            asset_class_aware_costs = st.checkbox("Asset-Class-Aware Costs", value=bool(st.session_state.get("asset_class_aware_costs", True)), key="asset_class_aware_costs")
        with cost2:
            aum_cost_scaling = st.checkbox("AUM-Scaled Costs", value=bool(st.session_state.get("aum_cost_scaling", True)), key="aum_cost_scaling")
        with cost3:
            crypto_cost_multiplier = st.number_input("Crypto Cost Multiplier", min_value=1.0, max_value=20.0, value=float(st.session_state.get("crypto_cost_multiplier", 4.0)), step=0.5, key="crypto_cost_multiplier")
        with cost4:
            aum_cost_scaling_strength = st.number_input("AUM Scaling Strength", min_value=0.0, max_value=1.0, value=float(st.session_state.get("aum_cost_scaling_strength", 0.25)), step=0.05, key="aum_cost_scaling_strength")

    with st.expander("Benchmark Settings", expanded=show_benchmark_overlay):
        b1, b2 = st.columns(2)
        with b1:
            benchmark_ticker = st.text_input("Benchmark Ticker", value=st.session_state.get("benchmark_ticker", "SPY"), key="benchmark_ticker").strip().upper()
        with b2:
            benchmark_type = st.selectbox("Benchmark Type", ["Stock", "Crypto"], index=0 if st.session_state.get("benchmark_type", "Stock") == "Stock" else 1, key="benchmark_type")

    with st.expander("Monte Carlo Settings", expanded=analysis_mode in ("Monte Carlo", "Both")):
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            monte_carlo_sims = int(st.number_input("Monte Carlo Simulations", min_value=100, max_value=10000, value=int(st.session_state.get("monte_carlo_sims", 2000)), step=100, key="monte_carlo_sims"))
        with mc2:
            monte_carlo_years = int(st.number_input("Monte Carlo Horizon (Years)", min_value=1, max_value=100, value=int(st.session_state.get("monte_carlo_years", 30)), step=1, key="monte_carlo_years"))
        with mc3:
            monte_carlo_seed = int(st.number_input("Monte Carlo Seed", min_value=0, max_value=999999, value=int(st.session_state.get("monte_carlo_seed", 42)), step=1, key="monte_carlo_seed"))
        with mc4:
            monte_carlo_block_size_months = int(st.number_input("Average Block Size (Months)", min_value=1, max_value=60, value=int(st.session_state.get("monte_carlo_block_size_months", 12)), step=1, key="monte_carlo_block_size_months"))

        mc5, mc6, mc7 = st.columns(3)
        with mc5:
            monte_carlo_bootstrap_method = st.selectbox(
                "Bootstrap Method",
                ["Block Bootstrap", "Stationary Bootstrap"],
                index=["Block Bootstrap", "Stationary Bootstrap"].index(st.session_state.get("monte_carlo_bootstrap_method", "Block Bootstrap")),
                key="monte_carlo_bootstrap_method",
            )
        with mc6:
            monte_carlo_regime_mode = st.selectbox(
                "MC Sampling Regime",
                ["All History", "Stress Blocks", "Calm Blocks"],
                index=["All History", "Stress Blocks", "Calm Blocks"].index(st.session_state.get("monte_carlo_regime_mode", "All History")),
                key="monte_carlo_regime_mode",
            )
        with mc7:
            monte_carlo_regime_window_months = int(st.number_input("Regime Scoring Window (Months)", min_value=3, max_value=24, value=int(st.session_state.get("monte_carlo_regime_window_months", 6)), step=1, key="monte_carlo_regime_window_months"))

        mc8, mc9, mc10 = st.columns(3)
        with mc8:
            monte_carlo_regime_strength = float(st.slider("Regime Tilt Strength", min_value=0.0, max_value=3.0, value=float(st.session_state.get("monte_carlo_regime_strength", 1.0)), step=0.1, key="monte_carlo_regime_strength"))
        with mc9:
            monte_carlo_adaptive_convergence = bool(st.checkbox("Adaptive Convergence Stop", value=bool(st.session_state.get("monte_carlo_adaptive_convergence", True)), key="monte_carlo_adaptive_convergence"))
        with mc10:
            monte_carlo_target_stderr_pct = float(st.number_input("Target Failure StdErr (%)", min_value=0.05, max_value=5.0, value=float(st.session_state.get("monte_carlo_target_stderr_pct", 0.35)), step=0.05, key="monte_carlo_target_stderr_pct"))

    assets = collect_assets(num_assets)
    run_clicked = st.button("Run simulation", type="primary")

    widget_payload = {
        "initial_investment": float(initial_investment),
        "withdrawal_rate": float(withdrawal_rate),
        "reinvest_dividends": bool(reinvest_dividends),
        "contribution_amount": float(contribution_amount),
        "contribution_end_year_input": str(contribution_end_year_input).strip(),
        "annual_fee_rate": float(annual_fee_rate),
        "inflation_rate": float(inflation_rate),
        "enable_real_dollars": bool(enable_real_dollars),
        "tax_rate_dividends": float(tax_rate_dividends),
        "tax_rate_withdrawals": float(tax_rate_withdrawals),
        "withdrawal_mode": withdrawal_mode,
        "show_benchmark_overlay": bool(show_benchmark_overlay),
        "benchmark_ticker": benchmark_ticker,
        "benchmark_type": benchmark_type,
        "rebalancing_method": rebalancing_method,
        "rebalance_band": float(rebalance_band),
        "rebalance_cost_bps": float(rebalance_cost_bps),
        "cashflow_trade_cost_bps": float(cashflow_trade_cost_bps),
        "asset_class_aware_costs": bool(asset_class_aware_costs),
        "aum_cost_scaling": bool(aum_cost_scaling),
        "aum_cost_scaling_strength": float(aum_cost_scaling_strength),
        "crypto_cost_multiplier": float(crypto_cost_multiplier),
        "analysis_mode": analysis_mode,
        "monte_carlo_sims": int(monte_carlo_sims),
        "monte_carlo_years": int(monte_carlo_years),
        "monte_carlo_seed": int(monte_carlo_seed),
        "monte_carlo_block_size_months": int(monte_carlo_block_size_months),
        "monte_carlo_regime_mode": monte_carlo_regime_mode,
        "monte_carlo_regime_window_months": int(monte_carlo_regime_window_months),
        "monte_carlo_bootstrap_method": monte_carlo_bootstrap_method,
        "monte_carlo_regime_strength": float(monte_carlo_regime_strength),
        "monte_carlo_adaptive_convergence": bool(monte_carlo_adaptive_convergence),
        "monte_carlo_target_stderr_pct": float(monte_carlo_target_stderr_pct),
    }
    raw_signature = build_raw_signature(_build_raw_state_payload(assets, widget_payload, tiingo_token))

    current_valid_snapshot = None
    current_core_signature = None
    current_validation_error = None
    try:
        valid_assets = validate_assets(assets)
        portfolio_inputs = _build_portfolio_inputs(
            initial_investment=initial_investment,
            withdrawal_rate=withdrawal_rate,
            reinvest_dividends=reinvest_dividends,
            contribution_amount=contribution_amount,
            contribution_end_year_input=contribution_end_year_input,
            annual_fee_rate=annual_fee_rate,
            inflation_rate=inflation_rate,
            enable_real_dollars=enable_real_dollars,
            tax_rate_dividends=tax_rate_dividends,
            tax_rate_withdrawals=tax_rate_withdrawals,
            withdrawal_mode=withdrawal_mode,
            show_benchmark_overlay=show_benchmark_overlay,
            benchmark_ticker=benchmark_ticker,
            benchmark_type=benchmark_type,
            rebalancing_method=rebalancing_method,
            rebalance_band=rebalance_band,
            rebalance_cost_bps=rebalance_cost_bps,
            cashflow_trade_cost_bps=cashflow_trade_cost_bps,
            asset_class_aware_costs=asset_class_aware_costs,
            aum_cost_scaling=aum_cost_scaling,
            aum_cost_scaling_strength=aum_cost_scaling_strength,
            crypto_cost_multiplier=crypto_cost_multiplier,
            analysis_mode=analysis_mode,
            monte_carlo_sims=monte_carlo_sims,
            monte_carlo_years=monte_carlo_years,
            monte_carlo_seed=monte_carlo_seed,
            monte_carlo_block_size_months=monte_carlo_block_size_months,
            monte_carlo_regime_mode=monte_carlo_regime_mode,
            monte_carlo_regime_window_months=monte_carlo_regime_window_months,
            monte_carlo_bootstrap_method=monte_carlo_bootstrap_method,
            monte_carlo_regime_strength=monte_carlo_regime_strength,
            monte_carlo_adaptive_convergence=monte_carlo_adaptive_convergence,
            monte_carlo_target_stderr_pct=monte_carlo_target_stderr_pct,
        )
        current_valid_snapshot = orch.make_run_snapshot(portfolio_inputs, valid_assets, tiingo_token, raw_signature)
        current_core_signature = orch.snapshot_core_signature(current_valid_snapshot)
    except Exception as exc:  # noqa: BLE001
        current_validation_error = str(exc)
        if run_clicked:
            st.error(current_validation_error)

    if run_clicked and current_valid_snapshot is not None:
        previous_signature = st.session_state.get("active_core_signature")
        if previous_signature != current_core_signature:
            reset_year_state()
            clear_cached_artifacts()
        mark_run_snapshot(current_valid_snapshot, current_core_signature)

    active_snapshot = get_active_run_snapshot()
    if active_snapshot is None:
        if current_validation_error and not run_clicked:
            st.info("Enter a valid portfolio and click Run simulation to generate results.")
        return

    try:
        active_core_signature = orch.snapshot_core_signature(active_snapshot)
        active_inputs = orch.snapshot_portfolio_inputs(active_snapshot)
        active_assets = [AssetConfig(**asset) for asset in active_snapshot.assets]
        asset_specs = tuple((asset["ticker"], float(asset["allocation"]), asset["asset_type"]) for asset in active_snapshot.assets)
        dataset = prepare_historical_dataset(asset_specs, active_snapshot.token)
        st.session_state["data_diagnostics"] = dataset.diagnostics
        st.session_state["last_config"] = {"portfolio_inputs": active_snapshot.portfolio_inputs, "assets": active_snapshot.assets}

        ignored_zero = st.session_state.get("ignored_zero_allocation_tickers", [])
        if ignored_zero:
            st.info(f"Ignoring zero-allocation tickers: {', '.join(ignored_zero)}")

        st.subheader("Historical Range")
        canonical_start, canonical_end = normalize_year_state(dataset.years)
        min_year, max_year = min(dataset.years), max(dataset.years)
        yc1, yc2 = st.columns(2)
        with yc1:
            st.number_input("Start Year", min_value=int(min_year), max_value=int(max_year), step=1, key="start_year_box", on_change=_sync_slider_from_year_boxes, args=(dataset.years,))
        with yc2:
            st.number_input("End Year", min_value=int(min_year), max_value=int(max_year), step=1, key="end_year_box", on_change=_sync_slider_from_year_boxes, args=(dataset.years,))
        st.slider("Select historical year range", min_value=int(min_year), max_value=int(max_year), value=(int(canonical_start), int(canonical_end)), key="year_range_slider", on_change=_sync_year_boxes_from_slider, args=(dataset.years,))
        year_range = tuple(st.session_state["canonical_year_range"])
        render_overlap_alerts(dataset, build_overlap_warning_lines(dataset))
        if current_core_signature is not None and current_core_signature != active_core_signature:
            render_input_change_warning()

        core_render_signature = _core_render_signature(active_core_signature, year_range)
        core_artifacts = get_rendered_artifacts(core_render_signature)
        if core_artifacts is None:
            with st.spinner("Preparing historical results..."):
                core_artifacts = orch.build_core_run_artifacts(snapshot=active_snapshot, year_range=year_range)
            store_rendered_artifacts(core_render_signature, core_artifacts)

        st.subheader("Summary")
        render_metric_tabs(core_artifacts.metrics)
        st.caption(
            "Historical results are cached and rendered first. Benchmark comparison, Monte Carlo, scenario comparison, and workbook export reuse cached artifacts and only compute when requested."
        )

        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            scenario_name_to_save = st.text_input("Save current scenario as", value="", key="scenario_name_to_save")
        with save_col2:
            st.write("")
            st.write("")
            if st.button("Save Scenario"):
                try:
                    save_current_scenario(scenario_name_to_save, active_inputs, active_assets, year_range)
                    st.success(f"Saved scenario: {scenario_name_to_save}")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        saved_scenarios = st.session_state.get("saved_scenarios", {})
        benchmark_signature = _benchmark_signature(active_core_signature, active_inputs, year_range)
        mc_signature = _mc_signature(active_core_signature, active_inputs, year_range)
        scenario_signature = _scenario_signature(active_core_signature, year_range, list(compare_saved), saved_scenarios)

        cached_benchmark = get_bucket_artifact("benchmark", benchmark_signature)
        benchmark_results_df = benchmark_comparison_df = benchmark_summary_table = None
        if isinstance(cached_benchmark, dict):
            benchmark_results_df = cached_benchmark.get("results_df")
            benchmark_comparison_df = cached_benchmark.get("comparison_df")
            benchmark_summary_table = cached_benchmark.get("summary_table")

        cached_mc = get_bucket_artifact("mc", mc_signature)
        mc_percentiles = mc_summary = mc_paths = mc_convergence = None
        if isinstance(cached_mc, dict):
            mc_percentiles = cached_mc.get("percentiles_df")
            mc_summary = cached_mc.get("summary_df")
            mc_paths = cached_mc.get("paths_df")
            mc_convergence = cached_mc.get("convergence_df")

        cached_scenario = get_bucket_artifact("scenario", scenario_signature)
        scenario_comparison_df = cached_scenario if isinstance(cached_scenario, pd.DataFrame) else None

        results_tab, risk_tab, scenario_tab, benchmark_tab, component_tab, monte_tab = st.tabs(
            ["Overview", "Risk", "Scenarios", "Benchmark", "Per-Asset", "Monte Carlo"]
        )

        with results_tab:
            st.subheader("Results Table")
            overview_table_view = st.radio("Overview table view", options=["Monthly", "Yearly"], horizontal=True, key="overview_table_view")
            st.caption("Display only. Calculations, cashflows, and simulations remain monthly under the hood.")
            render_table(core_artifacts.selection.results_df, view_mode=overview_table_view)
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Portfolio", "Cashflow", "Drawdown"])
            with chart_tab1:
                st.line_chart(core_artifacts.selection.results_df.set_index("Period")[["Balance (USD)", "Real Balance (USD)"]])
            with chart_tab2:
                st.line_chart(core_artifacts.selection.results_df.set_index("Period")[["Withdrawal (USD)", "Dividend Yield (USD)", "Total Withdrawal + Dividend (USD)"]])
            with chart_tab3:
                drawdown_df = pd.DataFrame(
                    {
                        "Period": core_artifacts.selection.results_df["Period"],
                        "Drawdown (%)": ((core_artifacts.selection.results_df["Balance (USD)"] / core_artifacts.selection.results_df["Balance (USD)"].cummax()) - 1.0) * 100.0,
                    }
                ).set_index("Period")
                st.line_chart(drawdown_df)
            st.download_button(
                "Download results as CSV",
                data=simulation_to_csv(core_artifacts.selection.results_df),
                file_name="investment_simulation_results.csv",
                mime="text/csv",
            )

        with risk_tab:
            if not core_artifacts.rolling3.empty:
                st.subheader("Rolling 3-Year Returns")
                st.line_chart(core_artifacts.rolling3.set_index("End Period"))
            if not core_artifacts.rolling5.empty:
                st.subheader("Rolling 5-Year Returns")
                st.line_chart(core_artifacts.rolling5.set_index("End Period"))
            st.dataframe(core_artifacts.risk_table, use_container_width=True)

        with scenario_tab:
            st.subheader("Scenario Comparison")
            run_scenarios_now = bool(compare_saved) and scenario_comparison_df is None and st.button(
                "Build scenario comparison", key=f"build_scenarios_{scenario_signature[:12]}"
            )
            if run_scenarios_now:
                with st.spinner("Building scenario comparison..."):
                    scenario_comparison_df = orch.build_scenario_comparison(
                        current_metrics=core_artifacts.metrics,
                        compare_saved=compare_saved,
                        saved_scenarios=saved_scenarios,
                        token=active_snapshot.token,
                    )
                store_bucket_artifact("scenario", scenario_signature, scenario_comparison_df)
            elif compare_saved and scenario_comparison_df is None:
                st.info("Scenario comparison is prepared on demand so the core results stay fast.")

            if scenario_comparison_df is not None and not scenario_comparison_df.empty:
                st.dataframe(scenario_comparison_df, use_container_width=True)
                if len(scenario_comparison_df) > 1:
                    st.bar_chart(scenario_comparison_df.set_index("Scenario")[["Final Balance", "Real Final Balance"]])
            else:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Scenario": "Current",
                                "Final Balance": core_artifacts.metrics["Final Balance"],
                                "Real Final Balance": core_artifacts.metrics["Real Final Balance"],
                                "CAGR": core_artifacts.metrics["CAGR"],
                                "Max Drawdown": core_artifacts.metrics["Max Drawdown"],
                                "Volatility": core_artifacts.metrics["Volatility"],
                            }
                        ]
                    ),
                    use_container_width=True,
                )

        with benchmark_tab:
            if active_inputs.benchmark_enabled:
                run_benchmark_now = benchmark_results_df is None and benchmark_comparison_df is None and benchmark_summary_table is None and st.button(
                    "Build benchmark comparison", key=f"build_benchmark_{benchmark_signature[:12]}"
                )
                if run_benchmark_now:
                    with st.spinner("Building benchmark comparison..."):
                        benchmark_results_df, benchmark_comparison_df, benchmark_summary_table = orch.build_benchmark_artifacts(
                            snapshot=active_snapshot,
                            year_range=year_range,
                            results_df=core_artifacts.selection.results_df,
                            metrics=core_artifacts.metrics,
                        )
                    store_bucket_artifact(
                        "benchmark",
                        benchmark_signature,
                        {
                            "results_df": benchmark_results_df,
                            "comparison_df": benchmark_comparison_df,
                            "summary_table": benchmark_summary_table,
                        },
                    )
                elif benchmark_results_df is None and benchmark_comparison_df is None and benchmark_summary_table is None:
                    st.info("Benchmark comparison now runs separately so the overview and risk tabs do not wait for another full simulation pass.")

                if benchmark_comparison_df is None or benchmark_comparison_df.empty:
                    if benchmark_results_df is not None:
                        st.warning("No overlapping benchmark months were available for comparison in the selected range.")
                else:
                    benchmark_table_view = st.radio("Benchmark table view", options=["Monthly", "Yearly"], horizontal=True, key="benchmark_table_view")
                    benchmark_display_df = _format_benchmark_display_table(benchmark_comparison_df, view_mode=benchmark_table_view)
                    st.caption("Charts stay monthly. The toggle below changes only the detailed benchmark table.")
                    st.line_chart(benchmark_comparison_df.set_index("Period")[["Balance (USD)", "Benchmark Balance (USD)"]])
                    st.line_chart(benchmark_comparison_df.set_index("Period")[["Relative Wealth (%)"]])
                    if benchmark_summary_table is not None:
                        st.dataframe(benchmark_summary_table, use_container_width=True)
                    st.dataframe(benchmark_display_df, use_container_width=True)
            else:
                st.info("Benchmark comparison is disabled for the active run.")

        with component_tab:
            if not core_artifacts.overlap_summary_df.empty:
                st.subheader("Overlap / Sample Diagnostics")
                st.dataframe(core_artifacts.overlap_summary_df, use_container_width=True)
            if show_component_table and not core_artifacts.selection.component_df.empty:
                component_display = core_artifacts.selection.component_df.rename(
                    columns={
                        "price_return": "Monthly Price Return (%)",
                        "dividend_yield": "Monthly Dividend Yield (%)",
                        "total_return": "Monthly Total Return (%)",
                        "weighted_price_return": "Weighted Price Contribution (%)",
                        "weighted_dividend_yield": "Weighted Dividend Contribution (%)",
                        "weighted_total_return": "Weighted Total Contribution (%)",
                    }
                )
                component_display["Period"] = pd.to_datetime(component_display["Period"]).dt.strftime("%Y-%m")
                st.dataframe(component_display, use_container_width=True)
            else:
                st.info("Enable the per-asset monthly table from the sidebar to view this tab.")

        with monte_tab:
            if active_inputs.analysis_mode in ("Monte Carlo", "Both"):
                run_mc_now = (mc_percentiles is None or mc_summary is None) and st.button(
                    "Run Monte Carlo now", key=f"run_mc_{mc_signature[:12]}"
                )
                if run_mc_now:
                    with st.spinner("Running Monte Carlo..."):
                        mc_outputs = orch.build_monte_carlo_artifacts(snapshot=active_snapshot, selection=core_artifacts.selection)
                    mc_percentiles = mc_outputs.get("percentiles_df")
                    mc_summary = mc_outputs.get("summary_df")
                    mc_paths = mc_outputs.get("paths_df")
                    mc_convergence = mc_outputs.get("convergence_df")
                    store_bucket_artifact("mc", mc_signature, mc_outputs)
                elif mc_percentiles is None or mc_summary is None:
                    st.info("Historical results are already ready. Monte Carlo runs separately so the overview and benchmark tabs do not wait on the full simulation count.")

                if mc_percentiles is not None and mc_summary is not None:
                    st.caption(
                        f"Bootstrap: {active_inputs.monte_carlo_bootstrap_method}; regime: {active_inputs.monte_carlo_regime_mode}; tilt strength {active_inputs.monte_carlo_regime_strength:.1f}; block size {active_inputs.monte_carlo_block_size_months} months; regime window {active_inputs.monte_carlo_regime_window_months} months; adaptive convergence {'on' if active_inputs.monte_carlo_adaptive_convergence else 'off'} (target stderr {active_inputs.monte_carlo_target_stderr_pct:.2f}%)."
                    )
                    st.subheader("Monte Carlo Probability Bands")
                    balance_cols = [col for col in ["P10", "Median", "P90", "Real P10", "Real Median", "Real P90"] if col in mc_percentiles.columns]
                    if balance_cols:
                        st.line_chart(mc_percentiles.set_index("Year")[balance_cols])
                    probability_cols = [col for col in ["Survival Probability (%)", "Failure by Year (%)", "Ruin by Year (%)", "Shortfall by Year (%)"] if col in mc_percentiles.columns]
                    if probability_cols:
                        st.subheader("Monte Carlo Survival / Failure Curves")
                        st.line_chart(mc_percentiles.set_index("Year")[probability_cols])
                    summary_display = mc_summary.copy()
                    summary_display["Display Value"] = summary_display.apply(
                        lambda row: (
                            f"${row['Value']:,.2f}" if row["Unit"] == "USD"
                            else (f"{row['Value']:.2f}%" if row["Unit"] == "%"
                            else (f"{int(row['Value'])}" if row["Unit"] == "Count" and pd.notna(row["Value"])
                            else (f"{row['Value']:.1f}" if row["Unit"] == "Years" and pd.notna(row["Value"]) else "N/A")))
                        ),
                        axis=1,
                    )
                    st.dataframe(summary_display[["Metric", "Display Value"]], use_container_width=True)
                    if mc_convergence is not None and not mc_convergence.empty:
                        st.subheader("Monte Carlo Convergence")
                        st.line_chart(mc_convergence.set_index("Sim Count")[["P05 Ending Balance", "Median Ending Balance", "P95 Ending Balance"]])
                        st.line_chart(mc_convergence.set_index("Sim Count")[["Failure Rate (%)", "Ruin Rate (%)", "Spending Shortfall Rate (%)", "Failure StdErr (%)"]])
                        if "Max Quantile Drift (%)" in mc_convergence.columns:
                            st.line_chart(mc_convergence.set_index("Sim Count")[["Max Quantile Drift (%)"]])
                        st.dataframe(mc_convergence, use_container_width=True)
            else:
                st.info("Set Analysis Mode to Monte Carlo or Both to enable this tab.")

        st.subheader("Workbook Export")
        export_signature = _export_signature(
            active_core_signature=active_core_signature,
            year_range=year_range,
            benchmark_signature=benchmark_signature if (benchmark_results_df is not None or benchmark_comparison_df is not None or benchmark_summary_table is not None) else None,
            mc_signature=mc_signature if (mc_percentiles is not None and mc_summary is not None) else None,
            scenario_signature=scenario_signature if scenario_comparison_df is not None else None,
        )
        export_bytes = get_bucket_artifact("export", export_signature)

        prepare_export_now = export_bytes is None and st.button(
            "Prepare FULL Simulation Workbook", key=f"prepare_export_{export_signature[:12]}"
        )
        if export_bytes is None:
            st.caption(
                "Prepare the workbook on demand. It reuses cached historical results and only adds optional sheets for benchmark, scenario comparison, and Monte Carlo if those outputs have been prepared for this run."
            )

        if prepare_export_now:
            with st.spinner("Preparing workbook export..."):
                if active_inputs.benchmark_enabled and benchmark_results_df is None and benchmark_summary_table is None:
                    benchmark_results_df, benchmark_comparison_df, benchmark_summary_table = orch.build_benchmark_artifacts(
                        snapshot=active_snapshot,
                        year_range=year_range,
                        results_df=core_artifacts.selection.results_df,
                        metrics=core_artifacts.metrics,
                    )
                    store_bucket_artifact(
                        "benchmark",
                        benchmark_signature,
                        {
                            "results_df": benchmark_results_df,
                            "comparison_df": benchmark_comparison_df,
                            "summary_table": benchmark_summary_table,
                        },
                    )
                if compare_saved and scenario_comparison_df is None:
                    scenario_comparison_df = orch.build_scenario_comparison(
                        current_metrics=core_artifacts.metrics,
                        compare_saved=compare_saved,
                        saved_scenarios=saved_scenarios,
                        token=active_snapshot.token,
                    )
                    store_bucket_artifact("scenario", scenario_signature, scenario_comparison_df)
                if active_inputs.analysis_mode in ("Monte Carlo", "Both") and (mc_percentiles is None or mc_summary is None):
                    mc_outputs = orch.build_monte_carlo_artifacts(snapshot=active_snapshot, selection=core_artifacts.selection)
                    mc_percentiles = mc_outputs.get("percentiles_df")
                    mc_summary = mc_outputs.get("summary_df")
                    mc_paths = mc_outputs.get("paths_df")
                    mc_convergence = mc_outputs.get("convergence_df")
                    store_bucket_artifact("mc", mc_signature, mc_outputs)

                export_signature = _export_signature(
                    active_core_signature=active_core_signature,
                    year_range=year_range,
                    benchmark_signature=benchmark_signature if (benchmark_results_df is not None or benchmark_comparison_df is not None or benchmark_summary_table is not None) else None,
                    mc_signature=mc_signature if (mc_percentiles is not None and mc_summary is not None) else None,
                    scenario_signature=scenario_signature if scenario_comparison_df is not None else None,
                )
                export_bytes = get_bucket_artifact("export", export_signature)
                if export_bytes is None:
                    export_bytes = orch.build_export_bytes_cached(
                        portfolio_input_dict=active_snapshot.portfolio_inputs,
                        asset_specs=orch.snapshot_asset_specs(active_snapshot),
                        year_range=year_range,
                        results_df=core_artifacts.selection.results_df,
                        component_df=core_artifacts.selection.component_df,
                        selected_returns_df=core_artifacts.selection.selected_returns_df,
                        selected_divs_df=core_artifacts.selection.selected_divs_df,
                        weighted_returns=core_artifacts.selection.weighted_returns,
                        weighted_divs=core_artifacts.selection.weighted_divs,
                        metrics=core_artifacts.metrics,
                        diagnostics_df=core_artifacts.diagnostics_df,
                        risk_table=core_artifacts.risk_table,
                        rolling3_df=core_artifacts.rolling3,
                        rolling5_df=core_artifacts.rolling5,
                        benchmark_comparison_df=benchmark_comparison_df,
                        benchmark_results_df=benchmark_results_df,
                        mc_percentiles_df=mc_percentiles,
                        mc_summary_df=mc_summary,
                        mc_paths_df=mc_paths,
                        mc_convergence_df=mc_convergence,
                        scenario_comparison_df=scenario_comparison_df,
                    )
                    store_bucket_artifact("export", export_signature, export_bytes)

        if export_bytes is not None:
            st.download_button(
                "Download FULL Simulation Workbook",
                data=export_bytes,
                file_name="portfolio_simulation_full_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as exc:  # noqa: BLE001
        set_latest_error(str(exc))
        st.error(f"Error fetching data for the provided tickers or running the simulation: {exc}")
