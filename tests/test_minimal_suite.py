from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from v241_refactor_app.engine import _apply_withdrawal_trade, maybe_rebalance, run_historical_simulation
from v241_refactor_app.mc_kernel import build_mc_path_plan, simulate_portfolio_path
from v241_refactor_app.models import HistoricalSelection
from v241_refactor_app.monte_carlo import simulate_monte_carlo

from conftest import make_assets, make_portfolio_inputs


def _first_positive_month(series: pd.Series) -> int | None:
    positive = series.gt(1e-9)
    if not positive.any():
        return None
    return int(np.flatnonzero(positive.to_numpy())[0] + 1)


def _first_zero_or_lower_month(series: pd.Series) -> int | None:
    non_positive = series.le(0.0)
    if not non_positive.any():
        return None
    return int(np.flatnonzero(non_positive.to_numpy())[0] + 1)


def _year_end_rows(selection: HistoricalSelection) -> pd.DataFrame:
    return selection.results_df.groupby("Year", sort=True).tail(1).reset_index(drop=True)


def test_monte_carlo_seed_regression_is_deterministic(base_inputs, base_assets, base_dataset):
    selection = run_historical_simulation(
        portfolio_inputs=base_inputs,
        assets=base_assets,
        dataset=base_dataset,
        selected_range=(2020, 2021),
    )

    first = simulate_monte_carlo(
        portfolio_inputs=base_inputs,
        assets=base_assets,
        historical_returns_df=selection.selected_returns_df,
        historical_dividends_df=selection.selected_divs_df,
        start_period=selection.filtered_periods[0],
    )
    second = simulate_monte_carlo(
        portfolio_inputs=base_inputs,
        assets=base_assets,
        historical_returns_df=selection.selected_returns_df,
        historical_dividends_df=selection.selected_divs_df,
        start_period=selection.filtered_periods[0],
    )

    for left, right in zip(first, second):
        pdt.assert_frame_equal(left, right, check_dtype=False, check_exact=False, atol=1e-12, rtol=0.0)


def test_engine_and_kernel_are_invariant_for_same_path(base_inputs, base_assets, base_dataset):
    selection = run_historical_simulation(
        portfolio_inputs=base_inputs,
        assets=base_assets,
        dataset=base_dataset,
        selected_range=(2020, 2021),
    )
    plan = build_mc_path_plan(base_inputs, base_assets, selection.filtered_periods[0])
    result = simulate_portfolio_path(
        plan=plan,
        portfolio_inputs=base_inputs,
        returns_matrix=selection.selected_returns_df.to_numpy(dtype=float),
        dividends_matrix=selection.selected_divs_df.to_numpy(dtype=float),
    )

    year_end = _year_end_rows(selection)
    assert np.allclose(year_end["Balance (USD)"].to_numpy(dtype=float), result.yearly_balances, atol=1e-9)
    assert np.allclose(year_end["Real Balance (USD)"].to_numpy(dtype=float), result.yearly_real_balances, atol=1e-9)

    expected_shortfall_month = _first_positive_month(selection.results_df["Withdrawal Shortfall (USD)"])
    expected_depletion_month = _first_zero_or_lower_month(selection.results_df["Balance (USD)"])
    expected_failure_month = min(
        [month for month in [expected_shortfall_month, expected_depletion_month] if month is not None],
        default=None,
    )

    assert result.shortfall_month == expected_shortfall_month
    assert result.depletion_month == expected_depletion_month
    assert result.failure_month == expected_failure_month


def test_threshold_band_does_not_rebalance_just_inside_band(base_inputs):
    balances = {"AAA": 54.999, "BBB": 45.001}
    target_weights = {"AAA": 50.0, "BBB": 50.0}
    asset_type_map = {"AAA": "Stock", "BBB": "Bond"}

    updated, ending_total, did_rebalance, cost_usd, turnover_pct = maybe_rebalance(
        balances=balances,
        target_weights=target_weights,
        method="Threshold Band",
        band=5.0,
        stage="month_end",
        rebalance_cost_bps=base_inputs.rebalance_cost_bps,
        asset_type_map=asset_type_map,
        portfolio_inputs=base_inputs,
    )

    assert updated == balances
    assert ending_total == 100.0
    assert did_rebalance is False
    assert cost_usd == 0.0
    assert turnover_pct == 0.0


def test_zero_balance_rebalance_is_safe_noop(base_inputs):
    updated, ending_total, did_rebalance, cost_usd, turnover_pct = maybe_rebalance(
        balances={"AAA": 0.0, "BBB": 0.0},
        target_weights={"AAA": 50.0, "BBB": 50.0},
        method="Annual",
        band=5.0,
        stage="year_end",
        rebalance_cost_bps=base_inputs.rebalance_cost_bps,
        asset_type_map={"AAA": "Stock", "BBB": "Bond"},
        portfolio_inputs=base_inputs,
    )

    assert updated == {"AAA": 0.0, "BBB": 0.0}
    assert ending_total == 0.0
    assert did_rebalance is False
    assert cost_usd == 0.0
    assert turnover_pct == 0.0


def test_withdrawal_shortfall_and_zero_balance_paths_never_go_negative():
    stressed_inputs = make_portfolio_inputs(
        initial_investment=1200.0,
        contribution_amount=0.0,
        withdrawal_mode="Fixed Dollar",
        withdrawal_rate=120.0,
        annual_fee_rate=0.0,
        tax_rate_dividends=0.0,
        tax_rate_withdrawals=0.0,
        cashflow_trade_cost_bps=0.0,
        rebalance_cost_bps=0.0,
        rebalancing_method="None",
        monte_carlo_years=1,
    )
    one_asset = make_assets((("AAA", 100.0, "Stock"),))
    plan = build_mc_path_plan(stressed_inputs, one_asset, pd.Timestamp("2020-01-31"))
    result = simulate_portfolio_path(
        plan=plan,
        portfolio_inputs=stressed_inputs,
        returns_matrix=np.zeros((12, 1), dtype=float),
        dividends_matrix=np.zeros((12, 1), dtype=float),
    )

    assert result.ending_balance == 0.0
    assert result.min_balance >= 0.0
    assert result.min_real_balance >= 0.0
    assert result.depletion_month == 10
    assert result.failure_month == 10

    remaining, gross_withdrawal, trade_cost, shortfall = _apply_withdrawal_trade(
        balances={"AAA": 100.0},
        desired_gross_amount=100.0,
        target_weights={"AAA": 100.0},
        asset_type_map={"AAA": "Stock"},
        base_bps=1000.0,
        portfolio_inputs=make_portfolio_inputs(cashflow_trade_cost_bps=1000.0),
    )
    assert gross_withdrawal < 100.0
    assert trade_cost > 0.0
    assert shortfall > 0.0
    assert remaining["AAA"] >= 0.0


def test_stock_data_filters_corporate_action_like_dividend_contamination(monkeypatch):
    import sys
    import types

    fake_streamlit = types.ModuleType("streamlit")
    def _cache_data(**_kwargs):
        def decorator(func):
            func.clear = lambda: None
            return func
        return decorator
    fake_streamlit.cache_data = _cache_data
    sys.modules.setdefault("streamlit", fake_streamlit)

    from v241_refactor_app import data_layer

    sample = [
        {"date": "2020-01-31T00:00:00.000Z", "adjClose": 100.0, "close": 100.0, "divCash": 0.0, "splitFactor": 1.0},
        {"date": "2020-02-28T00:00:00.000Z", "adjClose": 110.0, "close": 110.0, "divCash": 0.0, "splitFactor": 1.0},
        {"date": "2020-03-31T00:00:00.000Z", "adjClose": 121.0, "close": 60.5, "divCash": 2000.0, "splitFactor": 2.0},
        {"date": "2020-04-30T00:00:00.000Z", "adjClose": 133.1, "close": 66.55, "divCash": 0.0, "splitFactor": 1.0},
    ]

    monkeypatch.setattr(data_layer, "fetch_with_retry", lambda *args, **kwargs: sample)
    data_layer.get_stock_data.clear()

    price_returns, dividend_yields, years = data_layer.get_stock_data("GOOGL", "token")

    assert years == [2020]
    assert len(price_returns) == 3
    assert float(dividend_yields.max()) == 0.0
    assert float(price_returns.iloc[1]) > 0.0
    assert float(price_returns.iloc[1]) < 20.0


def test_parametric_var_95_is_reported_as_loss_sign():
    from v241_refactor_app.analytics import compute_risk_metrics

    periods = pd.date_range("2020-01-31", periods=6, freq="ME")
    results_df = pd.DataFrame(
        {
            "Period": periods,
            "Balance (USD)": [100, 105, 95, 110, 108, 115],
            "Portfolio Total Return (%)": [5.0, -9.5238095238, 15.7894736842, -1.8181818182, 6.4814814815, 0.0],
        }
    )

    metrics = compute_risk_metrics(results_df)
    assert metrics["Parametric VaR 95"] <= 0.0
