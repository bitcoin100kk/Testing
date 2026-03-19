from __future__ import annotations

from typing import Sequence

import pandas as pd
import pytest

from v241_refactor_app.models import (
    AssetConfig,
    HistoricalDataset,
    PORTFOLIO_INPUT_DEFAULTS,
    PortfolioInputs,
)


def make_portfolio_inputs(**overrides: object) -> PortfolioInputs:
    payload = {**PORTFOLIO_INPUT_DEFAULTS, **overrides}
    return PortfolioInputs(**payload)


def make_assets(specs: Sequence[tuple[str, float, str]] | None = None) -> list[AssetConfig]:
    specs = specs or (("AAA", 60.0, "Stock"), ("BBB", 40.0, "Bond"))
    return [AssetConfig(ticker=t, allocation=w, asset_type=a) for t, w, a in specs]


def make_historical_dataset(
    months: int = 24,
    *,
    returns_a: float = 1.0,
    returns_b: float = 0.5,
    dividends_a: float = 0.1,
    dividends_b: float = 0.0,
) -> HistoricalDataset:
    periods = pd.date_range("2020-01-31", periods=months, freq="ME")
    returns_df = pd.DataFrame(
        {
            "AAA": [returns_a] * months,
            "BBB": [returns_b] * months,
        },
        index=periods,
    )
    dividends_df = pd.DataFrame(
        {
            "AAA": [dividends_a] * months,
            "BBB": [dividends_b] * months,
        },
        index=periods,
    )
    component_df = pd.DataFrame({"Year": [period.year for period in periods]})
    return HistoricalDataset(
        returns_df=returns_df,
        dividends_df=dividends_df,
        years=sorted(component_df["Year"].unique().tolist()),
        component_df=component_df,
        diagnostics=[],
        overlap_start=periods[0],
        overlap_end=periods[-1],
        overlap_months=len(periods),
    )


@pytest.fixture
def base_assets() -> list[AssetConfig]:
    return make_assets()


@pytest.fixture
def base_inputs() -> PortfolioInputs:
    return make_portfolio_inputs(
        contribution_amount=1200.0,
        withdrawal_rate=4.0,
        annual_fee_rate=0.0,
        tax_rate_dividends=0.0,
        tax_rate_withdrawals=0.0,
        rebalance_cost_bps=0.0,
        cashflow_trade_cost_bps=0.0,
        rebalancing_method="Annual",
        monte_carlo_sims=32,
        monte_carlo_years=2,
        monte_carlo_seed=42,
        monte_carlo_adaptive_convergence=False,
    )


@pytest.fixture
def base_dataset() -> HistoricalDataset:
    return make_historical_dataset()
