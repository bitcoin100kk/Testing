from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .engine import (
    _allocate_cashflow_proportionally,
    _apply_contribution_trade,
    _apply_withdrawal_trade,
    _portfolio_total,
    _weight_map,
    maybe_rebalance,
)
from .models import AssetConfig, MonteCarloPathResult, PortfolioInputs
from .utils import _annual_pct_to_monthly_decimal


@dataclass(frozen=True)
class MonteCarloPathPlan:
    tickers: tuple[str, ...]
    target_weights: Dict[str, float]
    asset_type_map: Dict[str, str]
    initial_balances: Dict[str, float]
    start_period: pd.Timestamp
    monthly_fee_rate: float
    monthly_inflation_rate: float
    monthly_withdrawal_rate: float
    fixed_monthly_withdrawal_base: float


def build_mc_path_plan(
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    start_period: pd.Timestamp,
) -> MonteCarloPathPlan:
    tickers = tuple(asset.ticker for asset in assets)
    return MonteCarloPathPlan(
        tickers=tickers,
        target_weights={asset.ticker: float(asset.allocation) for asset in assets},
        asset_type_map={asset.ticker: str(asset.asset_type) for asset in assets},
        initial_balances={
            asset.ticker: float(portfolio_inputs.initial_investment) * (float(asset.allocation) / 100.0)
            for asset in assets
        },
        start_period=pd.Timestamp(start_period),
        monthly_fee_rate=_annual_pct_to_monthly_decimal(portfolio_inputs.annual_fee_rate),
        monthly_inflation_rate=_annual_pct_to_monthly_decimal(portfolio_inputs.inflation_rate),
        monthly_withdrawal_rate=_annual_pct_to_monthly_decimal(portfolio_inputs.withdrawal_rate),
        fixed_monthly_withdrawal_base=float(portfolio_inputs.initial_investment)
        * (float(portfolio_inputs.withdrawal_rate) / 100.0)
        / 12.0,
    )


def _calendar_year_month(start_period: pd.Timestamp, offset_months: int) -> tuple[int, int]:
    start_year = int(pd.Timestamp(start_period).year)
    start_month = int(pd.Timestamp(start_period).month)
    absolute_month = (start_month - 1) + int(offset_months)
    year = start_year + (absolute_month // 12)
    month = (absolute_month % 12) + 1
    return year, month


def simulate_portfolio_path(
    portfolio_inputs: PortfolioInputs,
    plan: MonteCarloPathPlan,
    returns_matrix: np.ndarray,
    dividends_matrix: np.ndarray,
) -> MonteCarloPathResult:
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    dividends_matrix = np.asarray(dividends_matrix, dtype=float)
    if returns_matrix.shape != dividends_matrix.shape:
        raise ValueError("Monte Carlo return and dividend matrices must have identical shapes.")
    if returns_matrix.ndim != 2:
        raise ValueError("Monte Carlo matrices must be 2-D with shape [months, assets].")

    horizon_months, asset_count = returns_matrix.shape
    if asset_count != len(plan.tickers):
        raise ValueError("Monte Carlo matrices do not match the asset count in the simulation plan.")
    if horizon_months <= 0:
        raise ValueError("Monte Carlo horizon must be positive.")

    balances = plan.initial_balances.copy()
    yearly_balances = []
    yearly_real_balances = []
    yearly_survival_flags = []
    yearly_ruin_flags = []
    yearly_shortfall_flags = []

    min_balance = _portfolio_total(balances)
    min_real_balance = min_balance
    first_ruin_month = None
    first_shortfall_month = None
    growth_product = 1.0
    cagr_valid = True

    for i in range(horizon_months):
        year, month = _calendar_year_month(plan.start_period, i)

        contribution_allowed = portfolio_inputs.contribution_end_year is None or year <= int(portfolio_inputs.contribution_end_year)
        contribution_gross = float(portfolio_inputs.contribution_amount) / 12.0 if contribution_allowed else 0.0
        balances, _, contribution_trade_cost = _apply_contribution_trade(
            balances=balances,
            gross_amount=contribution_gross,
            target_weights=plan.target_weights,
            asset_type_map=plan.asset_type_map,
            base_bps=portfolio_inputs.cashflow_trade_cost_bps,
            portfolio_inputs=portfolio_inputs,
        )

        total_trading_cost_usd = float(contribution_trade_cost)
        balances, _, _, rebalance_cost_contrib, _ = maybe_rebalance(
            balances,
            plan.target_weights,
            portfolio_inputs.rebalancing_method,
            portfolio_inputs.rebalance_band,
            "post_contribution",
            portfolio_inputs.rebalance_cost_bps,
            plan.asset_type_map,
            portfolio_inputs,
        )
        total_trading_cost_usd += float(rebalance_cost_contrib)

        current_total_after_contrib = _portfolio_total(balances)
        gross_dividend_usd = 0.0
        for j, ticker in enumerate(plan.tickers):
            div_val = float(dividends_matrix[i, j])
            if not math.isfinite(div_val):
                raise ValueError(f"Invalid dividend data for {ticker} at simulation month {i + 1}.")
            gross_dividend_usd += balances[ticker] * (div_val / 100.0)
        dividend_tax = gross_dividend_usd * (float(portfolio_inputs.tax_rate_dividends) / 100.0)
        net_dividend_usd = gross_dividend_usd - dividend_tax

        if portfolio_inputs.withdrawal_mode == "Percent of Balance":
            desired_gross_withdrawal = current_total_after_contrib * plan.monthly_withdrawal_rate
        elif portfolio_inputs.withdrawal_mode == "Fixed Dollar":
            desired_gross_withdrawal = plan.fixed_monthly_withdrawal_base
        elif portfolio_inputs.withdrawal_mode == "Inflation-Adjusted Dollar":
            desired_gross_withdrawal = plan.fixed_monthly_withdrawal_base * ((1.0 + plan.monthly_inflation_rate) ** i)
        elif portfolio_inputs.withdrawal_mode == "Dividend First":
            target_cash_need = current_total_after_contrib * plan.monthly_withdrawal_rate
            desired_gross_withdrawal = max(target_cash_need - net_dividend_usd, 0.0)
        else:
            raise ValueError(f"Unsupported withdrawal mode: {portfolio_inputs.withdrawal_mode}")

        balances, gross_withdrawal, withdrawal_trade_cost, withdrawal_shortfall = _apply_withdrawal_trade(
            balances=balances,
            desired_gross_amount=desired_gross_withdrawal,
            target_weights=plan.target_weights,
            asset_type_map=plan.asset_type_map,
            base_bps=portfolio_inputs.cashflow_trade_cost_bps,
            portfolio_inputs=portfolio_inputs,
        )
        total_trading_cost_usd += float(withdrawal_trade_cost)
        _ = gross_withdrawal * (float(portfolio_inputs.tax_rate_withdrawals) / 100.0)

        balances, _, _, rebalance_cost_withdrawal, _ = maybe_rebalance(
            balances,
            plan.target_weights,
            portfolio_inputs.rebalancing_method,
            portfolio_inputs.rebalance_band,
            "post_withdrawal",
            portfolio_inputs.rebalance_cost_bps,
            plan.asset_type_map,
            portfolio_inputs,
        )
        total_trading_cost_usd += float(rebalance_cost_withdrawal)

        pre_market_balance = _portfolio_total(balances)
        pre_market_weights = _weight_map(balances)
        weighted_price_return = 0.0
        for j, ticker in enumerate(plan.tickers):
            return_val = float(returns_matrix[i, j])
            if not math.isfinite(return_val):
                raise ValueError(f"Invalid return data for {ticker} at simulation month {i + 1}.")
            weighted_price_return += return_val * (pre_market_weights[ticker] / 100.0)
        net_dividend_yield_pct = ((net_dividend_usd / pre_market_balance) * 100.0) if pre_market_balance > 0 else 0.0

        if portfolio_inputs.reinvest_dividends and net_dividend_usd > 0:
            balances, _, dividend_reinvestment_cost = _apply_contribution_trade(
                balances=balances,
                gross_amount=net_dividend_usd,
                target_weights=plan.target_weights,
                asset_type_map=plan.asset_type_map,
                base_bps=portfolio_inputs.cashflow_trade_cost_bps,
                portfolio_inputs=portfolio_inputs,
            )
            total_trading_cost_usd += float(dividend_reinvestment_cost)

        pre_fee_balance = _portfolio_total(balances)
        fee_usd = pre_fee_balance * plan.monthly_fee_rate
        if fee_usd > 0:
            balances = _allocate_cashflow_proportionally(balances, fee_usd, positive=False)

        for j, ticker in enumerate(plan.tickers):
            balances[ticker] *= 1.0 + (float(returns_matrix[i, j]) / 100.0)
            if not math.isfinite(balances[ticker]):
                raise ValueError(f"Non-finite balance produced for {ticker} at simulation month {i + 1}.")

        month_end_stage = "year_end" if month == 12 else "month_end"
        balances, ending_balance, _, rebalance_cost_end, _ = maybe_rebalance(
            balances,
            plan.target_weights,
            portfolio_inputs.rebalancing_method,
            portfolio_inputs.rebalance_band,
            month_end_stage,
            portfolio_inputs.rebalance_cost_bps,
            plan.asset_type_map,
            portfolio_inputs,
        )
        total_trading_cost_usd += float(rebalance_cost_end)
        if ending_balance <= 0:
            ending_balance = _portfolio_total(balances)

        internal_cost_base = pre_fee_balance if pre_fee_balance > 0 else max(pre_market_balance, 1.0)
        cost_drag_pct = ((fee_usd + total_trading_cost_usd) / internal_cost_base) * 100.0 if internal_cost_base > 0 else 0.0
        portfolio_total_return = weighted_price_return + net_dividend_yield_pct - cost_drag_pct
        monthly_growth = 1.0 + (portfolio_total_return / 100.0)
        if monthly_growth > 0 and math.isfinite(monthly_growth) and cagr_valid:
            growth_product *= monthly_growth
        else:
            cagr_valid = False

        discount = (1.0 + plan.monthly_inflation_rate) ** i
        real_balance = ending_balance / discount if discount else ending_balance
        min_balance = min(min_balance, float(ending_balance))
        min_real_balance = min(min_real_balance, float(real_balance))

        if first_ruin_month is None and float(ending_balance) <= 0.0:
            first_ruin_month = i + 1
        if first_shortfall_month is None and float(withdrawal_shortfall) > 1e-9:
            first_shortfall_month = i + 1

        ruin_so_far = 1.0 if first_ruin_month is not None else 0.0
        shortfall_so_far = 1.0 if first_shortfall_month is not None else 0.0
        survival_so_far = 1.0 if (ruin_so_far == 0.0 and shortfall_so_far == 0.0) else 0.0

        if ((i + 1) % 12 == 0) or (i == horizon_months - 1):
            yearly_balances.append(float(ending_balance))
            yearly_real_balances.append(float(real_balance))
            yearly_survival_flags.append(float(survival_so_far))
            yearly_ruin_flags.append(float(ruin_so_far))
            yearly_shortfall_flags.append(float(shortfall_so_far))

    failure_flag = 1.0 if first_ruin_month is not None or first_shortfall_month is not None else 0.0
    failure_month = None
    if first_ruin_month is not None and first_shortfall_month is not None:
        failure_month = min(first_ruin_month, first_shortfall_month)
    elif first_ruin_month is not None:
        failure_month = first_ruin_month
    elif first_shortfall_month is not None:
        failure_month = first_shortfall_month

    cagr = ((growth_product ** (12.0 / horizon_months)) - 1.0) * 100.0 if cagr_valid else np.nan
    return MonteCarloPathResult(
        yearly_balances=np.asarray(yearly_balances, dtype=float),
        yearly_real_balances=np.asarray(yearly_real_balances, dtype=float),
        yearly_survival_flags=np.asarray(yearly_survival_flags, dtype=float),
        yearly_ruin_flags=np.asarray(yearly_ruin_flags, dtype=float),
        yearly_shortfall_flags=np.asarray(yearly_shortfall_flags, dtype=float),
        ending_balance=float(yearly_balances[-1]),
        real_ending_balance=float(yearly_real_balances[-1]),
        ruin=1.0 if first_ruin_month is not None else 0.0,
        shortfall=1.0 if first_shortfall_month is not None else 0.0,
        failure=float(failure_flag),
        cagr=float(cagr) if not pd.isna(cagr) else np.nan,
        min_balance=float(min_balance),
        min_real_balance=float(min_real_balance),
        depletion_month=first_ruin_month,
        shortfall_month=first_shortfall_month,
        failure_month=failure_month,
    )
