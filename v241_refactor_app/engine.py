from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .analytics import add_real_dollar_columns
from .models import AssetConfig, HistoricalDataset, HistoricalSelection, PortfolioInputs
from .utils import _annual_pct_to_monthly_decimal, _validate_matrix

def _allocate_cashflow_proportionally(
    balances: Dict[str, float],
    amount: float,
    positive: bool = True,
) -> Dict[str, float]:
    total = float(sum(max(v, 0.0) for v in balances.values()))
    if total <= 0 or amount == 0:
        return balances.copy()
    updated = balances.copy()
    for ticker, balance in balances.items():
        weight = max(balance, 0.0) / total if total > 0 else 0.0
        delta = amount * weight
        updated[ticker] = balance + delta if positive else balance - delta
    return updated

def _portfolio_total(balances: Dict[str, float]) -> float:
    return float(sum(max(v, 0.0) for v in balances.values()))

def _weight_map(balances: Dict[str, float]) -> Dict[str, float]:
    total = _portfolio_total(balances)
    if total <= 0:
        return {ticker: 0.0 for ticker in balances}
    return {ticker: (max(balance, 0.0) / total) * 100.0 for ticker, balance in balances.items()}

def _target_dollar_map(total: float, target_weights: Dict[str, float]) -> Dict[str, float]:
    return {ticker: max(total, 0.0) * (float(target_weights[ticker]) / 100.0) for ticker in target_weights}

def _proportional_split(source_map: Dict[str, float], amount: float, eligible: Optional[Sequence[str]] = None) -> Dict[str, float]:
    keys = list(eligible) if eligible is not None else list(source_map.keys())
    out = {key: 0.0 for key in keys}
    amount = float(amount)
    if amount <= 0 or not keys:
        return out

    positive_map = {key: max(float(source_map.get(key, 0.0)), 0.0) for key in keys}
    total = float(sum(positive_map.values()))
    if total <= 0:
        equal_amount = amount / len(keys)
        return {key: equal_amount for key in keys}

    running = 0.0
    last_key = keys[-1]
    for key in keys[:-1]:
        alloc = amount * (positive_map[key] / total)
        out[key] = alloc
        running += alloc
    out[last_key] = max(amount - running, 0.0)
    return out

def _targeted_contribution_trades(
    balances: Dict[str, float],
    target_weights: Dict[str, float],
    amount: float,
) -> Dict[str, float]:
    trades = {ticker: 0.0 for ticker in balances}
    amount = max(float(amount), 0.0)
    if amount <= 0:
        return trades

    total = _portfolio_total(balances)
    desired_total = total + amount
    target_dollars = _target_dollar_map(desired_total, target_weights)
    deficits = {ticker: max(target_dollars[ticker] - balances[ticker], 0.0) for ticker in balances}
    deficit_total = float(sum(deficits.values()))
    allocated = 0.0

    if deficit_total > 1e-12:
        first_pass = min(amount, deficit_total)
        split = _proportional_split(deficits, first_pass)
        for ticker, alloc in split.items():
            trades[ticker] += alloc
        allocated += first_pass

    remaining = max(amount - allocated, 0.0)
    if remaining > 1e-12:
        split = _proportional_split(target_weights, remaining)
        for ticker, alloc in split.items():
            trades[ticker] += alloc

    diff = amount - float(sum(trades.values()))
    if abs(diff) > 1e-8 and trades:
        last_key = list(trades.keys())[-1]
        trades[last_key] += diff
    return {ticker: max(value, 0.0) for ticker, value in trades.items()}

def _targeted_withdrawal_trades(
    balances: Dict[str, float],
    target_weights: Dict[str, float],
    amount: float,
) -> Dict[str, float]:
    trades = {ticker: 0.0 for ticker in balances}
    total = _portfolio_total(balances)
    amount = min(max(float(amount), 0.0), total)
    if amount <= 0 or total <= 0:
        return trades

    desired_total = max(total - amount, 0.0)
    target_dollars = _target_dollar_map(desired_total, target_weights)
    excesses = {ticker: max(balances[ticker] - target_dollars[ticker], 0.0) for ticker in balances}
    excess_total = float(sum(excesses.values()))
    allocated = 0.0

    if excess_total > 1e-12:
        first_pass = min(amount, excess_total)
        split = _proportional_split(excesses, first_pass)
        for ticker, alloc in split.items():
            trades[ticker] += min(alloc, max(balances[ticker], 0.0))
        allocated = float(sum(trades.values()))

    remaining = max(amount - allocated, 0.0)
    if remaining > 1e-12:
        availability = {ticker: max(balances[ticker] - trades[ticker], 0.0) for ticker in balances}
        eligible = [ticker for ticker, available in availability.items() if available > 1e-12]
        if eligible:
            split = _proportional_split(availability, remaining, eligible=eligible)
            for ticker, alloc in split.items():
                trades[ticker] += min(alloc, availability[ticker])

    current_total = float(sum(trades.values()))
    if current_total + 1e-8 < amount:
        availability = {ticker: max(balances[ticker] - trades[ticker], 0.0) for ticker in balances}
        eligible = [ticker for ticker, available in availability.items() if available > 1e-12]
        if eligible:
            split = _proportional_split(availability, amount - current_total, eligible=eligible)
            for ticker, alloc in split.items():
                trades[ticker] += min(alloc, availability[ticker])

    return {
        ticker: min(max(value, 0.0), max(balances[ticker], 0.0))
        for ticker, value in trades.items()
    }

def _asset_class_cost_multiplier(asset_type: str, portfolio_inputs: PortfolioInputs) -> float:
    if not portfolio_inputs.asset_class_aware_costs:
        return 1.0
    return float(portfolio_inputs.crypto_cost_multiplier) if str(asset_type) == "Crypto" else 1.0

def _aum_cost_multiplier(total_balance: float, portfolio_inputs: PortfolioInputs) -> float:
    if not portfolio_inputs.aum_cost_scaling or total_balance <= 1_000_000:
        return 1.0
    decades_above_million = math.log10(max(total_balance, 1.0) / 1_000_000.0)
    return 1.0 + max(decades_above_million, 0.0) * float(portfolio_inputs.aum_cost_scaling_strength)

def _effective_trade_bps(
    base_bps: float,
    asset_type: str,
    total_balance: float,
    portfolio_inputs: PortfolioInputs,
) -> float:
    return float(base_bps) * _asset_class_cost_multiplier(asset_type, portfolio_inputs) * _aum_cost_multiplier(total_balance, portfolio_inputs)

def _cashflow_cost_from_trades(
    trades: Dict[str, float],
    asset_type_map: Dict[str, str],
    base_bps: float,
    total_balance: float,
    portfolio_inputs: PortfolioInputs,
) -> float:
    total_cost = 0.0
    for ticker, trade_amount in trades.items():
        notional = abs(float(trade_amount))
        if notional <= 0:
            continue
        eff_bps = _effective_trade_bps(base_bps, asset_type_map.get(ticker, "Stock"), total_balance, portfolio_inputs)
        total_cost += notional * (eff_bps / 10000.0)
    return float(total_cost)

def _rebalance_cost_from_target_shift(
    old_balances: Dict[str, float],
    new_balances: Dict[str, float],
    asset_type_map: Dict[str, str],
    base_bps: float,
    total_balance: float,
    portfolio_inputs: PortfolioInputs,
) -> float:
    total_cost = 0.0
    for ticker in old_balances:
        delta = abs(float(new_balances[ticker]) - float(old_balances[ticker]))
        if delta <= 0:
            continue
        eff_bps = _effective_trade_bps(base_bps, asset_type_map.get(ticker, "Stock"), total_balance, portfolio_inputs)
        total_cost += 0.5 * delta * (eff_bps / 10000.0)
    return float(total_cost)

def _apply_contribution_trade(
    balances: Dict[str, float],
    gross_amount: float,
    target_weights: Dict[str, float],
    asset_type_map: Dict[str, str],
    base_bps: float,
    portfolio_inputs: PortfolioInputs,
) -> Tuple[Dict[str, float], float, float]:
    gross_amount = max(float(gross_amount), 0.0)
    if gross_amount <= 0:
        return balances.copy(), 0.0, 0.0

    investable = gross_amount
    current_total = _portfolio_total(balances)
    for _ in range(3):
        trades = _targeted_contribution_trades(balances, target_weights, investable)
        trade_cost = _cashflow_cost_from_trades(
            trades,
            asset_type_map=asset_type_map,
            base_bps=base_bps,
            total_balance=current_total + investable,
            portfolio_inputs=portfolio_inputs,
        )
        updated_investable = max(gross_amount - trade_cost, 0.0)
        if abs(updated_investable - investable) < 1e-8:
            investable = updated_investable
            break
        investable = updated_investable

    trades = _targeted_contribution_trades(balances, target_weights, investable)
    trade_cost = _cashflow_cost_from_trades(
        trades,
        asset_type_map=asset_type_map,
        base_bps=base_bps,
        total_balance=current_total + investable,
        portfolio_inputs=portfolio_inputs,
    )
    updated = balances.copy()
    for ticker, trade_amount in trades.items():
        updated[ticker] = updated.get(ticker, 0.0) + trade_amount
    return updated, float(sum(trades.values())), float(trade_cost)

def _withdrawal_trade_package(
    balances: Dict[str, float],
    gross_amount: float,
    target_weights: Dict[str, float],
    asset_type_map: Dict[str, str],
    base_bps: float,
    portfolio_inputs: PortfolioInputs,
) -> Tuple[Dict[str, float], float, float]:
    trades = _targeted_withdrawal_trades(balances, target_weights, gross_amount)
    trade_cost = _cashflow_cost_from_trades(
        trades,
        asset_type_map=asset_type_map,
        base_bps=base_bps,
        total_balance=_portfolio_total(balances),
        portfolio_inputs=portfolio_inputs,
    )
    gross_withdrawal = float(sum(trades.values()))
    return trades, float(trade_cost), gross_withdrawal + float(trade_cost)

def _apply_withdrawal_trade(
    balances: Dict[str, float],
    desired_gross_amount: float,
    target_weights: Dict[str, float],
    asset_type_map: Dict[str, str],
    base_bps: float,
    portfolio_inputs: PortfolioInputs,
) -> Tuple[Dict[str, float], float, float, float]:
    total = _portfolio_total(balances)
    desired_gross_amount = max(float(desired_gross_amount), 0.0)
    if desired_gross_amount <= 0 or total <= 0:
        return balances.copy(), 0.0, 0.0, 0.0

    desired_gross_amount = min(desired_gross_amount, total)

    def package_for(gross_amount: float) -> Tuple[Dict[str, float], float, float]:
        return _withdrawal_trade_package(
            balances=balances,
            gross_amount=gross_amount,
            target_weights=target_weights,
            asset_type_map=asset_type_map,
            base_bps=base_bps,
            portfolio_inputs=portfolio_inputs,
        )

    trades, trade_cost, total_needed = package_for(desired_gross_amount)
    gross_withdrawal = float(sum(trades.values()))
    if total_needed > total + 1e-9:
        low, high = 0.0, desired_gross_amount
        for _ in range(30):
            mid = (low + high) / 2.0
            _, mid_cost, mid_total = package_for(mid)
            if mid_total <= total:
                low = mid
            else:
                high = mid
        trades, trade_cost, total_needed = package_for(low)
        gross_withdrawal = float(sum(trades.values()))

    updated = balances.copy()
    for ticker, trade_amount in trades.items():
        updated[ticker] = max(updated.get(ticker, 0.0) - trade_amount, 0.0)
    if trade_cost > 0:
        updated = _allocate_cashflow_proportionally(updated, trade_cost, positive=False)
    shortfall = max(desired_gross_amount - gross_withdrawal, 0.0)
    return updated, gross_withdrawal, float(trade_cost), float(shortfall)

def maybe_rebalance(
    balances: Dict[str, float],
    target_weights: Dict[str, float],
    method: str,
    band: float,
    stage: str,
    rebalance_cost_bps: float,
    asset_type_map: Dict[str, str],
    portfolio_inputs: PortfolioInputs,
) -> Tuple[Dict[str, float], float, bool, float, float]:
    total = _portfolio_total(balances)
    if total <= 0:
        return balances.copy(), 0.0, False, 0.0, 0.0

    method = method or "None"
    should_rebalance = False
    if method == "Annual":
        should_rebalance = stage == "year_end"
    elif method == "Monthly":
        should_rebalance = stage in ("month_end", "year_end")
    elif method == "Threshold Band":
        if stage in ("month_end", "year_end"):
            actual = _weight_map(balances)
            should_rebalance = any(abs(actual[k] - target_weights[k]) > band for k in balances)
    elif method == "Contributions Only":
        should_rebalance = stage == "post_contribution"
    elif method == "Withdrawals Only":
        should_rebalance = stage == "post_withdrawal"

    if not should_rebalance:
        return balances.copy(), total, False, 0.0, 0.0

    new_balances = {ticker: total * (target_weights[ticker] / 100.0) for ticker in balances}
    turnover_usd = sum(abs(new_balances[ticker] - balances[ticker]) for ticker in balances) / 2.0
    turnover_pct = (turnover_usd / total * 100.0) if total > 0 else 0.0
    cost_usd = _rebalance_cost_from_target_shift(
        old_balances=balances,
        new_balances=new_balances,
        asset_type_map=asset_type_map,
        base_bps=rebalance_cost_bps,
        total_balance=total,
        portfolio_inputs=portfolio_inputs,
    )
    if cost_usd > 0:
        new_balances = _allocate_cashflow_proportionally(new_balances, cost_usd, positive=False)
    return new_balances, _portfolio_total(new_balances), True, cost_usd, turnover_pct

def simulate_portfolio(
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    returns_df: pd.DataFrame,
    dividends_df: pd.DataFrame,
    filtered_periods: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    tickers = [asset.ticker for asset in assets]
    target_weights = {asset.ticker: asset.allocation for asset in assets}
    asset_type_map = {asset.ticker: asset.asset_type for asset in assets}
    balances = {
        asset.ticker: portfolio_inputs.initial_investment * (asset.allocation / 100.0)
        for asset in assets
    }

    monthly_fee_rate = _annual_pct_to_monthly_decimal(portfolio_inputs.annual_fee_rate)
    monthly_inflation_rate = _annual_pct_to_monthly_decimal(portfolio_inputs.inflation_rate)
    monthly_withdrawal_rate = _annual_pct_to_monthly_decimal(portfolio_inputs.withdrawal_rate)
    fixed_monthly_withdrawal_base = portfolio_inputs.initial_investment * (portfolio_inputs.withdrawal_rate / 100.0) / 12.0
    rows: List[Dict[str, float]] = []

    for i, period in enumerate(filtered_periods):
        period = pd.Timestamp(period)
        year = int(period.year)
        month = int(period.month)
        beginning_balance = _portfolio_total(balances)

        contribution_allowed = portfolio_inputs.contribution_end_year is None or year <= portfolio_inputs.contribution_end_year
        contribution_gross = portfolio_inputs.contribution_amount / 12.0 if contribution_allowed else 0.0
        balances, net_contribution, contribution_trade_cost = _apply_contribution_trade(
            balances=balances,
            gross_amount=contribution_gross,
            target_weights=target_weights,
            asset_type_map=asset_type_map,
            base_bps=portfolio_inputs.cashflow_trade_cost_bps,
            portfolio_inputs=portfolio_inputs,
        )

        total_trading_cost_usd = contribution_trade_cost
        total_turnover_pct = 0.0
        balances, _, did_rebalance_contrib, rebalance_cost_contrib, turnover_contrib = maybe_rebalance(
            balances,
            target_weights,
            portfolio_inputs.rebalancing_method,
            portfolio_inputs.rebalance_band,
            "post_contribution",
            portfolio_inputs.rebalance_cost_bps,
            asset_type_map,
            portfolio_inputs,
        )
        total_trading_cost_usd += rebalance_cost_contrib
        total_turnover_pct += turnover_contrib

        current_total_after_contrib = _portfolio_total(balances)
        gross_dividend_usd = 0.0
        for ticker in tickers:
            div_val = float(dividends_df.loc[period, ticker])
            if not math.isfinite(div_val):
                raise ValueError(f"Invalid dividend data for {ticker} in {period:%Y-%m}.")
            gross_dividend_usd += balances[ticker] * (div_val / 100.0)
        dividend_tax = gross_dividend_usd * (portfolio_inputs.tax_rate_dividends / 100.0)
        net_dividend_usd = gross_dividend_usd - dividend_tax

        if portfolio_inputs.withdrawal_mode == "Percent of Balance":
            desired_gross_withdrawal = current_total_after_contrib * monthly_withdrawal_rate
        elif portfolio_inputs.withdrawal_mode == "Fixed Dollar":
            desired_gross_withdrawal = fixed_monthly_withdrawal_base
        elif portfolio_inputs.withdrawal_mode == "Inflation-Adjusted Dollar":
            desired_gross_withdrawal = fixed_monthly_withdrawal_base * ((1.0 + monthly_inflation_rate) ** i)
        elif portfolio_inputs.withdrawal_mode == "Dividend First":
            target_cash_need = current_total_after_contrib * monthly_withdrawal_rate
            desired_gross_withdrawal = max(target_cash_need - net_dividend_usd, 0.0)
        else:
            raise ValueError(f"Unsupported withdrawal mode: {portfolio_inputs.withdrawal_mode}")

        balances, gross_withdrawal, withdrawal_trade_cost, withdrawal_shortfall = _apply_withdrawal_trade(
            balances=balances,
            desired_gross_amount=desired_gross_withdrawal,
            target_weights=target_weights,
            asset_type_map=asset_type_map,
            base_bps=portfolio_inputs.cashflow_trade_cost_bps,
            portfolio_inputs=portfolio_inputs,
        )
        total_trading_cost_usd += withdrawal_trade_cost
        withdrawal_tax = gross_withdrawal * (portfolio_inputs.tax_rate_withdrawals / 100.0)
        net_cash_from_withdrawal = gross_withdrawal - withdrawal_tax

        balances, _, did_rebalance_withdrawal, rebalance_cost_withdrawal, turnover_withdrawal = maybe_rebalance(
            balances,
            target_weights,
            portfolio_inputs.rebalancing_method,
            portfolio_inputs.rebalance_band,
            "post_withdrawal",
            portfolio_inputs.rebalance_cost_bps,
            asset_type_map,
            portfolio_inputs,
        )
        total_trading_cost_usd += rebalance_cost_withdrawal
        total_turnover_pct += turnover_withdrawal

        pre_market_balance = _portfolio_total(balances)
        pre_market_weights = _weight_map(balances)
        weighted_price_return = sum(float(returns_df.loc[period, ticker]) * (pre_market_weights[ticker] / 100.0) for ticker in tickers)
        net_dividend_yield_pct = ((net_dividend_usd / pre_market_balance) * 100.0) if pre_market_balance > 0 else 0.0

        dividend_reinvestment_cost = 0.0
        dividend_reinvested_usd = 0.0
        if portfolio_inputs.reinvest_dividends and net_dividend_usd > 0:
            balances, dividend_reinvested_usd, dividend_reinvestment_cost = _apply_contribution_trade(
                balances=balances,
                gross_amount=net_dividend_usd,
                target_weights=target_weights,
                asset_type_map=asset_type_map,
                base_bps=portfolio_inputs.cashflow_trade_cost_bps,
                portfolio_inputs=portfolio_inputs,
            )
            total_trading_cost_usd += dividend_reinvestment_cost
            total_cashflow_to_user = net_cash_from_withdrawal
        else:
            total_cashflow_to_user = net_cash_from_withdrawal + net_dividend_usd

        pre_fee_balance = _portfolio_total(balances)
        fee_usd = pre_fee_balance * monthly_fee_rate
        if fee_usd > 0:
            balances = _allocate_cashflow_proportionally(balances, fee_usd, positive=False)

        pre_return_balance = _portfolio_total(balances)
        for ticker in tickers:
            return_val = float(returns_df.loc[period, ticker])
            if not math.isfinite(return_val):
                raise ValueError(f"Invalid return data for {ticker} in {period:%Y-%m}.")
            balances[ticker] *= 1.0 + (return_val / 100.0)
            if not math.isfinite(balances[ticker]):
                raise ValueError(f"Non-finite balance produced for {ticker} in {period:%Y-%m}.")

        month_end_stage = "year_end" if month == 12 else "month_end"
        balances, ending_balance, did_rebalance_end, rebalance_cost_end, turnover_end = maybe_rebalance(
            balances,
            target_weights,
            portfolio_inputs.rebalancing_method,
            portfolio_inputs.rebalance_band,
            month_end_stage,
            portfolio_inputs.rebalance_cost_bps,
            asset_type_map,
            portfolio_inputs,
        )
        total_trading_cost_usd += rebalance_cost_end
        total_turnover_pct += turnover_end
        if ending_balance <= 0:
            ending_balance = _portfolio_total(balances)

        internal_cost_base = pre_fee_balance if pre_fee_balance > 0 else max(pre_market_balance, 1.0)
        cost_drag_pct = ((fee_usd + total_trading_cost_usd) / internal_cost_base) * 100.0 if internal_cost_base > 0 else 0.0
        portfolio_total_return = weighted_price_return + net_dividend_yield_pct - cost_drag_pct
        actual_weights = _weight_map(balances)

        rows.append(
            {
                "Period": period,
                "Year": year,
                "Month": month,
                "Balance Before Contributions (USD)": beginning_balance,
                "Contribution (USD)": net_contribution,
                "Contribution Trade Cost (USD)": contribution_trade_cost,
                "Gross Dividend Yield (USD)": gross_dividend_usd,
                "Dividend Tax (USD)": dividend_tax,
                "Dividend Yield (USD)": net_dividend_usd,
                "Dividend Reinvestment (USD)": dividend_reinvested_usd,
                "Dividend Reinvestment Cost (USD)": dividend_reinvestment_cost,
                "Withdrawal Target (USD)": desired_gross_withdrawal,
                "Gross Withdrawal (USD)": gross_withdrawal,
                "Withdrawal Trade Cost (USD)": withdrawal_trade_cost,
                "Withdrawal Tax (USD)": withdrawal_tax,
                "Withdrawal (USD)": net_cash_from_withdrawal,
                "Withdrawal Shortfall (USD)": withdrawal_shortfall,
                "Fee (USD)": fee_usd,
                "Trading Cost (USD)": total_trading_cost_usd,
                "Turnover (%)": total_turnover_pct,
                "Balance (USD)": ending_balance,
                "Portfolio Price Return (%)": weighted_price_return,
                "Portfolio Dividend Yield (%)": net_dividend_yield_pct,
                "Portfolio Total Return (%)": portfolio_total_return,
                "Total Withdrawal + Dividend (USD)": total_cashflow_to_user,
                "Rebalanced": bool(did_rebalance_contrib or did_rebalance_withdrawal or did_rebalance_end),
                "Ending Weights": ", ".join(f"{ticker}:{actual_weights[ticker]:.1f}%" for ticker in tickers),
            }
        )

    return add_real_dollar_columns(pd.DataFrame(rows), portfolio_inputs.inflation_rate)

def select_historical_window(
    dataset: HistoricalDataset,
    assets: Sequence[AssetConfig],
    selected_range: Tuple[int, int],
) -> HistoricalSelection:
    filter_start, filter_end = selected_range
    filtered_periods = [period for period in dataset.returns_df.index if filter_start <= period.year <= filter_end]
    if not filtered_periods:
        raise ValueError("No months remain after applying the selected year range.")

    selected_returns_df = _validate_matrix(dataset.returns_df.loc[filtered_periods], "selected return")
    selected_divs_df = _validate_matrix(dataset.dividends_df.loc[filtered_periods], "selected dividend", allow_fill_zero=True)
    weights = [asset.allocation / 100.0 for asset in assets]
    weighted_returns = selected_returns_df.mul(weights, axis=1).sum(axis=1)
    weighted_divs = selected_divs_df.mul(weights, axis=1).sum(axis=1)
    component_df = dataset.component_df[dataset.component_df["Year"].between(filter_start, filter_end)].copy()
    return HistoricalSelection(
        results_df=pd.DataFrame(),
        component_df=component_df,
        weighted_returns=weighted_returns,
        weighted_divs=weighted_divs,
        filtered_periods=list(filtered_periods),
        selected_returns_df=selected_returns_df,
        selected_divs_df=selected_divs_df,
    )


def run_historical_simulation(
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    dataset: HistoricalDataset,
    selected_range: Tuple[int, int],
) -> HistoricalSelection:
    selection = select_historical_window(dataset=dataset, assets=assets, selected_range=selected_range)
    results_df = simulate_portfolio(
        portfolio_inputs=portfolio_inputs,
        assets=assets,
        returns_df=selection.selected_returns_df,
        dividends_df=selection.selected_divs_df,
        filtered_periods=selection.filtered_periods,
    )
    return HistoricalSelection(
        results_df=results_df,
        component_df=selection.component_df,
        weighted_returns=selection.weighted_returns,
        weighted_divs=selection.weighted_divs,
        filtered_periods=selection.filtered_periods,
        selected_returns_df=selection.selected_returns_df,
        selected_divs_df=selection.selected_divs_df,
    )
