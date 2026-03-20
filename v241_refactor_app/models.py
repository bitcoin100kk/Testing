from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AssetConfig:
    ticker: str
    allocation: float
    asset_type: str


@dataclass
class PortfolioInputs:
    initial_investment: float
    withdrawal_rate: float
    reinvest_dividends: bool
    contribution_amount: float
    contribution_end_year: Optional[int]
    annual_fee_rate: float
    inflation_rate: float
    tax_rate_dividends: float
    tax_rate_withdrawals: float
    withdrawal_mode: str
    benchmark_enabled: bool
    benchmark_ticker: str
    benchmark_type: str
    rebalancing_method: str
    rebalance_band: float
    rebalance_cost_bps: float
    cashflow_trade_cost_bps: float
    asset_class_aware_costs: bool
    aum_cost_scaling: bool
    aum_cost_scaling_strength: float
    crypto_cost_multiplier: float
    analysis_mode: str
    monte_carlo_sims: int
    monte_carlo_years: int
    monte_carlo_seed: int
    monte_carlo_block_size_months: int
    monte_carlo_regime_mode: str
    monte_carlo_regime_window_months: int
    monte_carlo_bootstrap_method: str
    monte_carlo_regime_strength: float
    monte_carlo_adaptive_convergence: bool
    monte_carlo_target_stderr_pct: float
    monte_carlo_forward_mode: str
    monte_carlo_return_haircut_pct: float
    monte_carlo_return_shift_pct: float
    monte_carlo_vol_multiplier: float
    monte_carlo_growth_extra_haircut_pct: float
    monte_carlo_crypto_extra_haircut_pct: float
    monte_carlo_dividend_multiplier: float
    monte_carlo_bucket_cma_defensive_return_pct: float
    monte_carlo_bucket_cma_core_return_pct: float
    monte_carlo_bucket_cma_aggressive_return_pct: float
    monte_carlo_bucket_cma_crypto_return_pct: float
    monte_carlo_bucket_cma_defensive_vol_multiplier: float
    monte_carlo_bucket_cma_core_vol_multiplier: float
    monte_carlo_bucket_cma_aggressive_vol_multiplier: float
    monte_carlo_bucket_cma_crypto_vol_multiplier: float


PORTFOLIO_INPUT_DEFAULTS = {
    "initial_investment": 100000.0,
    "withdrawal_rate": 4.0,
    "reinvest_dividends": False,
    "contribution_amount": 0.0,
    "contribution_end_year": None,
    "annual_fee_rate": 0.0,
    "inflation_rate": 3.0,
    "tax_rate_dividends": 0.0,
    "tax_rate_withdrawals": 0.0,
    "withdrawal_mode": "Percent of Balance",
    "benchmark_enabled": True,
    "benchmark_ticker": "SPY",
    "benchmark_type": "Stock",
    "rebalancing_method": "Annual",
    "rebalance_band": 5.0,
    "rebalance_cost_bps": 5.0,
    "cashflow_trade_cost_bps": 2.0,
    "asset_class_aware_costs": True,
    "aum_cost_scaling": True,
    "aum_cost_scaling_strength": 0.25,
    "crypto_cost_multiplier": 4.0,
    "analysis_mode": "Both",
    "monte_carlo_sims": 2000,
    "monte_carlo_years": 30,
    "monte_carlo_seed": 42,
    "monte_carlo_block_size_months": 12,
    "monte_carlo_regime_mode": "All History",
    "monte_carlo_regime_window_months": 6,
    "monte_carlo_bootstrap_method": "Block Bootstrap",
    "monte_carlo_regime_strength": 1.0,
    "monte_carlo_adaptive_convergence": True,
    "monte_carlo_target_stderr_pct": 0.35,
    "monte_carlo_forward_mode": "Historical Base",
    "monte_carlo_return_haircut_pct": 35.0,
    "monte_carlo_return_shift_pct": -2.0,
    "monte_carlo_vol_multiplier": 1.15,
    "monte_carlo_growth_extra_haircut_pct": 15.0,
    "monte_carlo_crypto_extra_haircut_pct": 10.0,
    "monte_carlo_dividend_multiplier": 0.85,
    "monte_carlo_bucket_cma_defensive_return_pct": 4.0,
    "monte_carlo_bucket_cma_core_return_pct": 6.0,
    "monte_carlo_bucket_cma_aggressive_return_pct": 8.0,
    "monte_carlo_bucket_cma_crypto_return_pct": 12.0,
    "monte_carlo_bucket_cma_defensive_vol_multiplier": 0.90,
    "monte_carlo_bucket_cma_core_vol_multiplier": 1.00,
    "monte_carlo_bucket_cma_aggressive_vol_multiplier": 1.10,
    "monte_carlo_bucket_cma_crypto_vol_multiplier": 1.25,
}


@dataclass
class HistoricalDataset:
    returns_df: pd.DataFrame
    dividends_df: pd.DataFrame
    years: List[int]
    component_df: pd.DataFrame
    diagnostics: List[Dict[str, object]]
    overlap_start: pd.Timestamp
    overlap_end: pd.Timestamp
    overlap_months: int


@dataclass
class HistoricalSelection:
    results_df: pd.DataFrame
    component_df: pd.DataFrame
    weighted_returns: pd.Series
    weighted_divs: pd.Series
    filtered_periods: List[pd.Timestamp]
    selected_returns_df: pd.DataFrame
    selected_divs_df: pd.DataFrame


@dataclass
class RunSnapshot:
    portfolio_inputs: Dict[str, object]
    assets: List[Dict[str, object]]
    token: str
    raw_signature: str
    ui_preferences: Dict[str, object] | None = None


@dataclass
class MonteCarloPathResult:
    yearly_balances: np.ndarray
    yearly_real_balances: np.ndarray
    yearly_survival_flags: np.ndarray
    yearly_ruin_flags: np.ndarray
    yearly_shortfall_flags: np.ndarray
    ending_balance: float
    real_ending_balance: float
    ruin: float
    shortfall: float
    failure: float
    cagr: float
    min_balance: float
    min_real_balance: float
    depletion_month: Optional[int]
    shortfall_month: Optional[int]
    failure_month: Optional[int]


@dataclass
class RunArtifacts:
    portfolio_inputs: PortfolioInputs
    assets: List[AssetConfig]
    year_range: Tuple[int, int]
    dataset: HistoricalDataset
    selection: HistoricalSelection
    metrics: Dict[str, float]
    rolling3: pd.DataFrame
    rolling5: pd.DataFrame
    risk_table: pd.DataFrame
    diagnostics_df: pd.DataFrame
    overlap_warning_lines: List[str]
    overlap_summary_df: pd.DataFrame
    benchmark_comparison_df: Optional[pd.DataFrame] = None
    benchmark_results_df: Optional[pd.DataFrame] = None
    benchmark_summary_table: Optional[pd.DataFrame] = None
    scenario_comparison_df: Optional[pd.DataFrame] = None
    mc_percentiles: Optional[pd.DataFrame] = None
    mc_summary: Optional[pd.DataFrame] = None
    mc_paths: Optional[pd.DataFrame] = None
    mc_convergence: Optional[pd.DataFrame] = None
    export_bytes: Optional[bytes] = None
