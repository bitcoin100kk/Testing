from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .mc_kernel import build_mc_path_plan, simulate_portfolio_path
from .models import AssetConfig, PortfolioInputs


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    probs = np.where(np.isfinite(probs), probs, 0.0)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0.0:
        return np.full(len(probs), 1.0 / max(len(probs), 1), dtype=float)
    return probs / total


def _draw_start_index(n_obs: int, rng: np.random.Generator, start_probabilities: Optional[np.ndarray]) -> int:
    if start_probabilities is None:
        return int(rng.integers(0, n_obs))
    return int(rng.choice(n_obs, p=start_probabilities))


def _sample_block_bootstrap_indices(
    n_obs: int,
    horizon: int,
    block_size: int,
    rng: np.random.Generator,
    start_probabilities: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n_obs <= 0:
        raise ValueError("Historical observations are required for Monte Carlo simulation.")
    block_size = max(1, min(int(block_size), n_obs))
    blocks_needed = int(math.ceil(horizon / block_size))
    sampled = np.empty(horizon, dtype=int)
    write_pos = 0
    for _ in range(blocks_needed):
        start = _draw_start_index(n_obs=n_obs, rng=rng, start_probabilities=start_probabilities)
        block = (start + np.arange(block_size, dtype=int)) % n_obs
        take = min(block_size, horizon - write_pos)
        sampled[write_pos : write_pos + take] = block[:take]
        write_pos += take
        if write_pos >= horizon:
            break
    return sampled


def _sample_stationary_bootstrap_indices(
    n_obs: int,
    horizon: int,
    mean_block_size: int,
    rng: np.random.Generator,
    start_probabilities: Optional[np.ndarray] = None,
) -> np.ndarray:
    if n_obs <= 0:
        raise ValueError("Historical observations are required for Monte Carlo simulation.")
    mean_block_size = max(1, min(int(mean_block_size), n_obs))
    restart_probability = 1.0 / float(mean_block_size)
    sampled = np.empty(horizon, dtype=int)
    current_idx = _draw_start_index(n_obs=n_obs, rng=rng, start_probabilities=start_probabilities)
    for i in range(horizon):
        if i == 0:
            sampled[i] = current_idx
            continue
        if float(rng.random()) < restart_probability:
            current_idx = _draw_start_index(n_obs=n_obs, rng=rng, start_probabilities=start_probabilities)
        else:
            current_idx = (current_idx + 1) % n_obs
        sampled[i] = current_idx
    return sampled


def _build_regime_scores(
    portfolio_total_return_series: pd.Series,
    block_size: int,
    regime_window_months: int,
) -> np.ndarray:
    returns = pd.to_numeric(portfolio_total_return_series, errors="coerce").fillna(0.0).to_numpy(dtype=float) / 100.0
    n_obs = len(returns)
    if n_obs <= 0:
        return np.array([], dtype=float)

    block_size = max(1, min(int(block_size), n_obs))
    window = max(3, min(int(regime_window_months), n_obs))
    scores = np.empty(n_obs, dtype=float)

    for start in range(n_obs):
        block_idx = (start + np.arange(block_size, dtype=int)) % n_obs
        trailing_idx = (start - np.arange(window, dtype=int)) % n_obs
        block = returns[block_idx]
        trailing = returns[trailing_idx]

        block_vol = float(np.std(block, ddof=0))
        block_downside = float(np.sqrt(np.mean(np.square(np.minimum(block, 0.0)))))
        trailing_vol = float(np.std(trailing, ddof=0))
        safe_block = np.maximum(1.0 + block, 1e-12)
        safe_trailing = np.maximum(1.0 + trailing, 1e-12)
        block_cumulative = float(np.prod(safe_block) - 1.0)
        trailing_cumulative = float(np.prod(safe_trailing) - 1.0)
        block_drawdown = max(0.0, -block_cumulative)
        trailing_drawdown = max(0.0, -trailing_cumulative)
        worst_month = max(0.0, -float(np.min(block)))

        scores[start] = (
            (1.50 * block_vol)
            + (1.00 * block_downside)
            + (1.25 * block_drawdown)
            + (1.00 * trailing_drawdown)
            + (0.75 * trailing_vol)
            + (0.50 * worst_month)
        )
    return scores


def _build_regime_start_probabilities(
    regime_scores: np.ndarray,
    regime_mode: str,
    regime_strength: float,
) -> Optional[np.ndarray]:
    n_obs = len(regime_scores)
    if n_obs <= 0 or regime_mode == "All History":
        return None

    scores = np.asarray(regime_scores, dtype=float)
    score_std = float(np.std(scores, ddof=0))
    if not math.isfinite(score_std) or score_std <= 1e-12:
        return None

    z_scores = (scores - float(np.mean(scores))) / score_std
    tilt = max(float(regime_strength), 0.0)
    if tilt <= 1e-12:
        return None

    direction = 1.0 if regime_mode == "Stress Blocks" else -1.0 if regime_mode == "Calm Blocks" else 0.0
    if direction == 0.0:
        return None

    logits = np.clip(direction * tilt * z_scores, -50.0, 50.0)
    weights = np.exp(logits - np.max(logits))
    return _normalize_probabilities(weights)


def _build_mc_convergence_df(
    ending_df: pd.DataFrame,
    *,
    step: int = 250,
    quantile_tolerance_pct: float = 1.0,
    target_stderr_pct: float = 0.35,
) -> pd.DataFrame:
    if ending_df.empty:
        return pd.DataFrame(
            columns=[
                "Sim Count",
                "P05 Ending Balance",
                "Median Ending Balance",
                "P95 Ending Balance",
                "Failure Rate (%)",
                "Failure StdErr (%)",
                "Ruin Rate (%)",
                "Spending Shortfall Rate (%)",
                "Max Quantile Drift (%)",
                "Stop Eligible",
            ]
        )

    step = max(50, int(step))
    checkpoints = sorted(set(list(range(step, len(ending_df) + 1, step)) + [len(ending_df)]))
    rows: List[Dict[str, float | bool]] = []
    previous_quantiles: Optional[np.ndarray] = None

    for count in checkpoints:
        sample = ending_df.iloc[:count]
        failure_rate = float(sample["Failure"].mean())
        std_err = math.sqrt(max(failure_rate * (1.0 - failure_rate), 0.0) / count) * 100.0
        quantiles = np.array(
            [
                float(sample["Ending Balance"].quantile(0.05)),
                float(sample["Ending Balance"].median()),
                float(sample["Ending Balance"].quantile(0.95)),
            ],
            dtype=float,
        )
        max_drift = float("nan")
        if previous_quantiles is not None:
            drifts = []
            for current_val, prev_val in zip(quantiles, previous_quantiles):
                scale = max(abs(float(current_val)), abs(float(prev_val)), 1000.0)
                drifts.append(abs(float(current_val) - float(prev_val)) / scale * 100.0)
            max_drift = float(max(drifts)) if drifts else float("nan")
        stop_eligible = (
            count >= max(500, step * 2)
            and std_err <= float(target_stderr_pct)
            and (math.isnan(max_drift) or max_drift <= float(quantile_tolerance_pct))
        )
        rows.append(
            {
                "Sim Count": float(count),
                "P05 Ending Balance": quantiles[0],
                "Median Ending Balance": quantiles[1],
                "P95 Ending Balance": quantiles[2],
                "Failure Rate (%)": failure_rate * 100.0,
                "Failure StdErr (%)": std_err,
                "Ruin Rate (%)": float(sample["Ruin"].mean() * 100.0),
                "Spending Shortfall Rate (%)": float(sample["Shortfall"].mean() * 100.0),
                "Max Quantile Drift (%)": max_drift,
                "Stop Eligible": bool(stop_eligible),
            }
        )
        previous_quantiles = quantiles
    return pd.DataFrame(rows)


def _should_stop_early(
    ending_df: pd.DataFrame,
    *,
    sim_count: int,
    checkpoint_step: int,
    target_stderr_pct: float,
    quantile_tolerance_pct: float,
) -> bool:
    if sim_count < max(500, checkpoint_step * 2):
        return False
    convergence_df = _build_mc_convergence_df(
        ending_df.iloc[:sim_count],
        step=checkpoint_step,
        quantile_tolerance_pct=quantile_tolerance_pct,
        target_stderr_pct=target_stderr_pct,
    )
    if convergence_df.empty:
        return False
    latest = convergence_df.iloc[-1]
    return bool(latest.get("Stop Eligible", False))


def _year_from_month(month_number: Optional[int]) -> float:
    if month_number is None or pd.isna(month_number):
        return float("nan")
    return float(math.ceil(float(month_number) / 12.0))




@dataclass(frozen=True)
class MonteCarloForwardSpec:
    mode: str
    return_haircut_pct: float
    return_shift_pct: float
    vol_multiplier: float
    aggressive_extra_haircut_pct: float
    crypto_extra_haircut_pct: float
    dividend_multiplier: float
    target_return_by_bucket: Optional[Dict[str, float]] = None
    vol_multiplier_by_bucket: Optional[Dict[str, float]] = None



@dataclass(frozen=True)
class MonteCarloForwardContext:
    mode: str
    hist_monthly_mu: np.ndarray
    target_monthly_mu: np.ndarray
    bucket_vol: np.ndarray
    dividend_multiplier: float
    bucket_map: Dict[str, str]


def _effective_forward_spec(portfolio_inputs: PortfolioInputs) -> MonteCarloForwardSpec:
    mode = str(getattr(portfolio_inputs, "monte_carlo_forward_mode", "Historical Base"))
    presets = {
        "Historical Base": MonteCarloForwardSpec(
            mode="Historical Base",
            return_haircut_pct=0.0,
            return_shift_pct=0.0,
            vol_multiplier=1.0,
            aggressive_extra_haircut_pct=0.0,
            crypto_extra_haircut_pct=0.0,
            dividend_multiplier=1.0,
        ),
        "Post-Bull Haircut": MonteCarloForwardSpec(
            mode="Post-Bull Haircut",
            return_haircut_pct=35.0,
            return_shift_pct=-2.0,
            vol_multiplier=1.15,
            aggressive_extra_haircut_pct=15.0,
            crypto_extra_haircut_pct=10.0,
            dividend_multiplier=0.85,
        ),
        "Stagnation & De-Rating": MonteCarloForwardSpec(
            mode="Stagnation & De-Rating",
            return_haircut_pct=65.0,
            return_shift_pct=-4.0,
            vol_multiplier=1.35,
            aggressive_extra_haircut_pct=25.0,
            crypto_extra_haircut_pct=20.0,
            dividend_multiplier=0.70,
        ),
    }
    if mode == "Bucket CMA Targets":
        return MonteCarloForwardSpec(
            mode="Bucket CMA Targets",
            return_haircut_pct=0.0,
            return_shift_pct=0.0,
            vol_multiplier=1.0,
            aggressive_extra_haircut_pct=0.0,
            crypto_extra_haircut_pct=0.0,
            dividend_multiplier=float(getattr(portfolio_inputs, "monte_carlo_dividend_multiplier", 1.0)),
            target_return_by_bucket={
                "defensive": float(getattr(portfolio_inputs, "monte_carlo_bucket_cma_defensive_return_pct", 4.0)),
                "core": float(getattr(portfolio_inputs, "monte_carlo_bucket_cma_core_return_pct", 6.0)),
                "aggressive": float(getattr(portfolio_inputs, "monte_carlo_bucket_cma_aggressive_return_pct", 8.0)),
                "crypto": float(getattr(portfolio_inputs, "monte_carlo_bucket_cma_crypto_return_pct", 12.0)),
            },
            vol_multiplier_by_bucket={
                "defensive": float(getattr(portfolio_inputs, "monte_carlo_bucket_cma_defensive_vol_multiplier", 0.90)),
                "core": float(getattr(portfolio_inputs, "monte_carlo_bucket_cma_core_vol_multiplier", 1.00)),
                "aggressive": float(getattr(portfolio_inputs, "monte_carlo_bucket_cma_aggressive_vol_multiplier", 1.10)),
                "crypto": float(getattr(portfolio_inputs, "monte_carlo_bucket_cma_crypto_vol_multiplier", 1.25)),
            },
        )
    if mode != "Custom Forward Stress":
        return presets.get(mode, presets["Historical Base"])
    return MonteCarloForwardSpec(
        mode="Custom Forward Stress",
        return_haircut_pct=float(getattr(portfolio_inputs, "monte_carlo_return_haircut_pct", 0.0)),
        return_shift_pct=float(getattr(portfolio_inputs, "monte_carlo_return_shift_pct", 0.0)),
        vol_multiplier=float(getattr(portfolio_inputs, "monte_carlo_vol_multiplier", 1.0)),
        aggressive_extra_haircut_pct=float(getattr(portfolio_inputs, "monte_carlo_growth_extra_haircut_pct", 0.0)),
        crypto_extra_haircut_pct=float(getattr(portfolio_inputs, "monte_carlo_crypto_extra_haircut_pct", 0.0)),
        dividend_multiplier=float(getattr(portfolio_inputs, "monte_carlo_dividend_multiplier", 1.0)),
    )


def materialize_forward_assumption_overrides(portfolio_inputs: PortfolioInputs) -> Dict[str, object]:
    forward_spec = _effective_forward_spec(portfolio_inputs)
    overrides: Dict[str, object] = {
        "monte_carlo_return_haircut_pct": float(forward_spec.return_haircut_pct),
        "monte_carlo_return_shift_pct": float(forward_spec.return_shift_pct),
        "monte_carlo_vol_multiplier": float(forward_spec.vol_multiplier),
        "monte_carlo_growth_extra_haircut_pct": float(forward_spec.aggressive_extra_haircut_pct),
        "monte_carlo_crypto_extra_haircut_pct": float(forward_spec.crypto_extra_haircut_pct),
        "monte_carlo_dividend_multiplier": float(forward_spec.dividend_multiplier),
    }
    if forward_spec.target_return_by_bucket:
        overrides.update(
            {
                "monte_carlo_bucket_cma_defensive_return_pct": float(forward_spec.target_return_by_bucket.get("defensive", getattr(portfolio_inputs, "monte_carlo_bucket_cma_defensive_return_pct", 4.0))),
                "monte_carlo_bucket_cma_core_return_pct": float(forward_spec.target_return_by_bucket.get("core", getattr(portfolio_inputs, "monte_carlo_bucket_cma_core_return_pct", 6.0))),
                "monte_carlo_bucket_cma_aggressive_return_pct": float(forward_spec.target_return_by_bucket.get("aggressive", getattr(portfolio_inputs, "monte_carlo_bucket_cma_aggressive_return_pct", 8.0))),
                "monte_carlo_bucket_cma_crypto_return_pct": float(forward_spec.target_return_by_bucket.get("crypto", getattr(portfolio_inputs, "monte_carlo_bucket_cma_crypto_return_pct", 12.0))),
            }
        )
    if forward_spec.vol_multiplier_by_bucket:
        overrides.update(
            {
                "monte_carlo_bucket_cma_defensive_vol_multiplier": float(forward_spec.vol_multiplier_by_bucket.get("defensive", getattr(portfolio_inputs, "monte_carlo_bucket_cma_defensive_vol_multiplier", 0.90))),
                "monte_carlo_bucket_cma_core_vol_multiplier": float(forward_spec.vol_multiplier_by_bucket.get("core", getattr(portfolio_inputs, "monte_carlo_bucket_cma_core_vol_multiplier", 1.00))),
                "monte_carlo_bucket_cma_aggressive_vol_multiplier": float(forward_spec.vol_multiplier_by_bucket.get("aggressive", getattr(portfolio_inputs, "monte_carlo_bucket_cma_aggressive_vol_multiplier", 1.10))),
                "monte_carlo_bucket_cma_crypto_vol_multiplier": float(forward_spec.vol_multiplier_by_bucket.get("crypto", getattr(portfolio_inputs, "monte_carlo_bucket_cma_crypto_vol_multiplier", 1.25))),
            }
        )
    return overrides


def _classify_forward_buckets(
    historical_returns_df: pd.DataFrame,
    assets: Sequence[AssetConfig],
) -> Dict[str, str]:
    tickers = [asset.ticker for asset in assets]
    bucket_map = {ticker: "core" for ticker in tickers}

    for asset in assets:
        asset_type = str(asset.asset_type)
        if asset_type == "Crypto":
            bucket_map[asset.ticker] = "crypto"
            continue
        if asset_type in {"Bond", "Treasury", "Cash", "TIPS"}:
            bucket_map[asset.ticker] = "defensive"
            continue
        if asset.ticker not in historical_returns_df.columns:
            continue
        monthly_vol = float(pd.to_numeric(historical_returns_df[asset.ticker], errors="coerce").std(ddof=0))
        annualized_vol = monthly_vol * math.sqrt(12.0)
        bucket_map[asset.ticker] = "aggressive" if annualized_vol >= 18.0 else "core"
    return bucket_map


def _annualized_geometric_return_pct(monthly_series: pd.Series) -> float:
    monthly = pd.to_numeric(monthly_series, errors="coerce").dropna().astype(float) / 100.0
    if monthly.empty or not (1.0 + monthly).gt(0.0).all():
        return float("nan")
    return float((np.prod(1.0 + monthly.to_numpy(dtype=float)) ** (12.0 / len(monthly)) - 1.0) * 100.0)


def _monthly_target_from_annual_pct(annual_pct: float) -> float:
    annual_pct = max(float(annual_pct), -99.999999)
    return (((1.0 + (annual_pct / 100.0)) ** (1.0 / 12.0)) - 1.0) * 100.0


def _prepare_forward_overlay_context(
    historical_returns_df: pd.DataFrame,
    assets: Sequence[AssetConfig],
    forward_spec: MonteCarloForwardSpec,
) -> Optional[MonteCarloForwardContext]:
    if forward_spec.mode == "Historical Base":
        return None

    hist_returns = historical_returns_df.to_numpy(dtype=float)
    if hist_returns.ndim != 2:
        raise ValueError("Historical returns must be a 2D matrix for forward-overlay preparation.")

    hist_mu = hist_returns.mean(axis=0)
    bucket_map = _classify_forward_buckets(historical_returns_df, assets)
    target_monthly_mu = np.empty(len(assets), dtype=float)
    bucket_vol = np.empty(len(assets), dtype=float)

    for j, asset in enumerate(assets):
        bucket = bucket_map.get(asset.ticker, "core")
        bucket_extra = forward_spec.aggressive_extra_haircut_pct if bucket == "aggressive" else 0.0
        crypto_extra = forward_spec.crypto_extra_haircut_pct if bucket == "crypto" else 0.0
        total_haircut = min(max(forward_spec.return_haircut_pct + bucket_extra + crypto_extra, 0.0), 100.0)

        hist_monthly_mu = float(hist_mu[j])
        hist_annual = ((1.0 + (hist_monthly_mu / 100.0)) ** 12 - 1.0) * 100.0
        if forward_spec.target_return_by_bucket and bucket in forward_spec.target_return_by_bucket:
            target_annual = float(forward_spec.target_return_by_bucket[bucket])
        else:
            target_annual = (hist_annual * (1.0 - (total_haircut / 100.0))) + float(forward_spec.return_shift_pct)
        target_monthly_mu[j] = _monthly_target_from_annual_pct(target_annual)
        bucket_vol[j] = (
            float(forward_spec.vol_multiplier_by_bucket.get(bucket, forward_spec.vol_multiplier))
            if forward_spec.vol_multiplier_by_bucket
            else float(forward_spec.vol_multiplier)
        )

    return MonteCarloForwardContext(
        mode=str(forward_spec.mode),
        hist_monthly_mu=np.asarray(hist_mu, dtype=float),
        target_monthly_mu=target_monthly_mu,
        bucket_vol=bucket_vol,
        dividend_multiplier=float(forward_spec.dividend_multiplier),
        bucket_map=bucket_map,
    )


def _apply_forward_overlay(
    *,
    sampled_returns: np.ndarray,
    sampled_dividends: np.ndarray,
    historical_returns_df: Optional[pd.DataFrame] = None,
    assets: Optional[Sequence[AssetConfig]] = None,
    forward_spec: Optional[MonteCarloForwardSpec] = None,
    forward_context: Optional[MonteCarloForwardContext] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if forward_context is None:
        if historical_returns_df is None and assets is None and forward_spec is None:
            return sampled_returns, sampled_dividends
        if historical_returns_df is None or assets is None or forward_spec is None:
            raise ValueError("Forward overlay requires either a prepared forward_context or historical data, assets, and forward_spec.")
        forward_context = _prepare_forward_overlay_context(historical_returns_df, assets, forward_spec)

    if forward_context is None or forward_context.mode == "Historical Base":
        return sampled_returns, sampled_dividends

    returns_adj = np.asarray(sampled_returns, dtype=float).copy()
    dividends_adj = np.asarray(sampled_dividends, dtype=float).copy()
    returns_adj = forward_context.target_monthly_mu + ((returns_adj - forward_context.hist_monthly_mu) * forward_context.bucket_vol)
    returns_adj = np.clip(returns_adj, -99.999, None)
    dividends_adj = np.clip(dividends_adj * float(forward_context.dividend_multiplier), 0.0, None)
    return returns_adj, dividends_adj



def build_forward_assumption_audit(
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    historical_returns_df: pd.DataFrame,
) -> pd.DataFrame:
    if historical_returns_df.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Asset Type",
                "Forward Bucket",
                "Historical Annual Return (%)",
                "Historical Monthly Vol (%)",
                "Target Annual Return (%)",
                "Applied Vol Multiplier",
                "Dividend Multiplier",
                "Forward Mode",
            ]
        )

    forward_spec = _effective_forward_spec(portfolio_inputs)
    bucket_map = _classify_forward_buckets(historical_returns_df, assets)
    hist_mu = historical_returns_df.mean(axis=0)
    hist_vol = historical_returns_df.std(axis=0, ddof=0)

    rows = []
    for asset in assets:
        hist_monthly_mu = float(hist_mu.get(asset.ticker, 0.0))
        hist_annual = _annualized_geometric_return_pct(historical_returns_df.get(asset.ticker, pd.Series(dtype=float)))
        bucket = bucket_map.get(asset.ticker, "core")
        if forward_spec.target_return_by_bucket and bucket in forward_spec.target_return_by_bucket:
            target_annual = float(forward_spec.target_return_by_bucket[bucket])
        else:
            bucket_extra = forward_spec.aggressive_extra_haircut_pct if bucket == "aggressive" else 0.0
            crypto_extra = forward_spec.crypto_extra_haircut_pct if bucket == "crypto" else 0.0
            total_haircut = min(max(forward_spec.return_haircut_pct + bucket_extra + crypto_extra, 0.0), 100.0)
            target_annual = (hist_annual * (1.0 - (total_haircut / 100.0))) + float(forward_spec.return_shift_pct)
        bucket_vol = float(forward_spec.vol_multiplier_by_bucket.get(bucket, forward_spec.vol_multiplier)) if forward_spec.vol_multiplier_by_bucket else float(forward_spec.vol_multiplier)
        rows.append(
            {
                "Ticker": asset.ticker,
                "Asset Type": str(asset.asset_type),
                "Forward Bucket": bucket.title(),
                "Historical Annual Return (%)": float(hist_annual),
                "Historical Monthly Vol (%)": float(hist_vol.get(asset.ticker, 0.0)),
                "Target Annual Return (%)": float(target_annual),
                "Applied Vol Multiplier": bucket_vol,
                "Dividend Multiplier": float(forward_spec.dividend_multiplier),
                "Forward Mode": str(forward_spec.mode),
            }
        )
    return pd.DataFrame(rows)


def _build_mc_summary_df(
    *,
    actual_sims: int,
    ending_balance_arr: np.ndarray,
    real_ending_balance_arr: np.ndarray,
    ruin_arr: np.ndarray,
    shortfall_arr: np.ndarray,
    failure_arr: np.ndarray,
    cagr_arr: np.ndarray,
    min_balance_arr: np.ndarray,
    min_real_balance_arr: np.ndarray,
    depletion_month_arr: np.ndarray,
    shortfall_month_arr: np.ndarray,
    failure_month_arr: np.ndarray,
    horizon_years: int,
    percentiles_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    bottom_tail = pd.Series(ending_balance_arr).nsmallest(max(1, int(math.ceil(actual_sims * 0.05))))
    cvar_balance = float(bottom_tail.mean()) if not bottom_tail.empty else 0.0
    failure_rate = float(np.mean(failure_arr)) if actual_sims > 0 else 0.0
    failure_stderr = math.sqrt(max(failure_rate * (1.0 - failure_rate), 0.0) / actual_sims) * 100.0 if actual_sims > 0 else 0.0
    failure_month_series = pd.Series(failure_month_arr).dropna()
    depletion_month_series = pd.Series(depletion_month_arr).dropna()
    shortfall_month_series = pd.Series(shortfall_month_arr).dropna()
    median_failure_year = _year_from_month(float(failure_month_series.median())) if not failure_month_series.empty else float("nan")
    median_depletion_year = _year_from_month(float(depletion_month_series.median())) if not depletion_month_series.empty else float("nan")
    median_shortfall_year = _year_from_month(float(shortfall_month_series.median())) if not shortfall_month_series.empty else float("nan")

    summary_rows = [
        ("Simulations Completed", float(actual_sims), "Count"),
        ("Median Ending Balance", float(np.median(ending_balance_arr)), "USD"),
        ("Real Median Ending Balance", float(np.median(real_ending_balance_arr)), "USD"),
        ("10th Percentile Ending Balance", float(np.quantile(ending_balance_arr, 0.10)), "USD"),
        ("Real 10th Percentile Ending Balance", float(np.quantile(real_ending_balance_arr, 0.10)), "USD"),
        ("90th Percentile Ending Balance", float(np.quantile(ending_balance_arr, 0.90)), "USD"),
        ("Real 90th Percentile Ending Balance", float(np.quantile(real_ending_balance_arr, 0.90)), "USD"),
        ("Failure Rate (Ruin or Shortfall)", failure_rate * 100.0, "%"),
        ("Failure StdErr", float(failure_stderr), "%"),
        ("Ruin Rate", float(np.mean(ruin_arr) * 100.0), "%"),
        ("Spending Shortfall Rate", float(np.mean(shortfall_arr) * 100.0), "%"),
        ("Median Portfolio CAGR", float(np.nanmedian(cagr_arr)), "%"),
        ("CVaR 5% Ending Balance", cvar_balance, "USD"),
        ("Median Minimum Balance", float(np.median(min_balance_arr)), "USD"),
        ("Median Minimum Real Balance", float(np.median(min_real_balance_arr)), "USD"),
        ("Median Failure Year (conditional)", median_failure_year, "Years"),
        ("Median Depletion Year (conditional)", median_depletion_year, "Years"),
        ("Median Shortfall Year (conditional)", median_shortfall_year, "Years"),
    ]
    if percentiles_df is not None and not percentiles_df.empty:
        for checkpoint_year in (10, 20, 30):
            if checkpoint_year <= horizon_years:
                summary_rows.append(
                    (
                        f"Survival Probability Through Year {checkpoint_year}",
                        float(percentiles_df.loc[percentiles_df["Year"] == checkpoint_year, "Survival Probability (%)"].iloc[0]),
                        "%",
                    )
                )
    return pd.DataFrame(summary_rows, columns=["Metric", "Value", "Unit"])


def _simulate_monte_carlo_internal(
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    historical_returns_df: pd.DataFrame,
    historical_dividends_df: pd.DataFrame,
    start_period: pd.Timestamp,
    *,
    summary_only: bool,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    rng = np.random.default_rng(portfolio_inputs.monte_carlo_seed)
    if historical_returns_df.empty:
        raise ValueError("Historical returns are required for Monte Carlo simulation.")

    horizon_months = int(portfolio_inputs.monte_carlo_years) * 12
    if horizon_months <= 0:
        raise ValueError("Monte Carlo horizon must be at least one year.")
    n_obs = len(historical_returns_df)
    horizon_years = int(math.ceil(horizon_months / 12.0))
    max_sims = int(portfolio_inputs.monte_carlo_sims)

    returns_arr = historical_returns_df.to_numpy(dtype=float)
    dividends_arr = historical_dividends_df.to_numpy(dtype=float)
    if returns_arr.shape != dividends_arr.shape:
        raise ValueError("Return and dividend history must share the same shape for Monte Carlo simulation.")

    weights = np.array([asset.allocation / 100.0 for asset in assets], dtype=float)
    joint_total_return_series = historical_returns_df.add(historical_dividends_df, fill_value=0.0).mul(weights, axis=1).sum(axis=1)
    regime_scores = _build_regime_scores(
        portfolio_total_return_series=joint_total_return_series,
        block_size=portfolio_inputs.monte_carlo_block_size_months,
        regime_window_months=portfolio_inputs.monte_carlo_regime_window_months,
    )
    start_probabilities = _build_regime_start_probabilities(
        regime_scores=regime_scores,
        regime_mode=str(portfolio_inputs.monte_carlo_regime_mode),
        regime_strength=float(portfolio_inputs.monte_carlo_regime_strength),
    )
    plan = build_mc_path_plan(portfolio_inputs, assets, start_period)
    forward_spec = _effective_forward_spec(portfolio_inputs)
    forward_context = _prepare_forward_overlay_context(historical_returns_df, assets, forward_spec)

    nominal_paths = real_paths = survival_paths = ruin_paths = shortfall_paths = None
    if not summary_only:
        nominal_paths = np.empty((max_sims, horizon_years), dtype=float)
        real_paths = np.empty((max_sims, horizon_years), dtype=float)
        survival_paths = np.empty((max_sims, horizon_years), dtype=float)
        ruin_paths = np.empty((max_sims, horizon_years), dtype=float)
        shortfall_paths = np.empty((max_sims, horizon_years), dtype=float)

    ending_balance_arr = np.empty(max_sims, dtype=float)
    real_ending_balance_arr = np.empty(max_sims, dtype=float)
    ruin_arr = np.empty(max_sims, dtype=float)
    shortfall_arr = np.empty(max_sims, dtype=float)
    failure_arr = np.empty(max_sims, dtype=float)
    cagr_arr = np.empty(max_sims, dtype=float)
    min_balance_arr = np.empty(max_sims, dtype=float)
    min_real_balance_arr = np.empty(max_sims, dtype=float)
    depletion_month_arr = np.full(max_sims, np.nan, dtype=float)
    shortfall_month_arr = np.full(max_sims, np.nan, dtype=float)
    failure_month_arr = np.full(max_sims, np.nan, dtype=float)

    checkpoint_step = max(100, min(250, max_sims))
    quantile_tolerance_pct = 1.0
    actual_sims = 0

    for sim in range(max_sims):
        if portfolio_inputs.monte_carlo_bootstrap_method == "Stationary Bootstrap":
            sampled_idx = _sample_stationary_bootstrap_indices(
                n_obs=n_obs,
                horizon=horizon_months,
                mean_block_size=portfolio_inputs.monte_carlo_block_size_months,
                rng=rng,
                start_probabilities=start_probabilities,
            )
        else:
            sampled_idx = _sample_block_bootstrap_indices(
                n_obs=n_obs,
                horizon=horizon_months,
                block_size=portfolio_inputs.monte_carlo_block_size_months,
                rng=rng,
                start_probabilities=start_probabilities,
            )

        sampled_returns = returns_arr[sampled_idx]
        sampled_dividends = dividends_arr[sampled_idx]
        sampled_returns, sampled_dividends = _apply_forward_overlay(
            sampled_returns=sampled_returns,
            sampled_dividends=sampled_dividends,
            forward_context=forward_context,
        )

        path_result = simulate_portfolio_path(
            portfolio_inputs=portfolio_inputs,
            plan=plan,
            returns_matrix=sampled_returns,
            dividends_matrix=sampled_dividends,
        )

        if not summary_only:
            nominal_paths[sim, :] = path_result.yearly_balances
            real_paths[sim, :] = path_result.yearly_real_balances
            survival_paths[sim, :] = path_result.yearly_survival_flags
            ruin_paths[sim, :] = path_result.yearly_ruin_flags
            shortfall_paths[sim, :] = path_result.yearly_shortfall_flags

        ending_balance_arr[sim] = path_result.ending_balance
        real_ending_balance_arr[sim] = path_result.real_ending_balance
        ruin_arr[sim] = path_result.ruin
        shortfall_arr[sim] = path_result.shortfall
        failure_arr[sim] = path_result.failure
        cagr_arr[sim] = path_result.cagr
        min_balance_arr[sim] = path_result.min_balance
        min_real_balance_arr[sim] = path_result.min_real_balance
        depletion_month_arr[sim] = float(path_result.depletion_month) if path_result.depletion_month is not None else np.nan
        shortfall_month_arr[sim] = float(path_result.shortfall_month) if path_result.shortfall_month is not None else np.nan
        failure_month_arr[sim] = float(path_result.failure_month) if path_result.failure_month is not None else np.nan
        actual_sims = sim + 1

        if bool(portfolio_inputs.monte_carlo_adaptive_convergence) and actual_sims < max_sims and (actual_sims % checkpoint_step == 0):
            checkpoint_df = pd.DataFrame(
                {
                    "Ending Balance": ending_balance_arr[:actual_sims],
                    "Real Ending Balance": real_ending_balance_arr[:actual_sims],
                    "Ruin": ruin_arr[:actual_sims],
                    "Shortfall": shortfall_arr[:actual_sims],
                    "Failure": failure_arr[:actual_sims],
                    "CAGR": cagr_arr[:actual_sims],
                }
            )
            if _should_stop_early(
                checkpoint_df,
                sim_count=actual_sims,
                checkpoint_step=checkpoint_step,
                target_stderr_pct=float(portfolio_inputs.monte_carlo_target_stderr_pct),
                quantile_tolerance_pct=quantile_tolerance_pct,
            ):
                break

    ending_balance_arr = ending_balance_arr[:actual_sims]
    real_ending_balance_arr = real_ending_balance_arr[:actual_sims]
    ruin_arr = ruin_arr[:actual_sims]
    shortfall_arr = shortfall_arr[:actual_sims]
    failure_arr = failure_arr[:actual_sims]
    cagr_arr = cagr_arr[:actual_sims]
    min_balance_arr = min_balance_arr[:actual_sims]
    min_real_balance_arr = min_real_balance_arr[:actual_sims]
    depletion_month_arr = depletion_month_arr[:actual_sims]
    shortfall_month_arr = shortfall_month_arr[:actual_sims]
    failure_month_arr = failure_month_arr[:actual_sims]

    if summary_only:
        summary_df = _build_mc_summary_df(
            actual_sims=actual_sims,
            ending_balance_arr=ending_balance_arr,
            real_ending_balance_arr=real_ending_balance_arr,
            ruin_arr=ruin_arr,
            shortfall_arr=shortfall_arr,
            failure_arr=failure_arr,
            cagr_arr=cagr_arr,
            min_balance_arr=min_balance_arr,
            min_real_balance_arr=min_real_balance_arr,
            depletion_month_arr=depletion_month_arr,
            shortfall_month_arr=shortfall_month_arr,
            failure_month_arr=failure_month_arr,
            horizon_years=horizon_years,
            percentiles_df=None,
        )
        return None, summary_df, None, None

    nominal_paths = nominal_paths[:actual_sims, :]
    real_paths = real_paths[:actual_sims, :]
    survival_paths = survival_paths[:actual_sims, :]
    ruin_paths = ruin_paths[:actual_sims, :]
    shortfall_paths = shortfall_paths[:actual_sims, :]

    years = np.arange(1, horizon_years + 1, dtype=int)
    percentiles_df = pd.DataFrame(
        {
            "Year": years,
            "P10": np.quantile(nominal_paths, 0.10, axis=0),
            "Median": np.quantile(nominal_paths, 0.50, axis=0),
            "P90": np.quantile(nominal_paths, 0.90, axis=0),
            "Real P10": np.quantile(real_paths, 0.10, axis=0),
            "Real Median": np.quantile(real_paths, 0.50, axis=0),
            "Real P90": np.quantile(real_paths, 0.90, axis=0),
            "Survival Probability (%)": survival_paths.mean(axis=0) * 100.0,
            "Ruin by Year (%)": ruin_paths.mean(axis=0) * 100.0,
            "Shortfall by Year (%)": shortfall_paths.mean(axis=0) * 100.0,
            "Failure by Year (%)": np.maximum(ruin_paths, shortfall_paths).mean(axis=0) * 100.0,
        }
    )

    paths_df = pd.DataFrame(
        {
            "Year": np.tile(years, actual_sims),
            "Balance (USD)": nominal_paths.reshape(-1),
            "Real Balance (USD)": real_paths.reshape(-1),
            "Simulation": np.repeat(np.arange(1, actual_sims + 1, dtype=int), horizon_years),
        }
    )

    ending_df = pd.DataFrame(
        {
            "Ending Balance": ending_balance_arr,
            "Real Ending Balance": real_ending_balance_arr,
            "Ruin": ruin_arr,
            "Shortfall": shortfall_arr,
            "Failure": failure_arr,
            "CAGR": cagr_arr,
            "Minimum Balance": min_balance_arr,
            "Minimum Real Balance": min_real_balance_arr,
            "Depletion Month": depletion_month_arr,
            "Shortfall Month": shortfall_month_arr,
            "Failure Month": failure_month_arr,
        }
    )

    summary_df = _build_mc_summary_df(
        actual_sims=actual_sims,
        ending_balance_arr=ending_balance_arr,
        real_ending_balance_arr=real_ending_balance_arr,
        ruin_arr=ruin_arr,
        shortfall_arr=shortfall_arr,
        failure_arr=failure_arr,
        cagr_arr=cagr_arr,
        min_balance_arr=min_balance_arr,
        min_real_balance_arr=min_real_balance_arr,
        depletion_month_arr=depletion_month_arr,
        shortfall_month_arr=shortfall_month_arr,
        failure_month_arr=failure_month_arr,
        horizon_years=horizon_years,
        percentiles_df=percentiles_df,
    )
    convergence_df = _build_mc_convergence_df(
        ending_df,
        step=checkpoint_step,
        quantile_tolerance_pct=quantile_tolerance_pct,
        target_stderr_pct=float(portfolio_inputs.monte_carlo_target_stderr_pct),
    )
    return percentiles_df, summary_df, paths_df, convergence_df


def simulate_monte_carlo_summary(
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    historical_returns_df: pd.DataFrame,
    historical_dividends_df: pd.DataFrame,
    start_period: pd.Timestamp,
) -> pd.DataFrame:
    _, summary_df, _, _ = _simulate_monte_carlo_internal(
        portfolio_inputs=portfolio_inputs,
        assets=assets,
        historical_returns_df=historical_returns_df,
        historical_dividends_df=historical_dividends_df,
        start_period=start_period,
        summary_only=True,
    )
    return summary_df


def _summary_metric_value(summary_df: pd.DataFrame, metric: str, default: float = float("nan")) -> float:
    if summary_df is None or summary_df.empty:
        return default
    match = summary_df.loc[summary_df["Metric"].eq(metric), "Value"]
    if match.empty:
        return default
    return float(match.iloc[0])


def _wilson_interval(success_rate: float, n: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    n = int(max(n, 0))
    if n <= 0:
        return float("nan"), float("nan")
    p_hat = min(max(float(success_rate), 0.0), 1.0)
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + ((z * z) / (4.0 * n * n)))
    return max(0.0, center - spread), min(1.0, center + spread)


def _bootstrap_stat_interval(values: np.ndarray, stat_fn, *, seed: int, n_resamples: int = 250) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        val = float(stat_fn(arr))
        return val, val
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats[i] = float(stat_fn(sample))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def build_monte_carlo_validation_report(
    *,
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    historical_returns_df: pd.DataFrame,
    historical_dividends_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    paths_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=["Metric", "Value", "Unit", "Assessment"])

    requested_sims = int(max(getattr(portfolio_inputs, "monte_carlo_sims", 0), 0))
    actual_sims = int(round(_summary_metric_value(summary_df, "Simulations Completed", 0.0)))
    failure_rate_pct = _summary_metric_value(summary_df, "Failure Rate (Ruin or Shortfall)", 0.0)
    ruin_rate_pct = _summary_metric_value(summary_df, "Ruin Rate", 0.0)
    shortfall_rate_pct = _summary_metric_value(summary_df, "Spending Shortfall Rate", 0.0)
    real_median_pct = _summary_metric_value(summary_df, "Real Median Ending Balance", float("nan"))
    real_p10_pct = _summary_metric_value(summary_df, "Real 10th Percentile Ending Balance", float("nan"))

    failure_low, failure_high = _wilson_interval(failure_rate_pct / 100.0, actual_sims)
    ruin_low, ruin_high = _wilson_interval(ruin_rate_pct / 100.0, actual_sims)
    shortfall_low, shortfall_high = _wilson_interval(shortfall_rate_pct / 100.0, actual_sims)

    final_year = int(paths_df["Year"].max()) if paths_df is not None and not paths_df.empty else 0
    final_real_balances = np.array([], dtype=float)
    if paths_df is not None and not paths_df.empty and final_year > 0:
        final_real_balances = paths_df.loc[paths_df["Year"].eq(final_year), "Real Balance (USD)"].to_numpy(dtype=float)
    median_low, median_high = _bootstrap_stat_interval(
        final_real_balances,
        np.median,
        seed=int(getattr(portfolio_inputs, "monte_carlo_seed", 0)) + 17,
    )
    p10_low, p10_high = _bootstrap_stat_interval(
        final_real_balances,
        lambda arr: np.quantile(arr, 0.10),
        seed=int(getattr(portfolio_inputs, "monte_carlo_seed", 0)) + 31,
    )

    weights = np.array([float(asset.allocation) / 100.0 for asset in assets], dtype=float)
    joint_total_return_series = historical_returns_df.add(historical_dividends_df, fill_value=0.0).mul(weights, axis=1).sum(axis=1)
    regime_scores = _build_regime_scores(
        portfolio_total_return_series=joint_total_return_series,
        block_size=int(portfolio_inputs.monte_carlo_block_size_months),
        regime_window_months=int(portfolio_inputs.monte_carlo_regime_window_months),
    )
    start_probabilities = _build_regime_start_probabilities(
        regime_scores=regime_scores,
        regime_mode=str(portfolio_inputs.monte_carlo_regime_mode),
        regime_strength=float(portfolio_inputs.monte_carlo_regime_strength),
    )
    n_obs = len(historical_returns_df)
    entropy_ratio = 1.0
    effective_start_months = float(n_obs)
    if start_probabilities is not None and len(start_probabilities) > 0:
        probs = np.asarray(start_probabilities, dtype=float)
        probs = probs[probs > 0.0]
        if probs.size > 0:
            entropy_ratio = float(-(probs * np.log(probs)).sum() / max(math.log(len(start_probabilities)), 1e-12))
            effective_start_months = float(1.0 / np.square(probs).sum())

    final_failure_stderr = float("nan")
    final_quantile_drift = float("nan")
    stop_eligible = False
    if convergence_df is not None and not convergence_df.empty:
        latest = convergence_df.iloc[-1]
        final_failure_stderr = float(latest.get("Failure StdErr (%)", float("nan")))
        final_quantile_drift = float(latest.get("Max Quantile Drift (%)", float("nan")))
        stop_eligible = bool(latest.get("Stop Eligible", False))

    target_stderr_pct = float(getattr(portfolio_inputs, "monte_carlo_target_stderr_pct", 0.0))
    rows = [
        {"Metric": "Convergence status", "Value": "Pass" if stop_eligible or actual_sims >= requested_sims else "Monitor", "Unit": "", "Assessment": "Pass means the final checkpoint met either the requested simulation count or the configured stopping rule."},
        {"Metric": "Requested simulations", "Value": requested_sims, "Unit": "Count", "Assessment": "Configured Monte Carlo budget."},
        {"Metric": "Completed simulations", "Value": actual_sims, "Unit": "Count", "Assessment": "Actual number of simulated paths used in the report."},
        {"Metric": "Simulation utilization", "Value": (actual_sims / requested_sims * 100.0) if requested_sims > 0 else float("nan"), "Unit": "%", "Assessment": "Low utilization means adaptive convergence stopped the run early."},
        {"Metric": "Failure rate 95% CI lower", "Value": failure_low * 100.0, "Unit": "%", "Assessment": "Wilson interval for terminal failure probability."},
        {"Metric": "Failure rate 95% CI upper", "Value": failure_high * 100.0, "Unit": "%", "Assessment": "Wilson interval for terminal failure probability."},
        {"Metric": "Ruin rate 95% CI lower", "Value": ruin_low * 100.0, "Unit": "%", "Assessment": "Wilson interval for terminal ruin probability."},
        {"Metric": "Ruin rate 95% CI upper", "Value": ruin_high * 100.0, "Unit": "%", "Assessment": "Wilson interval for terminal ruin probability."},
        {"Metric": "Shortfall rate 95% CI lower", "Value": shortfall_low * 100.0, "Unit": "%", "Assessment": "Wilson interval for terminal spending-shortfall probability."},
        {"Metric": "Shortfall rate 95% CI upper", "Value": shortfall_high * 100.0, "Unit": "%", "Assessment": "Wilson interval for terminal spending-shortfall probability."},
        {"Metric": "Real median ending balance 95% CI lower", "Value": median_low, "Unit": "USD", "Assessment": "Bootstrap interval across simulated terminal real balances."},
        {"Metric": "Real median ending balance 95% CI upper", "Value": median_high, "Unit": "USD", "Assessment": "Bootstrap interval across simulated terminal real balances."},
        {"Metric": "Real P10 ending balance 95% CI lower", "Value": p10_low, "Unit": "USD", "Assessment": "Bootstrap interval across simulated terminal real balances."},
        {"Metric": "Real P10 ending balance 95% CI upper", "Value": p10_high, "Unit": "USD", "Assessment": "Bootstrap interval across simulated terminal real balances."},
        {"Metric": "Final failure stderr", "Value": final_failure_stderr, "Unit": "%", "Assessment": "Sampling error at the final convergence checkpoint."},
        {"Metric": "Target failure stderr", "Value": target_stderr_pct, "Unit": "%", "Assessment": "Configured convergence target for failure-rate error."},
        {"Metric": "Final max quantile drift", "Value": final_quantile_drift, "Unit": "%", "Assessment": "Largest checkpoint-over-checkpoint drift across P05/Median/P95 ending balance."},
        {"Metric": "Effective distinct start months", "Value": effective_start_months, "Unit": "Count", "Assessment": "Inverse Herfindahl measure of how concentrated the regime tilt makes start-block selection."},
        {"Metric": "Start-weight entropy", "Value": entropy_ratio * 100.0, "Unit": "%", "Assessment": "100% means near-uniform block starts; lower values mean heavier concentration on selected regimes."},
        {"Metric": "Historical months available", "Value": n_obs, "Unit": "Count", "Assessment": "Amount of monthly history available for bootstrap sampling."},
        {"Metric": "Monte Carlo horizon", "Value": int(getattr(portfolio_inputs, "monte_carlo_years", 0)) * 12, "Unit": "Months", "Assessment": "Simulated investment horizon length."},
        {"Metric": "Point estimate :: real median ending balance", "Value": real_median_pct, "Unit": "USD", "Assessment": "Point estimate shown in the summary table."},
        {"Metric": "Point estimate :: real P10 ending balance", "Value": real_p10_pct, "Unit": "USD", "Assessment": "Point estimate shown in the summary table."},
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit", "Assessment"])


def simulate_monte_carlo(
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    historical_returns_df: pd.DataFrame,
    historical_dividends_df: pd.DataFrame,
    start_period: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    percentiles_df, summary_df, paths_df, convergence_df = _simulate_monte_carlo_internal(
        portfolio_inputs=portfolio_inputs,
        assets=assets,
        historical_returns_df=historical_returns_df,
        historical_dividends_df=historical_dividends_df,
        start_period=start_period,
        summary_only=False,
    )
    assert percentiles_df is not None and paths_df is not None and convergence_df is not None
    return percentiles_df, summary_df, paths_df, convergence_df


__all__ = [
    "build_forward_assumption_audit",
    "build_monte_carlo_validation_report",
    "materialize_forward_assumption_overrides",
    "simulate_monte_carlo",
    "simulate_monte_carlo_summary",
]
