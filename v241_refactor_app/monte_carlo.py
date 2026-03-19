from __future__ import annotations

import math
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


def simulate_monte_carlo(
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    historical_returns_df: pd.DataFrame,
    historical_dividends_df: pd.DataFrame,
    start_period: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        regime_mode=portfolio_inputs.monte_carlo_regime_mode,
        regime_strength=portfolio_inputs.monte_carlo_regime_strength,
    )
    plan = build_mc_path_plan(portfolio_inputs=portfolio_inputs, assets=assets, start_period=start_period)

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

        path_result = simulate_portfolio_path(
            portfolio_inputs=portfolio_inputs,
            plan=plan,
            returns_matrix=returns_arr[sampled_idx],
            dividends_matrix=dividends_arr[sampled_idx],
        )

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

        if (
            bool(portfolio_inputs.monte_carlo_adaptive_convergence)
            and actual_sims < max_sims
            and (actual_sims % checkpoint_step == 0)
        ):
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

    nominal_paths = nominal_paths[:actual_sims, :]
    real_paths = real_paths[:actual_sims, :]
    survival_paths = survival_paths[:actual_sims, :]
    ruin_paths = ruin_paths[:actual_sims, :]
    shortfall_paths = shortfall_paths[:actual_sims, :]
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

    bottom_tail = pd.Series(ending_balance_arr).nsmallest(max(1, int(math.ceil(actual_sims * 0.05))))
    cvar_balance = float(bottom_tail.mean()) if not bottom_tail.empty else 0.0
    failure_rate = float(np.mean(failure_arr)) if actual_sims > 0 else 0.0
    failure_stderr = math.sqrt(max(failure_rate * (1.0 - failure_rate), 0.0) / actual_sims) * 100.0 if actual_sims > 0 else 0.0
    median_failure_year = _year_from_month(float(pd.Series(failure_month_arr).dropna().median())) if pd.Series(failure_month_arr).notna().any() else float("nan")
    median_depletion_year = _year_from_month(float(pd.Series(depletion_month_arr).dropna().median())) if pd.Series(depletion_month_arr).notna().any() else float("nan")
    median_shortfall_year = _year_from_month(float(pd.Series(shortfall_month_arr).dropna().median())) if pd.Series(shortfall_month_arr).notna().any() else float("nan")

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
    for checkpoint_year in (10, 20, 30):
        if checkpoint_year <= horizon_years:
            summary_rows.append(
                (
                    f"Survival Probability Through Year {checkpoint_year}",
                    float(percentiles_df.loc[percentiles_df["Year"] == checkpoint_year, "Survival Probability (%)"].iloc[0]),
                    "%",
                )
            )

    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value", "Unit"])
    convergence_df = _build_mc_convergence_df(
        ending_df,
        step=checkpoint_step,
        quantile_tolerance_pct=quantile_tolerance_pct,
        target_stderr_pct=float(portfolio_inputs.monte_carlo_target_stderr_pct),
    )
    return percentiles_df, summary_df, paths_df, convergence_df
