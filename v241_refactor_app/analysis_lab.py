from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import AssetConfig, PortfolioInputs
from .monte_carlo import build_forward_assumption_audit, materialize_forward_assumption_overrides, simulate_monte_carlo


SUMMARY_METRICS = {
    "Median Ending Balance": "median_ending_balance",
    "Real Median Ending Balance": "real_median_ending_balance",
    "10th Percentile Ending Balance": "p10_ending_balance",
    "Real 10th Percentile Ending Balance": "real_p10_ending_balance",
    "90th Percentile Ending Balance": "p90_ending_balance",
    "Real 90th Percentile Ending Balance": "real_p90_ending_balance",
    "Failure Rate (Ruin or Shortfall)": "failure_rate_pct",
    "Ruin Rate": "ruin_rate_pct",
    "Spending Shortfall Rate": "shortfall_rate_pct",
    "Median Portfolio CAGR": "median_cagr_pct",
    "CVaR 5% Ending Balance": "cvar_ending_balance",
    "Median Minimum Balance": "median_min_balance",
    "Median Minimum Real Balance": "median_min_real_balance",
}


def _summary_to_dict(summary_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if summary_df is None or summary_df.empty:
        return out
    for _, row in summary_df.iterrows():
        key = SUMMARY_METRICS.get(str(row.get("Metric")))
        if key is None:
            continue
        try:
            out[key] = float(row.get("Value"))
        except Exception:  # noqa: BLE001
            continue
    return out


def _run_mc_summary(
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    historical_returns_df: pd.DataFrame,
    historical_dividends_df: pd.DataFrame,
    start_period: pd.Timestamp,
) -> Dict[str, float]:
    _, summary_df, _, _ = simulate_monte_carlo(
        portfolio_inputs=portfolio_inputs,
        assets=assets,
        historical_returns_df=historical_returns_df,
        historical_dividends_df=historical_dividends_df,
        start_period=start_period,
    )
    return _summary_to_dict(summary_df)


def _analysis_sims(base_sims: int, *, floor: int = 250, ceiling: int = 800) -> int:
    base_sims = int(max(base_sims, 1))
    if base_sims <= floor:
        return base_sims
    return int(min(ceiling, max(base_sims // 4, floor)))


def _analysis_inputs(portfolio_inputs: PortfolioInputs) -> PortfolioInputs:
    return replace(
        portfolio_inputs,
        monte_carlo_sims=_analysis_sims(int(portfolio_inputs.monte_carlo_sims)),
        monte_carlo_adaptive_convergence=False,
    )


def build_fragility_analysis(
    *,
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    historical_returns_df: pd.DataFrame,
    historical_dividends_df: pd.DataFrame,
    start_period: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    analysis_inputs = _analysis_inputs(portfolio_inputs)
    base = _run_mc_summary(
        analysis_inputs,
        assets,
        historical_returns_df,
        historical_dividends_df,
        start_period,
    )

    stress_cases: List[Tuple[str, Dict[str, object]]] = [
        ("Base Case", {}),
        ("Spending +10%", {"withdrawal_rate": float(portfolio_inputs.withdrawal_rate) * 1.10}),
        ("Spending -10%", {"withdrawal_rate": max(float(portfolio_inputs.withdrawal_rate) * 0.90, 0.0)}),
        ("Fees +0.50%", {"annual_fee_rate": float(portfolio_inputs.annual_fee_rate) + 0.50}),
        ("Return Shift -2%", {"monte_carlo_return_shift_pct": float(portfolio_inputs.monte_carlo_return_shift_pct) - 2.0}),
        ("Volatility x1.15", {"monte_carlo_vol_multiplier": float(portfolio_inputs.monte_carlo_vol_multiplier) * 1.15}),
        (
            "Harder Forward Regime",
            {
                "monte_carlo_forward_mode": "Post-Bull Haircut"
                if str(portfolio_inputs.monte_carlo_forward_mode) == "Historical Base"
                else "Stagnation & De-Rating"
            },
        ),
    ]

    rows: List[Dict[str, float | str]] = []
    for label, overrides in stress_cases:
        needs_explicit_forward = any(
            k in overrides
            for k in {
                "monte_carlo_return_shift_pct",
                "monte_carlo_vol_multiplier",
                "monte_carlo_return_haircut_pct",
                "monte_carlo_dividend_multiplier",
            }
        )
        merged_overrides: Dict[str, object] = {}
        if needs_explicit_forward:
            merged_overrides.update(materialize_forward_assumption_overrides(analysis_inputs))
            merged_overrides["monte_carlo_forward_mode"] = "Custom Forward Stress"
        merged_overrides.update(overrides)
        stressed_inputs = replace(analysis_inputs, **merged_overrides)
        metrics = base if label == "Base Case" else _run_mc_summary(
            stressed_inputs,
            assets,
            historical_returns_df,
            historical_dividends_df,
            start_period,
        )
        rows.append(
            {
                "Scenario": label,
                "Failure Rate (%)": float(metrics.get("failure_rate_pct", np.nan)),
                "Ruin Rate (%)": float(metrics.get("ruin_rate_pct", np.nan)),
                "Shortfall Rate (%)": float(metrics.get("shortfall_rate_pct", np.nan)),
                "Median Ending Balance (USD)": float(metrics.get("median_ending_balance", np.nan)),
                "Real Median Ending Balance (USD)": float(metrics.get("real_median_ending_balance", np.nan)),
                "Real P10 Ending Balance (USD)": float(metrics.get("real_p10_ending_balance", np.nan)),
                "Median CAGR (%)": float(metrics.get("median_cagr_pct", np.nan)),
            }
        )

    fragility_df = pd.DataFrame(rows)
    if not fragility_df.empty:
        base_row = fragility_df.iloc[0]
        for col in [
            "Failure Rate (%)",
            "Ruin Rate (%)",
            "Shortfall Rate (%)",
            "Median Ending Balance (USD)",
            "Real Median Ending Balance (USD)",
            "Real P10 Ending Balance (USD)",
            "Median CAGR (%)",
        ]:
            fragility_df[f"Delta vs Base :: {col}"] = fragility_df[col] - float(base_row[col])
        fragility_scale = max(abs(float(base_row["Real P10 Ending Balance (USD)"])), abs(float(base_row["Real Median Ending Balance (USD)"])), 25000.0)
        fragility_df["Fragility Score (heuristic)"] = np.where(
            fragility_df["Scenario"].eq("Base Case"),
            0.0,
            (
                fragility_df["Delta vs Base :: Failure Rate (%)"].abs() * 4.0
                + (fragility_df["Delta vs Base :: Real P10 Ending Balance (USD)"].abs() / fragility_scale) * 100.0
            ),
        )
        fragility_df["Fragility Rank"] = fragility_df["Fragility Score (heuristic)"]
        fragility_df = pd.concat(
            [fragility_df.iloc[[0]], fragility_df.iloc[1:].sort_values("Fragility Score (heuristic)", ascending=False)],
            ignore_index=True,
        )

    explicit_forward = materialize_forward_assumption_overrides(analysis_inputs)
    wr_levels = sorted({max(float(portfolio_inputs.withdrawal_rate) + delta, 0.0) for delta in (-2.0, -1.0, 0.0, 1.0, 2.0)})
    shift_levels = sorted({float(portfolio_inputs.monte_carlo_return_shift_pct) + delta for delta in (-3.0, -1.5, 0.0, 1.5, 3.0)})
    grid_rows: List[Dict[str, float]] = []
    for wr in wr_levels:
        for shift in shift_levels:
            grid_overrides: Dict[str, object] = dict(explicit_forward)
            grid_overrides.update(
                {
                    "monte_carlo_forward_mode": "Custom Forward Stress",
                    "withdrawal_rate": float(wr),
                    "monte_carlo_return_shift_pct": float(shift),
                }
            )
            stressed_inputs = replace(analysis_inputs, **grid_overrides)
            metrics = _run_mc_summary(
                stressed_inputs,
                assets,
                historical_returns_df,
                historical_dividends_df,
                start_period,
            )
            grid_rows.append(
                {
                    "Withdrawal Rate (%)": float(wr),
                    "Explicit Forward Return Shift (%)": float(shift),
                    "Failure Rate (%)": float(metrics.get("failure_rate_pct", np.nan)),
                    "Real Median Ending Balance (USD)": float(metrics.get("real_median_ending_balance", np.nan)),
                    "Real P10 Ending Balance (USD)": float(metrics.get("real_p10_ending_balance", np.nan)),
                }
            )
    fragility_grid_df = pd.DataFrame(grid_rows)
    fragility_pivot_df = pd.DataFrame()
    if not fragility_grid_df.empty:
        fragility_pivot_df = fragility_grid_df.pivot(
            index="Withdrawal Rate (%)",
            columns="Explicit Forward Return Shift (%)",
            values="Failure Rate (%)",
        ).sort_index().sort_index(axis=1)
        fragility_pivot_df.index.name = "Withdrawal Rate (%)"

    return {
        "fragility_df": fragility_df,
        "fragility_grid_df": fragility_grid_df,
        "fragility_pivot_df": fragility_pivot_df,
    }


def _policy_label(base_inputs: PortfolioInputs, candidate_inputs: PortfolioInputs, *, is_current: bool = False) -> str:
    if is_current:
        return "Current"
    if candidate_inputs.withdrawal_rate < base_inputs.withdrawal_rate - 1e-9:
        delta = (1.0 - (float(candidate_inputs.withdrawal_rate) / max(float(base_inputs.withdrawal_rate), 1e-9))) * 100.0
        return f"Spend -{delta:.0f}%"
    if candidate_inputs.withdrawal_mode != base_inputs.withdrawal_mode:
        return f"Mode: {candidate_inputs.withdrawal_mode}"
    if candidate_inputs.rebalancing_method != base_inputs.rebalancing_method:
        return f"Rebalance: {candidate_inputs.rebalancing_method}"
    return "Alternative"


def _policy_candidates(portfolio_inputs: PortfolioInputs) -> List[Tuple[str, PortfolioInputs]]:
    base = _analysis_inputs(portfolio_inputs)
    candidates: List[Tuple[str, PortfolioInputs]] = [("Current", base)]
    for cut in (0.90, 0.80):
        if float(base.withdrawal_rate) > 0:
            candidate = replace(base, withdrawal_rate=max(float(base.withdrawal_rate) * cut, 0.0))
            candidates.append((_policy_label(base, candidate), candidate))
    if str(base.withdrawal_mode) != "Dividend First":
        candidate = replace(base, withdrawal_mode="Dividend First")
        candidates.append((_policy_label(base, candidate), candidate))
    if str(base.rebalancing_method) != "Threshold Band":
        candidate = replace(base, rebalancing_method="Threshold Band", rebalance_band=min(max(float(base.rebalance_band), 5.0), 10.0))
        candidates.append((_policy_label(base, candidate), candidate))
    if str(base.rebalancing_method) != "Annual":
        candidate = replace(base, rebalancing_method="Annual")
        candidates.append((_policy_label(base, candidate), candidate))

    unique: List[Tuple[str, PortfolioInputs]] = []
    seen = set()
    for label, candidate in candidates:
        sig = (
            round(float(candidate.withdrawal_rate), 6),
            str(candidate.withdrawal_mode),
            str(candidate.rebalancing_method),
            round(float(candidate.rebalance_band), 6),
        )
        if sig in seen:
            continue
        seen.add(sig)
        unique.append((label, candidate))
    return unique


def _robustness_score(row: pd.Series, initial_investment: float) -> float:
    failure_component = max(0.0, 100.0 - float(row["Failure Rate (%)"])) * 0.45
    ruin_component = max(0.0, 100.0 - float(row["Ruin Rate (%)"])) * 0.15
    shortfall_component = max(0.0, 100.0 - float(row["Shortfall Rate (%)"])) * 0.15
    p10_component = max(float(row["Real P10 Ending Balance (USD)"]), 0.0) / max(float(initial_investment), 1.0)
    p10_component = min(p10_component, 3.0) * 8.0
    median_component = max(float(row["Real Median Ending Balance (USD)"]), 0.0) / max(float(initial_investment), 1.0)
    median_component = min(median_component, 5.0) * 3.0
    return float(failure_component + ruin_component + shortfall_component + p10_component + median_component)


def build_decision_policy_analysis(
    *,
    portfolio_inputs: PortfolioInputs,
    assets: Sequence[AssetConfig],
    historical_returns_df: pd.DataFrame,
    historical_dividends_df: pd.DataFrame,
    start_period: pd.Timestamp,
    objective: str,
) -> Dict[str, pd.DataFrame | str]:
    rows: List[Dict[str, float | str]] = []
    candidates = _policy_candidates(portfolio_inputs)
    for label, candidate_inputs in candidates:
        metrics = _run_mc_summary(
            candidate_inputs,
            assets,
            historical_returns_df,
            historical_dividends_df,
            start_period,
        )
        rows.append(
            {
                "Policy": label,
                "Failure Rate (%)": float(metrics.get("failure_rate_pct", np.nan)),
                "Ruin Rate (%)": float(metrics.get("ruin_rate_pct", np.nan)),
                "Shortfall Rate (%)": float(metrics.get("shortfall_rate_pct", np.nan)),
                "Median Ending Balance (USD)": float(metrics.get("median_ending_balance", np.nan)),
                "Real Median Ending Balance (USD)": float(metrics.get("real_median_ending_balance", np.nan)),
                "Real P10 Ending Balance (USD)": float(metrics.get("real_p10_ending_balance", np.nan)),
                "Median CAGR (%)": float(metrics.get("median_cagr_pct", np.nan)),
                "Withdrawal Rate (%)": float(candidate_inputs.withdrawal_rate),
                "Withdrawal Mode": str(candidate_inputs.withdrawal_mode),
                "Rebalancing Method": str(candidate_inputs.rebalancing_method),
            }
        )
    policy_df = pd.DataFrame(rows)
    if policy_df.empty:
        return {"policy_df": policy_df, "recommendation_df": pd.DataFrame(), "recommendation_text": "No policy candidates available."}

    policy_df["Robustness Score"] = policy_df.apply(_robustness_score, axis=1, initial_investment=float(portfolio_inputs.initial_investment))

    objective = str(objective)
    if objective == "Maximize legacy":
        ranked = policy_df.sort_values(
            ["Real Median Ending Balance (USD)", "Real P10 Ending Balance (USD)", "Failure Rate (%)"],
            ascending=[False, False, True],
        )
    elif objective == "Maximize downside resilience":
        ranked = policy_df.sort_values(
            ["Real P10 Ending Balance (USD)", "Failure Rate (%)", "Real Median Ending Balance (USD)"],
            ascending=[False, True, False],
        )
    elif objective == "Balanced robustness":
        ranked = policy_df.sort_values(["Robustness Score", "Failure Rate (%)"], ascending=[False, True])
    else:
        ranked = policy_df.sort_values(
            ["Failure Rate (%)", "Real P10 Ending Balance (USD)", "Real Median Ending Balance (USD)"],
            ascending=[True, False, False],
        )

    ranked = ranked.reset_index(drop=True)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1, dtype=int))
    winner = ranked.iloc[0]
    current = ranked.loc[ranked["Policy"].eq("Current")].iloc[0] if ranked["Policy"].eq("Current").any() else winner
    recommendation_text = (
        f"Best policy for {objective.lower()}: {winner['Policy']}. "
        f"Failure rate {winner['Failure Rate (%)']:.2f}% vs current {current['Failure Rate (%)']:.2f}%; "
        f"real median ending balance ${winner['Real Median Ending Balance (USD)']:,.0f} vs current ${current['Real Median Ending Balance (USD)']:,.0f}."
    )
    recommendation_df = pd.DataFrame(
        [
            {
                "Objective": objective,
                "Recommended Policy": winner["Policy"],
                "Failure Improvement vs Current (pp)": float(current["Failure Rate (%)"]) - float(winner["Failure Rate (%)"]),
                "Real Median Improvement vs Current (USD)": float(winner["Real Median Ending Balance (USD)"]) - float(current["Real Median Ending Balance (USD)"]),
                "Real P10 Improvement vs Current (USD)": float(winner["Real P10 Ending Balance (USD)"]) - float(current["Real P10 Ending Balance (USD)"]),
                "Robustness Score": float(winner["Robustness Score"]),
            }
        ]
    )
    return {
        "policy_df": ranked,
        "recommendation_df": recommendation_df,
        "recommendation_text": recommendation_text,
    }


__all__ = [
    "build_decision_policy_analysis",
    "build_forward_assumption_audit",
    "build_fragility_analysis",
]
