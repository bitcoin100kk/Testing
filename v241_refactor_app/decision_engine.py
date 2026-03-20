from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    ordered_metrics: Tuple[Tuple[str, bool], ...]
    description: str
    recommendation_basis: str


OBJECTIVE_SPECS: Dict[str, ObjectiveSpec] = {
    "Minimize failure rate": ObjectiveSpec(
        name="Minimize failure rate",
        ordered_metrics=(
            ("Failure Rate (%)", True),
            ("Ruin Rate (%)", True),
            ("Shortfall Rate (%)", True),
            ("Real P10 Ending Balance (USD)", False),
            ("Real Median Ending Balance (USD)", False),
        ),
        description=(
            "Lexicographic safety-first ranking: minimize total failure first, then ruin, then spending shortfall, "
            "and only then prefer stronger tail and median wealth."
        ),
        recommendation_basis="Pareto-efficient policies ranked by failure, ruin, shortfall, then tail/median wealth.",
    ),
    "Maximize downside resilience": ObjectiveSpec(
        name="Maximize downside resilience",
        ordered_metrics=(
            ("Real P10 Ending Balance (USD)", False),
            ("Failure Rate (%)", True),
            ("Ruin Rate (%)", True),
            ("Shortfall Rate (%)", True),
            ("Real Median Ending Balance (USD)", False),
        ),
        description=(
            "Downside-first ranking: maximize the 10th-percentile real ending balance while preserving low failure, ruin, and shortfall rates."
        ),
        recommendation_basis="Pareto-efficient policies ranked by real P10 wealth first, then failure, ruin, shortfall, and median wealth.",
    ),
    "Balanced robustness": ObjectiveSpec(
        name="Balanced robustness",
        ordered_metrics=(
            ("Failure Rate (%)", True),
            ("Real P10 Ending Balance (USD)", False),
            ("Ruin Rate (%)", True),
            ("Shortfall Rate (%)", True),
            ("Real Median Ending Balance (USD)", False),
        ),
        description=(
            "Balanced robust ranking: start with low failure probability, then prefer higher tail wealth, lower ruin and shortfall risk, and finally higher median wealth."
        ),
        recommendation_basis="Pareto-efficient policies ranked by failure first, then tail wealth, ruin, shortfall, and median wealth.",
    ),
    "Maximize legacy": ObjectiveSpec(
        name="Maximize legacy",
        ordered_metrics=(
            ("Real Median Ending Balance (USD)", False),
            ("Real P10 Ending Balance (USD)", False),
            ("Failure Rate (%)", True),
            ("Ruin Rate (%)", True),
            ("Shortfall Rate (%)", True),
        ),
        description=(
            "Legacy-first ranking: maximize median real ending wealth while still preferring stronger tail wealth and lower failure-side risks."
        ),
        recommendation_basis="Pareto-efficient policies ranked by median wealth first, then tail wealth, failure, ruin, and shortfall.",
    ),
}


DEFAULT_OBJECTIVE = OBJECTIVE_SPECS["Balanced robustness"]


def _dominates(a: pd.Series, b: pd.Series, metric_directions: Sequence[Tuple[str, bool]], *, tolerance: float = 1e-9) -> bool:
    strictly_better = False
    for metric, ascending in metric_directions:
        a_val = float(a[metric])
        b_val = float(b[metric])
        if ascending:
            if a_val > b_val + tolerance:
                return False
            if a_val < b_val - tolerance:
                strictly_better = True
        else:
            if a_val < b_val - tolerance:
                return False
            if a_val > b_val + tolerance:
                strictly_better = True
    return strictly_better


def compute_pareto_frontier(policy_df: pd.DataFrame, metric_directions: Sequence[Tuple[str, bool]]) -> pd.DataFrame:
    if policy_df.empty:
        out = policy_df.copy()
        out["Pareto Efficient"] = pd.Series(dtype=bool)
        out["Pareto Dominated By"] = pd.Series(dtype=int)
        return out

    dominated_by = np.zeros(len(policy_df), dtype=int)
    records = policy_df.reset_index(drop=True)
    for i in range(len(records)):
        for j in range(len(records)):
            if i == j:
                continue
            if _dominates(records.iloc[j], records.iloc[i], metric_directions):
                dominated_by[i] += 1
    out = records.copy()
    out["Pareto Dominated By"] = dominated_by.astype(int)
    out["Pareto Efficient"] = out["Pareto Dominated By"].eq(0)
    return out


def _metric_sort_key(frame: pd.DataFrame, metric: str, ascending: bool) -> pd.Series:
    series = pd.to_numeric(frame[metric], errors="coerce")
    return series if ascending else -series


def rank_policy_candidates(policy_df: pd.DataFrame, *, objective: str) -> tuple[pd.DataFrame, pd.Series, ObjectiveSpec]:
    spec = OBJECTIVE_SPECS.get(str(objective), DEFAULT_OBJECTIVE)
    frontier = compute_pareto_frontier(policy_df, spec.ordered_metrics)
    feasible = frontier[frontier["Pareto Efficient"]].copy()
    if feasible.empty:
        feasible = frontier.copy()

    sort_keys = [_metric_sort_key(feasible, metric, ascending) for metric, ascending in spec.ordered_metrics]
    ranked = feasible.iloc[np.lexsort(tuple(key.to_numpy() for key in reversed(sort_keys)))].copy().reset_index(drop=True)
    ranked.insert(0, "Objective Rank", np.arange(1, len(ranked) + 1, dtype=int))
    ranked.insert(1, "Objective", spec.name)
    ranked["Recommendation Basis"] = spec.recommendation_basis

    non_frontier = frontier[~frontier["Pareto Efficient"]].copy()
    if not non_frontier.empty:
        non_frontier = non_frontier.sort_values(
            ["Pareto Dominated By"] + [metric for metric, _ in spec.ordered_metrics],
            ascending=[True] + [ascending for _, ascending in spec.ordered_metrics],
        ).reset_index(drop=True)
        non_frontier.insert(0, "Objective Rank", np.arange(len(ranked) + 1, len(ranked) + len(non_frontier) + 1, dtype=int))
        non_frontier.insert(1, "Objective", spec.name)
        non_frontier["Recommendation Basis"] = spec.recommendation_basis
        ranked = pd.concat([ranked, non_frontier], ignore_index=True)

    winner = ranked.iloc[0]
    return ranked, winner, spec


__all__ = [
    "DEFAULT_OBJECTIVE",
    "OBJECTIVE_SPECS",
    "ObjectiveSpec",
    "compute_pareto_frontier",
    "rank_policy_candidates",
]
