from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import AssetConfig, PORTFOLIO_INPUT_DEFAULTS, PortfolioInputs


def normalize_ticker(ticker: str) -> str:
    return (ticker or "").strip().upper()


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _validate_matrix(df: pd.DataFrame, label: str, *, allow_fill_zero: bool = False) -> pd.DataFrame:
    cleaned = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if allow_fill_zero:
        cleaned = cleaned.fillna(0.0)
    if cleaned.isna().any().any():
        bad = cleaned.columns[cleaned.isna().any()].tolist()
        raise ValueError(
            f"Invalid {label} data detected for: {', '.join(map(str, bad))}. Choose a range with valid overlapping history or review the ticker type."
        )
    return cleaned.astype(float)


def _annual_pct_to_monthly_decimal(annual_pct: float) -> float:
    annual_decimal = max(float(annual_pct), -99.999999) / 100.0
    return (1.0 + annual_decimal) ** (1.0 / 12.0) - 1.0


def coerce_portfolio_input_dict(data: Dict[str, object]) -> Dict[str, object]:
    merged = {**PORTFOLIO_INPUT_DEFAULTS}
    merged.update(data or {})
    return merged


def serialize_assets(assets: Sequence[AssetConfig]) -> Tuple[Tuple[str, float, str], ...]:
    return tuple((normalize_ticker(asset.ticker), float(asset.allocation), str(asset.asset_type)) for asset in assets)


def deserialize_assets(asset_specs: Sequence[Tuple[str, float, str]]) -> list[AssetConfig]:
    return [AssetConfig(ticker=str(ticker), allocation=float(allocation), asset_type=str(asset_type)) for ticker, allocation, asset_type in asset_specs]


def serialize_portfolio_inputs(portfolio_inputs: PortfolioInputs) -> Dict[str, object]:
    return asdict(portfolio_inputs)


def deserialize_portfolio_inputs(data: Dict[str, object]) -> PortfolioInputs:
    return PortfolioInputs(**coerce_portfolio_input_dict(data))


def stable_hash_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_raw_signature(payload: Dict[str, object]) -> str:
    return stable_hash_payload(payload)


def build_core_signature(input_dict: Dict[str, object], asset_specs: Sequence[Tuple[str, float, str]], token: str) -> str:
    payload = {"portfolio_inputs": input_dict, "assets": list(asset_specs), "token": token}
    return stable_hash_payload(payload)


def build_render_signature(core_signature: str, year_range: Tuple[int, int], compare_saved: Sequence[str]) -> str:
    payload = {
        "core_signature": core_signature,
        "year_range": [int(year_range[0]), int(year_range[1])],
        "compare_saved": list(sorted(compare_saved)),
    }
    return stable_hash_payload(payload)


def format_currency(value: float) -> str:
    return f"{value:,.2f}"


def highlight_changes(val: object) -> str:
    try:
        numeric_val = float(val)
    except Exception:  # noqa: BLE001
        return ""
    return f"color: {'green' if numeric_val > 0 else 'red'}"


def _format_percentage_or_na(value: object) -> str:
    try:
        numeric_val = float(value)
    except Exception:  # noqa: BLE001
        return "N/A"
    return f"{numeric_val:.2f}" if math.isfinite(numeric_val) else "N/A"
