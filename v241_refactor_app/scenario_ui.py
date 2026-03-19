from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

import streamlit as st

from .config import MAX_SCENARIOS
from .models import AssetConfig, PortfolioInputs
from .utils import coerce_portfolio_input_dict, normalize_ticker


def get_preset(name: str) -> List[Dict[str, object]]:
    presets = {
        "100% SPY": [{"ticker": "SPY", "allocation": 100.0, "asset_type": "Stock"}],
        "60/40 Classic": [
            {"ticker": "SPY", "allocation": 60.0, "asset_type": "Stock"},
            {"ticker": "AGG", "allocation": 40.0, "asset_type": "Stock"},
        ],
        "Dividend + Bitcoin": [
            {"ticker": "SCHD", "allocation": 70.0, "asset_type": "Stock"},
            {"ticker": "BTC", "allocation": 30.0, "asset_type": "Crypto"},
        ],
        "Global 3-Fund": [
            {"ticker": "VTI", "allocation": 50.0, "asset_type": "Stock"},
            {"ticker": "VXUS", "allocation": 30.0, "asset_type": "Stock"},
            {"ticker": "BND", "allocation": 20.0, "asset_type": "Stock"},
        ],
        "Mega Cap Growth + Defensives": [
            {"ticker": "MSFT", "allocation": 13.0, "asset_type": "Stock"},
            {"ticker": "NVDA", "allocation": 9.0, "asset_type": "Stock"},
            {"ticker": "GOOGL", "allocation": 10.0, "asset_type": "Stock"},
            {"ticker": "AMZN", "allocation": 7.0, "asset_type": "Stock"},
            {"ticker": "META", "allocation": 7.0, "asset_type": "Stock"},
            {"ticker": "KO", "allocation": 11.0, "asset_type": "Stock"},
            {"ticker": "COST", "allocation": 10.0, "asset_type": "Stock"},
            {"ticker": "WMT", "allocation": 8.0, "asset_type": "Stock"},
            {"ticker": "XOM", "allocation": 12.0, "asset_type": "Stock"},
            {"ticker": "JNJ", "allocation": 13.0, "asset_type": "Stock"},
        ],
    }
    return presets.get(name, [])


def apply_preset_if_requested() -> None:
    preset_name = st.session_state.get("preset_name", "Custom")
    if st.button("Load preset") and preset_name != "Custom":
        preset_assets = get_preset(preset_name)
        st.session_state["num_assets"] = len(preset_assets)
        for idx, asset in enumerate(preset_assets):
            st.session_state[f"ticker_{idx}"] = str(asset["ticker"])
            st.session_state[f"allocation_{idx}"] = float(asset["allocation"])
            st.session_state[f"asset_type_{idx}"] = str(asset["asset_type"])
        st.rerun()


def load_saved_scenario(name: str) -> None:
    scenarios = st.session_state.get("saved_scenarios", {})
    if name not in scenarios:
        return
    cfg = scenarios[name]
    st.session_state["num_assets"] = len(cfg["assets"])
    for idx, asset in enumerate(cfg["assets"]):
        st.session_state[f"ticker_{idx}"] = asset["ticker"]
        st.session_state[f"allocation_{idx}"] = float(asset["allocation"])
        st.session_state[f"asset_type_{idx}"] = asset["asset_type"]
    for key, value in coerce_portfolio_input_dict(cfg.get("inputs", {})).items():
        st.session_state[key] = value
    st.session_state["contribution_end_year_text"] = "" if st.session_state.get("contribution_end_year") is None else str(st.session_state.get("contribution_end_year"))
    year_range = cfg.get("year_range")
    if year_range:
        st.session_state["canonical_year_range"] = tuple(year_range)
    st.rerun()


def save_current_scenario(name: str, portfolio_inputs: PortfolioInputs, assets: Sequence[AssetConfig], year_range: Tuple[int, int]) -> None:
    if not name.strip():
        raise ValueError("Enter a scenario name before saving.")
    scenarios = st.session_state.get("saved_scenarios", {})
    if len(scenarios) >= MAX_SCENARIOS and name not in scenarios:
        raise ValueError(f"You can save up to {MAX_SCENARIOS} scenarios in one session.")
    scenarios[name] = {
        "inputs": {**asdict(portfolio_inputs)},
        "assets": [asdict(asset) for asset in assets],
        "year_range": list(year_range),
    }
    st.session_state["saved_scenarios"] = scenarios


def collect_assets(num_assets: int) -> List[AssetConfig]:
    assets: List[AssetConfig] = []
    st.subheader("Asset Builder")
    for i in range(num_assets):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input(f"Ticker {i + 1}", value=st.session_state.get(f"ticker_{i}", ""), key=f"ticker_{i}").strip().upper()
        with col2:
            allocation = st.number_input(
                f"Allocation {i + 1} (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get(f"allocation_{i}", round(100.0 / max(num_assets, 1), 2))),
                key=f"allocation_{i}",
            )
        with col3:
            asset_type = st.selectbox(
                f"Type {i + 1}",
                options=["Stock", "Crypto"],
                index=0 if st.session_state.get(f"asset_type_{i}", "Stock") == "Stock" else 1,
                key=f"asset_type_{i}",
            )
        assets.append(AssetConfig(ticker=ticker, allocation=float(allocation), asset_type=asset_type))
    return assets


def validate_assets(assets: Sequence[AssetConfig]) -> List[AssetConfig]:
    normalized_assets = [AssetConfig(ticker=normalize_ticker(asset.ticker), allocation=float(asset.allocation), asset_type=asset.asset_type) for asset in assets]
    filled_assets = [asset for asset in normalized_assets if asset.ticker]
    if not filled_assets:
        raise ValueError("Please enter at least one ticker.")

    positive_assets = [asset for asset in filled_assets if asset.allocation > 0]
    if not positive_assets:
        raise ValueError("Please assign a positive allocation to at least one ticker.")

    allocation_total = sum(asset.allocation for asset in positive_assets)
    if abs(allocation_total - 100.0) > 0.01:
        raise ValueError(f"Allocations for tickers with positive weights must add up to 100%. Current total: {allocation_total:.2f}%")

    duplicate_tickers = sorted({asset.ticker for asset in positive_assets if sum(1 for a in positive_assets if a.ticker == asset.ticker) > 1})
    if duplicate_tickers:
        raise ValueError(f"Duplicate tickers detected: {', '.join(duplicate_tickers)}")

    ignored_zero = [asset.ticker for asset in filled_assets if asset.allocation <= 0]
    st.session_state["ignored_zero_allocation_tickers"] = ignored_zero
    return positive_assets
