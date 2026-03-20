from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from .streamlit_compat import st

from .config import (
    BASE_CRYPTO_URL,
    BASE_STOCK_URL,
    CACHE_TTL_SECONDS,
    DEFAULT_START_DATE,
    HEADERS,
)
from .models import AssetConfig, HistoricalDataset
from .utils import _safe_numeric_series, _validate_matrix, normalize_ticker


def fetch_with_retry(url: str, params: Dict[str, str], max_attempts: int = 3) -> List[dict]:
    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_err = exc
            if attempt < max_attempts - 1:
                time.sleep(2**attempt)
            else:
                raise RuntimeError(str(last_err)) from last_err
    raise RuntimeError("Unknown request failure")


def _clean_monthly_history(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"No usable monthly history returned for '{ticker}'.")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["price_return", "dividend_yield"])
    if df.empty:
        raise ValueError(f"Not enough monthly history to calculate returns for '{ticker}'.")
    return df.astype(float)


def _build_monthly_stock_history_from_daily(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError(f"Unexpected Tiingo response for '{ticker}': missing date field.")

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], utc=True, errors="coerce").dt.tz_localize(None)
    working = working.dropna(subset=["date"]).set_index("date").sort_index()

    if "adjClose" in working.columns and _safe_numeric_series(working["adjClose"]).notna().any():
        adj_price = _safe_numeric_series(working["adjClose"])
    elif "close" in working.columns and _safe_numeric_series(working["close"]).notna().any():
        adj_price = _safe_numeric_series(working["close"])
    else:
        raise ValueError(f"No usable price column returned for '{ticker}'.")

    raw_close = _safe_numeric_series(working["close"] if "close" in working.columns else working.get("adjClose"))
    div_cash = _safe_numeric_series(working["divCash"]) if "divCash" in working.columns else pd.Series(0.0, index=working.index)
    split_factor = _safe_numeric_series(working["splitFactor"]) if "splitFactor" in working.columns else pd.Series(1.0, index=working.index)
    split_factor = split_factor.fillna(1.0)

    daily = pd.DataFrame(
        {
            "adj_price": adj_price,
            "raw_close": raw_close,
            "divCash": div_cash.fillna(0.0),
            "splitFactor": split_factor,
        },
        index=working.index,
    ).dropna(subset=["adj_price", "raw_close"])
    daily = daily[(daily["adj_price"] > 0) & (daily["raw_close"] > 0)]
    if daily.empty:
        raise ValueError(f"No positive adjusted prices returned for '{ticker}'.")

    prev_raw_close = daily["raw_close"].shift(1)
    daily_dividend_yield = (daily["divCash"] / prev_raw_close.where(prev_raw_close > 0)).replace([np.inf, -np.inf], np.nan)
    suspicious_dividend_event = (daily["splitFactor"].sub(1.0).abs() > 1e-9) | daily_dividend_yield.abs().gt(0.10)
    filtered_event_count = int(suspicious_dividend_event.fillna(False).sum())
    daily["filtered_divCash"] = daily["divCash"].where(~suspicious_dividend_event, 0.0)

    monthly = daily.resample("ME").agg({"adj_price": "last", "raw_close": "last", "filtered_divCash": "sum"})
    monthly["total_return"] = monthly["adj_price"].pct_change(fill_method=None)
    monthly["dividend_yield"] = (
        monthly["filtered_divCash"] / monthly["raw_close"].shift(1).where(monthly["raw_close"].shift(1) > 0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    clipped_months = int(monthly["dividend_yield"].abs().gt(0.10).sum())
    monthly.loc[monthly["dividend_yield"].abs() > 0.10, "dividend_yield"] = 0.0
    monthly["price_return"] = ((1.0 + monthly["total_return"]) / (1.0 + monthly["dividend_yield"].clip(lower=-0.99))) - 1.0
    monthly["price_return"] = monthly["price_return"].replace([np.inf, -np.inf], np.nan)
    monthly["price_return"] = monthly["price_return"].fillna(monthly["total_return"])
    cleaned = _clean_monthly_history(monthly[["price_return", "dividend_yield"]], ticker)
    cleaned.attrs["filtered_dividend_events"] = filtered_event_count
    cleaned.attrs["clipped_dividend_months"] = clipped_months
    return cleaned


def _get_stock_data_with_metadata(ticker: str, token: str) -> Tuple[pd.Series, pd.Series, List[int], Dict[str, object]]:
    ticker = normalize_ticker(ticker)
    params = {
        "token": token,
        "resampleFreq": "daily",
        "startDate": DEFAULT_START_DATE,
        "columns": "date,adjClose,divCash,close,splitFactor",
    }
    data = fetch_with_retry(BASE_STOCK_URL.format(ticker=ticker), params)
    if not data:
        raise ValueError(f"No Tiingo price data returned for stock/fund ticker '{ticker}'.")

    df = pd.DataFrame(data)
    monthly = _build_monthly_stock_history_from_daily(df, ticker)
    years = sorted(monthly.index.year.unique().tolist())
    metadata = {
        "data_source": "stock_api",
        "filtered_dividend_events": int(monthly.attrs.get("filtered_dividend_events", 0)),
        "clipped_dividend_months": int(monthly.attrs.get("clipped_dividend_months", 0)),
        "fallback_used": False,
        "fallback_reason": "",
    }
    return monthly["price_return"] * 100.0, monthly["dividend_yield"] * 100.0, years, metadata


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_stock_data(ticker: str, token: str) -> Tuple[pd.Series, pd.Series, List[int]]:
    price_returns, dividend_yields, years, _ = _get_stock_data_with_metadata(ticker, token)
    return price_returns, dividend_yields, years


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _get_stock_like_crypto_fallback(ticker: str, token: str) -> Tuple[pd.Series, List[int]]:
    price_returns, _, years = get_stock_data(ticker, token)
    return price_returns, years


def _get_crypto_data_with_metadata(ticker: str, token: str) -> Tuple[pd.Series, List[int], Dict[str, object]]:
    ticker = normalize_ticker(ticker)
    crypto_result = None
    crypto_err = None
    try:
        params = {
            "tickers": ticker.lower(),
            "resampleFreq": "1day",
            "startDate": DEFAULT_START_DATE,
            "token": token,
        }
        data = fetch_with_retry(BASE_CRYPTO_URL, params)
        if not data or not isinstance(data, list) or "priceData" not in data[0]:
            raise ValueError(f"No Tiingo crypto data returned for '{ticker}'.")
        prices = data[0]["priceData"]
        if not prices:
            raise ValueError(f"Empty Tiingo crypto history for '{ticker}'.")

        df = pd.DataFrame(prices)
        if "date" not in df.columns or "close" not in df.columns:
            raise ValueError(f"Unexpected Tiingo crypto response format for '{ticker}'.")
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        close_series = _safe_numeric_series(df["close"]).dropna()
        if close_series.empty:
            raise ValueError(f"No usable crypto prices returned for '{ticker}'.")

        monthly = close_series.resample("ME").last().to_frame("close")
        monthly["price_return"] = monthly["close"].pct_change(fill_method=None)
        monthly["dividend_yield"] = 0.0
        monthly = _clean_monthly_history(monthly[["price_return", "dividend_yield"]], ticker)
        years = sorted(monthly.index.year.unique().tolist())
        crypto_result = (monthly["price_return"] * 100.0, years, {"data_source": "crypto_api", "fallback_used": False, "fallback_reason": ""})
    except Exception as exc:  # noqa: BLE001
        crypto_err = str(exc)

    stock_like_result = None
    try:
        stock_price_returns, _stock_dividend_yields, stock_years, stock_meta = _get_stock_data_with_metadata(ticker, token)
        stock_like_result = (
            stock_price_returns,
            stock_years,
            {
                "data_source": "stock_api_fallback",
                "fallback_used": True,
                "fallback_reason": crypto_err or "crypto endpoint unavailable",
                "filtered_dividend_events": int(stock_meta.get("filtered_dividend_events", 0)),
                "clipped_dividend_months": int(stock_meta.get("clipped_dividend_months", 0)),
            },
        )
    except Exception:
        stock_like_result = None

    if crypto_result and stock_like_result:
        if len(stock_like_result[0]) > len(crypto_result[0]):
            return stock_like_result
        return crypto_result
    if crypto_result:
        return crypto_result
    if stock_like_result:
        return stock_like_result
    raise ValueError(crypto_err or f"No usable crypto history returned for '{ticker}'.")


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_crypto_data(ticker: str, token: str) -> Tuple[pd.Series, List[int]]:
    price_returns, years, _ = _get_crypto_data_with_metadata(ticker, token)
    return price_returns, years


def build_asset_matrices(
    assets: Sequence[AssetConfig],
    token: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], pd.DataFrame, List[Dict[str, object]]]:
    component_frames: List[pd.DataFrame] = []
    returns_map: Dict[str, pd.Series] = {}
    div_map: Dict[str, pd.Series] = {}
    diagnostics: List[Dict[str, object]] = []
    common_index: Optional[pd.DatetimeIndex] = None

    for asset in assets:
        if asset.asset_type == "Crypto":
            price_returns, _years, asset_metadata = _get_crypto_data_with_metadata(asset.ticker, token)
            dividend_yields = pd.Series(0.0, index=price_returns.index)
        else:
            price_returns, dividend_yields, _years, asset_metadata = _get_stock_data_with_metadata(asset.ticker, token)

        price_returns = _safe_numeric_series(price_returns).dropna().astype(float)
        dividend_yields = _safe_numeric_series(dividend_yields).reindex(price_returns.index).fillna(0.0).astype(float)
        asset_index = pd.DatetimeIndex(price_returns.index)
        if asset_index.empty:
            raise ValueError(f"No usable overlapping monthly history returned for '{asset.ticker}'.")

        common_index = asset_index if common_index is None else common_index.intersection(asset_index)
        returns_map[asset.ticker] = price_returns
        div_map[asset.ticker] = dividend_yields
        diagnostics.append(
            {
                "Ticker": asset.ticker,
                "Type": asset.asset_type,
                "Allocation (%)": asset.allocation,
                "First Month": asset_index.min().strftime("%Y-%m"),
                "Last Month": asset_index.max().strftime("%Y-%m"),
                "Monthly Points": int(len(asset_index)),
                "Data Source": str(asset_metadata.get("data_source", asset.asset_type.lower())),
                "Filtered Dividend Events": int(asset_metadata.get("filtered_dividend_events", 0)),
                "Clipped Dividend Months": int(asset_metadata.get("clipped_dividend_months", 0)),
                "Fallback Used": bool(asset_metadata.get("fallback_used", False)),
                "Fallback Reason": str(asset_metadata.get("fallback_reason", "")),
            }
        )

    if not returns_map or common_index is None or common_index.empty:
        raise ValueError("No overlapping monthly history exists across the selected assets.")

    common_index = common_index.sort_values()
    if len(common_index) < 12:
        raise ValueError("At least 12 overlapping monthly data points are required across all selected assets.")

    returns_df = pd.DataFrame({ticker: series.reindex(common_index) for ticker, series in returns_map.items()}, index=common_index)
    dividends_df = pd.DataFrame({ticker: series.reindex(common_index).fillna(0.0) for ticker, series in div_map.items()}, index=common_index)
    returns_df = _validate_matrix(returns_df, "monthly return")
    dividends_df = _validate_matrix(dividends_df, "monthly dividend", allow_fill_zero=True)

    for row in diagnostics:
        row["Overlap Start"] = common_index.min().strftime("%Y-%m")
        row["Overlap End"] = common_index.max().strftime("%Y-%m")
        row["Overlap Months"] = int(len(common_index))
        row["History Lost To Overlap (Months)"] = int(max(row["Monthly Points"] - len(common_index), 0))

    for asset in assets:
        component_frames.append(
            pd.DataFrame(
                {
                    "Period": common_index,
                    "Year": common_index.year.astype(int),
                    "Ticker": asset.ticker,
                    "Type": asset.asset_type,
                    "Allocation (%)": asset.allocation,
                    "price_return": returns_df[asset.ticker].values,
                    "dividend_yield": dividends_df[asset.ticker].values,
                    "total_return": (returns_df[asset.ticker] + dividends_df[asset.ticker]).values,
                    "weighted_price_return": returns_df[asset.ticker].values * (asset.allocation / 100.0),
                    "weighted_dividend_yield": dividends_df[asset.ticker].values * (asset.allocation / 100.0),
                    "weighted_total_return": (returns_df[asset.ticker] + dividends_df[asset.ticker]).values * (asset.allocation / 100.0),
                }
            )
        )

    component_df = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()
    years = sorted(pd.Index(common_index.year).unique().tolist())
    return returns_df, dividends_df, years, component_df, diagnostics


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def prepare_historical_dataset(asset_specs: Tuple[Tuple[str, float, str], ...], token: str) -> HistoricalDataset:
    assets = [AssetConfig(ticker=ticker, allocation=float(allocation), asset_type=asset_type) for ticker, allocation, asset_type in asset_specs]
    returns_df, dividends_df, years, component_df, diagnostics = build_asset_matrices(assets, token)
    return HistoricalDataset(
        returns_df=returns_df,
        dividends_df=dividends_df,
        years=years,
        component_df=component_df,
        diagnostics=diagnostics,
        overlap_start=pd.Timestamp(returns_df.index.min()),
        overlap_end=pd.Timestamp(returns_df.index.max()),
        overlap_months=int(len(returns_df)),
    )
