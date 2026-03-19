from __future__ import annotations

from typing import Optional, Sequence, Tuple

import streamlit as st

ARTIFACT_BUCKETS = ("core", "benchmark", "scenario", "mc", "export")
MAX_BUCKET_ITEMS = 12


def initialize_state() -> None:
    defaults = {
        "saved_scenarios": {},
        "scenario_to_load": "",
        "canonical_year_range": None,
        "available_years": [],
        "year_range_slider": None,
        "start_year_box": None,
        "end_year_box": None,
        "last_config": None,
        "ignored_zero_allocation_tickers": [],
        "data_diagnostics": [],
        "active_run_snapshot": None,
        "active_core_signature": None,
        "latest_run_error": None,
        "artifact_cache": {bucket: {} for bucket in ARTIFACT_BUCKETS},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    _ensure_artifact_cache()


def _ensure_artifact_cache() -> dict:
    cache = st.session_state.get("artifact_cache")
    if not isinstance(cache, dict):
        cache = {}
    changed = False
    for bucket in ARTIFACT_BUCKETS:
        if not isinstance(cache.get(bucket), dict):
            cache[bucket] = {}
            changed = True
    if changed or st.session_state.get("artifact_cache") is not cache:
        st.session_state["artifact_cache"] = cache
    return cache


def clamp_year_pair(start: int, end: int, years: Sequence[int]) -> Tuple[int, int]:
    min_year = min(years)
    max_year = max(years)
    start = max(min_year, min(start, max_year))
    end = max(min_year, min(end, max_year))
    if start > end:
        start, end = end, start
    return int(start), int(end)


def _sync_year_boxes_from_slider(years: Sequence[int]) -> None:
    slider_start, slider_end = st.session_state.get("year_range_slider", (min(years), max(years)))
    new_start, new_end = clamp_year_pair(int(slider_start), int(slider_end), years)
    st.session_state["canonical_year_range"] = (new_start, new_end)
    st.session_state["start_year_box"] = new_start
    st.session_state["end_year_box"] = new_end


def _sync_slider_from_year_boxes(years: Sequence[int]) -> None:
    start_box = int(st.session_state.get("start_year_box", min(years)))
    end_box = int(st.session_state.get("end_year_box", max(years)))
    new_start, new_end = clamp_year_pair(start_box, end_box, years)
    st.session_state["canonical_year_range"] = (new_start, new_end)
    st.session_state["start_year_box"] = new_start
    st.session_state["end_year_box"] = new_end
    st.session_state["year_range_slider"] = (new_start, new_end)


def normalize_year_state(years: Sequence[int]) -> Tuple[int, int]:
    min_year = int(min(years))
    max_year = int(max(years))
    available_years = list(years)
    current_range = st.session_state.get("canonical_year_range")

    if not current_range or st.session_state.get("available_years") != available_years:
        current_range = (min_year, max_year)

    canonical_start, canonical_end = clamp_year_pair(int(current_range[0]), int(current_range[1]), years)
    st.session_state["canonical_year_range"] = (canonical_start, canonical_end)
    st.session_state["available_years"] = available_years
    st.session_state["start_year_box"] = canonical_start
    st.session_state["end_year_box"] = canonical_end
    st.session_state["year_range_slider"] = (canonical_start, canonical_end)
    return canonical_start, canonical_end


def reset_year_state() -> None:
    st.session_state["canonical_year_range"] = None
    st.session_state["available_years"] = []
    st.session_state["year_range_slider"] = None
    st.session_state["start_year_box"] = None
    st.session_state["end_year_box"] = None


def mark_run_snapshot(snapshot: object, core_signature: str) -> None:
    st.session_state["active_run_snapshot"] = snapshot
    st.session_state["active_core_signature"] = core_signature
    st.session_state["latest_run_error"] = None


def clear_cached_artifacts(bucket: Optional[str] = None) -> None:
    if bucket is None:
        st.session_state["artifact_cache"] = {name: {} for name in ARTIFACT_BUCKETS}
        return
    cache = _ensure_artifact_cache()
    cache[bucket] = {}
    st.session_state["artifact_cache"] = cache


def _prune_bucket(bucket_cache: dict) -> dict:
    while len(bucket_cache) > MAX_BUCKET_ITEMS:
        oldest_key = next(iter(bucket_cache))
        bucket_cache.pop(oldest_key, None)
    return bucket_cache


def store_bucket_artifact(bucket: str, render_signature: str, artifacts: object) -> None:
    cache = _ensure_artifact_cache()
    bucket_cache = dict(cache.get(bucket, {}))
    bucket_cache[render_signature] = artifacts
    cache[bucket] = _prune_bucket(bucket_cache)
    st.session_state["artifact_cache"] = cache


def get_bucket_artifact(bucket: str, render_signature: str) -> Optional[object]:
    cache = _ensure_artifact_cache()
    return cache.get(bucket, {}).get(render_signature)


def store_rendered_artifacts(render_signature: str, artifacts: object) -> None:
    store_bucket_artifact("core", render_signature, artifacts)


def get_rendered_artifacts(render_signature: str) -> Optional[object]:
    return get_bucket_artifact("core", render_signature)


def set_latest_error(message: str) -> None:
    st.session_state["latest_run_error"] = message


def get_active_run_snapshot() -> Optional[object]:
    return st.session_state.get("active_run_snapshot")
