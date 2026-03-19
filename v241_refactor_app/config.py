import os
import streamlit as st

DEFAULT_TIINGO_API_TOKEN = os.getenv(
    "TIINGO_API_TOKEN",
    "ea59607dd8ff30a1493d798ce53e515a547c9b63",
)
BASE_STOCK_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
BASE_CRYPTO_URL = "https://api.tiingo.com/tiingo/crypto/prices"
HEADERS = {"Content-Type": "application/json"}
DEFAULT_START_DATE = "1990-01-01"
CACHE_TTL_SECONDS = 3600
MAX_SCENARIOS = 8

APP_TITLE = "Interactive Investment Balance Calculator"
APP_CAPTION = (
    "Upgraded for fully monthly simulation, asset-aware + AUM-aware trading frictions, "
    "multivariate block-bootstrap Monte Carlo, and convergence diagnostics."
)


def configure_page() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
