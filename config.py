"""
Configuration for P2-ETF-PERSISTENT-EXCITATION-RL engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-pe-rl-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM", "IWD"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}
ACTIVE_UNIVERSE = "COMBINED"

# --- Macro Features (available in dataset) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- RL Hyperparameters ---
STATE_DIM = 64           # feature dimension (from returns + macro)
ACTION_DIM = None        # set automatically = number of assets
HIDDEN_DIM = 128
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.005

# --- Persistent Excitation ---
PE_LAMBDA_MIN = 0.01
PE_NOISE_STD = 0.01
PE_ADAPTIVE = True
PE_COV_WINDOW = 1000

# --- Lyapunov ---
LYAPUNOV_BETA = 0.95
LYAPUNOV_THRESHOLD = 1.0

# --- Online Training ---
TRAIN_EPOCHS = 10
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000

# --- Backtest ---
TRAIN_START_DATE = "2008-01-01"
TRAIN_END_DATE = "2025-12-31"
TEST_START_DATE = "2026-01-01"
TEST_END_DATE = None

# --- Results ---
LOCAL_RESULTS_DIR = "results"

# --- Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# --- Run ID ---
TODAY = datetime.now().strftime("%Y-%m-%d")
