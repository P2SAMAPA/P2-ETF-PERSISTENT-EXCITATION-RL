import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from config import HF_DATA_REPO, HF_DATA_FILE, HF_TOKEN, MACRO_COLS

def load_master_data():
    path = hf_hub_download(
        repo_id=HF_DATA_REPO,
        filename=HF_DATA_FILE,
        repo_type="dataset",
        token=HF_TOKEN if HF_TOKEN else None
    )
    df = pd.read_parquet(path)
    if df.index.name != 'date':
        df.index.name = 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    return df

class DataManager:
    def __init__(self, tickers):
        self.df = load_master_data()
        self.tickers = [t for t in tickers if t in self.df.columns]
        if not self.tickers:
            raise ValueError(f"No tickers found: {tickers}")
        prices = self.df[self.tickers].ffill().bfill()
        self.returns = prices.pct_change().dropna()
        self.macro = self.df[MACRO_COLS].ffill().bfill() if all(c in self.df.columns for c in MACRO_COLS) else None
    
    def get_state(self, date, lookback=20):
        """Return state vector: recent returns + macro (if available) + time features."""
        # Get returns for past lookback days
        past_rets = self.returns.loc[:date].iloc[-lookback:].values.flatten()
        # Normalise
        past_rets = (past_rets - past_rets.mean()) / (past_rets.std() + 1e-8)
        state = past_rets.tolist()
        
        if self.macro is not None:
            macro_vals = self.macro.loc[date].values
            state.extend(macro_vals)
        # Add time features (sin/cos of day of year)
        doy = date.dayofyear
        state.extend([np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)])
        return np.array(state, dtype=np.float32)
