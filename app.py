import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import re
from config import HF_OUTPUT_REPO, UNIVERSES, HF_TOKEN

st.set_page_config(layout="wide", page_title="PE-RL Trading Dashboard")
st.title("⚡ Persistent Excitation RL – Online Actor‑Critic")
st.markdown("Adaptive control with persistent excitation for ETF portfolio trading")

# ---------------------------
# Load data from HF
# ---------------------------
@st.cache_data(ttl=3600)
def get_latest_run_folder():
    if not HF_TOKEN:
        return None
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_OUTPUT_REPO, repo_type="dataset", token=HF_TOKEN)
        run_folders = set()
        for f in files:
            match = re.match(r"(\d{8}_\d{6})/", f)
            if match:
                run_folders.add(match.group(1))
        if not run_folders:
            return None
        return sorted(run_folders, reverse=True)[0]
    except:
        return None

@st.cache_data
def load_performance(run_folder):
    try:
        cum = pd.read_parquet(
            hf_hub_download(repo_id=HF_OUTPUT_REPO, filename=f"{run_folder}/cumulative_returns.parquet", 
                            repo_type="dataset", token=HF_TOKEN)
        )
        daily = pd.read_parquet(
            hf_hub_download(repo_id=HF_OUTPUT_REPO, filename=f"{run_folder}/daily_returns.parquet", 
                            repo_type="dataset", token=HF_TOKEN)
        )
        # Convert to Series if DataFrame
        if isinstance(cum, pd.DataFrame):
            cum = cum.iloc[:, 0]
        if isinstance(daily, pd.DataFrame):
            daily = daily.iloc[:, 0]
        return cum, daily
    except Exception as e:
        st.warning(f"Could not load performance data: {e}")
        return None, None

@st.cache_data
def compute_etf_stats(universe_name, tickers):
    from data_manager import DataManager
    dm = DataManager(tickers)
    rets = dm.returns
    mean_ret = rets.mean() * 252
    vol = rets.std() * np.sqrt(252)
    sharpe = mean_ret / vol
    stats = pd.DataFrame({"Sharpe": sharpe, "Volatility": vol, "MeanReturn": mean_ret})
    return stats.sort_values("Sharpe", ascending=False)

def max_drawdown(cum_series):
    """Return max drawdown as a float."""
    if cum_series is None or len(cum_series) == 0:
        return 0.0
    cum_series = cum_series.copy()
    cum_series = cum_series / cum_series.iloc[0]
    running_max = cum_series.cummax()
    drawdown = (cum_series - running_max) / running_max
    result = drawdown.min()
    # Ensure scalar
    if isinstance(result, pd.Series):
        result = result.iloc[0]
    if np.isnan(result):
        result = 0.0
    return float(result)

def plot_cumulative(cum_series, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_series.index, y=cum_series.values, mode='lines', name='RL Strategy'))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Cumulative Return", yaxis_type="log")
    return fig

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Configuration")
selected_universe = st.sidebar.selectbox("Select Universe", list(UNIVERSES.keys()))
tickers = UNIVERSES[selected_universe]

run_folder = get_latest_run_folder()
if run_folder is None:
    st.error("No results found. Please run the backtest first.")
    st.stop()

cum, daily = load_performance(run_folder)
if cum is None:
    st.error("Could not load performance data from HF.")
    st.stop()

# Ensure datetime index
cum.index = pd.to_datetime(cum.index)
daily.index = pd.to_datetime(daily.index)

# ---------------------------
# Main dashboard
# ---------------------------
st.header(f"Universe: {selected_universe}")

# Performance metrics
strat_ret = float(cum.iloc[-1] - 1)
strat_sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() != 0 else 0.0
strat_mdd = max_drawdown(cum)

col1, col2, col3 = st.columns(3)
col1.metric("Strategy Total Return", f"{strat_ret:.2%}")
col2.metric("Sharpe Ratio", f"{strat_sharpe:.2f}")
col3.metric("Max Drawdown", f"{strat_mdd:.2%}")

st.plotly_chart(plot_cumulative(cum, "Cumulative Strategy Returns"), use_container_width=True)

# ETF analysis
st.subheader("ETF Performance & Top Picks")
with st.spinner("Computing ETF statistics..."):
    etf_stats = compute_etf_stats(selected_universe, tickers)

top3 = etf_stats.head(3)
st.markdown("#### 🏆 Top 3 ETFs by Sharpe Ratio")
st.dataframe(top3[["Sharpe", "Volatility", "MeanReturn"]].style.format("{:.2f}"))

other_etfs = [t for t in tickers if t not in top3.index]
if other_etfs:
    selected_etf = st.selectbox("Select any other ETF to view its performance", other_etfs)
    etf_data = compute_etf_stats(selected_universe, [selected_etf])
    st.metric(f"{selected_etf} Sharpe Ratio", f"{etf_data.loc[selected_etf, 'Sharpe']:.2f}")
    from data_manager import DataManager
    dm = DataManager([selected_etf])
    prices = dm.df[selected_etf].dropna()
    if not prices.empty:
        fig_etf = go.Figure()
        fig_etf.add_trace(go.Scatter(x=prices.index, y=prices / prices.iloc[0], mode='lines', name=selected_etf))
        fig_etf.update_layout(title=f"{selected_etf} Cumulative Price", xaxis_title="Date")
        st.plotly_chart(fig_etf, use_container_width=True)

with st.expander("Full ETF Universe Table (sorted by Sharpe)"):
    st.dataframe(etf_stats.style.format("{:.3f}"))

st.subheader("Strategy Daily Returns Distribution")
fig_hist = px.histogram(daily, x=daily.values, nbins=50, title="Daily Returns")
st.plotly_chart(fig_hist, use_container_width=True)

st.caption("Data loaded from Hugging Face. Top ETFs by Sharpe ratio are based on historical returns (2008–present).")
