import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import re
from config import HF_OUTPUT_REPO, UNIVERSES, HF_TOKEN

st.set_page_config(layout="wide", page_title="PE-RL ETF Dashboard")
st.title("⚡ Persistent Excitation RL – ETF Return Rankings")
st.markdown("Top ETFs by annualised return for each universe")

# ---------------------------
# Load data from HF (optional – we may not need strategy results)
# But we keep for reference; the main display uses live DataManager
# ---------------------------
@st.cache_data(ttl=3600)
def compute_etf_stats(tickers):
    """Compute annual return and volatility for each ETF."""
    from data_manager import DataManager
    dm = DataManager(tickers)
    rets = dm.returns
    # Annualised metrics (252 trading days)
    ann_return = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    stats = pd.DataFrame({
        "Annual Return (%)": ann_return * 100,
        "Annual Volatility (%)": ann_vol * 100,
        "Sharpe Ratio": sharpe
    }).sort_values("Annual Return (%)", ascending=False)
    return stats

# ---------------------------
# Sidebar universe selection
# ---------------------------
st.sidebar.header("Configuration")
selected_universe = st.sidebar.selectbox("Select Universe", list(UNIVERSES.keys()))
tickers = UNIVERSES[selected_universe]

# Compute stats
with st.spinner(f"Loading ETF data for {selected_universe}..."):
    etf_stats = compute_etf_stats(tickers)

# ---------------------------
# Top 3 Hero Cards (by Annual Return)
# ---------------------------
st.subheader(f"🏆 Top 3 ETFs in {selected_universe} by Annual Return")
top3 = etf_stats.head(3)

cols = st.columns(3)
for i, (ticker, row) in enumerate(top3.iterrows()):
    with cols[i]:
        st.metric(
            label=f"{ticker}",
            value=f"{row['Annual Return (%)']:.2f}%",
            delta=f"Vol: {row['Annual Volatility (%)']:.2f}%"
        )

# ---------------------------
# Dropdown for all ETFs (full list)
# ---------------------------
st.subheader("🔍 Explore Any ETF")
selected_etf = st.selectbox("Choose an ETF", tickers)

# Display selected ETF stats
etf_row = etf_stats.loc[selected_etf]
col1, col2, col3 = st.columns(3)
col1.metric("Annual Return", f"{etf_row['Annual Return (%)']:.2f}%")
col2.metric("Annual Volatility", f"{etf_row['Annual Volatility (%)']:.2f}%")
col3.metric("Sharpe Ratio", f"{etf_row['Sharpe Ratio']:.3f}")

# Plot price chart for selected ETF
from data_manager import DataManager
dm = DataManager([selected_etf])
prices = dm.df[selected_etf].dropna()
if not prices.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices / prices.iloc[0], mode='lines', name=selected_etf))
    fig.update_layout(title=f"{selected_etf} Normalised Price", xaxis_title="Date", yaxis_title="Price (starting at 1)")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Expandable full table
# ---------------------------
with st.expander("📊 Full Universe Table (sorted by Annual Return)"):
    st.dataframe(etf_stats.style.format("{:.2f}"))

st.caption("Annualised returns based on daily data from 2008 to present. No strategy performance shown – only ETF rankings.")
