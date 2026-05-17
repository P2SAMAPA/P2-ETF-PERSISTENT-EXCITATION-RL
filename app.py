import streamlit as st
import pandas as pd
import plotly.express as px
from huggingface_hub import HfApi, hf_hub_download
import re
from config import HF_OUTPUT_REPO, HF_TOKEN

st.set_page_config(layout="wide")
st.title("⚡ Persistent Excitation RL – Online Actor‑Critic with PE Condition")
st.markdown("Persistent excitation ensures parameter identifiability and convergence.")

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
def load_results(run_folder):
    try:
        cum = pd.read_parquet(hf_hub_download(repo_id=HF_OUTPUT_REPO, filename=f"{run_folder}/cumulative_returns.parquet", repo_type="dataset", token=HF_TOKEN))
        daily = pd.read_parquet(hf_hub_download(repo_id=HF_OUTPUT_REPO, filename=f"{run_folder}/daily_returns.parquet", repo_type="dataset", token=HF_TOKEN))
        return cum, daily
    except:
        return None, None

latest = get_latest_run_folder()
if latest:
    cum, daily = load_results(latest)
    if cum is not None:
        fig = px.line(cum, title="Cumulative Strategy Returns")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cum.tail())
else:
    st.info("No results found. Run backtest first.")
