# Persistent Excitation RL – Online Actor‑Critic for ETF Trading

Adaptive control + RL with persistent excitation condition to guarantee parameter identifiability. Lyapunov-based stability certificate.

## Features
- Actor‑critic with adaptive exploration noise (maintains Fisher information matrix rank)
- Lyapunov stability monitoring
- Online training on ETF returns + macro data
- Daily backtest via GitHub Actions

## Setup
1. Clone repo
2. `pip install -r requirements.txt`
3. Set `HF_TOKEN` secret
4. Run `python run_backtest.py`

## Dashboard
`streamlit run app.py`
