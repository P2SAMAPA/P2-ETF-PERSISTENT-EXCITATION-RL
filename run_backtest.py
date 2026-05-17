from backtest import run_backtest
from results_uploader import upload_results
import pandas as pd

def main():
    print("Running Persistent Excitation RL backtest...")
    cum_returns, daily_returns, model = run_backtest()
    
    results = {
        "cumulative_returns": pd.Series(cum_returns, name="strategy"),
        "daily_returns": pd.Series(daily_returns, name="strategy_return")
    }
    upload_results(results)
    print("Backtest completed and uploaded.")

if __name__ == "__main__":
    main()
