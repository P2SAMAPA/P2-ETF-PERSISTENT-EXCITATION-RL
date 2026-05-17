from backtest import run_backtest
from results_uploader import upload_results
import pandas as pd

def main():
    print("Running Persistent Excitation RL backtest...")
    cum_returns, daily_returns, model, dates = run_backtest()
    
    # Convert to DataFrames with datetime index
    cum_df = pd.Series(cum_returns, index=dates, name="strategy").to_frame()
    daily_df = pd.Series(daily_returns, index=dates, name="strategy_return").to_frame()
    
    results = {
        "cumulative_returns": cum_df,
        "daily_returns": daily_df
    }
    upload_results(results)
    print("Backtest completed and uploaded.")

if __name__ == "__main__":
    main()
