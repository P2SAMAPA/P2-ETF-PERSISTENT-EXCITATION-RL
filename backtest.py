import pandas as pd
import numpy as np
from data_manager import DataManager
from pe_actor_critic import PE_ActorCritic
from persistent_excitation import PersistentExcitation
from lyapunov import LyapunovMonitor
from train_online import online_training
from config import UNIVERSES, ACTIVE_UNIVERSE, STATE_DIM, TRAIN_END_DATE, TEST_START_DATE

class TradingEnv:
    def __init__(self, returns_df, start_idx, end_idx, dm, lookback=20, initial_capital=1.0):
        self.returns = returns_df
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.initial_capital = initial_capital
        self.current_step = start_idx
        self.capital = initial_capital
        self.dm = dm
        self.lookback = lookback
        self.action_dim = returns_df.shape[1]
        self.dates = returns_df.index
    
    def reset(self):
        self.current_step = self.start_idx
        self.capital = self.initial_capital
        return self._get_state()
    
    def _get_state(self):
        if self.current_step >= len(self.returns):
            return np.zeros(STATE_DIM, dtype=np.float32)
        date = self.dates[self.current_step]
        state = self.dm.get_state(date, lookback=self.lookback)
        if len(state) > STATE_DIM:
            state = state[:STATE_DIM]
        elif len(state) < STATE_DIM:
            state = np.pad(state, (0, STATE_DIM - len(state)))
        return state.astype(np.float32)
    
    def step(self, action):
        weights = np.clip(action, 0, 1)
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(self.action_dim) / self.action_dim
        if self.current_step >= len(self.returns):
            return self._get_state(), 0.0, True, {}
        ret = self.returns.iloc[self.current_step].values
        port_ret = np.dot(weights, ret)
        self.capital *= (1 + port_ret)
        self.current_step += 1
        done = self.current_step >= self.end_idx
        reward = port_ret
        next_state = self._get_state() if not done else np.zeros(STATE_DIM, dtype=np.float32)
        return next_state, reward, done, {}

def run_backtest():
    tickers = UNIVERSES[ACTIVE_UNIVERSE]
    dm = DataManager(tickers)
    returns = dm.returns
    
    # Split train/test
    train_returns = returns.loc[:TRAIN_END_DATE] if TRAIN_END_DATE else returns
    test_returns = returns.loc[TEST_START_DATE:] if TEST_START_DATE else returns.iloc[-252:]
    
    # Training
    train_env = TradingEnv(train_returns, start_idx=20, end_idx=len(train_returns)-1, dm=dm)
    action_dim = len(tickers)
    actor_critic = PE_ActorCritic(STATE_DIM, action_dim)
    pe = PersistentExcitation(STATE_DIM + action_dim, lambda_min=0.01, window=1000)
    lyap = LyapunovMonitor()
    online_training(train_env, actor_critic, pe, lyap, epochs=10, batch_size=32)
    
    # Test backtest
    test_env = TradingEnv(test_returns, start_idx=20, end_idx=len(test_returns)-1, dm=dm)
    state = test_env.reset()
    done = False
    portfolio_returns = []
    dates = []
    while not done:
        action = actor_critic.act(state, noise_std=0.0)
        next_state, reward, done, _ = test_env.step(action)
        # Record date of the return used
        idx = test_env.current_step - 1
        if idx < len(test_returns):
            dates.append(test_returns.index[idx])
        portfolio_returns.append(reward)
        state = next_state
    
    cum_returns = np.cumprod(1 + np.array(portfolio_returns))
    return cum_returns, portfolio_returns, actor_critic, dates
