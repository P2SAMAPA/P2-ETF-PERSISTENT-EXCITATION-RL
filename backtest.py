import pandas as pd
import numpy as np
from data_manager import DataManager
from pe_actor_critic import PE_ActorCritic
from persistent_excitation import PersistentExcitation
from lyapunov import LyapunovMonitor
from config import UNIVERSES, ACTIVE_UNIVERSE, STATE_DIM, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE

class TradingEnv:
    def __init__(self, returns_df, start_idx, end_idx, initial_capital=1.0):
        self.returns = returns_df
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.initial_capital = initial_capital
        self.current_step = start_idx
        self.capital = initial_capital
        self.portfolio = None  # weights
        self.action_dim = returns_df.shape[1]
    
    def reset(self):
        self.current_step = self.start_idx
        self.capital = self.initial_capital
        self.portfolio = np.ones(self.action_dim) / self.action_dim
        return self._get_state()
    
    def _get_state(self):
        # Simplified state: last 20 returns + macro (if available) + time
        # For brevity, use random; in practice call DataManager.get_state
        return np.random.randn(STATE_DIM).astype(np.float32)
    
    def step(self, action):
        # Action is portfolio weights (softmax or clip)
        weights = np.clip(action, 0, 1)
        weights /= weights.sum()
        ret = self.returns.iloc[self.current_step].values
        port_ret = np.dot(weights, ret)
        self.capital *= (1 + port_ret)
        self.current_step += 1
        done = self.current_step >= self.end_idx
        reward = port_ret
        next_state = self._get_state() if not done else None
        return next_state, reward, done, {}
    
    def get_weights_history(self):
        # Not implemented fully; for backtest we record actions.
        pass

def run_backtest():
    tickers = UNIVERSES[ACTIVE_UNIVERSE]
    dm = DataManager(tickers)
    returns = dm.returns
    
    # Split train/test
    train_returns = returns.loc[:TRAIN_END_DATE] if TRAIN_END_DATE else returns
    test_returns = returns.loc[TEST_START_DATE:] if TEST_START_DATE else returns.iloc[-252:]
    
    # Build training environment
    train_env = TradingEnv(train_returns, 0, len(train_returns)-1)
    action_dim = len(tickers)
    actor_critic = PE_ActorCritic(STATE_DIM, action_dim)
    pe = PersistentExcitation(STATE_DIM + action_dim, lambda_min=0.01, window=1000)
    lyap = LyapunovMonitor()
    
    # Train
    online_training(train_env, actor_critic, pe, lyap, epochs=10, batch_size=32)
    
    # Backtest on test period
    test_env = TradingEnv(test_returns, 0, len(test_returns)-1)
    state = test_env.reset()
    done = False
    portfolio_returns = []
    while not done:
        action = actor_critic.act(state, noise_std=0.0)
        next_state, reward, done, _ = test_env.step(action)
        portfolio_returns.append(reward)
        state = next_state
    
    # Cumulative returns
    cum_returns = np.cumprod(1 + np.array(portfolio_returns))
    return cum_returns, portfolio_returns, actor_critic
