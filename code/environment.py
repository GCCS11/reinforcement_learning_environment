import gymnasium as gym
import numpy as np
from gymnasium import spaces

from code.features import feature_columns

transaction_cost = 0.00125


class BTCTradingEnv(gym.Env):
    """
    Semi-Markov trading environment for BTC/USDT hourly data.

    The agent decides at each bar whether to open a long, open a short,
    or stay flat. Once a position is open the environment simulates forward
    bar by bar until one of four closing conditions is met:
        1. Stop-loss triggered within a bar
        2. Take-profit triggered within a bar
        3. Maximum holding period reached (exit at close)
        4. Episode boundary reached (exit at close)

    OHLC assumption: if both stop-loss and take-profit are touched within
    the same bar, the stop-loss fills first. This is the conservative default.

    Entry price: open of the bar immediately after the decision bar.
    This avoids any within-bar lookahead on the decision bar itself.

    Observation: normalized feature vector at the current decision bar.
    Reward: net log PnL at position close (gross log return minus 2x transaction cost).
            Zero for flat actions.
    """

    metadata = {"render_modes": []}

    def __init__(self, df, stop_loss=0.02, take_profit=0.04, max_hold=48, episode_length=720):
        super().__init__()

        self.df             = df.reset_index(drop=True)
        self.stop_loss      = stop_loss
        self.take_profit    = take_profit
        self.max_hold       = max_hold
        self.episode_length = episode_length

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(feature_columns),),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.current_step  = 0
        self.episode_end   = 0

    def _obs(self):
        return self.df.iloc[self.current_step][feature_columns].values.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.df) - self.episode_length - self.max_hold - 2
        if max_start <= 0:
            max_start = 1

        self.current_step = int(self.np_random.integers(0, max_start))
        self.episode_end  = self.current_step + self.episode_length

        return self._obs(), {}

    def _simulate_position(self, direction):
        """
        Simulates a position opened at self.current_step.
        direction: 1 for long, -1 for short.
        Returns a dict with trade details, or None if no room to simulate.
        """
        entry_bar_idx = self.current_step + 1
        if entry_bar_idx >= len(self.df):
            return None

        entry_price = float(self.df.iloc[entry_bar_idx]["open"])
        sl_price    = entry_price * (1.0 - direction * self.stop_loss)
        tp_price    = entry_price * (1.0 + direction * self.take_profit)

        horizon_end = min(
            entry_bar_idx + self.max_hold,
            self.episode_end,
            len(self.df) - 1
        )

        exit_price  = float(self.df.iloc[horizon_end]["close"])
        exit_idx    = horizon_end
        exit_reason = "horizon"

        for i in range(entry_bar_idx, horizon_end + 1):
            bar  = self.df.iloc[i]
            high = float(bar["high"])
            low  = float(bar["low"])

            if direction == 1:
                sl_hit = low  <= sl_price
                tp_hit = high >= tp_price
            else:
                sl_hit = high >= sl_price
                tp_hit = low  <= tp_price

            if sl_hit:
                exit_price  = sl_price
                exit_reason = "stop_loss"
                exit_idx    = i
                break
            if tp_hit:
                exit_price  = tp_price
                exit_reason = "take_profit"
                exit_idx    = i
                break

        gross_pnl = direction * np.log(exit_price / entry_price)
        net_pnl   = gross_pnl - 2.0 * transaction_cost

        return {
            "entry_price":      entry_price,
            "exit_price":       exit_price,
            "exit_idx":         exit_idx,
            "exit_reason":      exit_reason,
            "gross_pnl":        gross_pnl,
            "net_pnl":          net_pnl,
            "direction":        direction,
            "entry_timestamp":  self.df.iloc[entry_bar_idx]["date"],
            "exit_timestamp":   self.df.iloc[exit_idx]["date"],
        }

    def step(self, action):
        terminated = False
        truncated  = False
        info       = {"exit_reason": "flat"}

        if action == 0:
            reward = 0.0
            self.current_step += 1
        else:
            direction = 1 if action == 1 else -1
            trade     = self._simulate_position(direction)

            if trade is None:
                reward = 0.0
                self.current_step += 1
            else:
                reward            = float(trade["net_pnl"])
                self.current_step = trade["exit_idx"] + 1
                info              = trade

        if self.current_step >= self.episode_end or self.current_step >= len(self.df) - 1:
            terminated = True
            self.current_step = min(self.current_step, len(self.df) - 1)

        return self._obs(), reward, terminated, truncated, info

    def render(self):
        pass