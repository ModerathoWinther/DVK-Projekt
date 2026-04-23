from collections import deque

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

import init_state
from action_space import Direction, ACTION_SPACE, HOLD_ACTION, UNIT_TEST_ACTION_SPACE

class TradingEnvironment(gym.Env):

    def __init__(self, params):
        super().__init__()
        self.split = params.get('split')
        self.num_trades = params.get('num_trades')
        self.initial_capital = params.get('initial_capital')
        self.transaction_cost = params.get('transaction_cost')
        self.normalize = params.get('normalize')
        self.atr = params.get('atr')
        self.macd = params.get('macd')
        self.rsi = params.get('rsi')
        self.episode_length = params.get('episode_length')

        if params.get("unit_test"):
            self.action_list = UNIT_TEST_ACTION_SPACE
        else:
            self.action_list = ACTION_SPACE

        self.column_count = params.get('column_count')
        self.input_data, self.prices = init_state.run(**params)

        self.col_high_wick = 0
        self.col_low_wick = 1
        self.col_trend = 2
        self.col_input_vol = 3
        self.col_open, self.col_high, self.col_low, self.col_close, self.col_vol = range(5)
        self.col_atr = 5
        self.col_macd = 6
        self.col_rsi = 9

        self.data_length = len(self.input_data)

        if self.episode_length is None or self.episode_length >= self.data_length:
            self.episode_length = self.data_length - 1
            self.max_start = 0
        else:
            self.max_start = self.data_length - self.episode_length - 1

        self.visible_cols = [self.column_count]

        if params.get('atr'):
            self.visible_cols.append(self.col_atr)
        if params.get('macd'):
            self.visible_cols.extend([self.col_macd, self.col_macd + 1, self.col_macd + 2])
        if params.get('rsi'):
            self.visible_cols.append(self.col_rsi)

        self.current_step = 0
        self.episode_end = self.episode_length

        # Portfolio state
        self.current_equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.closed_trades = []
        self.open_slots = self.num_trades
        self.trades_state = np.zeros((self.num_trades, 4), dtype=np.float32)
        self.trades_obs = np.zeros((self.num_trades, 3), dtype=np.float32)

        self.episode_results = []

        num_obs_features = len(self.visible_cols) + (self.num_trades * 4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(num_obs_features,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.action_list))
        self._log_init_diagnostics()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.split == "train":
            self.max_start = len(self.input_data) - self.episode_length - 1
            self.current_step = np.random.randint(0, self.max_start)
            self.episode_end = self.current_step + self.episode_length
        else:
            self.current_step = 0
            self.episode_end = self.max_start

        if len(self.closed_trades) > 0:
            episode_stats = self._calc_episode_stats()
            self.episode_results.append(episode_stats)
        self.closed_trades = []

        self.current_equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.open_slots = self.num_trades
        self.trades_state.fill(0.0)
        return self._get_observation(), {}

    def calculate_sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        equity = np.array(self.equity_curve)
        if np.any(equity <= 0):
            return -999.0
        returns = np.diff(equity) / equity[:-1]
        active_returns = returns[returns != 0.0]
        if len(active_returns) < 2:
            return 0.0
        std_ret = np.std(active_returns, ddof=1)
        if std_ret < 1e-8:
            return 0.0
        mean_ret = np.mean(active_returns)

        total_bars = len(self.equity_curve)
        n_trades = len(active_returns)
        trades_per_year = (n_trades / total_bars) * (252 * 92)

        sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)
        return float(sharpe)

    def step(self, action: int):
        current_prices = self.prices[self.current_step]
        high, low, close = current_prices[self.col_high], current_prices[self.col_low], current_prices[self.col_close]

        realized_pnl, _ = self._process_trades(high, low)
        reward = realized_pnl

        self.current_step += 1
        self.current_equity += realized_pnl

        is_last_step = (self.current_step + 1) >= self.episode_end
        if is_last_step:
            current_prices = self.prices[self.current_step]
            high, low = current_prices[self.col_high], current_prices[self.col_low]
            high_low_hit_r, _ = self._process_trades(high, low)
            self.current_equity += high_low_hit_r
            reward += high_low_hit_r

            term_r = self._calculate_terminated_reward()
            self.current_equity += term_r
            reward += term_r

        else:
            act = self.action_list[action]
            if act.direction != Direction.HOLD and self.open_slots > 0:
                open = self.prices[self.current_step][self.col_open]
                entry_price = open
                sl = entry_price - act.direction.value * (entry_price * act.sl)
                tp = entry_price + act.direction.value * (entry_price * act.tp)
                for i in range(self.num_trades):
                    if self.trades_state[i, 0] == 0:
                        self.trades_state[i] = [act.direction.value, entry_price, sl, tp]
                        self.open_slots -= 1
                        break

        self._update_trades_obs(close)
        self.equity_curve.append(self.current_equity)
        return self._get_observation(), reward, is_last_step, False, {}

    def _get_observation(self):
        current_prices = self.prices[self.current_step]
        close = current_prices[self.col_close]
        current_md = self.input_data[self.current_step]

        for i in range(len(self.trades_state)):
            direction, _, sl, tp = self.trades_state[i]
            if(direction != 0):
                tp_dist = abs(tp - close)
                sl_dist = abs(sl - close)
                self.trades_obs[i] = direction, sl_dist, tp_dist

        flat_trades = self.trades_obs.flatten()
        return np.concatenate([current_md, flat_trades]).astype(np.float32)

    def _process_trades(self, high: float, low: float) -> tuple[float, int]:
        total_realized_pnl = 0.0
        closed = 0
        for i in range(self.num_trades):
            if self.trades_state[i, 0] == 0:
                continue

            direction, entry_price, sl, tp = self.trades_state[i]

            hit_sl = (direction > 0 and low  <= sl) or (direction < 0 and high >= sl)
            hit_tp = (direction > 0 and high >= tp) or (direction < 0 and low  <= tp)

            if hit_sl or hit_tp:
                pnl = (tp - entry_price) * direction if hit_tp else (sl - entry_price) * direction
                realized_pnl = pnl - self.transaction_cost * abs(entry_price)
                total_realized_pnl += realized_pnl
                self.trades_state[i] = [0, 0, 0, 0]
                self.open_slots += 1
                closed += 1
                self.closed_trades.append(realized_pnl)

        return total_realized_pnl, closed

    def _update_trades_obs(self, close: float) -> None:
        for i in range(self.num_trades):
            direction, _, sl, tp = self.trades_state[i]
            if direction != 0:
                self.trades_obs[i] = [direction, abs(sl - close), abs(tp - close)]
            else:
                self.trades_obs[i] = [0.0, 0.0, 0.0]

    def _log_init_diagnostics(self):
        closes = self.input_data[:, self.col_close]
        bar_ranges = self.input_data[:, self.col_high] - self.input_data[:, self.col_low]
        median_range = float(np.median(bar_ranges))
        print(f"Input data shape  : {self.input_data.shape}")
        print(f"Observation shape : {self.observation_space.shape}")
        print(f"Close range       : {closes.min():.4f} to {closes.max():.4f}")
        print(f"Median close      : {np.median(closes):.4f}")
        print(f"Median bar range  : {median_range:.6f}")
        if self.atr:
            print(f"Median ATR        : {np.median(self.input_data[:, self.col_atr]):.6f}")
        sl_levels = [a.sl for a in ACTION_SPACE if a.direction != Direction.HOLD]
        tp_levels = [a.tp for a in ACTION_SPACE if a.direction != Direction.HOLD]
        print(f"SL levels         : {sorted(set(sl_levels))}")
        print(f"TP levels         : {sorted(set(tp_levels))}")

        print(f"Tightest SL / bar range: {min(sl_levels) / median_range:.2f}×")

    def _calculate_terminated_reward(self):
        close = self.prices[self.current_step][self.col_close]
        total_realized_pnl = 0.0
        for i in range(self.num_trades):
            if self.trades_state[i, 0] == 0:
                continue

            direction, entry_price, _, _ = self.trades_state[i]
            pnl = (close - entry_price) * direction
            realized_pnl = pnl - self.transaction_cost * abs(entry_price)
            total_realized_pnl += realized_pnl
            self.trades_state[i] = [0, 0, 0, 0]
            self.closed_trades.append(realized_pnl)
            print("realized_pnl ", realized_pnl, "close", close, "entry_price", entry_price)

        return total_realized_pnl


    def _calc_episode_stats(self):
        closed_trades = len(self.closed_trades)
        total_profit = 0
        total_gain = 0
        total_loss = 0
        num_gain = 0
        num_loss = 0
        win_rate = 0.0
        loss_rate = 0.0
        avg_gain = 0.0
        avg_loss = 0.0
        profit_factor = 0.0

        for pnl in self.closed_trades:
            total_profit += pnl
            if pnl > 0:
                num_gain += 1
                total_gain += pnl
            if pnl < 0:
                num_loss += 1
                total_loss += pnl * -1

        if closed_trades > 0:
            win_rate = num_gain / closed_trades
            loss_rate = num_loss / closed_trades

        if num_gain > 0:
            avg_gain = total_gain / num_gain
        if num_loss > 0:
            avg_loss = total_loss / num_loss
        expectancy = (win_rate * avg_gain) - (loss_rate * avg_loss)

        if total_loss != 0:
            profit_factor = total_gain / total_loss

        peak = max(self.equity_curve)
        trough = min(self.equity_curve)

        max_drawdown = (trough - peak) / peak
        sharpe_ratio = self.calculate_sharpe_ratio()

        stats = {
            "closed_trades": len(self.closed_trades),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }
        return stats

    def get_episode_stats(self):
        return self.episode_results

if __name__ == "__main__":
    with open('../hyperparameters.yml', 'r') as file:
        all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets["midas-train1"]
    env_make_params = hyperparameters.get('env_make_params', {})

    env = TradingEnvironment(env_make_params)

    # Test here
    env.step(1)
    env.step(1)
    env.step(1)
    print(env.trades_state)
    print(env._get_observation())
    obs = None
    for i in range(1000):
        obs_last = obs
        obs, reward, _, _, _ = env.step(HOLD_ACTION.index)
        if(reward != 0):
            print(i)
            print(obs_last)
            print(reward)
            print(env.closed_trades)

    env.reset()
    print(env.get_episode_stats())