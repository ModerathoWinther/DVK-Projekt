from collections import deque

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

import action_space
import init_state
from action_space import Direction, ACTION_SPACE, HOLD_ACTION


class TradingEnvironment(gym.Env):

    # todo reset: close trades that haven't hit sl/tp yet, profit being (price bought - close) * direction

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

        self.input_data, self.prices = init_state.run(**params)

        self.col_high_wick = 0
        self.col_low_wick = 1
        self.col_trend = 2
        self.col_input_vol = 3
        col = 4
        if self.atr:
            self.col_atr = col
            col += 1
        if self.macd:
            self.col_macd = col
            col += 3
        if self.rsi:
            self.col_rsi = col
            col += 1

        self.col_high = 0
        self.col_low = 1
        self.col_close = 2

        self.data_length = len(self.input_data)

        if self.episode_length is None or self.episode_length >= self.data_length:
            self.episode_length = self.data_length - 1
            self.max_start = 0
        else:
            self.max_start = self.data_length - self.episode_length - 1

        self.current_step = 0
        self.episode_end = self.episode_length

        self.pnl_mean_estimate, self.pnl_scale = self._calc_pnl_mean_and_scale()
        self.tp_hits = self.sl_hits = 0

        # Portfolio state
        self.current_equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.closed_trades = []
        self.open_slots = self.num_trades
        self.trades_state = np.zeros((self.num_trades, 4), dtype=np.float32)
        self.trades_obs = np.zeros((self.num_trades, 3), dtype=np.float32)

        self.episode_results = []

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.input_data.shape[1] + self.num_trades * 3,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(len(ACTION_SPACE))
        print(f"\n\nepisode_length={self.episode_length}, max_start={self.max_start}, data_length={self.data_length}\n\n")

        print(
            f"Close range: {self.input_data[:, self.col_close].min():.4f} to {self.input_data[:, self.col_close].max():.4f}")
        print(f"Median close: {np.median(self.input_data[:, self.col_close]):.4f}")
        print(f"pnl_mean={self.pnl_mean_estimate:.6f}, pnl_scale={self.pnl_scale:.6f}")
        print(f"SL distances: {[a.sl for a in ACTION_SPACE if a.direction != Direction.HOLD]}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.split == 'train' and self.max_start > 0:
            self.current_step = np.random.randint(0, self.max_start)
        else:
            self.current_step = 0

        episode_stats = self._calc_episode_stats()
        self.episode_results.append(episode_stats)
        self.closed_trades = []

        self.tp_hits = 0
        self.sl_hits = 0
        self.current_step = 0
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

        act = ACTION_SPACE[action]
        if act.direction != Direction.HOLD and self.open_slots > 0:
            sl = close - act.direction.value * (close * act.sl)
            tp = close + act.direction.value * (close * act.tp)
            for i in range(self.num_trades):
                if self.trades_state[i, 0] == 0:
                    self.trades_state[i] = [act.direction.value, close, sl, tp]
                    self.open_slots -= 1
                    break

        reward = realized_pnl

        self.current_equity += realized_pnl
        self.equity_curve.append(self.current_equity)
        self.current_step += 1
        terminated = self.current_step >= self.episode_end

        if terminated:
            total_realized_pnl = 0
            for i in range(self.num_trades):
                direction, entry_price, _, _ = self.trades_state[i]
                if direction != 0:
                    pnl = (entry_price - close) * direction
                    realized_pnl = pnl - self.transaction_cost * entry_price
                    total_realized_pnl += realized_pnl
                    self.closed_trades.append(realized_pnl)
            reward += total_realized_pnl


        return self._get_observation(), reward, terminated, False, {}

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
                if hit_tp:
                    self.tp_hits += 1
                else:
                    self.sl_hits += 1
                self.closed_trades.append(realized_pnl)

        return total_realized_pnl, closed

    def _unrealized_pnl(self, current_price: float) -> float:
        total = 0.0
        for i in range(self.num_trades):
            if self.trades_state[i, 0] != 0:
                direction, entry_price, sl, tp = self.trades_state[i]
                total += (current_price - entry_price) * direction
        return total

    def _sharpe_reward(self, pnl: float) -> float:
        if pnl == 0.0 or self.pnl_scale < 1e-8:
            return 0.0
        normalised = (pnl - self.pnl_mean_estimate) / self.pnl_scale
        return float(np.clip(normalised, -10.0, 10.0))

    def _calc_pnl_mean_and_scale(self) -> tuple[float, float]:
        non_hold = [a for a in ACTION_SPACE if a.direction != Direction.HOLD]
        avg_tp = float(np.mean([a.tp for a in non_hold]))
        avg_sl = float(np.mean([a.sl for a in non_hold]))

        assumed_win_rate = 0.50
        expected_cost = self.transaction_cost * float(
            np.median(np.abs(self.input_data[:, self.col_close]))
        )

        pnl_mean  = assumed_win_rate * avg_tp - (1 - assumed_win_rate) * avg_sl - expected_cost
        pnl_scale = assumed_win_rate * avg_tp + (1 - assumed_win_rate) * avg_sl

        return pnl_mean, pnl_scale

    def _holding_penalty(self) -> float:
        penalty = 0.0
        close = self.input_data[self.current_step, self.col_close]
        for i in range(self.num_trades):
            if self.trades_state[i, 0] != 0:
                direction, entry_price, sl, tp = self.trades_state[i]
                unrealised = (close - entry_price) * direction
                if unrealised < 0:
                    penalty -= self.pnl_scale * 0.05
        return penalty

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
                total_loss += pnl

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