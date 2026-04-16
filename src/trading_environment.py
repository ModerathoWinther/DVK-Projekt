from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import init_state
from action_space import Direction, ACTION_SPACE

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

        self.market_data = init_state.run(**params)
        col = 0
        self.col_open = col
        col += 1
        self.col_high = col
        col += 1
        self.col_low = col
        col += 1
        self.col_close = col
        col += 1
        self.col_vol = col
        col +=1
        if self.atr:
            self.col_atr = col
            col += 1
        if self.macd:
            self.col_macd = col
            col += 3
        if self.rsi:
            self.col_rsi = col
            col += 1

        self._recent_returns = deque(maxlen=96)
        self.current_step = 0
        self.max_steps = len(self.market_data) - 1

        # Portfolio state
        self.current_equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.open_slots = self.num_trades
        self.trades = np.zeros((self.num_trades, 4), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.market_data.shape[1] + self.num_trades * 4,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(len(ACTION_SPACE))

        self.SL_ATR_MULT = 1.5
        self.TP_ATR_MULT = 3.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.open_slots = self.num_trades
        self.trades.fill(0.0)
        return self._get_observation(), {}

    def step(self, action: int):
        current_md = self.market_data[self.current_step]
        high, low = current_md[self.col_high], current_md[self.col_low]
        close = current_md[self.col_close]

        realized_pnl, _ = self._process_trades(high, low)

        act = ACTION_SPACE[action]

        if act.direction != Direction.HOLD and self.open_slots > 0:
            price = close
            atr = current_md[self.col_atr]
            sl = price - act.direction.value * self.SL_ATR_MULT * atr
            tp = price + act.direction.value * self.TP_ATR_MULT * atr
            for i in range(self.num_trades):
                if self.trades[i, 0] == 0:
                    self.trades[i] = [act.direction.value, price, sl, tp]
                    self.open_slots -= 1
                    break

        sharpe_shaped = self._sharpe_reward(realized_pnl)
        reward = sharpe_shaped

        self.current_equity += realized_pnl
        self.equity_curve.append(self.current_equity)
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        return self._get_observation(), sharpe_shaped, terminated, False, {}

    def _process_trades(self, high: float, low: float) -> tuple[float, int]:
        total_reward = 0.0
        closed = 0
        for i in range(self.num_trades):
            if self.trades[i, 0] == 0:
                continue

            direction, entry_price, sl, tp = self.trades[i]

            hit_sl = (direction > 0 and low <= sl) or (direction < 0 and high >= sl)
            hit_tp = (direction > 0 and high >= tp) or (direction < 0 and low <= tp)

            if hit_sl or hit_tp:
                pnl = (tp - entry_price) * direction if hit_tp else (sl - entry_price) * direction
                total_reward += pnl - self.transaction_cost * abs(entry_price)
                self.trades[i] = [0, 0, 0, 0]
                self.open_slots += 1
                closed += 1


        return total_reward, closed

    def _sharpe_reward(self, pnl: float) -> float:
        self._recent_returns.append(pnl)
        if len(self._recent_returns) < 2:
            return pnl
        std = np.std(self._recent_returns)
        if std < 1e-8:
            return 0.0
        shaped = pnl / std

        return shaped

    def _unrealized_pnl(self, current_price: float) -> float:
        total = 0.0
        for i in range(self.num_trades):
            if self.trades[i, 0] != 0:
                direction, entry_price, sl, tp = self.trades[i]
                total += (current_price - entry_price) * direction
        return total

    def _get_observation(self):
        current_md = self.market_data[self.current_step]
        flat_trades = self.trades.flatten()
        return np.concatenate([current_md, flat_trades]).astype(np.float32)

    def calculate_sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        periods_per_year = 252 * 96  # 15-minute bars, Trading days * 24h * 4 bars/h

        sharpe = (mean_ret - 0.0) / std_ret * np.sqrt(periods_per_year)
        return float(sharpe)



    def render(self):
        pass

    def close(self):
        pass