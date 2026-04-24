import shutil
import unittest

import os

from numpy.ma.testutils import assert_almost_equal
import numpy as np

import data_process as dp
from trading_environment import TradingEnvironment

SPLIT = "unit_test"

HOLD_ACTION = 0
BUY_ACTION = 1
SELL_ACTION = 2

PARAM_TCOST = 0.0002
# Test always trades on entry_price = 100
TRANSACTION_COST = PARAM_TCOST * 100
TP_WIN = 2
SL_LOSS = -1

PARAMS = {
    "split": SPLIT,
    "num_trades": 5,
    "atr": False,
    "macd": False,
    "rsi": False,
    "initial_capital": 10000.0,
    "transaction_cost": 0.0002,
    "episode_length": 33,
    "data_format": "ohlcv",
    "unit_test": True,
}

filepath = os.path.abspath("test/raw_unit_test.csv")
os.makedirs("data/raw/unit_test", exist_ok=True)
shutil.copyfile(filepath, "data/raw/unit_test/raw_unit_test.csv")
shutil.copyfile(filepath, "data/processed/stationary/indicators/unit_test.csv")
shutil.copyfile(filepath, "data/processed/stationary/ohlcv-normalized/unit_test.csv")

raw = dp.load_split(SPLIT)
stat = dp.make_stationary(raw)
dp.save_candlesticks(raw, SPLIT)
dp.save_stationary_data(stat, SPLIT)

class TestTradingEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = TradingEnvironment(PARAMS)

    def test_environment_episode(self):
        env = self.env

        # Buy, hit tp
        env.step(HOLD_ACTION)
        env.step(BUY_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        _, reward, _, _, _ = env.step(HOLD_ACTION)
        assert reward == TP_WIN - TRANSACTION_COST

        # Buy, hit sl
        env.step(BUY_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        _, reward, _, _, _ = env.step(HOLD_ACTION)
        assert reward == SL_LOSS - TRANSACTION_COST

        # Sell, hit tp
        env.step(SELL_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        _, reward, _, _, _ = env.step(HOLD_ACTION)
        assert reward == TP_WIN - TRANSACTION_COST

        # Sell, hit sl
        env.step(SELL_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        _, reward, _, _, _ = env.step(HOLD_ACTION)
        assert reward == SL_LOSS - TRANSACTION_COST

        # Buy, hit tp, with slippage
        env.step(BUY_ACTION)
        env.step(HOLD_ACTION)
        _, reward, _, _, _ = env.step(HOLD_ACTION)
        assert_almost_equal(TP_WIN + 0.5 - TRANSACTION_COST, reward)

        # Buy, hit sl, with slippage
        env.step(BUY_ACTION)
        env.step(HOLD_ACTION)
        _, reward, _, _, _ = env.step(HOLD_ACTION)
        assert_almost_equal(SL_LOSS - 0.5 - TRANSACTION_COST, reward)

        # End of episode closes trades
        env.step(BUY_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        _, reward, terminated, _, _ = env.step(HOLD_ACTION)
        assert terminated == True
        assert_almost_equal(0.5 - TRANSACTION_COST, reward)

        # Reset env
        env.reset()
        env.current_step = env.episode_length
        env.episode_end = len(env.prices)
        env.episode_length = env.current_step - env.episode_end

        # 3 tp hits plus slippage and termination trade
        total_win = (TP_WIN * 3) + 0.5 + 0.5 - (TRANSACTION_COST * 4)
        avg_win = total_win / 4
        win_rate = (4 / 7)

        # 3 sl hits plus slippage
        total_loss = abs((SL_LOSS * 3) - 0.5 - (TRANSACTION_COST * 3))
        avg_loss = total_loss / 3
        loss_rate = (3 / 7)

        # init + TP_WIN - SL_LOSS + TP_WIN - SL_LOSS + TP_WIN + 0.5
        peak = env.initial_capital + (TP_WIN * 2 + 0.5) - (TRANSACTION_COST * 5)
        trough = peak + SL_LOSS - 0.5 - TRANSACTION_COST

        profit_factor = total_win / total_loss
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        max_drawdown = (trough - peak) / peak

        ep_stats = env.get_episode_stats()[0]
        assert ep_stats.get('closed_trades') == 7
        assert ep_stats.get('win_rate') == win_rate
        assert ep_stats.get('loss_rate') == loss_rate
        assert_almost_equal(profit_factor, ep_stats.get('profit_factor'))
        assert_almost_equal(expectancy, ep_stats.get('expectancy'))
        assert_almost_equal(max_drawdown, ep_stats.get('max_drawdown'))
        #assert_almost_equal(ep_stats.get('sharpe_ratio'), sharpe_ratio)

        # Both sl and tp hit at same time, should hit sl
        env.step(BUY_ACTION)
        env.step(HOLD_ACTION)
        _, reward, _, _, _ = env.step(HOLD_ACTION)
        assert reward == SL_LOSS - TRANSACTION_COST

    def test_multiple_trades(self):
        env = self.env

        env.step(BUY_ACTION)
        env.step(BUY_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        _, reward, _, _, _ = env.step(HOLD_ACTION)
        assert reward == (TP_WIN * 2) - (TRANSACTION_COST * 2)
        assert len(env.closed_trades) == 2

    def test_equity_curve(self):
        return 0