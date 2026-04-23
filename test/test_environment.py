import shutil
import unittest

import os

from numpy.ma.testutils import assert_almost_equal

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
TP_LOSS = -1

PARAMS = {
    "split": SPLIT,
    "num_trades": 5,
    "atr": False,
    "macd": False,
    "rsi": False,
    "initial_capital": 10000.0,
    "transaction_cost": 0.0002,
    "episode_length": 33,
    "price_columns": 3,
    "unit_test": True,
}

filepath = os.path.abspath("test/raw_unit_test.csv")
os.makedirs("data/raw/unit_test", exist_ok=True)
shutil.copyfile(filepath, "data/raw/unit_test/raw_unit_test.csv")

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
        assert reward == TP_LOSS - TRANSACTION_COST

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
        assert reward == TP_LOSS - TRANSACTION_COST

        # Buy, hit tp, with slippage
        env.step(BUY_ACTION)
        env.step(HOLD_ACTION)
        _, reward, _, _, _ = env.step(HOLD_ACTION)
        #assert reward == tp_win + 0.5 - transaction_cost

        # Buy, hit sl, with slippage
        env.step(BUY_ACTION)
        env.step(HOLD_ACTION)
        _, reward, _, _, _ = env.step(HOLD_ACTION)
        #assert reward == tp_loss - 0.5 - transaction_cost

        # End of episode closes trades
        env.step(BUY_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        env.step(HOLD_ACTION)
        _, reward, terminated, _, _ = env.step(HOLD_ACTION)
        assert terminated == True
        assert_almost_equal(reward, 0.5 - TRANSACTION_COST)

        # todo check episode stats
        env.reset()
        env.current_step = env.episode_length
        env.episode_end = len(env.prices)
        env.episode_length = env.current_step - env.episode_end

        # Both sl and tp hit at same time, should hit sl
        env.step(BUY_ACTION)
        env.step(HOLD_ACTION)
        _, reward, _, _, _ = env.step(HOLD_ACTION)
        assert reward == TP_LOSS - TRANSACTION_COST

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
