import numpy
from enum import Enum

import init_state

class action(Enum):
    SELL = 1
    HOLD = 2
    BUY = 3

class environment:

    def __init__(self, split, num_trades, atr=False, macd=False, rsi=False):
        self.index = 0
        self.market_data, self.trades = init_state.run(split, num_trades, atr=atr, macd=macd, rsi=rsi)

    def get_current_state(self):
        current_md = self.market_data[self.index]
        return numpy.concatenate([current_md, self.trades])

    def perform_action(self, action):
        state = self.get_current_state()

        match(action):
            case action.HOLD:
                self.index += 1
                return self.get_current_state()
            case action.SELL:
                return self.get_current_state()
            case action.BUY:
                return self.get_current_state()


if __name__ == "__main__":
    env = environment("test", 1, atr=True, macd=True, rsi=True)
    print(env.get_current_state())
    print(env.perform_action(action.HOLD))
