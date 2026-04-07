import numpy
from enum import Enum

import init_state


ACTION_INDEX = 0
PRICE_INDEX = 1


class action(Enum):
    SELL = -1
    HOLD = 0
    BUY = 1

class environment:

    def __init__(self, split, num_trades, atr=False, macd=False, rsi=False):
        self.index = 0
        self.market_data, self.trades = init_state.run(split, num_trades, atr=atr, macd=macd, rsi=rsi)

    def get_current_state(self):
        current_md = self.market_data[self.index]
        return numpy.concatenate([current_md, self.trades])

    def perform_action(self, action):
        state = self.get_current_state()
        self.index += 1

        match action:
            case action.HOLD:
                return self.get_current_state()
            case _:
                self.trades[ACTION_INDEX] = action.value
                self.trades[PRICE_INDEX] = state[3]
                return self.get_current_state()


if __name__ == "__main__":
    env = environment("test", 1, atr=True, macd=True, rsi=True)
    print(env.get_current_state())
    print(env.perform_action(action.HOLD))
    print(env.perform_action(action.SELL))
