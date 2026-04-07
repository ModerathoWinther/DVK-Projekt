import numpy
from enum import Enum

import init_state


ACTION_INDEX = 0
PRICE_INDEX = 1

ENTRY_PER_TRADE = 2


class Action(Enum):
    SELL = -1
    HOLD = 0
    BUY = 1

class Environment:

    def __init__(self, split, num_trades, atr=False, macd=False, rsi=False):
        self.index = 0
        self.num_trades = num_trades
        self.open_slots = num_trades
        self.market_data, self.trades = init_state.run(split, num_trades, ENTRY_PER_TRADE, atr=atr, macd=macd, rsi=rsi)

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
                if not self.can_trade():
                    raise Exception("No open slots")

                empty_trade = -1
                for i in range(self.num_trades):
                    if self.trades[i * ENTRY_PER_TRADE] == 0:
                        empty_trade = i * ENTRY_PER_TRADE
                        break

                self.trades[empty_trade + ACTION_INDEX] = action.value
                self.trades[empty_trade + PRICE_INDEX] = state[3]
                self.open_slots -= 1
                return self.get_current_state()

    def can_trade(self):
        return self.open_slots > 0


if __name__ == "__main__":
    env = Environment("test", 2, atr=True, macd=True, rsi=True)
    print(env.get_current_state())
    print(env.perform_action(Action.HOLD))
    print(env.perform_action(Action.SELL))
    print(env.perform_action(Action.BUY))
    # Should throw an Exception because too many trades
    # print(env.perform_action(action.BUY))
