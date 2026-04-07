import numpy

import init_state


class environment:

    def __init__(self, split, num_trades, atr=False, macd=False, rsi=False):
        self.index = 0
        self.market_data, self.trades = init_state.run(split, num_trades, atr=atr, macd=macd, rsi=rsi)

    def get_current_state(self):
        current_md = self.market_data[self.index]
        return numpy.concatenate([current_md, self.trades])

    def perform_action(self, action):
        return

if __name__ == "__main__":
    env = environment("test", 1, atr=True, macd=True, rsi=True)
    print(env.get_current_state())