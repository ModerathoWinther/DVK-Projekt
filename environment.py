from enum import Enum

import numpy

import init_state

ENTRY_PER_TRADE = 4
ACTION_INDEX = 0
PRICE_INDEX = 1
SL_INDEX = 2
TP_INDEX = 3

MARKET_HIGH = 1
MARKET_LOW = 2
MARKET_CLOSE = 3


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

                self.__set_trade_info(empty_trade, action.value, state[MARKET_CLOSE], 0, 1065)
                self.open_slots -= 1
                return self.get_current_state()

    def get_reward_and_clear_trades(self):
        current_md = self.market_data[self.index]
        high = current_md[MARKET_HIGH]
        low = current_md[MARKET_LOW]

        sum_reward = 0
        for i in range(self.num_trades):
            trade = i * ENTRY_PER_TRADE
            reward, closed = self.__calculate_reward(high, low, trade)
            sum_reward += reward
            if closed:
                self.__set_trade_info(trade, 0, 0, 0, 0)
                self.open_slots += 1

        return sum_reward

    def can_trade(self):
        return self.open_slots > 0

    def __get_trade_info(self, start_index):
        action = self.trades[start_index + ACTION_INDEX]
        price = self.trades[start_index + PRICE_INDEX]
        sl = self.trades[start_index + SL_INDEX]
        tp = self.trades[start_index + TP_INDEX]
        return action, price, sl, tp

    def __set_trade_info(self, start_index, action, price, sl, tp):
        self.trades[start_index + ACTION_INDEX] = action
        self.trades[start_index + PRICE_INDEX] = price
        self.trades[start_index + SL_INDEX] = sl
        self.trades[start_index + TP_INDEX] = tp

    # todo Calculate reward using Sharpe ratio
    def __calculate_reward(self, high, low, trade):
        action, price, sl, tp = self.__get_trade_info(trade)

        closed = False
        reward = 0
        match action:
            case Action.SELL.value:
                if sl > high:
                    reward = price - sl
                    closed = True
                elif tp < low:
                    reward = price - tp
                    closed = True
            case Action.BUY.value:
                if sl > low:
                    reward = sl - price
                    closed = True
                if tp < high:
                    reward = tp - price
                    closed = True

        return reward, closed


if __name__ == "__main__":
    env = Environment("train", 2)
    print("reward:  ", env.get_reward_and_clear_trades())
    env.perform_action(Action.HOLD)
    print("reward:  ", env.get_reward_and_clear_trades())
    env.perform_action(Action.BUY)
    print("reward:  ", env.get_reward_and_clear_trades())
    for i in range(10):
        env.perform_action(Action.HOLD)
        print("reward:  ", env.get_reward_and_clear_trades())

    # Should throw an Exception because too many trades
    # print(env.perform_action(Action.BUY))
    # print(env.perform_action(action.BUY))
