from action_space import Direction, Action, HOLD_ACTION, ACTION_SPACE

import numpy
import init_state
import yaml

ENTRY_PER_TRADE = 4
ACTION_INDEX = 0
PRICE_INDEX = 1
SL_INDEX = 2
TP_INDEX = 3

MARKET_HIGH = 1
MARKET_LOW = 2
MARKET_CLOSE = 3


class Environment:

    def __init__(self, split, num_trades, atr=False, macd=False, rsi=False):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter = yaml.safe_load(file)
            hyperparameters = all_hyperparameter['midas-dqn-1']


        self.index = 0
        self.num_trades = num_trades
        self.open_slots = num_trades
        self.market_data, self.trades = init_state.run(split, num_trades, ENTRY_PER_TRADE, atr=atr, macd=macd, rsi=rsi)

        # Sharpe ratio
        self.initial_capital = 100_000.0
        self.current_equity = self.initial_capital
        self.equity_curve = [self.initial_capital]

    def get_current_state(self):
        current_md = self.market_data[self.index]
        return numpy.concatenate([current_md, self.trades])

    def perform_action(self, action: Action):
        state = self.get_current_state()
        self.index += 1

        direction = action.direction

        if direction == Direction.HOLD:
            self.equity_curve.append(self.current_equity)
            return self.get_current_state()

        if not self.can_trade():
            raise Exception("No open slots")
        empty_trade = self.__find_empty_trade()

        price = state[MARKET_CLOSE]
        tp, sl = 0, 0
        if direction == Direction.BUY:
            sl = price - price * action.sl
            tp = price + price * action.tp
        if direction == Direction.SELL:
            sl = price + price * action.sl
            tp = price - price * action.tp

        self.__set_trade_info(empty_trade, direction.value, state[MARKET_CLOSE], sl, tp)
        self.open_slots -= 1

        self.equity_curve.append(self.current_equity)
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

        self.current_equity += sum_reward
        self.equity_curve.append(self.current_equity)
        print(f"Sharpe ratio = {self.__calculate_sharpe_ratio()}")
        return sum_reward

    def can_trade(self):
        return self.open_slots > 0

    def __calculate_sharpe_ratio(self):

        if len(self.equity_curve) < 2:
            return 0.0

        equity = numpy.array(self.equity_curve)
        returns = numpy.diff(equity) / equity[:-1]

        if len(returns) == 0 or numpy.std(returns) == 0:
            return 0.0

        mean_ret = numpy.mean(returns)
        std_ret = numpy.std(returns)

        # 252 trading days in year multiplied by 96 = number of bars/day with 15M time frame.
        periods_per_year = 252 * 96
        sharpe = (mean_ret - 0.0) / std_ret * numpy.sqrt(periods_per_year)
        return float(sharpe)

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
            case Direction.SELL.value:
                if sl < high:
                    reward = price - sl
                    closed = True
                elif tp > low:
                    reward = price - tp
                    closed = True
            case Direction.BUY.value:
                if sl > low:
                    # sl - price när SL är högre än BUY-low ger ju ex. 1000 - 1020 = -20 reward
                    reward = sl - price
                    closed = True
                # if gör att båda kan aktivera, elif gör den exclusive
                elif tp < high:
                    reward = tp - price
                    closed = True

        return reward, closed

    def __find_empty_trade(self):
        empty_trade = -1
        for i in range(self.num_trades):
            if self.trades[i * ENTRY_PER_TRADE] == 0:
                empty_trade = i * ENTRY_PER_TRADE
                break
        return empty_trade

    def has_next(self):
        return self.index < len(self.market_data) - 1

if __name__ == "__main__":
    env = Environment("train", 2)
    print(env.get_current_state())
    env.perform_action(HOLD_ACTION)
    print(env.get_current_state())
    env.perform_action(ACTION_SPACE[1])
    print(env.get_current_state())
    env.perform_action(HOLD_ACTION)
    print(env.get_current_state())
    print("Reward:", env.get_reward_and_clear_trades())
    for i in range(100):
        env.perform_action(HOLD_ACTION)
        print(env.get_reward_and_clear_trades())

    # Should throw an Exception because too many trades
    # print(env.perform_action(Action.BUY))
    # print(env.perform_action(action.BUY))
