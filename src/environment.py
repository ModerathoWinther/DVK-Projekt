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


def set_sl_tp(price, action):
    tp, sl = 0, 0
    if action.direction == Direction.BUY:
        sl = price - price * action.sl
        tp = price + price * action.tp
    if action.direction == Direction.SELL:
        sl = price + price * action.sl
        tp = price - price * action.tp
    return sl, tp


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
        sl, tp = set_sl_tp(price, action)

        self.__set_trade_info(empty_trade, direction.value, state[MARKET_CLOSE], sl, tp)
        self.open_slots -= 1

        self.equity_curve.append(self.current_equity)
        return self.get_current_state()

    def get_reward_and_clear_trades(self):
        high, low = self.__get_high_low()

        sum_reward = 0
        to_clear = []
        for i in range(self.num_trades):
            trade = i * ENTRY_PER_TRADE
            reward, closed = self.__calculate_reward(high, low, trade)
            sum_reward += reward
            if closed:
                to_clear.append(trade)

        self.__clear(to_clear)

        self.current_equity += sum_reward
        self.equity_curve.append(self.current_equity)
        print(f"Sharpe ratio = {self.__calculate_sharpe_ratio()}")
        return sum_reward

    def can_trade(self):
        return self.open_slots > 0

    def has_next(self):
        return self.index < len(self.market_data) - 1


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

    def __find_empty_trade(self):
        empty_trade = -1
        for i in range(self.num_trades):
            if self.trades[i * ENTRY_PER_TRADE] == 0:
                empty_trade = i * ENTRY_PER_TRADE
                break
        return empty_trade

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

    def __clear(self, trades):
        for trade in trades:
            self.__set_trade_info(trade, 0, 0, 0, 0)
            self.open_slots += 1

    def __get_high_low(self):
        current_md = self.market_data[self.index]
        return current_md[MARKET_HIGH], current_md[MARKET_LOW]


class BackTestEnvironment:

    def __init__(self, split, num_trades, atr=False, macd=False, rsi=False):
        self.env = Environment(split, num_trades, atr, macd, rsi)

        self.closed_trades = 0
        self.total_profit = 0
        self.total_gain = 0
        self.total_loss = 0
        self.num_gain = 0
        self.num_loss = 0

    def get_current_state(self):
        return self.env.get_current_state()

    def perform_action(self, action):
        env = self.env

        env.perform_action(action)

        high, low = env.__get_high_low()

        for trade in range(env.num_trades):
            action, price, sl, tp = env.__get_trade_info(trade)

            to_clear = []
            match action:
                case Direction.SELL.value:
                    if sl < high:
                        profit = price - sl
                        self.__record_loss(profit * -1)
                        to_clear.append(trade)
                    elif tp > low:
                        profit = price - tp
                        self.__record_gain(profit)
                        to_clear.append(trade)
                case Direction.BUY.value:
                    if sl > low:
                        profit = sl - price
                        self.__record_loss(profit * -1)
                        to_clear.append(trade)
                    elif tp < high:
                        profit = tp - price
                        self.__record_gain(profit)
                        to_clear.append(trade)

        env.__clear(to_clear)

    def __record_loss(self, loss):
        self.closed_trades += 1
        self.total_profit -= loss
        self.total_loss += loss
        self.num_loss += 1

    def __record_gain(self, gain):
        self.closed_trades += 1
        self.total_profit += gain
        self.total_gain += gain
        self.num_gain += 1

    def get_results(self):
        win_rate = self.num_gain / self.closed_trades
        loss_rate = self.num_loss / self.closed_trades

        avg_gain = self.total_gain / self.num_gain
        avg_loss = self.total_loss / self.num_loss
        expectancy = (win_rate * avg_gain) - (loss_rate * avg_loss)

        profit_factor = self.total_gain / self.total_loss

        max_drawdown = 0
        sharpe_ratio = 0
        return win_rate, expectancy, profit_factor, max_drawdown, sharpe_ratio


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
