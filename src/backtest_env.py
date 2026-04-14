from environment import Environment
from action_space import Direction, Action, HOLD_ACTION, ACTION_SPACE

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
        sharpe_ratio = self.env.calc_sharpe_ratio()
        return win_rate, expectancy, profit_factor, max_drawdown, sharpe_ratio