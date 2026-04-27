import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.chdir('..')
import datetime
import MetaTrader5 as mt5
import pandas as pd
import torch
import yaml
import data_process as dp
from action_space import Direction, ACTION_SPACE
from dqn import DQN

DEVICE = 'cpu'
RESULTS_DIR = 'results'

class LiveTest:


    def __init__(self, params):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            params = all_hyperparameter_sets[params]

        self.time_start = datetime.now()
        self.next_time_frame = self.time_start
        self.time_end = self.time_start + datetime.timedelta(minutes=15)
        self.parameters = params.get('')
        self.env_id = params.get('env_id')
        self.num_trades = params.get('num_trades')
        self.fc1_nodes = params.get('fc1_nodes')
        self.enable_dueling_dqn = params.get('enable_dueling_dqn')
        self.atr = params.get('atr')
        self.macd = params.get('macd')
        self.rsi = params.get('rsi')
        self.data_format = params.get('data_format')



        self.MODEL_FILE = os.path.join(RESULTS_DIR, f'{self.env_id}.pt')
        self.LOG_FILE = os.path.join(RESULTS_DIR, f'{self.env_id}.log')

        self.num_actions = len(ACTION_SPACE)
        self.num_states = self._get_num_states()

        self.dqn = DQN(self.num_states, self.num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(DEVICE)

        self.dqn.load_state_dict(torch.load(self.MODEL_FILE))
        self.dqn.eval()


        print(self.time_start)
        print(self.time_end)
        print(self.get_market_data())
        self.send_order(ACTION_SPACE[2])

    def _get_num_states(self):
        n_states = 0
        if self.atr: n_states += 1
        if self.rsi: n_states += 1
        if self.macd: n_states += 1
        n_states += 5 if self.data_format == 'ohlcv' else 4
        n_states += self.num_trades * 3
        return n_states


    def get_market_data(self):
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, 1)
        df = pd.DataFrame(rates)
        df = df.rename(columns={'tick_volume': 'volume'})
        final_df = df[['open', 'high', 'low', 'close', 'volume']]
        return final_df

    def send_order(self, action):
        if action.direction == Direction.HOLD:
            return -1

        sl = tp = 0
        order_type = None
        if action.direction == Direction.BUY:
            price = mt5.symbol_info_tick("XAUUSD").ask
            order_type = mt5.ORDER_TYPE_BUY
            sl = price - (price * action.sl)
            tp = price + (price * action.tp)
        elif action.direction == Direction.SELL:
            price = mt5.symbol_info_tick("XAUUSD").bid
            order_type = mt5.ORDER_TYPE_SELL
            sl = price + (price * action.sl)
            tp = price - (price * action.tp)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": "XAUUSD",
            "type": order_type,
            "sl": sl,
            "tp": tp,
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": self.time_end
        }

    def run(self):
        # Loop

        # use get_market_data to fetch last candlestick

        # todo Translate in-data to format used in training (normalize, etc)

        # todo Let model decide action to take

        # use send_order to send orders via MetaTrader, only if mt5.positions_total() < 5

        # After loop
        # todo Use history_deals_get or history_orders_get to extract results (not really sure if they work)

        return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MT5 login details.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument(action='store_true')
    args = parser.parse_args()

    midas = LiveTest(params=args.hyperparameters)
    mt5.initialize(args.mt5_details)
