import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.chdir('..')
import json
import datetime
import MetaTrader5 as mt5
import pandas as pd
import torch
import yaml
from action_space import Direction, ACTION_SPACE
from dqn import DQN
import data_process as dp
import indicators as ind

DEVICE = 'cpu'
RESULTS_DIR = 'results'

class LiveTest:


    def __init__(self, params):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            params = all_hyperparameter_sets[params]

        self.time_start = datetime.datetime.now()
        self.next_time_frame = self.time_start
        self.time_end = self.time_start + datetime.timedelta(minutes=15)
        self.parameters = params.get('')
        self.env_id = params.get('env_id')
        self.env_params = params.get('env_make_params')
        self.fc1_nodes = params.get('fc1_nodes')
        self.enable_dueling_dqn = params.get('enable_dueling_dqn')
        self.num_trades = self.env_params['num_trades']
        self.atr = self.env_params['atr']
        self.macd = self.env_params['macd']
        self.rsi = self.env_params['rsi']
        self.data_format = self.env_params['data_format']
        self.price_mean = 0
        self.price_std = 0

        self.z_scores = self._load_z_score_params()

        self.MODEL_FILE = os.path.join(RESULTS_DIR, f'{self.env_id}.pt')
        self.LOG_FILE = os.path.join(RESULTS_DIR, f'{self.env_id}.log')

        self.num_actions = len(ACTION_SPACE)
        self.num_states = self._get_num_states()
        print(self.num_actions, self.num_states)
        self.dqn = DQN(self.num_states, self.num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(DEVICE)

        self.dqn.load_state_dict(torch.load(self.MODEL_FILE))
        self.dqn.eval()
        self.input_data = self._process_data(self.get_market_data())
        print(self.time_start)
        print(self.time_end)
        print(self.get_market_data())
        self.send_order(ACTION_SPACE[2])

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:

        return df

    def _get_num_states(self):
        n_states = 0
        if self.atr: n_states += 1
        if self.macd: n_states += 3
        if self.rsi: n_states += 1
        n_states += 5 if self.data_format == 'ohlcv' else 4
        n_states += self.num_trades * 3
        return n_states


    def get_market_data(self) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, dp.WARMUP_ROWS)
        df = pd.DataFrame(rates)
        df = df.rename(columns={'tick_volume': 'volume', 'time': 'date'})
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        return df

    def _load_z_score_params(self):

        path = os.path.join(f'{dp.NORMALIZED_DIR}/zscores.json')
        with open(path, 'r') as file:
            all_params = json.load(file)

        active_params = {}

        if self.data_format == 'ohlcv':
            active_params['ohlc'] = all_params['ohlc']
            active_params['volume'] = all_params['volume']
        elif self.data_format == 'wick':
            active_params['wick'] = all_params['wick']
            active_params['volume'] = all_params['volume']

        if self.atr: active_params['atr'] = all_params['atr']
        if self.macd: active_params['macd'] = all_params['macd']

        print(f'\n\nactive_params: {active_params}\n\n')
        return active_params

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
    # parser.add_argument(action='store_true')
    args = parser.parse_args()
    # mt5.initialize(args.mt5_details)
    mt5.initialize()
    midas = LiveTest(params=args.hyperparameters)

