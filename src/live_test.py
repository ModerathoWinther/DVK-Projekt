import argparse
import os
from time import sleep

import numpy as np

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
WARMUP_BARS = dp.WARMUP_ROWS + 1

class LiveTest:


    def __init__(self, params):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            params = all_hyperparameter_sets[params]

        live_test_params = params.get('live_test')

        self.time_start = datetime.datetime.strptime(
            live_test_params.get('time_start'),
            "%Y-%m-%d %H:%M:%S")
        self.next_time_frame = self.time_start
        self.time_frame_minute_size = live_test_params.get('time_frame_minute_size')
        self.time_end = (self.time_start +
                         datetime.timedelta(minutes=live_test_params.get('test_minute_length')))

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
        self.input_data = self.get_market_data()
        print(self.time_start)
        print(self.time_end)
        print(self.get_market_data())
        self.send_order(ACTION_SPACE[2])
        print(self._compute_indicators(self.input_data))
        print(self._normalize_ohlcv(self.input_data))

    def _get_num_states(self):
        n_states = 0
        if self.atr: n_states += 1
        if self.macd: n_states += 3
        if self.rsi: n_states += 1
        n_states += 5 if self.data_format == 'ohlcv' else 4
        n_states += self.num_trades * 3
        return n_states


    def get_market_data(self) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, WARMUP_BARS)
        df = pd.DataFrame(rates)
        df = df.rename(columns={'tick_volume': 'volume', 'time': 'date'})
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        return df

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ind.atr(df)
        if self.macd:
            ind.macd(df)
        if self.rsi:
            ind.rsi(df)
        df = df.dropna().reset_index(drop=True)
        return df

    def _normalize_ohlcv(self, df: pd.DataFrame) -> np.ndarray:
        p = self.z_scores
        row = df.iloc[-1]
        vol_mean, vol_std = p['volume'][0], p['volume'][1]

        if self.data_format == 'ohlcv':
            ohlc_mean, ohlc_std = p['ohlc'][0], p['ohlc'][1]
            return np.array([
                (row['open']   - ohlc_mean) / (ohlc_std  + 1e-8),
                (row['high']   - ohlc_mean) / (ohlc_std  + 1e-8),
                (row['low']    - ohlc_mean) / (ohlc_std  + 1e-8),
                (row['close']  - ohlc_mean) / (ohlc_std  + 1e-8),
                (row['volume'] - vol_mean)  / (vol_std   + 1e-8),
            ], dtype=np.float32)
        # wick
        else:
            wick_mean, wick_std = p['wick'][0], p['wick'][1]
            high_wick = row['high'] - row['open']
            low_wick = row['open'] - row['low']
            trend = row['close'] - row['open']

            return np.array([
                (high_wick - wick_mean) / (wick_std  + 1e-8),
                (low_wick   - wick_mean) / (wick_std  + 1e-8),
                (trend    - wick_mean) / (wick_std  + 1e-8),
                (row['volume'] - vol_mean)  / (vol_std   + 1e-8),
            ], dtype=np.float32)

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
        print()
        print(f"PROGRAM START: {datetime.datetime.now()}")
        print(f"TRADE START: {self.time_start}")
        print(f"TRADE END: {self.time_end}")
        print()

        while True:
            if datetime.datetime.now() > self.next_time_frame:
                # use get_market_data to fetch last candlestick

                # todo Translate in-data to format used in training (normalize, etc)

                # todo Let model decide action to take

                # use send_order to send orders via MetaTrader, only if mt5.positions_total() < 5

                print(f"[{datetime.datetime.now()}] TRADE DONE")
                if self.next_time_frame >= self.time_end:
                    break
                self.next_time_frame += datetime.timedelta(minutes=self.time_frame_minute_size)
            else:
                sleep(0.01)

        # todo Use history_deals_get or history_orders_get to extract results (not really sure if they work)
        print()
        print(f"TRADING FINISHED AT: {datetime.datetime.now()}")

        return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MT5 login details.')
    parser.add_argument('hyperparameters', help='')
    # parser.add_argument(action='store_true')
    args = parser.parse_args()
    # mt5.initialize(args.mt5_details)
    mt5.initialize()
    print(mt5.symbol_info("XAUUSD"))
    midas = LiveTest(params=args.hyperparameters)
    midas.run()

