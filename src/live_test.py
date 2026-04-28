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
        self.input_data = self._compute_indicators(self.input_data)
        print(f'\n\n after _compute_indicators(): {self.input_data}\n\n')
        self.input_data = self._normalize_input(self.input_data)
        print(f'Input data after _normalize_price(): {self.input_data}')

    def _get_num_states(self):
        n_states = 0
        if self.atr: n_states += 1
        if self.macd: n_states += 3
        if self.rsi: n_states += 1
        n_states += 5 if self.data_format == 'ohlcv' else 4
        n_states += self.num_trades * 3
        return n_states

    def get_market_data(self) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, WARMUP_BARS)
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

    def _normalize_input(self, df: pd.DataFrame) -> pd.DataFrame:
        params = self.z_scores

        dp.apply_zscore(df, params, 'volume', ['volume'])

        if self.data_format == 'ohlcv':
            dp.apply_zscore(df, params, 'ohlc', ['open', 'high', 'low', 'close'])
        else:
            dp.apply_zscore(df, params, 'wick', ['high_wick', 'low_wick', 'trend'])

        if self.atr:
            dp.apply_zscore(df, params, 'atr', ['atr'])

        if self.macd:
            dp.apply_zscore(df, params, 'macd', ['macd'])
            dp.apply_zscore(df, params, 'macd_signal', ['macd_signal'])
            dp.apply_zscore(df, params, 'macd_histogram', ['macd_histogram'])

        if self.rsi:
            df['rsi'] = df['rsi'] / 100.0

        return df

    def _load_z_score_params(self):

        path = os.path.join(f'{dp.NORMALIZED_DIR}/zscores.json')
        with open(path, 'r') as file:
            all_params = json.load(file)

        active_params = {}

        if self.data_format == 'ohlcv':
            active_params['ohlc'] = [all_params['ohlc'][0], all_params['ohlc'][1]]
        elif self.data_format == 'wick':
            active_params['wick'] = [all_params['wick'][0], all_params['wick'][1]]

        active_params['volume'] = [all_params['volume'][0], all_params['volume'][1]]

        if self.atr: active_params['atr'] = all_params['atr']
        if self.macd:
            active_params['macd'] = [all_params['macd'][0], all_params['macd'][1]]
            active_params['macd_signal'] = [all_params['macd_signal'][0], all_params['macd_signal'][1]]
            active_params['macd_histogram'] = [all_params['macd_histogram'][0], all_params['macd_histogram'][1]]

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
            "volume": self.volume,
            "type": order_type,
            "sl": sl,
            "tp": tp,
        }

        result = mt5.order_send(request)

        if result is None:
            print(f"[{datetime.datetime.now()}] ORDER_SEND FAILED, ERROR: {mt5.last_error()}")
            return -1
        elif result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[{datetime.datetime.now()}] ORDER REJECTED: {result.retcode}, {result.comment}")
            return -1
        else:
            print(f"ORDER PLACED")
            return 1

    def close_all_positions(self):
        for pos in mt5.positions_get():
            mt5.Close(symbol="XAUUSD", ticket=pos.ticket)

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
                action = ACTION_SPACE[0]
                if mt5.positions_total() < self.num_trades:
                    self.send_order(action)

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
