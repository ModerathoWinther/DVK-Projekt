import argparse
import os
from time import sleep
from util import load_z_score_params

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.chdir('..')
import datetime
from mt5linux import MetaTrader5
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
        port = live_test_params['port']
        self.mt5 = MetaTrader5(host='localhost', port=port)
        self.mt5.initialize()
        print(self.mt5.account_info())
        print(self.mt5.version())

        self.time_start = datetime.datetime.strptime(
            live_test_params.get('time_start'),
            "%Y-%m-%d %H:%M:%S")
        self.next_time_frame = self.time_start
        self.time_frame_minute_size = live_test_params.get('time_frame_minute_size')
        self.time_end = (self.time_start +
                         datetime.timedelta(minutes=live_test_params.get('test_minute_length')))
        self.trading_vol = live_test_params.get('trading_volume')

        self.env_id = params.get('env_id')
        self.env_params = params.get('env_make_params')
        self.fc1_nodes = params.get('fc1_nodes')
        self.enable_dueling_dqn = params.get('enable_dueling_dqn')
        self.num_trades = self.env_params['num_trades']
        self.atr = self.env_params['atr']
        self.macd = self.env_params['macd']
        self.rsi = self.env_params['rsi']
        self.data_format = self.env_params['data_format']

        self.z_scores = load_z_score_params(self.data_format, self.atr, self.macd)

        self.MODEL_FILE = os.path.join(RESULTS_DIR, f'{self.env_id}.pt')
        self.LOG_FILE = os.path.join(RESULTS_DIR, f'{self.env_id}.log')

        self.num_actions = len(ACTION_SPACE)
        self.num_states = self._get_num_states()
        self.dqn = DQN(self.num_states, self.num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(DEVICE)

        self.dqn.load_state_dict(torch.load(self.MODEL_FILE))
        self.dqn.eval()
        self.input_data = self._get_input_data()
        self.send_order(ACTION_SPACE[2])


    def _get_num_states(self):
        n_states = 0
        if self.atr: n_states += 1
        if self.macd: n_states += 3
        if self.rsi: n_states += 1
        n_states += 5 if self.data_format == 'ohlcv' else 4
        n_states += self.num_trades * 4
        return n_states

    def _get_input_data(self):
        df = self.get_market_data()
        df = self._compute_indicators(df)
        df = self._normalize_input(df)
        price_feature_cols = [col for col in df.columns if col != 'date']
        return df[price_feature_cols]

    def get_market_data(self) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos("XAUUSD", self.mt5.TIMEFRAME_M1, 0, WARMUP_BARS)
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

        if self.data_format == 'ohlcv':
            dp.apply_zscore(df, params, 'volume', ['volume'])
            dp.apply_zscore(df, params, 'ohlc', ['open', 'high', 'low', 'close'])
        else:
            wicks = dp.to_wick_format(df)
            dp.apply_zscore(wicks, params, 'wick', ['high_wick', 'low_wick', 'trend'])
            dp.apply_zscore(wicks, params, 'volume', ['volume'])
            drop_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
            non_ohlc_cols = [c for c in df.columns if c not in drop_cols]
            df = pd.concat([wicks, df[non_ohlc_cols].reset_index(drop=True)], axis=1)

            dp.apply_zscore(df, params, 'volume', ['volume'])

        if self.atr:
            dp.apply_zscore(df, params, 'atr', ['atr'])

        if self.macd:
            dp.apply_zscore(df, params, 'macd', ['macd'])
            dp.apply_zscore(df, params, 'macd_signal', ['macd_signal'])
            dp.apply_zscore(df, params, 'macd_histogram', ['macd_histogram'])

        if self.rsi:
            df['rsi'] = df['rsi'] / 100.0

        return df

    def _build_observation(self):


    def send_order(self, action):
        if action.direction == Direction.HOLD:
            print(f"[{datetime.datetime.now()}] HOLD POSITION:")
            return 1

        sl = tp = 0
        order_type = None
        if action.direction == Direction.BUY:
            price = self.mt5.symbol_info_tick("XAUUSD").ask
            order_type = self.mt5.ORDER_TYPE_BUY
            sl = price - (price * action.sl)
            tp = price + (price * action.tp)
        elif action.direction == Direction.SELL:
            price = self.mt5.symbol_info_tick("XAUUSD").bid
            order_type = self.mt5.ORDER_TYPE_SELL
            sl = price + (price * action.sl)
            tp = price - (price * action.tp)

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": "XAUUSD",
            "volume": self.trading_vol,
            "type": order_type,
            "sl": sl,
            "tp": tp,
        }

        result = self.mt5.order_send(request)

        if result is None:
            print(f"[{datetime.datetime.now()}] ORDER_SEND FAILED, ERROR: {self.mt5.last_error()}")
            return -1
        elif result.retcode != self.mt5.TRADE_RETCODE_DONE:
            print(f"[{datetime.datetime.now()}] ORDER REJECTED: {result.retcode}, {result.comment}")
            return -1
        else:
            print(f"[{datetime.datetime.now()}] ORDER PLACED")
            return 1

    def close_all_positions(self):
        for pos in self.mt5.positions_get():
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": self.mt5.ORDER_TYPE_BUY if pos.type == 1 else self.mt5.ORDER_TYPE_SELL,
                "type_time": self.mt5.ORDER_TIME_GTC
            }
            self.mt5.order_send(request)

    def run(self):
        if self.time_start < datetime.datetime.now():
            raise ValueError("TIME START IS IN THE PAST")

        print(f"\nPROGRAM START: {datetime.datetime.now()}")
        print(f"TRADE START: {self.time_start}")
        print(f"TRADE END: {self.time_end}\n")

        print(self.next_time_frame)
        while True:
            if datetime.datetime.now() > self.next_time_frame:
                # use get_market_data to fetch last candlestick

                # todo Translate in-data to format used in training (normalize, etc)

                # todo Let model decide action to take
                action = ACTION_SPACE[2]
                if self.mt5.positions_total() < self.num_trades:
                    self.send_order(action)

                if self.next_time_frame >= self.time_end:
                    break
                self.next_time_frame += datetime.timedelta(minutes=self.time_frame_minute_size)
                #print(self.next_time_frame)
                #print(self.time_end)
            else:
                sleep(0.01)


        self.close_all_positions()
        print(f"\nTRADING FINISHED AT: {datetime.datetime.now()}")

        # todo Use history_deals_get or history_orders_get to extract results (not really sure if they work)

        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MT5 login details.')
    parser.add_argument('hyperparameters', help='')
    # parser.add_argument(action='store_true')
    args = parser.parse_args()
    midas = LiveTest(params=args.hyperparameters)
    midas.run()