import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.chdir('..')

from util import load_z_score_params
import argparse
from time import sleep
import numpy as np
import datetime
from mt5linux import MetaTrader5
import MetaTrader5 as mt5_local
import pandas as pd
import torch
import yaml
from action_space import Direction, ACTION_SPACE
from dqn import DQN
import data_process as dp
import indicators as ind

DEVICE = 'cpu'
RESULTS_DIR = 'results'
OUTPUT_DIR = os.path.join(f'{RESULTS_DIR}/live')
WARMUP_BARS = dp.WARMUP_ROWS + 1
SYMBOL = 'XAUUSD'

class LiveTest:

    def __init__(self, params):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            params = all_hyperparameter_sets[params]

        self.live_test_params = params.get('live_test')
        port = self.live_test_params['port']
        self.is_containerized = self.live_test_params['is_containerized']


        if self.is_containerized: self.mt5 = MetaTrader5(host='localhost', port=port)
        else: self.mt5 = mt5_local

        self.mt5.initialize()
        print(self.mt5.account_info())
        print(self.mt5.version())


        self.time_start = datetime.datetime.strptime(
            self.live_test_params.get('time_start'),
            "%Y-%m-%d %H:%M:%S")
        self.next_time_frame = self.time_start
        self.time_frame_minute_size = self.live_test_params.get('time_frame_minute_size')
        self.time_end = (self.time_start +
                         datetime.timedelta(minutes=self.live_test_params.get('test_minute_length')))
        self.trading_vol = self.live_test_params.get('trading_volume')

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

        self.trades_state = np.zeros((self.num_trades, 4), dtype=np.float32)
        self.trades_obs = np.zeros((self.num_trades, 3), dtype=np.float32)
        self.open_slots = self.num_trades

        print(self.num_actions)
        print(self.num_states)

        self.dqn = DQN(self.num_states, self.num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(DEVICE)

        self.dqn.load_state_dict(torch.load(self.MODEL_FILE))
        self.dqn.eval()
        self.input_data = self._get_input_data()

    def _get_num_states(self):
        n_states = 0
        if self.atr: n_states += 1
        if self.macd: n_states += 3
        if self.rsi: n_states += 1
        n_states += 5 if self.data_format == 'ohlcv' else 4
        n_states += (self.num_trades * 3)
        print(f'n_states. {n_states}')
        return n_states

    def _get_input_data(self):
        df = self.get_market_data()
        df = self._compute_indicators(df)
        df = self._normalize_input(df)
        price_feature_cols = [col for col in df.columns if col != 'date']
        return df[price_feature_cols].values

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

    def _build_observation(self, df: pd.DataFrame) -> torch.Tensor:
        row = df.iloc[-1]

        if self.data_format == 'ohlcv':
            market_features = [
                row['open'], row['high'], row['low'], row['close'], row['volume']
            ]
        else:
            market_features = [
                row['high_wick'], row['low_wick'], row['trend'], row['volume']
            ]

        if self.atr:
            market_features.append(row['atr'])
        if self.macd:
            market_features.extend([row['macd'], row['macd_signal'], row['macd_histogram']])
        if self.rsi:
            market_features.append(row['rsi'])

        raw_close = self._get_raw_close()
        trades_obs = self._build_trades_obs(raw_close)

        obs = np.array(market_features, dtype=np.float32)
        obs = np.concatenate([obs, trades_obs])

        assert len(obs) == self.num_states, (
            f"Observation shape mismatch: got {len(obs)}, expected {self.num_states}. "
            f"market={len(market_features)}, trades={len(trades_obs)}"
        )

        return torch.tensor(obs, dtype=torch.float, device=DEVICE)

    def _build_trades_obs(self, current_price: float) -> np.ndarray:
        positions = self.mt5.positions_get(symbol="XAUUSD") or []
        trades_obs = np.zeros((self.num_trades, 3), dtype=np.float32)

        for i, pos in enumerate(positions[:self.num_trades]):
            direction = 1.0 if pos.type == self.mt5.POSITION_TYPE_BUY else -1.0
            sl_dist = abs(pos.sl - current_price) if pos.sl else 0.0
            tp_dist = abs(pos.tp - current_price) if pos.tp else 0.0
            trades_obs[i] = [direction, sl_dist, tp_dist]
        return trades_obs.flatten()

    def _sync_trades_state(self) -> None:
        open_positions = self.mt5.positions_get(symbol="XAUUSD") or []
        open_count = len(open_positions)
        our_count = int(np.sum(self.trades_state[:, 0] != 0))

        if open_count < our_count:
            to_clear = our_count - open_count
            cleared = 0
            for i in range(self.num_trades - 1, -1, -1):
                if self.trades_state[i, 0] != 0 and cleared < to_clear:
                    self.trades_state[i] = [0, 0, 0, 0]
                    self.open_slots += 1
                    cleared += 1

    def _open_positions_count(self) -> int:
        positions = self.mt5.positions_get(symbol=SYMBOL)
        return len(positions) if positions else 0

    def _get_raw_close(self) -> float:
        tick = self.mt5.symbol_info_tick("XAUUSD")
        return (tick.ask + tick.bid) / 2 if tick else 0.0

    def _process_action(self, action, price: float) -> None:
        sl = price - action.direction.value * (price * action.sl)
        tp = price + action.direction.value * (price * action.tp)
        for i in range(self.num_trades):
            if self.trades_state[i, 0] == 0:
                self.trades_state[i] = [action.direction.value, price, sl, tp]
                self.open_slots -= 1
                break

    def _log(self, message: str) -> None:
        ts = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line)
        with open(self.LOG_FILE, 'a') as f:
            f.write(line + '\n')

    def send_order(self, action, raw_atr: float) -> int:
        if action.direction == Direction.HOLD:
            self._log("HOLD")
            return 1

        if self.mt5.positions_total() >= self.num_trades:
            self._log("Max positions open — skipping")
            return 0

        tick = self.mt5.symbol_info_tick("XAUUSD")
        if tick is None:
            self._log(f"ERROR: no tick data — {self.mt5.last_error()}")
            return -1

        if action.direction == Direction.BUY:
            price = tick.ask
            order_type = self.mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            order_type = self.mt5.ORDER_TYPE_SELL

        sl = price - action.direction.value * action.sl * raw_atr
        tp = price + action.direction.value * action.tp * raw_atr

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": "XAUUSD",
            "volume": self.trading_vol,
            "type": order_type,
            "price": price,
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "deviation": 10,
            "magic": 20250430,
            "comment": f"midas {self.env_id}",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

        result = self.mt5.order_send(request)
        if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
            err = self.mt5.last_error() if result is None else result.comment
            self._log(f"ORDER REJECTED: {err}")
            return -1

        self._log(f"ORDER: {action.direction.name} | price={price:.2f} | "
                  f"sl={sl:.2f} | tp={tp:.2f} | atr={raw_atr:.4f}")
        return 1

    # def send_order(self, action):
    #     if action.direction == Direction.HOLD:
    #         print(f"[{datetime.datetime.now()}] HOLD POSITION:")
    #         return 1
    #
    #     sl = tp = 0
    #     order_type = None
    #     if action.direction == Direction.BUY:
    #         price = self.mt5.symbol_info_tick("XAUUSD").ask
    #         order_type = self.mt5.ORDER_TYPE_BUY
    #         sl = price - (price * action.sl)
    #         tp = price + (price * action.tp)
    #     elif action.direction == Direction.SELL:
    #         price = self.mt5.symbol_info_tick("XAUUSD").bid
    #         order_type = self.mt5.ORDER_TYPE_SELL
    #         sl = price + (price * action.sl)
    #         tp = price - (price * action.tp)
    #
    #     request = {
    #         "action": self.mt5.TRADE_ACTION_DEAL,
    #         "symbol": "XAUUSD",
    #         "volume": self.trading_vol,
    #         "type": order_type,
    #         "sl": sl,
    #         "tp": tp,
    #     }
    #
    #     result = self.mt5.order_send(request)
    #
    #     if result is None:
    #         print(f"[{datetime.datetime.now()}] ORDER_SEND FAILED, ERROR: {self.mt5.last_error()}")
    #         return -1
    #     elif result.retcode != self.mt5.TRADE_RETCODE_DONE:
    #         print(f"[{datetime.datetime.now()}] ORDER REJECTED: {result.retcode}, {result.comment}")
    #         return -1
    #     else:
    #         print(f"[{datetime.datetime.now()}] ORDER PLACED")
    #         return 1

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
        print(f'build_observation: {self._build_observation()}')
        if self.time_start < datetime.datetime.now():
            raise ValueError("TIME START IS IN THE PAST")

        print(f"\nPROGRAM START: {datetime.datetime.now()}")
        print(f"TRADE START: {self.time_start}")
        print(f"TRADE END: {self.time_end}\n")
        print(self.next_time_frame)

        while True:
            now = datetime.datetime.now()

            if now > self.next_time_frame:

                self.input_data = self._get_input_data()
                self._sync_trades_state()
                state = self._build_observation(self.input_data)

                with torch.no_grad():
                    q_values = self.dqn(state.unsqueeze(0)).squeeze()
                    action_index = int(q_values.argmax().item())

                action = ACTION_SPACE[action_index]

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