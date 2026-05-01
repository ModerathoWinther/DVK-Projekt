import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.chdir('..')

from util import load_z_score_params
import argparse
from time import sleep
import numpy as np
import datetime
import pytz
from mt5linux import MetaTrader5
import MetaTrader5 as mt5_local
import pandas as pd
import torch
import yaml
from action_space import Direction, ACTION_SPACE
from dqn import DQN
import data_process as dp
import indicators as ind
import util

DEVICE = 'cpu'
RESULTS_DIR = 'results'
WARMUP_BARS = dp.WARMUP_ROWS + 1
SYMBOL = "XAUUSD"

MT5_TIMEZONE = pytz.timezone("Europe/Helsinki")
BARS_PER_DAY_M1 = 92 * 15


class LiveTest:

    def __init__(self, params):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            params = all_hyperparameter_sets[params]

        self.live_test_params = params.get('live_test')
        port = self.live_test_params['port']
        self.is_containerized = self.live_test_params['is_containerized']
        self.env_make_params = params.get('env_make_params')
        self.is_buy_hold = self.env_make_params['is_buy_hold']

        if self.is_containerized:
            self.mt5 = MetaTrader5(host='localhost', port=port)

        else:
            self.mt5 = mt5_local
            mt5_local.initialize()

        self.mt5.initialize()
        print(self.mt5.account_info())
        print(self.mt5.version())

        self.num_tests = self.live_test_params.get('num_tests')
        self.time_start = datetime.datetime.strptime(
            self.live_test_params.get('time_start'),
            "%Y-%m-%d %H:%M:%S")
        self.next_time_frame = self.time_start
        self.time_frame_minute_size = self.live_test_params.get('time_frame_minute_size')
        self.test_minute_length = self.live_test_params.get('test_minute_length')
        self.time_end = (self.time_start +
                         datetime.timedelta(minutes=self.test_minute_length))
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
        self.slot_tickets = np.full(self.num_trades, -1, dtype=np.int64)
        self.open_slots = self.num_trades

        print(self.num_actions)
        print(self.num_states)

        self.dqn = DQN(self.num_states, self.num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(DEVICE)

        self.dqn.load_state_dict(torch.load(self.MODEL_FILE))
        self.dqn.eval()
        self._get_input_data()

        self.equity_curve = []

        self.env_id = params.get('env_id')

    def _get_num_states(self):
        n_states = 0
        if self.atr: n_states += 1
        if self.macd: n_states += 3
        if self.rsi: n_states += 1
        n_states += 5 if self.data_format == 'ohlcv' else 4
        n_states += (self.num_trades * 3)
        return n_states

    def _get_input_data(self):
        df = self.get_market_data()
        df = self._compute_indicators(df)
        df = self._normalize_input(df)
        price_feature_cols = [col for col in df.columns if col != 'date']
        self.input_data = df[price_feature_cols].values

    def get_market_data(self) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos(SYMBOL, self.mt5.TIMEFRAME_M1, 0, WARMUP_BARS)
        df = pd.DataFrame(rates)
        print(df)
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

    def _build_observation(self) -> torch.Tensor:
        market_features = self.input_data[-1]

        tick = self.mt5.symbol_info_tick(SYMBOL)
        current_price = tick.bid if tick else 0.0

        trades_features = self._build_trades_obs(current_price)

        obs = np.concatenate([
            market_features,
            trades_features
        ]).astype(np.float32)

        assert len(obs) == self.num_states, (
            f"Got obs count: {len(obs)} but expected: {self.num_states}. "
            f"Market: {len(market_features)}, Trades: {len(trades_features)}"
        )

        return torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    def _build_trades_obs(self, current_price: float) -> np.ndarray:
        trades_obs = np.zeros((self.num_trades, 3), dtype=np.float32)

        for i in range(self.num_trades):
            direction, entry_price, sl, tp = self.trades_state[i]
            if direction != 0:
                sl_dist = abs(sl - current_price) / current_price
                tp_dist = abs(tp - current_price) / current_price
                trades_obs[i] = [direction, sl_dist, tp_dist]
        return trades_obs.flatten()

    def _sync_mt5_trades(self) -> None:
        positions = self.mt5.positions_get(symbol=SYMBOL) or []
        open_tickets = {pos.ticket for pos in positions}
        for i in range(self.num_trades):
            if self.trades_state[i, 0] != 0 and self.slot_tickets[i] not in open_tickets:
                self.trades_state[i] = [0, 0, 0, 0]
                self.slot_tickets[i] = -1
                self.open_slots += 1

    def _store_trades(self, action, price: float, ticket: int) -> None:
        sl = price - action.direction.value * (price * action.sl)
        tp = price + action.direction.value * (price * action.tp)
        for i in range(self.num_trades):
            if self.trades_state[i, 0] == 0:
                self.trades_state[i] = [action.direction.value, price, sl, tp]
                self.slot_tickets[i] = ticket
                self.open_slots -= 1
                break

    def _log(self, message: str) -> None:
        ts = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line)
        with open(self.LOG_FILE, 'a') as f:
            f.write(line + '\n')

    def send_order(self, action):
        if action.direction == Direction.HOLD:
            print(f"[{datetime.datetime.now()}] HOLD POSITION:")
            return 1, None

        sl = tp = 0
        order_type = None
        if action.direction == Direction.BUY:
            price = self.mt5.symbol_info_tick(SYMBOL).ask
            order_type = self.mt5.ORDER_TYPE_BUY
            sl = price - (price * action.sl)
            tp = price + (price * action.tp)
        elif action.direction == Direction.SELL:
            price = self.mt5.symbol_info_tick(SYMBOL).bid
            order_type = self.mt5.ORDER_TYPE_SELL
            sl = price + (price * action.sl)
            tp = price - (price * action.tp)

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": self.trading_vol,
            "type": order_type,
            "sl": sl,
            "tp": tp,
        }

        result = self.mt5.order_send(request)

        if result is None:
            print(f"[{datetime.datetime.now()}] ORDER_SEND FAILED, ERROR: {self.mt5.last_error()}")
            return -1, None
        elif result.retcode != self.mt5.TRADE_RETCODE_DONE:
            print(f"[{datetime.datetime.now()}] ORDER REJECTED: {result.retcode}, {result.comment}")
            return -1, None
        else:
            print(f"[{datetime.datetime.now()}] ORDER PLACED")
            return 1, result.order

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

    def send_buy_hold_order(self):
        price = self.mt5.symbol_info_tick(SYMBOL).ask
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": self.trading_vol,
            "type": self.mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": 50,
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

    def close_all_positions_buy_hold(self):
        for pos in self.mt5.positions_get():
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": self.mt5.ORDER_TYPE_SELL,
                "type_time": self.mt5.ORDER_TIME_GTC,
                "price": self.mt5.symbol_info_tick(SYMBOL).bid,
                "deviation": 50,
            }
            self.mt5.order_send(request)

    def run(self):
        os.makedirs(f'results/live_test/{self.env_id}', exist_ok=True)
        has_traded = False

        for i in range(self.num_tests):
            print(f'build_observation: {self._build_observation()}')
            # if self.time_start < datetime.datetime.now():
            #     raise ValueError("TIME START IS IN THE PAST")

            print(f"\nPROGRAM START: {datetime.datetime.now()}")
            print(f"TRADE START: {self.time_start}")
            print(f"TRADE END: {self.time_end}\n")
            while True:
                now = datetime.datetime.now()
                if now > self.next_time_frame:
                    if self.next_time_frame >= self.time_end:
                        break

                    if self.is_buy_hold:
                        if not has_traded:
                            for _ in range(self.num_trades):
                                self.send_buy_hold_order()
                            has_traded = True
                    else:
                        self._get_input_data()

                        self._sync_mt5_trades()

                        state = self._build_observation()
                        print(state)
                        with torch.no_grad():
                            action_index = self.dqn(state.unsqueeze(0)).squeeze().argmax().item()

                        action = ACTION_SPACE[action_index]

                        print(f"[{now}] Action: {action.direction.name}"
                              f"  sl={action.sl}  tp={action.tp}"
                              f"  open_slots={self.open_slots}")

                        if action.direction != Direction.HOLD and self.open_slots > 0:
                            result, ticket = self.send_order(action)
                            if result == 1:
                                price = (self.mt5.symbol_info_tick(SYMBOL).ask
                                         if action.direction == Direction.BUY
                                         else self.mt5.symbol_info_tick(SYMBOL).bid)
                                self._store_trades(action, price, ticket=ticket)
                        else:
                            print(f"[{datetime.datetime.now()}] NO ORDER, TOO MANY TRADES OPEN")

                    self.next_time_frame += datetime.timedelta(minutes=self.time_frame_minute_size)
                    self.equity_curve.append(self.mt5.account_info().balance)
                else:
                    sleep(0.01)

            if self.is_buy_hold:
                self.close_all_positions_buy_hold()
            else:
                self.close_all_positions()
            print(f"\nTRADING FINISHED AT: {datetime.datetime.now()}")

            # Let terminal save deals
            sleep(1)

            self.equity_curve.append(self.mt5.account_info().balance)
            current_equity_curve = self.equity_curve[-(self.test_minute_length + 1):]
            print(self.equity_curve)
            print(current_equity_curve)

            # Translate to mt5 timezones to get correct window
            eest_now = datetime.datetime.now(MT5_TIMEZONE).replace(tzinfo=None)
            eest_start = self.time_start.astimezone(MT5_TIMEZONE)
            eest_start = eest_start.replace(tzinfo=None)
            deals = self.mt5.history_deals_get(eest_start, eest_now, group=SYMBOL)

            try:
                original_df = pd.read_csv(f'results/live_test/{self.env_id}/{self.env_id}.csv')
            except FileNotFoundError:
                original_df = pd.DataFrame(
                    columns=['id', 'win_rate', 'loss_rate', 'profit_factor', 'expectancy', 'max_drawdown'])

            if len(deals) > 0:
                df = pd.DataFrame([d._asdict() for d in deals])
                df = df[df['entry'] == 1]

                stats = self.calc_stats(current_equity_curve, df)
                stats['id'] = self.env_id
            else:
                stats = {'id': self.env_id, 'win_rate': None, 'loss_rate': None, 'profit_factor': None,
                         'expectancy': None, 'max_drawdown': None}

            print(stats)
            new_df = pd.concat([original_df, pd.DataFrame([stats])], ignore_index=True)
            new_df.to_csv(f'results/live_test/{self.env_id}/{self.env_id}.csv', index=False)

            self.time_start = datetime.datetime.now() + datetime.timedelta(minutes=1)
            self.time_start = self.time_start.replace(second=0, microsecond=0)
            self.next_time_frame = self.time_start
            self.time_end = self.time_start + datetime.timedelta(minutes=self.test_minute_length)

        print(util.calculate_sharpe_ratio(self.equity_curve, BARS_PER_DAY_M1))
        return 1

    def calc_stats(self, equity_curve, deals):

        profits = deals['profit'].values

        wins = profits[profits > 0]
        losses = profits[profits < 0]

        win_rate = len(wins) / len(profits)
        loss_rate = len(losses) / len(profits)
        profit_factor = wins.sum() / abs(losses.sum())
        expectancy = profits.mean()

        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdowns)

        return {
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_drawdown": max_drawdown,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MT5 login details.')
    parser.add_argument('hyperparameters', help='')
    # parser.add_argument(action='store_true')
    args = parser.parse_args()
    midas = LiveTest(params=args.hyperparameters)
    midas.run()
