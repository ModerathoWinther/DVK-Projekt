import os

import numpy as np

import data_process as dp
import json

def load_z_score_params(data_format, atr, macd):
    path = os.path.join(f'{dp.NORMALIZED_DIR}/zscores.json')
    with open(path, 'r') as file:
        all_params = json.load(file)

    active_params = {}

    if data_format == 'ohlcv':
        active_params['ohlc'] = [all_params['ohlc'][0], all_params['ohlc'][1]]
    elif data_format == 'wick':
        active_params['wick'] = [all_params['wick'][0], all_params['wick'][1]]

    active_params['volume'] = [all_params['volume'][0], all_params['volume'][1]]

    if atr: active_params['atr'] = all_params['atr']
    if macd:
        active_params['macd'] = [all_params['macd'][0], all_params['macd'][1]]
        active_params['macd_signal'] = [all_params['macd_signal'][0], all_params['macd_signal'][1]]
        active_params['macd_histogram'] = [all_params['macd_histogram'][0], all_params['macd_histogram'][1]]

    return active_params


def calculate_sharpe_ratio(equity_curve, bars_per_day) -> float:
    if len(equity_curve) < 2:
        return 0.0
    equity = np.array(equity_curve)
    if np.any(equity <= 0):
        return -999.0
    returns = np.diff(equity) / equity[:-1]
    active_returns = returns[returns != 0.0]
    if len(active_returns) < 2:
        return 0.0
    std_ret = np.std(active_returns, ddof=1)
    if std_ret < 1e-8:
        return 0.0
    mean_ret = np.mean(active_returns)

    total_bars = len(equity_curve)
    n_trades = len(active_returns)
    trades_per_year = (n_trades / total_bars) * (252 * bars_per_day)

    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)
    return float(sharpe)

