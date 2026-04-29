import os
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

