import pandas as pd

import data_process as dp

PRICE_DIR = dp.NORMAL_DIR
INDICATOR_DIR = dp.INDICATOR_DIR
STATIONARY_DIR = dp.STATIONARY_DIR

def get_input_data(split, atr=False, macd=False, rsi=False):
    price = pd.read_csv(f"{STATIONARY_DIR}/{split}.csv",
                        index_col="date", parse_dates=["date"])

    print(f'price.columns = {price.columns}')
    frames = [price]
    if atr:
        frames.append(pd.read_csv(f"{INDICATOR_DIR}/{split}/atr.csv",
                                  index_col="date", parse_dates=["date"]))
    if macd:
        frames.append(pd.read_csv(f"{INDICATOR_DIR}/{split}/macd.csv",
                                  index_col="date", parse_dates=["date"]))
    if rsi:
        frames.append(pd.read_csv(f"{INDICATOR_DIR}/{split}/rsi.csv",
                                  index_col="date", parse_dates=["date"]))
    input_data = pd.concat(frames, axis=1).dropna()
    print(f'input_data.columns = {input_data.columns}')

    return input_data.to_numpy()

def get_prices(split):
    price = pd.read_csv(f"{PRICE_DIR}/{split}.csv", index_col="date", parse_dates=["date"])
    price = price[['high', 'low', 'close']]
    return price

def run(**params):
    input_data = get_input_data(params.get('split'), atr=params.get('atr'), macd=params.get('macd'), rsi=params.get('rsi'))
    prices = get_prices(params.get('split'))
    return input_data, prices