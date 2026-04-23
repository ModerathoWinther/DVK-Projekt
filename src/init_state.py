import pandas as pd

import data_process as dp

PRICE_DIR = dp.NORMAL_DIR
INDICATOR_DIR = dp.INDICATOR_DIR
STATIONARY_DIR = dp.STATIONARY_DIR

def get_input_data(split):
    price = pd.read_csv(f"{STATIONARY_DIR}/{split}.csv",
                        index_col="date", parse_dates=["date"])

    frames = [price, pd.read_csv(f"{INDICATOR_DIR}/{split}/atr.csv",
                                 index_col="date", parse_dates=["date"]),
              pd.read_csv(f"{INDICATOR_DIR}/{split}/macd.csv",
                          index_col="date", parse_dates=["date"]),
              pd.read_csv(f"{INDICATOR_DIR}/{split}/rsi.csv",
                          index_col="date", parse_dates=["date"])]

    input_data = pd.concat(frames, axis=1).dropna()

    return input_data.to_numpy()

def get_prices(split):
    price = pd.read_csv(f"{PRICE_DIR}/{split}.csv", index_col="date", parse_dates=["date"])
    price = price[['open', 'high', 'low', 'close']]
    return price.to_numpy()

def run(**params):
    return get_input_data(params.get('split')), get_prices(params.get('split'))