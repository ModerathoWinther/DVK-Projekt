import numpy
import pandas as pd

import data_process as dp

PRICE_DIR = dp.NORMALIZED_OUTPUT
INDICATOR_DIR = dp.OUTPUT_DIR


def get_market_data(split, atr=False, macd=False, rsi=False):
    price = pd.read_csv(PRICE_DIR,
                     index_col="date", parse_dates=["date"])

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
    market_data = pd.concat(frames, axis=1).dropna()

    return market_data.to_numpy()

def init_trades(num_trades, entry_per_trade):
    return numpy.zeros(num_trades * entry_per_trade)

def run(**params):
    return get_market_data(params.get('split'), atr=params.get('atr'), macd=params.get('macd'), rsi=params.get('rsi'))

