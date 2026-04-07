import numpy
import pandas as pd
import data_process as dp


PRICE_DIR = dp.INPUT_DIR
INDICATOR_DIR = dp.OUTPUT_DIR
ENTRY_PER_TRADE = 2


def get_market_data(split, atr=False, macd=False, rsi=False):
    price = pd.read_csv(f"{PRICE_DIR}/{split}/raw_{split}.csv",
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

def init_trades(num_trades):
    return numpy.zeros(num_trades * ENTRY_PER_TRADE)

def run(split, num_trades, atr=False, macd=False, rsi=False):
    market_data = get_market_data(split, atr=atr, macd=macd, rsi=rsi)
    trades = init_trades(num_trades)
    return market_data, trades

