import numpy
import pandas as pd
import data_process as dp


PRICE_DIR = dp.INPUT_DIR
INDICATOR_DIR = dp.OUTPUT_DIR
ENTRY_PER_TRADE = 4

_train_mean = None
_train_std = None

def get_market_data(split, atr=False, macd=False, rsi=False, normalize=True):
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

    # calc Z-score: (value - mean) / std
    if normalize:
        global _train_mean, _train_std

        if split == "train":
            _train_mean = market_data.mean()
            _train_std = market_data.std()

            _train_mean.to_csv("data/processed/train_mean.csv")
            _train_std.to_csv("data/processed/train_std.csv")

        else:
            if _train_mean is None or _train_std is None:
                _train_mean = pd.read_csv("data/processed/train_mean.csv", index_col=0).squeeze()
                _train_std = pd.read_csv("data/processed/train_std.csv", index_col=0).squeeze()

        market_data = (market_data - _train_mean) / (_train_std + 1e-8)

    return market_data.to_numpy()

def init_trades(num_trades):
    return numpy.zeros(num_trades * ENTRY_PER_TRADE)

def run(split, num_trades, atr=False, macd=False, rsi=False):
    market_data = get_market_data(split, atr=atr, macd=macd, rsi=rsi)
    trades = init_trades(num_trades)
    return market_data, trades