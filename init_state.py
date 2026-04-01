import pandas as pd
import data_process as dp


PRICE_DIR = dp.INPUT_DIR
INDICATOR_DIR = dp.OUTPUT_DIR


def get_inputs(split, atr=False, macd=False, rsi=False):
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
    inputs = pd.concat(frames, axis=1).dropna()
    return inputs.to_numpy()

def run(split, atr=False, macd=False, rsi=False):
    inputs = get_inputs(split, atr=atr, macd=macd, rsi=rsi)
    return inputs

