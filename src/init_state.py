import pandas as pd
from data_process import OHLCV_NORMALIZED, INDICATOR_DIR, STATIONARY_DIR, NORMAL_DIR as PRICE_DIR

def get_input_data(split, dataset):
    dataset_path = OHLCV_NORMALIZED if dataset == 'ohlcv' else STATIONARY_DIR
    price = pd.read_csv(f"{dataset_path}/{split}.csv",
                        index_col="date", parse_dates=["date"])

    frames = [price, pd.read_csv(f"{INDICATOR_DIR}/{split}/atr.csv",
                                 index_col="date", parse_dates=["date"]),
              pd.read_csv(f"{INDICATOR_DIR}/{split}/macd.csv",
                          index_col="date", parse_dates=["date"]),
              pd.read_csv(f"{INDICATOR_DIR}/{split}/rsi.csv",
                          index_col="date", parse_dates=["date"])]

    input_data = pd.concat(frames, axis=1).dropna()
    print(f"INIT_STATE WITH: Dataset: {dataset}\t|\tinput_data cols: {input_data.columns}")
    return input_data.to_numpy()

def get_prices(split):
    price = pd.read_csv(f"{PRICE_DIR}/{split}.csv", index_col="date", parse_dates=["date"])
    price = price[['open', 'high', 'low', 'close']]
    return price.to_numpy()

def run(**params):
    return get_input_data(params.get('split'), params.get('data_format')), get_prices(params.get('split'))