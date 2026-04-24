import pandas as pd
from data_process import Z_SCORE_INDICATOR_DIR, Z_SCORE_OHLCV_DIR, Z_SCORE_WICK_DIR, NORMAL_DIR as PRICE_DIR


def get_input_data(split, dataset):
    dataset_path = Z_SCORE_OHLCV_DIR if dataset == 'ohlcv' else Z_SCORE_WICK_DIR
    price = pd.read_csv(f"{dataset_path}/{split}.csv",
                        index_col="date", parse_dates=["date"])

    frames = [price, pd.read_csv(f"{Z_SCORE_INDICATOR_DIR}/{split}.csv",
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