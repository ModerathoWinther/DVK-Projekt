import os

import pandas as pd

import data_fetch
import indicators as ind


INPUT_DIR = data_fetch.OUTPUT_DIR
INDICATOR_DIR = os.path.join(data_fetch.DATA_DIR, "processed/indicators")
PRICE_OUTPUT = os.path.join(data_fetch.DATA_DIR, "processed/non-normalized/test.csv")
NORMALIZED_OUTPUT = os.path.join(data_fetch.DATA_DIR, "processed/normalized/train.csv")
WARMUP_ROWS = 33
DATASET_SPLITS = ["train", "val", "test"]
SYMBOL = "XAUUSD"


def load_split(split: str) -> pd.DataFrame:
    print(INPUT_DIR)
    path = os.path.join(INPUT_DIR, f"{split}/raw_{split}.csv")
    if not os.path.isfile(path):
        data_fetch.run()
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Loaded {split:>10}: {len(df):>6} rows  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df


def validate_ohlcv(df: pd.DataFrame, split: str) -> None:
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{split}] Missing columns: {missing}")

    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl:
        print(f"  WARNING [{split}]: {bad_hl} rows where high < low")

    dupes = df.duplicated(subset="date").sum()
    if dupes:
        print(f"  WARNING [{split}]: {dupes} duplicate timestamps — dropping")
        df.drop_duplicates(subset="date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    gaps = df["date"].diff().dropna()
    modal_gap = gaps.mode()[0]
    large_gaps = (gaps > modal_gap * 3).sum()
    if large_gaps:
        print(f"  WARNING [{split}]: {large_gaps} time gaps larger than 3× "
              f"the modal interval ({modal_gap}) — possible missing bars")


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ind.macd(df)
    ind.atr(df)
    ind.rsi(df)

    indicator_cols = ["macd", "macd_signal", "macd_histogram", "atr", "rsi"]
    missing = [c for c in indicator_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"indicators.py did not produce expected columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    result = df[["date"] + indicator_cols].copy()
    n_before = len(result)
    result = result.dropna().reset_index(drop=True)
    n_dropped = n_before - len(result)
    if n_dropped:
        print(f"    Dropped {n_dropped} NaN warmup rows")
    return result


def fit_normalization_params(df: pd.DataFrame) -> dict:
    indicator_cols = [c for c in df.columns if c != "date"]
    params = {}
    for col in indicator_cols:
        params[col] = {"min": df[col].min(), "max": df[col].max()}
    return params


def apply_normalization(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    normalized = df[["date"]].copy()
    indicator_cols = [c for c in df.columns if c != "date"]
    for col in indicator_cols:
        min_val = params[col]["min"]
        max_val = params[col]["max"]
        if max_val == min_val:
            normalized[col] = 0.0
        else:
            normalized[col] = (df[col] - min_val) / (max_val - min_val)
            normalized[col] = normalized[col].clip(0.0, 1.0)
    return normalized


def save_separate_indicator_files(df: pd.DataFrame, split: str) -> None:
    sep_dir = os.path.join(INDICATOR_DIR, split)
    os.makedirs(sep_dir, exist_ok=True)
    grouped = {
        "macd": ["macd", "macd_signal", "macd_histogram"],
    }
    grouped_cols = {col for cols in grouped.values() for col in cols}
    singles = [c for c in df.columns if c != "date" and c not in grouped_cols]

    for name, cols in grouped.items():
        path = os.path.join(sep_dir, f"{name}.csv")
        df[["date"] + cols].to_csv(path, index=False)
    for col in singles:
        path = os.path.join(sep_dir, f"{col}.csv")
        df[["date", col]].to_csv(path, index=False)


def compute_price_zscore_params(df_train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    price_cols = ["open", "high", "low", "close", "volume"]
    price_mean = df_train[price_cols].mean()
    price_std = df_train[price_cols].std()

    for col in price_cols:
        print(f"    {col:<8}  mean={price_mean[col]:>12.4f}   std={price_std[col]:>12.4f}")

    return price_mean, price_std


def apply_price_zscore(df: pd.DataFrame, price_mean: pd.Series, price_std: pd.Series) -> pd.DataFrame:
    price_cols = ["open", "high", "low", "close", "volume"]
    df_out = df[["date"] + price_cols].copy()
    for col in price_cols:
        df_out[col] = (df[col] - price_mean[col]) / (price_std[col] + 1e-8)
    return df_out

def save_prices(df: pd.DataFrame) -> None:
    prices = df.drop(df.index[:WARMUP_ROWS])
    os.makedirs(os.path.dirname(PRICE_OUTPUT), exist_ok=True)
    prices.to_csv(PRICE_OUTPUT, index=False)
    print(f"  Saved back-testing prices → {PRICE_OUTPUT}")

def save_normalized_prices(df: pd.DataFrame) -> None:
    normalized_prices = df.drop(df.index[:WARMUP_ROWS])
    os.makedirs(os.path.dirname(NORMALIZED_OUTPUT), exist_ok=True)
    normalized_prices.to_csv(NORMALIZED_OUTPUT, index=False)
    print(f"  Saved Z-score normalized training prices → {NORMALIZED_OUTPUT}")


def run():
    splits_raw = {}
    for split in DATASET_SPLITS:
        splits_raw[split] = load_split(split)

    for split, df in splits_raw.items():
        validate_ohlcv(df, split)

    price_mean, price_std = compute_price_zscore_params(splits_raw["train"])

    save_prices(splits_raw["test"])

    for split in DATASET_SPLITS:
        splits_raw[split] = apply_price_zscore(splits_raw[split], price_mean, price_std)

    save_normalized_prices(splits_raw["train"])

    for split, df in splits_raw.items():
        splits_ind = build_indicators(df.copy())
        save_separate_indicator_files(splits_ind, split)

if __name__ == "__main__":
    run()
