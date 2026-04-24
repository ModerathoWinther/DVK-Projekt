import os

import pandas as pd

import data_fetch
import indicators as ind

INPUT_DIR = data_fetch.OUTPUT_DIR
NORMAL_DIR = os.path.join(data_fetch.DATA_DIR, "processed/normal")
STATIONARY_DIR = os.path.join(data_fetch.DATA_DIR, "processed/stationary")
OHLCV_NORMALIZED = os.path.join(STATIONARY_DIR, "ohlcv-normalized")
INDICATOR_DIR = os.path.join(data_fetch.DATA_DIR, "processed/indicators")

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


def make_stationary(df: pd.DataFrame) -> pd.DataFrame:
    stationary = pd.DataFrame({
        "date": df["date"],
        "high_wick": df['high'] - df['open'],
        "low_wick": df['open'] - df['low'],
        "trend": df['close'] - df['open'],
        "volume": df['volume'],
    }, index=df.index)
    return stationary


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


def compute_price_zscore_params(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    price_cols = ["open", "high", "low", "close", "volume"]
    price_mean = df[price_cols].mean()
    price_std = df[price_cols].std()

    for col in price_cols:
        print(f"    {col:<8}  mean={price_mean[col]:>12.4f}   std={price_std[col]:>12.4f}")

    return price_mean, price_std


def apply_price_zscore(df: pd.DataFrame, price_mean: pd.Series, price_std: pd.Series) -> pd.DataFrame:
    price_cols = ["open", "high", "low", "close", "volume"]
    df_out = df[["date"] + price_cols].copy()
    for col in price_cols:
        df_out[col] = (df[col] - price_mean[col]) / (price_std[col] + 1e-8)
    return df_out


def save_candlesticks(df: pd.DataFrame, split) -> None:
    os.makedirs(NORMAL_DIR, exist_ok=True)
    df.to_csv(f"{NORMAL_DIR}/{split}.csv", index=False)


def save_stationary_data(df: pd.DataFrame, split) -> None:
    os.makedirs(STATIONARY_DIR, exist_ok=True)
    df.to_csv(f"{STATIONARY_DIR}/{split}.csv", index=False)


def drop_warmup_rows(df: pd.DataFrame):
    return df.drop(df.index[:WARMUP_ROWS])

def save_frames_to_csv(df: pd.DataFrame, dir, split) -> None:
    os.makedirs(dir, exist_ok=True)
    df.to_csv(f'{dir}/{split}.csv', index=False)

def run():
    splits_raw = {}
    for split in DATASET_SPLITS:
        splits_raw[split] = load_split(split)

    for split, df in splits_raw.items():
        validate_ohlcv(df, split)

    for split in DATASET_SPLITS:
        without_warmup = drop_warmup_rows(splits_raw[split])
        save_candlesticks(without_warmup, split)
        save_stationary_data(make_stationary(without_warmup), split)

    for split, df in splits_raw.items():
        splits_ind = build_indicators(df.copy())
        save_separate_indicator_files(splits_ind, split)

    for split in DATASET_SPLITS:
        df_ohlcv = drop_warmup_rows(splits_raw[split])
        price_mean, price_std = compute_price_zscore_params(df_ohlcv)
        normalized_df = apply_price_zscore(df_ohlcv, price_mean=price_mean, price_std=price_std)
        save_frames_to_csv(normalized_df, f"{OHLCV_NORMALIZED}", split)
        print(f'Saving split: {split} of df: {normalized_df} to path :{OHLCV_NORMALIZED} ')



if __name__ == "__main__":
    run()
