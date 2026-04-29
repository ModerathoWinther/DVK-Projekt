import os
import json
import pandas as pd
import data_fetch
import indicators as ind

INPUT_DIR = data_fetch.OUTPUT_DIR
PROCESSED_DIR = os.path.join(data_fetch.DATA_DIR, "processed")

NON_NORMAL_DIR = os.path.join(PROCESSED_DIR, "non_normalized")
INDICATOR_DIR = os.path.join(NON_NORMAL_DIR, "indicators")

NORMALIZED_DIR = os.path.join(PROCESSED_DIR, "normalized")
Z_SCORE_OHLCV_DIR = os.path.join(NORMALIZED_DIR, "ohlcv")
Z_SCORE_WICK_DIR = os.path.join(NORMALIZED_DIR, "wick")
Z_SCORE_INDICATOR_DIR = os.path.join(NORMALIZED_DIR, "indicators")

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


def to_wick_format(df: pd.DataFrame) -> pd.DataFrame:
    wick = pd.DataFrame({
        "date": df["date"],
        "high_wick": df['high'] / df['open'],
        "low_wick": df['open'] / df['low'],
        "trend": df['close'] / df['open'],
        "volume": df['volume'],
    }, index=df.index)
    return wick


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


def save_separate_indicator_files(df: pd.DataFrame, split: str):
    sep_dir = os.path.join(INDICATOR_DIR, split)
    os.makedirs(sep_dir, exist_ok=True)
    grouped = {
        "macd": ["macd", "macd_signal", "macd_histogram"],
    }
    grouped_cols = {col for cols in grouped.values() for col in cols}
    singles = [c for c in df.columns if c != "date" and c not in grouped_cols]

    indicator_dfs = {}
    for name, cols in grouped.items():
        path = os.path.join(sep_dir, f"{name}.csv")
        df[["date"] + cols].to_csv(path, index=False)
        indicator_dfs = df
    for col in singles:
        path = os.path.join(sep_dir, f"{col}.csv")
        df[["date", col]].to_csv(path, index=False)

    return indicator_dfs


def compute_zscore_params(ohlc_df: pd.DataFrame, wick_df: pd.DataFrame, ind_df: pd.DataFrame):
    ohlc_values = ohlc_df[['open', 'high', 'low', 'close']].values.flatten()
    wick_values = wick_df[['high_wick', 'low_wick', 'trend']].values.flatten()

    params = {
        "ohlc": (ohlc_values.mean(), ohlc_values.std()),
        "wick": (wick_values.mean(), wick_values.std()),
        "volume": (ohlc_df['volume'].mean(), ohlc_df['volume'].std()),
        "atr": (ind_df['atr'].mean(), ind_df['atr'].std()),
        "macd": (ind_df['macd'].mean(), ind_df['macd'].std()),
        "macd_signal": (ind_df['macd_signal'].mean(), ind_df['macd_signal'].std()),
        "macd_histogram": (ind_df['macd_histogram'].mean(), ind_df['macd_histogram'].std()),
    }
    return params


def apply_zscore(df: pd.DataFrame, params, param_key: str, cols: list[str]):
    mean, std = params.get(param_key)
    df[cols] = (df[cols] - mean) / std + 1e-8


def apply_wick_zscore(df: pd.DataFrame, params):
    cols = ["high_wick", "low_wick", "trend"]
    df_wick = df[["date"] + cols].copy().set_index("date")
    df_vol = df[["date", "volume"]].copy().set_index("date")

    apply_zscore(df_wick, params, "wick", cols)
    apply_zscore(df_vol, params, "volume", ["volume"])
    return pd.concat([df_wick, df_vol], axis=1)


def apply_ohlcv_zscore(df: pd.DataFrame, params):
    cols = ["open", "high", "low", "close"]
    df_ohlc = df[["date"] + cols].copy().set_index("date")
    df_vol = df[["date", "volume"]].copy().set_index("date")

    apply_zscore(df_ohlc, params, "ohlc", cols)
    apply_zscore(df_vol, params, "volume", ["volume"])
    return pd.concat([df_ohlc, df_vol], axis=1)


def apply_indicator_zscore(df: pd.DataFrame, params):
    atr = df[["date", "atr"]].copy().set_index("date")
    macd = df[["date", "macd"]].copy().set_index("date")
    macd_signal = df[["date", "macd_signal"]].copy().set_index("date")
    macd_histogram = df[["date", "macd_histogram"]].copy().set_index("date")
    rsi = df[["date", "rsi"]].copy().set_index("date")

    apply_zscore(atr, params, "atr", ["atr"])
    apply_zscore(macd, params, "macd", ["macd"])
    apply_zscore(macd_signal, params, "macd_signal", ["macd_signal"])
    apply_zscore(macd_histogram, params, "macd_histogram", ["macd_histogram"])
    # RSI is divided by 100 instead
    return pd.concat([atr, macd, macd_signal, macd_histogram, rsi], axis=1)


def drop_warmup_rows(df: pd.DataFrame):
    return df.drop(df.index[:WARMUP_ROWS])


def save_frames_to_csv(df: pd.DataFrame, directory: str, split: str) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{split}.csv")
    df.to_csv(path)
    print(f"    Saved → {path}  ({len(df)} rows)")


def save_zscore_params_to_json(params):
    with open(f'{NORMALIZED_DIR}/zscores.json', 'w') as f:
        json.dump(params, f, ensure_ascii=False)


def run():
    splits_raw = {}
    for split in DATASET_SPLITS:
        splits_raw[split] = load_split(split)

    for split, df in splits_raw.items():
        validate_ohlcv(df, split)

    splits_indicators = {}
    for split, df in splits_raw.items():
        print(f"\n  Building indicators for {split}...")
        splits_indicators[split] = build_indicators(df.copy())
        save_separate_indicator_files(splits_indicators[split], split)

    splits_trimmed = {
        split: drop_warmup_rows(df) for split, df in splits_raw.items()
    }

    for split, df in splits_trimmed.items():
        save_frames_to_csv(df, NON_NORMAL_DIR, split)

    print("\n  Fitting normalisation params on train split...")
    train_wick = to_wick_format(splits_trimmed["train"])
    params = compute_zscore_params(splits_trimmed["train"], train_wick, splits_indicators["train"])
    save_zscore_params_to_json(params)

    for split in DATASET_SPLITS:
        print(f"\n  Normalising {split} with train mean/std...")

        ohlcv_norm = apply_ohlcv_zscore(splits_trimmed[split], params)
        save_frames_to_csv(ohlcv_norm, Z_SCORE_OHLCV_DIR, split)

        wick_df = to_wick_format(splits_trimmed[split])
        wick_norm = apply_wick_zscore(wick_df, params)
        save_frames_to_csv(wick_norm, Z_SCORE_WICK_DIR, split)

        ind_trimmed = drop_warmup_rows(splits_indicators[split])
        ind_norm = apply_indicator_zscore(ind_trimmed, params)
        # RSI is normalized by dividing by 100
        ind_norm["rsi"] = ind_norm["rsi"] / 100
        save_frames_to_csv(ind_norm, Z_SCORE_INDICATOR_DIR, split)


if __name__ == "__main__":
    run()
