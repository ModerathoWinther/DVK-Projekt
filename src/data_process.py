import os
import pandas as pd
import data_fetch
import indicators as ind

INPUT_DIR = data_fetch.OUTPUT_DIR
PROCESSED_DIR = os.path.join(data_fetch.DATA_DIR, "processed")

NORMAL_DIR = os.path.join(PROCESSED_DIR, "normal")
NORMAL_INDICATOR_DIR = os.path.join(NORMAL_DIR, "indicators")

STATIONARY_DIR = os.path.join(PROCESSED_DIR, "stationary")
Z_SCORE_OHLCV_DIR = os.path.join(STATIONARY_DIR, "ohlcv-normalized")
Z_SCORE_WICK_DIR = os.path.join(STATIONARY_DIR, "wick-normalized")
Z_SCORE_INDICATOR_DIR = os.path.join(STATIONARY_DIR, "indicators")

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


def save_separate_indicator_files(df: pd.DataFrame, split: str):
    sep_dir = os.path.join(NORMAL_INDICATOR_DIR, split)
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

def compute_wick_zscore_params(df_wick: pd.DataFrame) -> tuple[pd.Series, pd.Series]:

    ref_mean = df_wick["trend"].mean()
    ref_std  = df_wick["trend"].std()

    vol_mean = df_wick["volume"].mean()
    vol_std  = df_wick["volume"].std()

    wick_mean = pd.Series({
        "high_wick": 0.0,
        "low_wick":  0.0,
        "trend":     ref_mean,
        "volume":    vol_mean,
    })
    wick_std = pd.Series({
        "high_wick": ref_std,
        "low_wick":  ref_std,
        "trend":     ref_std,
        "volume":    vol_std,
    })

    print("\n  Wick Z-score params (fit on train):")
    for col in ["high_wick", "low_wick", "trend", "volume"]:
        print(f"    {col:<12}  mean={wick_mean[col]:>12.6f}   std={wick_std[col]:>12.6f}")

    return wick_mean, wick_std

def compute_price_zscore_params(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    price_cols = ["open", "high", "low", "close", "volume"]
    price_mean = df[price_cols].mean()
    price_std = df[price_cols].std()

    for col in price_cols:
        print(f"    {col:<8}  mean={price_mean[col]:>12.4f}   std={price_std[col]:>12.4f}")

    return price_mean, price_std

def compute_indicator_zscore_params(df: pd.DataFrame,
                                    price_mean: pd.Series,
                                    price_std: pd.Series) -> tuple[pd.Series, pd.Series]:

    ref_mean = price_mean["close"]
    ref_std  = price_std["close"]

    price_unit_cols = ["atr", "macd", "macd_signal", "macd_histogram"]

    rsi_mean = df["rsi"].mean()
    rsi_std  = df["rsi"].std()

    ind_mean = pd.Series({
        "atr":            ref_mean,
        "macd":           ref_mean,
        "macd_signal":    ref_mean,
        "macd_histogram": ref_mean,
        "rsi":            rsi_mean,
    })
    ind_std = pd.Series({
        "atr":            ref_std,
        "macd":           ref_std,
        "macd_signal":    ref_std,
        "macd_histogram": ref_std,
        "rsi":            rsi_std,
    })

    print("\n  Indicator Z-score params (fit on train):")
    for col in ["atr", "macd", "macd_signal", "macd_histogram", "rsi"]:
        print(f"    {col:<16}  mean={ind_mean[col]:>12.4f}   std={ind_std[col]:>12.4f}")

    return ind_mean, ind_std

def apply_wick_zscore(df_wick: pd.DataFrame,
                      wick_mean: pd.Series,
                      wick_std: pd.Series) -> pd.DataFrame:
    cols = ["high_wick", "low_wick", "trend", "volume"]
    df_out = df_wick[["date"] + cols].copy()

    for col in cols:
        df_out[col] = (df_wick[col] - wick_mean[col]) / (wick_std[col] + 1e-8)

    return df_out

def apply_price_zscore(df: pd.DataFrame, price_mean: pd.Series, price_std: pd.Series) -> pd.DataFrame:
    price_cols = ["open", "high", "low", "close", "volume"]
    df_out = df[["date"] + price_cols].copy()

    ref_mean = price_mean["close"]
    ref_std  = price_std["close"] + 1e-8

    for col in ["open", "high", "low", "close"]:
        df_out[col] = (df[col] - ref_mean) / ref_std

    df_out["volume"] = (df["volume"] - price_mean["volume"]) / (price_std["volume"] + 1e-8)

    return df_out


def apply_indicator_zscore(df: pd.DataFrame,
                           ind_mean: pd.Series,
                           ind_std: pd.Series) -> pd.DataFrame:
    indicator_cols = ["atr", "macd", "macd_signal", "macd_histogram", "rsi"]
    df_out = df[["date"] + indicator_cols].copy()

    for col in indicator_cols:
        df_out[col] = (df[col] - ind_mean[col]) / (ind_std[col] + 1e-8)

    return df_out


def save_candlesticks(df: pd.DataFrame, split) -> None:
    os.makedirs(NORMAL_DIR, exist_ok=True)
    df.to_csv(f"{NORMAL_DIR}/{split}.csv", index=False)


def save_stationary_data(df: pd.DataFrame, split) -> None:
    os.makedirs(STATIONARY_DIR, exist_ok=True)
    df.to_csv(f"{STATIONARY_DIR}/{split}.csv", index=False)


def drop_warmup_rows(df: pd.DataFrame):
    return df.drop(df.index[:WARMUP_ROWS])

def save_frames_to_csv(df: pd.DataFrame, directory: str, split: str) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{split}.csv")
    df.to_csv(path, index=False)
    print(f"    Saved → {path}  ({len(df)} rows)")

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

    splits_trimmed = {
        split: drop_warmup_rows(df) for split, df in splits_raw.items()
    }

    for split, df in splits_trimmed.items():
        save_candlesticks(df, split)
        save_stationary_data(make_stationary(df), split)

    print("\n  Fitting normalisation params on train split...")
    price_mean, price_std = compute_price_zscore_params(splits_trimmed["train"])
    ind_mean, ind_std     = compute_indicator_zscore_params(
                                splits_indicators["train"], price_mean, price_std)

    train_wick = make_stationary(splits_trimmed["train"])
    wick_mean, wick_std   = compute_wick_zscore_params(train_wick)

    for split in DATASET_SPLITS:
        print(f"\n  Normalising {split} with train mean/std...")

        ohlcv_norm = apply_price_zscore(splits_trimmed[split], price_mean, price_std)
        save_frames_to_csv(ohlcv_norm, Z_SCORE_OHLCV_DIR, split)

        wick_df   = make_stationary(splits_trimmed[split])
        wick_norm = apply_wick_zscore(wick_df, wick_mean, wick_std)
        save_frames_to_csv(wick_norm, Z_SCORE_WICK_DIR, split)

        ind_trimmed = drop_warmup_rows(splits_indicators[split])
        ind_norm    = apply_indicator_zscore(ind_trimmed, ind_mean, ind_std)
        save_frames_to_csv(ind_norm, Z_SCORE_INDICATOR_DIR, split)


if __name__ == "__main__":
    run()
