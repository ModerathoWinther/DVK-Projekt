import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime
import time
import os

# Constant configuration
TIMEFRAME = mt5.TIMEFRAME_M15
SYMBOL = "XAUUSD"
OUTPUT_DIR = "../data/raw"
TIMEZONE = pytz.timezone("Etc/UTC")
DATE_FROM = datetime(2016, 1, 1, tzinfo=TIMEZONE)
DATE_TO = datetime(2025, 12, 31, hour=23, tzinfo=TIMEZONE)
FILE_NAME = f"data/raw/{SYMBOL}_{TIMEFRAME}M_{DATE_FROM.date()}_{DATE_TO.date()}.csv"
SPLITS = ["train", "val", "test"]
#

# Change to ROOT dir
os.chdir("..")


# Downloads data via MetaTrader 5 and exports to .csv
def fetch_and_save_data():

    if not mt5.initialize():
        print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return

    selected = mt5.symbol_select(SYMBOL, True)
    if not selected:
        print(f"Failed to select {SYMBOL}")
        mt5.shutdown()
        return

    print(f"Synchronizing history for {SYMBOL}...")
    time.sleep(2)

    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, DATE_FROM, DATE_TO)

    if rates is None or len(rates) == 0:
        print(f"Error: No data retrieved. Error code: {mt5.last_error()}")
        mt5.shutdown()
        return

    mt5.shutdown()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'time': 'date', 'tick_volume': 'volume'})

    final_df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    actual_rows = len(final_df)

    final_df.to_csv(FILE_NAME, index=False)
    print(f"Successfully saved {actual_rows} rows to {FILE_NAME}")
    print(f"Data range: {final_df['date'].min()} to {final_df['date'].max()}")

# Parti
def partition_fetched_data():

    if not os.path.isfile(FILE_NAME):
        print(f"Error: Raw data file not found with file name {FILE_NAME}.")
        return

    df = pd.read_csv(FILE_NAME, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    train_end = pd.Timestamp('2022-12-31 23:59:59')
    val_end = pd.Timestamp('2023-12-31 23:59:59')


    train_df = df[df['date'] <= train_end].copy()
    val_df   = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
    test_df  = df[df['date'] > val_end].copy()

    for split in SPLITS:
        dirs = os.path.join(OUTPUT_DIR, split)
        os.makedirs(dirs, exist_ok=True)


    train_path = f"{OUTPUT_DIR}/train/raw_train.csv"
    val_path   = f"{OUTPUT_DIR}/val/raw_val.csv"
    test_path  = f"{OUTPUT_DIR}/test/raw_test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved splits to {OUTPUT_DIR}/X_path")
    return train_path, val_path, test_path

def run():

    if os.path.isfile(FILE_NAME):
        print("Raw files already exist.")
        return

    fetch_and_save_data()
    partition_fetched_data()

if __name__ == "__main__":
    run()
