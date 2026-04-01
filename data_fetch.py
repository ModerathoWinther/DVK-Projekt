import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime
import time
import os


def fetch_and_save_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M15):

    if not mt5.initialize():
        print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return

    selected = mt5.symbol_select(symbol, True)
    if not selected:
        print(f"Failed to select {symbol}")
        mt5.shutdown()
        return

    print(f"Synchronizing history for {symbol}...")
    time.sleep(2)


    timezone = pytz.timezone("Etc/UTC")

    date_from = datetime(2016, 1, 1, tzinfo=timezone)
    date_to = datetime(2025, 12, 31, hour=23, tzinfo=timezone)

    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    if rates is None or len(rates) == 0:
        print(f"Error: No data retrieved. Error code: {mt5.last_error()}")
        mt5.shutdown()
        return

    mt5.shutdown()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'time': 'date', 'tick_volume': 'volume'})
    df['tick'] = symbol

    final_df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'tick']]

    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')

    actual_rows = len(final_df)
    fn = fn = f"data/raw/{symbol}_{timeframe}M_{date_from.date()}_{date_to.date()}.csv"

    final_df.to_csv(fn, index=False)
    print(f"Successfully saved {actual_rows} rows to {fn}")
    print(f"Data range: {final_df['date'].min()} to {final_df['date'].max()}")
    return fn

def partition_fetched_data():

    if not os.path.isfile(file_name):
        print(f"Error: Raw data file not found with file name {file_name}.")
        return

    df = pd.read_csv(file_name, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    train_end = pd.Timestamp('2022-12-31 23:59:59')
    val_end = pd.Timestamp('2023-12-31 23:59:59')


    train_df = df[df['date'] <= train_end].copy()
    val_df   = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
    test_df  = df[df['date'] > val_end].copy()

    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)

    symbol = "XAUUSD"

    train_path = f"{processed_dir}/train.csv"
    val_path   = f"{processed_dir}/validation.csv"
    test_path  = f"{processed_dir}/test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved splits to {processed_dir}/")
    return train_path, val_path, test_path

file_name = fetch_and_save_data("XAUUSD", mt5.TIMEFRAME_M15)
partition_fetched_data()
