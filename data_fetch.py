import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime
import time
import os


def fetch_and_save_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M1):

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

    date_from = datetime(2016, 1, 5, tzinfo=timezone)
    date_to = datetime(2017, 12, 31, hour=23, tzinfo=timezone)

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

    if not os.path.exists('data'):
        os.makedirs('data')

    actual_rows = len(final_df)
    file_name = f"data/{symbol}_{timeframe}_{actual_rows}_rows.csv"

    final_df.to_csv(file_name, index=False)
    print(f"Successfully saved {actual_rows} rows to {file_name}")
    print(f"Data range: {final_df['date'].min()} to {final_df['date'].max()}")


fetch_and_save_data("XAUUSD", mt5.TIMEFRAME_M1)