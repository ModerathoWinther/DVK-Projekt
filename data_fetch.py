import MetaTrader5 as mt5
import pandas as pd



def fetch_and_save_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, n_bars=5000):
        if not mt5.initialize():
            print("MT5 initialize() failed, error code =", mt5.last_error())
            return

        print(f"Fetching {n_bars} bars for {symbol}...")


        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)

        mt5.shutdown()

        df = pd.DataFrame(rates)

        df['time'] = pd.to_datetime(df['time'], unit='s')

        df = df.rename(columns={'time': 'date', 'tick_volume': 'volume'})
        df['tic'] = symbol

        final_df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'tic']]

        file_name = f"data/{symbol}_{n_bars}_bars.csv"

        import os
        if not os.path.exists('data'):
            os.makedirs('data')

        final_df.to_csv(file_name, index=False)
        print(f"Successfully saved {len(final_df)} rows to {file_name}")


fetch_and_save_data()