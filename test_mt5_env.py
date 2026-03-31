from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
import indicators as ind

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

print("terminal info: ", mt5.terminal_info())
print("version: ", mt5.version())

xauusd_ticks = mt5.copy_rates_from("XAUUSD", mt5.TIMEFRAME_H1, datetime(2020, 1, 10), 100)
df = pd.DataFrame(xauusd_ticks)
ind.macd(df)
ind.atr(df)

print(df)

mt5.shutdown()