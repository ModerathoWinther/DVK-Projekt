import sys
import os
import MetaTrader5 as mt5


print("--- MT5 PYTHON CHECK ---")
print(f"Python Path: {sys.executable}")
print(f"Is FinRL installed? {'finrl' in sys.modules or 'finrl' in os.listdir(os.path.dirname(sys.executable))}")

if not mt5.initialize():
    print("Failed to connect to MT5 Terminal")
else:
    print("Successfully connected to MT5 Terminal!")
    mt5.shutdown()