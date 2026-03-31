import sys


def main():

    print("Python venv check")
    print(f"Interpreter: {sys.executable}")

    try:
        import finrl
        print("FinRL status: INSTALLED")
    except ImportError:
        print("FinRL status: NOT FOUND")

    try:
        import MetaTrader5 as mt5
        print("MT5 status: INSTALLED")
    except ImportError:
        print("MT5 status: NOT FOUND")

main()