import talib


def atr(dataframe):
    dataframe['atr'] = talib.ATR(
        dataframe['high'],
        dataframe['low'],
        dataframe['close'])

def macd(dataframe):
    dataframe['macd'], dataframe['macd_signal'], dataframe['macd_histogram'] = talib.MACD(
        dataframe['close'],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9)

def rsi(dataframe):
    return 0