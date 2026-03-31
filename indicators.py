import talib as ta


def macd(dataframe):
    dataframe['macd'], dataframe['macd_signal'], dataframe['macd_histogram'] = ta.MACD(
        dataframe['close'],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9)