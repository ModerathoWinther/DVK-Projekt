import datetime

import MetaTrader5 as mt5
import pandas as pd

from action_space import Action, Direction, ACTION_SPACE

TIME_START = datetime.datetime.now()
TIME_END = TIME_START + datetime.timedelta(minutes=15)

mt5.initialize()

def get_market_data():
    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, 1)
    df = pd.DataFrame(rates)
    df = df.rename(columns={'tick_volume': 'volume'})
    final_df = df[['open', 'high', 'low', 'close', 'volume']]
    return final_df

def send_order(action):
    if action.direction == Direction.BUY:
        price = mt5.symbol_info_tick("XAUUSD").ask
        order_type = mt5.ORDER_TYPE_BUY
        sl = price - (price * action.sl)
        tp = price + (price * action.tp)
    elif action.direction == Direction.SELL:
        price = mt5.symbol_info_tick("XAUUSD").bid
        order_type = mt5.ORDER_TYPE_SELL
        sl = price + (price * action.sl)
        tp = price - (price * action.tp)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "XAUUSD",
        "type": order_type,
        "sl": sl,
        "tp": tp,
        "type_time": mt5.ORDER_TIME_SPECIFIED,
        "expiration": TIME_END
    }
    mt5.order_send(request)

print(TIME_START)
print(TIME_END)
print(get_market_data())
send_order(ACTION_SPACE[2])


# todo Load trained model from memory

# Loop

# use get_market_data to fetch last candlestick

# todo Translate in-data to format used in training (normalize, etc)

# todo Let model decide action to take

# use send_order to send orders via MetaTrader

# After loop
# todo Use history_deals_get or history_orders_get to extract results (not really sure if they work)