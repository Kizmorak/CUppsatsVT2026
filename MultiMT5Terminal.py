import datetime
import os
import shutil

import numpy as np
import mplfinance as mpf
import pandas as pd
import MetaTrader5 as mt5
import multiprocessing
import time
import dataCollection
from dataCollection import ratesSymbol, ratesTimeFrame


# 1. Define the worker function
def run_trader(path, login, password, server):
    # Each process initializes its own connection
    if not mt5.initialize(path=path, login=login, password=password, server=server):
        print(f"Failed to initialize account {login}")
        return

    print(f"Worker started for account: {login}")
    last_min = datetime.datetime.now().minute

    try:
        while True:
            current_min = datetime.datetime.now().minute
            # Your trading logic goes here
            # Example: check prices or send orders
            if current_min != last_min:
                # --- DO SOMETHING HERE ---
                print(f"New minute detected! It is now {datetime.datetime.now().strftime('%H:%M')}")

                last_min = current_min

                ratesData = pd.DataFrame(mt5.copy_rates_from_pos(ratesSymbol, ratesTimeFrame, 0, 60))
                # TECHNICAL INDICATORS CALCULATIONS##################################################################################
                ratesData["time"] = pd.to_datetime(ratesData["time"], unit="s")
                ratesData = ratesData.set_index("time")
                # ratesData = ratesData.between_time(timeOfDayStart, timeOfDayEnd)
                ratesData = dataCollection.MACalculator(ratesData, dataCollection.MAWindowSize, dataCollection.MAPrice)
                ratesData = dataCollection.BollingerBandsCalculator(ratesData, dataCollection.BBPeriod,
                                                                    dataCollection.BBStandardDeviations)
                ratesData = dataCollection.RelativeStrengthIndexCalculator(ratesData, dataCollection.RSIPeriod)
                ratesData = dataCollection.OnBalanceVolume(ratesData)
                ratesData = dataCollection.AverageTrueRangeCalculator(ratesData, dataCollection.atrPeriod)
                # Prepares data for chart
                ratesData.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
                ####################################################################################################################
                print(ratesData.columns)

                ratesData = ratesData.tail(30)

                saveFolderName = os.path.join("inputGraph", str(login))
                if os.path.exists(saveFolderName):
                    shutil.rmtree(saveFolderName)
                    print(f"Deleted existing folder: {"datasetNew"}")
                os.makedirs(saveFolderName)

                dataCollection.makePlot(ratesData, saveFolderName)

                #GET MODEL PREDICTION HERE!
                direction = "buy"
                # GET MODEL PREDICTION HERE!

                atr = ratesData["averageTrueRange"].iloc[-1]

                if direction == "buy":
                    order_type = mt5.ORDER_TYPE_BUY
                    price = mt5.symbol_info_tick(ratesSymbol).ask  # Buy at the Ask
                    sl = price - (atr * 5)  # SL is below price
                    tp = price + (atr * 5)  # TP is above price

                elif direction == "sell":
                    order_type = mt5.ORDER_TYPE_SELL
                    price = mt5.symbol_info_tick(ratesSymbol).bid  # Sell at the Bid
                    sl = price + (atr * 5)  # SL is above price
                    tp = price - (atr * 5)  # TP is below price

                expiration_time = int(time.time() + 60)


                if order_type == mt5.ORDER_TYPE_BUY or order_type == mt5.ORDER_TYPE_SELL:
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": ratesSymbol,
                        "volume": 0.1,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 0,
                        "comment": "python script open",
                        "type_time": mt5.ORDER_TIME_SPECIFIED,
                        "expiration": expiration_time,
                        "type_filling": mt5.ORDER_FILLING_FOK,
                    }

                    tick = mt5.symbol_info_tick(ratesSymbol)
                    print(f"Buying {ratesSymbol} | Chart Price: {price} | Actual Spread: {tick.ask - tick.bid}")
                    # send a trading request
                    result = mt5.order_send(request)

                    #ERROR CODE FROM MT5
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print("2. order_send failed, retcode={}".format(result.retcode))
                        # request the result as a dictionary and display it element by element
                        result_dict = result._asdict()
                        for field in result_dict.keys():
                            print("   {}={}".format(field, result_dict[field]))
                            # if this is a trading request structure, display it element by element as well
                            if field == "request":
                                traderequest_dict = result_dict[field]._asdict()
                                for tradereq_filed in traderequest_dict:
                                    print("       traderequest: {}={}".format(tradereq_filed,
                                                                              traderequest_dict[tradereq_filed]))
                    # ERROR CODE FROM MT5

            time.sleep(1)  # Wait 1 second before next loop
    except KeyboardInterrupt:
        print(f"Shutting down {login}")
    finally:
        mt5.shutdown()

if __name__ == '__main__':
    # Define your account configurations
    account_configs = [
        {
            "path": r"N:\MT5Terminals\Terminal1\MetaTrader 5\terminal64.exe",
            "login": 5048694908, "password": "MlW_G6Mj", "server": "MetaQuotes-Demo"
        },
        {
            "path": r"N:\MT5Terminals\Terminal2\MetaTrader 5\terminal64.exe",
            "login": 5048694942, "password": "-1QeDlGw", "server": "MetaQuotes-Demo"
        },
        {
            "path": r"N:\MT5Terminals\Terminal3\MetaTrader 5\terminal64.exe",
            "login": 5048694953, "password": "!q4wIpQy", "server": "MetaQuotes-Demo"
        }
    ]

    processes = []

    # Launch a process for each config
    for config in account_configs:
        p = multiprocessing.Process(target=run_trader, kwargs=config)
        p.start()
        processes.append(p)

    # Keep the main script alive while workers run
    for p in processes:
        p.join()



# Connect to Account A (Terminal 1)
#mt5.initialize(path=r"N:\MT5Terminals\Terminal1\MetaTrader 5\terminal64.exe",
#               login=5048694908, password="MlW_G6Mj", server="MetaQuotes-Demo")
#E_LhU3Ih  (Read only password)
#mt5.shutdown()
#print("Terminal 1 online")

# Connect to Account B (Terminal 2)
#mt5.initialize(path=r"N:\MT5Terminals\Terminal2\MetaTrader 5\terminal64.exe",
#               login=5048694942, password="-1QeDlGw", server="MetaQuotes-Demo")
#I!Km0aWm  (Read only password)
#mt5.shutdown()
#print("Terminal 2 online")

# Connect to Account C (Terminal 3)
#mt5.initialize(path=r"N:\MT5Terminals\Terminal3\MetaTrader 5\terminal64.exe",
#               login=5048694953, password="!q4wIpQy", server="MetaQuotes-Demo")
#@bK0OdUm  (Read only password)
#mt5.shutdown()
#print("Terminal 3 online")